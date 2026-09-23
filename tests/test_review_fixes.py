"""Regression tests for directory ownership, quotas, cloud parsing and launching."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.model_repo.hf_client import HuggingFaceClient
from src.model_repo.manifest_validator import ManifestValidator, ManifestValidationError
from src.data_parsers.parser_factory import OCIParser, DataSourceConfig
from src.runtime import colibri_launcher as launcher
from src.inference.server_builder import InferenceConfig, InferenceServerBuilder


@pytest.fixture
def client(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hf_token':'test-only', 'model_storage_path':str(tmp_path / 'models')}))
    client = HuggingFaceClient(str(config))
    client.api = MagicMock()
    client.api.model_info.return_value = SimpleNamespace(
        id='org/model', author='org', likes=0, downloads=0, tags=[], pipeline_tag=None,
        sha='a' * 40, siblings=[SimpleNamespace(size=4)])
    return client


def fake_download(**kwargs):
    path = Path(kwargs['local_dir'])
    (path / 'weights.bin').write_bytes(b'data')
    return str(path)


def test_colliding_ids_get_independent_directories(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=fake_download):
        first = client.download_model('team_a/model')
        second = client.download_model('team/a_model')
    assert first['local_path'] != second['local_path']
    assert client.delete_model(first['model_id'])
    assert (Path(second['local_path']) / 'weights.bin').read_bytes() == b'data'


def test_legacy_shared_folder_is_preserved_and_redownload_rejected(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=fake_download):
        first = client.download_model('org/one')
    client._update_manifest('org/two', first)
    with patch('src.model_repo.hf_client.snapshot_download') as download:
        with pytest.raises(ValueError, match='overlaps'):
            client.download_model('org/one')
        download.assert_not_called()
    client.delete_model('org/one')
    assert (Path(first['local_path']) / 'weights.bin').exists()


def test_explicit_destination_cannot_overwrite_another_model(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=fake_download):
        first = client.download_model('org/one')
        with pytest.raises(ValueError, match='overlaps'):
            client.download_model('org/two', local_dir=first['local_path'])


def test_resume_keeps_registered_directory_and_pins_revision(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=fake_download) as download:
        first = client.download_model('org/one')
        second = client.download_model('org/one')
        assert first['local_path'] == second['local_path']
        assert download.call_args.kwargs['revision'] == 'a' * 40


def test_quota_preflight_blocks_without_download(client, monkeypatch):
    monkeypatch.setattr(ManifestValidator, 'MAX_STORAGE_QUOTA_BYTES', 3)
    with patch('src.model_repo.hf_client.snapshot_download') as download:
        with pytest.raises(ManifestValidationError, match='Insufficient storage'):
            client.download_model('org/model')
        download.assert_not_called()
    assert client.list_models() == []


def test_metadata_without_size_fails_closed(client):
    client.api.model_info.return_value.siblings[0].size = None
    with patch('src.model_repo.hf_client.snapshot_download') as download:
        with pytest.raises(ManifestValidationError, match='determine model size'):
            client.download_model('org/model')
        download.assert_not_called()


def test_partial_downloads_count_toward_quota(client, monkeypatch):
    partial = client._model_storage_path / 'partial'
    partial.mkdir()
    (partial / 'weights.part').write_bytes(b'1234')
    monkeypatch.setattr(ManifestValidator, 'MAX_STORAGE_QUOTA_BYTES', 7)
    with patch('src.model_repo.hf_client.snapshot_download') as download:
        with pytest.raises(ManifestValidationError):
            client.download_model('org/model')
        download.assert_not_called()


def test_quota_counts_shared_files_once(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=fake_download):
        entry = client.download_model('org/one')
    client._update_manifest('org/two', entry)
    validator = ManifestValidator(str(client._manifest_path))
    assert validator._get_current_storage_usage() == 4
    assert validator.get_storage_usage()['quota_bytes'] == 1024 ** 4


def test_oci_uses_sdk_signature_and_real_stream(tmp_path):
    class Storage:
        def list_objects(self, namespace_name, bucket_name, prefix):
            assert namespace_name == 'namespace'
            return SimpleNamespace(data=SimpleNamespace(objects=[SimpleNamespace(name='data.txt', size=5)]))
        def get_object(self, namespace_name, bucket_name, object_name):
            assert namespace_name == 'namespace'
            response = requests.Response()
            response._content = b'hello'
            response._content_consumed = True
            return SimpleNamespace(data=response)
    sdk = SimpleNamespace(pagination=SimpleNamespace(list_call_get_all_results=lambda call, **kwargs: call(**kwargs)))
    storage = SimpleNamespace(ObjectStorageClient=lambda config: Storage())
    parser = OCIParser()
    parser._client = {'user':'not-the-namespace'}
    with patch.dict(sys.modules, {'oci':sdk, 'oci.object_storage':storage}):
        stats = parser.parse(DataSourceConfig('oci', '', oci_namespace='namespace', oci_bucket='bucket'), tmp_path)
    assert stats['errors'] == []
    assert stats['files_processed'] == 1
    assert (tmp_path / 'data.txt').read_bytes() == b'hello'


@pytest.mark.parametrize('runtime, expected', [('docker',['docker','compose']), ('podman',['podman-compose'])])
def test_compose_command_and_environment(runtime, expected, monkeypatch):
    monkeypatch.setenv('EXISTING_SETTING','preserve')
    with patch.object(launcher.shutil, 'which', return_value='/usr/bin/podman-compose'), patch.object(launcher, 'run_command') as run:
        launcher.run_container(runtime, 'deployment.json', {'MODEL_PATH':'/folder with spaces'})
    assert run.call_args.args[0] == expected + ['-f','deployment.json','up','--detach']
    assert run.call_args.kwargs['env']['MODEL_PATH'] == '/folder with spaces'
    assert run.call_args.kwargs['env']['EXISTING_SETTING'] == 'preserve'


def test_launcher_passes_built_image_and_health_port(tmp_path):
    compose = tmp_path / 'compose.yaml'
    compose.write_text('services: {}')
    with patch.object(launcher,'build_image') as build, patch.object(launcher,'run_container') as run, patch.object(launcher,'health_check',return_value=True) as health:
        assert launcher.launch_colibri('podman','custom/image:v1',compose,{'HOST_PORT':'8123'})
    assert build.call_args.args[1] == 'custom/image:v1'
    assert run.call_args.args[2]['INFERENCE_IMAGE'] == 'custom/image:v1'
    health.assert_called_once_with('podman', port=8123)


def test_fedora_gpu_config_uses_cdi_and_selinux_mount(tmp_path):
    config = InferenceConfig('model',str(tmp_path),container_runtime='podman',device='cuda',gpu_count=2)
    service = json.loads(InferenceServerBuilder().generate_config(config))['services']['inference-server']
    assert service['devices'] == ['nvidia.com/gpu=0','nvidia.com/gpu=1']
    assert 'deploy' not in service
    assert service['volumes'][0]['bind']['selinux'] == 'z'
    assert service['volumes'][0]['read_only'] is True


def test_build_chooses_podman_config_from_selected_engine(tmp_path):
    builder = InferenceServerBuilder(str(tmp_path / 'compose.json'))
    with patch.object(builder,'_get_compose_command',return_value=['podman-compose']), patch('src.inference.server_builder.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')):
        builder.build_container(InferenceConfig('model',str(tmp_path),device='cuda'))
    service = json.loads((tmp_path / 'compose.json').read_text())['services']['inference-server']
    assert service['devices'] == ['nvidia.com/gpu=0']
