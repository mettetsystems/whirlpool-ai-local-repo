import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src import credentials
from src.settings import read_settings, save_settings, PATH_KEYS
from src.model_repo.hf_client import HuggingFaceClient
from src.inference.server_builder import InferenceConfig, InferenceServerBuilder


@pytest.fixture
def vault():
    values = {}
    store = MagicMock()
    store.set_password.side_effect = lambda service, user, value: values.__setitem__((service, user), value)
    store.get_password.side_effect = lambda service, user: values.get((service, user))
    with patch('src.credentials.backend', return_value=store):
        yield store


def defaults(tmp_path):
    return {key: str(tmp_path / key) for key in PATH_KEYS}


def test_save_migrates_legacy_token_to_keyring(tmp_path, vault):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hf_token': 'test-secret', 'custom': 'keep'}))
    values = save_settings(config, defaults(tmp_path))
    assert values['hf_token'] == 'test-secret'
    assert 'test-secret' not in config.read_text()
    assert 'hf_token' not in json.loads(config.read_text())
    assert read_settings(config)['hf_token'] == 'test-secret'
    assert read_settings(config)['custom'] == 'keep'
    vault.set_password.assert_called_once_with(credentials.SERVICE, credentials.account(config), 'test-secret')
    assert config.stat().st_mode & 0o777 == 0o600
    assert 'credentials_path' not in PATH_KEYS


def test_failed_keyring_save_keeps_config_unchanged(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"model_storage_path": "./models"}')
    original = config.read_text()
    with patch('src.credentials.backend', side_effect=RuntimeError('unavailable')):
        with pytest.raises(RuntimeError, match='No plaintext'):
            save_settings(config, {**defaults(tmp_path), 'hf_token': 'new-secret'})
    assert config.read_text() == original
    assert not any('new-secret' in file.read_text() for file in tmp_path.rglob('*.json'))


def test_keyring_unavailable_does_not_clear_existing_token(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"credential_backend": "secret-service"}')
    with patch('src.credentials.backend', side_effect=RuntimeError('locked')):
        assert 'credential_error' in read_settings(config)
        with pytest.raises(RuntimeError, match='Unlock GNOME'):
            save_settings(config, {**defaults(tmp_path), 'hf_token': ''})
    assert json.loads(config.read_text()) == {'credential_backend': 'secret-service'}


def test_profile_credentials_are_isolated(tmp_path, vault):
    first, second = tmp_path / 'one.json', tmp_path / 'two.json'
    credentials.set_token(first, 'one')
    credentials.set_token(second, 'two')
    assert credentials.get_token(first) == 'one'
    assert credentials.get_token(second) == 'two'


def test_new_download_directory_preserves_model_registrations(tmp_path, vault):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'model_storage_path': str(tmp_path / 'old')}))
    client = HuggingFaceClient(str(config))
    source = tmp_path / 'existing-model'
    source.mkdir()
    (source / 'config.json').write_text('{}')
    original = client.add_local_model(str(source))
    client.update_settings({**defaults(tmp_path), 'hf_token': 'new-secret'})
    assert client.get_token() == 'new-secret'
    assert client._model_storage_path == tmp_path / 'model_storage_path'
    assert client.list_models() == [original]
    assert source.exists()
    restarted = HuggingFaceClient(str(config))
    assert restarted.list_models() == [original]
    assert restarted.get_token() == 'new-secret'


def test_env_token_overrides_keyring(tmp_path, vault, monkeypatch):
    config = tmp_path / 'config.json'
    save_settings(config, {**defaults(tmp_path), 'hf_token': 'saved-secret'})
    monkeypatch.setenv('HF_TOKEN', 'env-secret')
    assert HuggingFaceClient(str(config)).get_token() == 'env-secret'
    assert read_settings(config)['hf_token'] == 'saved-secret'


def test_export_uses_chosen_directory_and_image(tmp_path):
    builder = InferenceServerBuilder()
    config = InferenceConfig('model', str(tmp_path), inference_image='example/custom:v1')
    def save(command, **kwargs):
        Path(command[3]).write_bytes(b'image archive')
        return MagicMock(returncode=0)
    with patch.object(builder, '_get_compose_command', return_value=['podman-compose']), patch(
            'src.inference.server_builder.subprocess.run', side_effect=save) as run:
        result = builder.export_image(config, str(tmp_path / 'exports'))
    assert result.parent == tmp_path / 'exports'
    assert result.read_bytes() == b'image archive'
    assert run.call_args.args[0][-1] == 'example/custom:v1'
    assert not list(result.parent.glob('*.partial'))


def test_export_failure_leaves_no_archive(tmp_path):
    builder = InferenceServerBuilder()
    with patch.object(builder, '_get_compose_command', return_value=['docker', 'compose']), patch(
            'src.inference.server_builder.subprocess.run', return_value=MagicMock(returncode=1, stderr='failed')):
        with pytest.raises(RuntimeError, match='Image export failed'):
            builder.export_image(InferenceConfig('model', str(tmp_path)), str(tmp_path))
    assert not list(tmp_path.glob('*.tar'))
