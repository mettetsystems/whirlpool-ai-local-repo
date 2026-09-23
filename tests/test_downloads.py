import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from src.model_repo.hf_client import HuggingFaceClient
from src.model_repo.download_state import DownloadState
from src.model_repo.download_process import DownloadPaused
from src.model_repo.manifest_validator import ManifestValidator
from src.ui.model_browser import DownloadModelDialog
from src.ui.downloads_tab import DownloadsTab, DownloadThread


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def client(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'hf_token':'test-only-token','model_storage_path':str(tmp_path/'models')}))
    client = HuggingFaceClient(str(config))
    client.api = MagicMock()
    client.api.model_info.return_value = SimpleNamespace(id='org/model', author='org', likes=0, downloads=0,
        tags=[], pipeline_tag=None, sha='a'*40, siblings=[SimpleNamespace(size=10, rfilename='weights.bin')])
    return client


def interrupt_download(**kwargs):
    cache = Path(kwargs['local_dir']) / '.cache/huggingface/download'
    cache.mkdir(parents=True)
    (cache/'weights.etag.incomplete').write_bytes(b'1234')
    raise ConnectionError('connection lost')


def test_resume_after_restart_preserves_path_and_revision(client):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=interrupt_download):
        with pytest.raises(ConnectionError):
            client.download_model('org/model')
    state = DownloadState(client._model_storage_path)
    failed = state.get('org/model')
    assert failed['status'] == 'failed'
    assert client.list_models() == []
    restarted = HuggingFaceClient(str(client.auth_config_path))
    restarted.api = client.api
    def complete(**kwargs):
        assert kwargs['local_dir'] == failed['local_path']
        assert kwargs['revision'] == 'a'*40
        assert (Path(kwargs['local_dir'])/'.cache/huggingface/download/weights.etag.incomplete').exists()
        return kwargs['local_dir']
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=complete):
        restarted.download_model('org/model')
    assert restarted.api.model_info.call_args.kwargs['revision'] == 'a'*40
    assert state.get('org/model')['status'] == 'completed'
    assert restarted.model_exists('org/model')
    assert 'test-only-token' not in state.path('org/model').read_text()


def test_resume_quota_reserves_remaining_bytes(client, monkeypatch):
    with patch('src.model_repo.hf_client.snapshot_download', side_effect=interrupt_download):
        with pytest.raises(ConnectionError):
            client.download_model('org/model')
    monkeypatch.setattr(ManifestValidator,'MAX_STORAGE_QUOTA_BYTES',10)
    with patch('src.model_repo.hf_client.snapshot_download',return_value='/unused') as download:
        client.download_model('org/model')
    download.assert_called_once()


def test_unowned_partial_directory_still_rejected(client):
    import hashlib
    folder=client._model_storage_path/hashlib.sha256(b'org/model').hexdigest()
    folder.mkdir()
    (folder/'unrelated').write_text('keep')
    with pytest.raises(ValueError,match='must be empty'):
        client.download_model('org/model')


def test_pause_preserves_record_and_does_not_register_model(client):
    with patch('src.model_repo.hf_client.run_download',side_effect=DownloadPaused('Paused')):
        with pytest.raises(DownloadPaused):
            client.download_model('org/model',progress_callback=lambda event:None)
    assert DownloadState(client._model_storage_path).get('org/model')['status'] == 'paused'
    assert not client.model_exists('org/model')


def test_progress_callback_is_forwarded_and_token_is_only_in_request(client):
    events=[]
    def transfer(request,callback,stop_event):
        assert request['token']=='test-only-token'
        callback({'bytes':4,'total':10,'type':'progress','file':'weights.bin','speed':2,'eta':3})
        return request['local_dir']
    with patch('src.model_repo.hf_client.run_download',side_effect=transfer):
        client.download_model('org/model',progress_callback=events.append)
    assert events[0]['bytes']==4
    assert 'test-only-token' not in DownloadState(client._model_storage_path).path('org/model').read_text()


def test_queued_record_can_begin(client):
    DownloadState(client._model_storage_path).save({'model_id':'org/model','status':'queued'})
    with patch('src.model_repo.hf_client.snapshot_download',return_value='/unused'):
        client.download_model('org/model')
    assert client.model_exists('org/model')


def test_single_search_only_downloads_selected_result(client,app):
    dialog=DownloadModelDialog(client)
    dialog.searched=True
    dialog._search_ready([{'id':'org/one'},{'id':'org/two'}])
    assert not dialog.download_button.isEnabled()
    dialog.search_results.setCurrentRow(1)
    assert dialog.get_model_ids()==['org/two']
    assert dialog.download_button.isEnabled()
    dialog.close()


def test_batch_only_checked_models_and_choices_survive_search(client,app):
    dialog=DownloadModelDialog(client,batch=True)
    dialog._search_ready([{'id':'org/one'},{'id':'org/two'}])
    dialog.search_results.setCurrentRow(1)
    assert dialog.get_model_ids()==[]
    dialog.search_results.item(0).setCheckState(Qt.Checked)
    assert dialog.get_model_ids()==['org/one']
    dialog.search_results.clear()
    dialog._search_ready([{'id':'org/three'}])
    dialog.search_results.item(0).setCheckState(Qt.Checked)
    assert dialog.get_model_ids()==['org/one','org/three']
    dialog._clear_checked()
    assert not dialog.download_button.isEnabled()
    dialog.close()


def test_batch_failures_do_not_abort_other_selected_models(client,app):
    worker=DownloadThread(client,['org/one','org/two'])
    calls=[]
    def download(model_id,**kwargs):
        calls.append(model_id)
        if model_id=='org/one':
            raise RuntimeError('Failed test-only-token')
        return {'model_id':model_id}
    with patch.object(client,'download_model',side_effect=download):
        worker.run()
    assert calls==['org/one','org/two']
    assert 'test-only-token' not in DownloadState(client._model_storage_path).path('org/one').read_text()


def test_batch_pause_stops_remaining_models(client,app):
    worker=DownloadThread(client,['org/one','org/two'])
    with patch.object(client,'download_model',side_effect=DownloadPaused('Paused')) as download:
        worker.run()
    assert download.call_count==1


def test_download_tracker_progress_and_interrupted_state(client,app):
    DownloadState(client._model_storage_path).save({'model_id':'org/model','status':'downloading','bytes':4,'total':10})
    tab=DownloadsTab(client)
    item,bar=tab.rows['org/model']
    assert item.text(1)=='interrupted'
    assert bar.value()==40
    tab.update_row('org/model',{'bytes':6,'total':10,'file':'weights.bin','speed':2,'eta':2})
    assert bar.value()==60
    assert item.text(6)=='weights.bin'
    assert item.text(4)=='2.0 B/s'
    tab.close()
