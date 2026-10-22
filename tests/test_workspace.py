import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from email.message import Message

from skills import workspace, web_research as wr, research_documents as docs
from skills import material_policy
from webui import workspace_jobs


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = wr.validate_config({})

    def code(self, data=b'print("saved only, not executed")\n', url='https://example.com/main.py'):
        fetcher = Mock()
        fetcher.get_bytes.return_value = data, url, 'application/octet-stream', 'utf-8'
        return workspace.import_material(self.root, url, 'code', config=self.cfg, fetcher=fetcher)

    def test_existing_documents_and_new_code_share_library_filter_and_paging(self):
        doc = docs.store_document(self.root, 'https://example.com/article',
            b'<title>Article</title><p>This is public research material with enough words to read safely.</p>',
            'text/html', 'utf-8', self.cfg, time.monotonic()+5)
        code = self.code()
        result = workspace.list_materials(self.root, limit=1)
        self.assertEqual(result['all_total'], 2)
        self.assertEqual(result['counts']['code'], 1)
        self.assertEqual(result['counts']['article'], 1)
        second = workspace.list_materials(self.root, offset=result['next_offset'], limit=1)
        self.assertEqual(len(second['items']), 1)
        self.assertNotEqual(second['items'][0]['document_id'], result['items'][0]['document_id'])
        self.assertEqual(workspace.list_materials(self.root, kind='article')['items'][0]['document_id'], doc['document_id'])
        self.assertEqual(workspace.list_materials(self.root, query='MAIN.PY')['items'][0]['document_id'], code['document_id'])
        self.assertIn('saved only', docs.document_page(self.root, code['document_id'])['text'])
        self.code()  # same source/content does not create another copy
        self.assertEqual(workspace.list_materials(self.root)['all_total'], 2)

    def test_browsing_pages_are_rejected_before_creating_files_and_real_articles_remain(self):
        for url, title in [('https://www.ctrip.com/', '携程旅行网'),
                           ('https://dictionary.cambridge.org/travel', 'TRAVEL中文翻译'),
                           ('https://www.iciba.com/word?w=travel', 'travel是什么意思'),
                           ('https://baike.baidu.com/item/Travel', '百度百科'),
                           ('https://example.com/search?q=test', '搜索结果'),
                           ('https://xgboost.readthedocs.io/en/stable/', 'XGBoost Documentation')]:
            with self.assertRaises(wr.ResearchError):
                docs.store_document(self.root, url, f'<title>{title}</title><p>{"body " * 30}</p>'.encode(),
                                    'text/html', 'utf-8', self.cfg, time.monotonic()+5)
        self.assertFalse((self.root / workspace.STORAGE).exists())
        self.assertIsNone(material_policy.exclusion_reason({'format': 'html', 'url': 'https://any-blog.example/a-real-article', 'title': 'Research article'}))
        self.assertIsNone(material_policy.exclusion_reason({'format': 'html', 'url': 'https://xgboost.readthedocs.io/en/stable/parameter.html', 'title': 'Parameters'}))

    def test_legacy_browsing_pages_hidden_and_identical_code_merged_without_deletion(self):
        item = self.code()
        duplicate = docs.store_document(self.root, 'https://example.com/main.py',
            b'print("saved only, not executed")\n', 'text/plain', 'utf-8', self.cfg, time.monotonic()+5)
        self.assertEqual(duplicate['category'], 'code')
        folder = self.root / workspace.STORAGE / ('f' * 64); folder.mkdir()
        (folder / 'original.html').write_text('old browsing cache')
        (folder / 'metadata.json').write_text(json.dumps({'document_id': 'f'*64, 'url': 'https://www.ctrip.com/',
            'title': '携程首页', 'format': 'html', 'original_file': 'original.html', 'download_bytes': 10}))
        listing = workspace.list_materials(self.root)
        self.assertEqual(listing['all_total'], 1)
        self.assertEqual(listing['items'][0]['document_id'], item['document_id'])
        self.assertEqual(listing['excluded'], 2)
        self.assertEqual(len(list((self.root / workspace.STORAGE).iterdir())), 3)

    def test_zip_is_saved_without_extracting_and_invalid_code_is_rejected(self):
        zipped = self.code(data=b'PK\x03\x04archive preserved without reading entries', url='https://example.com/repo.zip')
        self.assertEqual(zipped['category'], 'archive')
        self.assertEqual(zipped['extraction_status'], 'unavailable')
        self.assertEqual(docs.document_page(self.root, zipped['document_id'])['text'], '')
        self.assertTrue(docs.document_file(self.root, zipped['document_id']).read_bytes().startswith(b'PK'))
        for data in (b'\x7fELF\x00binary', b'\x01\x02other binary', b'\xff\xfe'):
            with self.assertRaises(wr.ResearchError): self.code(data=data)

    def test_github_links_and_private_url_rejection(self):
        self.assertEqual(workspace.code_url('https://github.com/org/repo'), 'https://codeload.github.com/org/repo/zip/HEAD')
        self.assertEqual(workspace.code_url('https://github.com/org/repo/blob/main/code.py'), 'https://raw.githubusercontent.com/org/repo/main/code.py')
        with self.assertRaises(ValueError): workspace.code_url('https://github.com/org/repo/tree/main')
        fetcher = Mock()
        for url in ('file:///etc/passwd', 'https://127.0.0.1/a', 'http://example.com/a', 'https://user:secret@example.com/a'):
            with self.assertRaises(wr.ResearchError): workspace.import_material(self.root, url, 'code', config=self.cfg, fetcher=fetcher)
        fetcher.get_bytes.assert_not_called()

    def test_corrupt_metadata_missing_files_and_symlinks_are_skipped(self):
        item = self.code(); folder = self.root / workspace.STORAGE / item['document_id']
        original = folder / item['original_file']; original.unlink()
        original.symlink_to('/etc/passwd')
        self.assertEqual(workspace.list_materials(self.root)['skipped'], 1)
        with self.assertRaises(FileNotFoundError): docs.document_file(self.root, item['document_id'])
        (folder / 'metadata.json').write_text('{broken')
        self.assertEqual(workspace.list_materials(self.root)['all_total'], 0)
        for kwargs in ({'offset': -1}, {'limit': 200}, {'kind': 'secret'}, {'query': 'x'*301}):
            with self.assertRaises(ValueError): workspace.list_materials(self.root, **kwargs)

    def test_fetcher_artifact_mode_does_not_relax_normal_document_binary_policy(self):
        fetcher = wr.PublicFetcher(self.cfg)
        response = Mock(); response.status = 200; response.headers = Message()
        response.headers['Content-Type'] = 'application/octet-stream'
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        fetcher.opener = Mock(); fetcher.opener.open.return_value = response
        with patch.object(wr, 'validate_url'):
            response.read = io.BytesIO(b'PK\x03\x04zip').read
            self.assertEqual(fetcher.get_bytes('https://example.com/code', time.monotonic()+5, artifact=True)[0], b'PK\x03\x04zip')
            response.read = io.BytesIO(b'PK\x03\x04zip').read
            with self.assertRaises(wr.ResearchError): fetcher.get_bytes('https://example.com/code', time.monotonic()+5, document=True)

    def test_library_api_and_forced_safe_code_download(self):
        from webui import app as module
        item = self.code(); client = module.app.test_client()
        with patch.object(module, '_ROOT', self.root):
            self.assertEqual(client.get('/api/workspace').get_json()['scope'], 'project')
            self.assertEqual(client.get('/api/workspace?offset=-1').status_code, 400)
            response = client.get(item['download_url'])
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertIn('main.py', response.headers['Content-Disposition'])
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            response.close()
            self.assertEqual(client.get('/api/workspace/imports/invalid').status_code, 400)
            self.assertEqual(client.get('/api/workspace/imports/' + 'a'*32).status_code, 404)
            self.assertEqual(client.post('/api/workspace/imports', json={'request_id': 'b'*32, 'url': 'https://127.0.0.1/a'}).status_code, 400)

    def test_display_names_from_verified_reports_and_remote_path(self):
        item = self.code()
        listing = workspace.list_materials(self.root)['items'][0]
        self.assertEqual(listing['display_name'], 'main.py')
        self.assertEqual(listing['editor_path'], str(docs.document_file(self.root, item['document_id']).resolve()))
        self.assertEqual(workspace.display_name({'format': 'pdf', 'url': 'https://arxiv.org/pdf/1804.08130v4'}), '1804.08130v4.pdf')
        report = self.root / 'artifacts/web_research/eta.json'
        report.write_text(json.dumps({'sources': [{'title': 'Readable paper title', 'document': {'document_id': item['document_id']}}]}))
        self.assertEqual(workspace.list_materials(self.root, query='Readable')['total'], 1)
        self.assertNotIn('/', workspace.display_name({'title': '../../secret', 'format': 'pdf'}))

    def test_pdf_inline_preview_is_pdf_only_and_download_is_still_attachment(self):
        from webui import app as module
        from tests.test_research_documents import pdf_fixture
        document = docs.store_document(self.root, 'https://example.com/paper.pdf', pdf_fixture(),
            'application/pdf', 'utf-8', self.cfg, time.monotonic()+10)
        code = self.code()
        client = module.app.test_client()
        with patch.object(module, '_ROOT', self.root):
            base = '/api/web-research/documents/' + document['document_id']
            response = client.get(base + '/preview')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'application/pdf')
            self.assertIn('inline', response.headers['Content-Disposition'])
            self.assertIn('paper.pdf', response.headers['Content-Disposition'])
            response.close()
            response = client.get(base + '/download')
            self.assertIn('attachment', response.headers['Content-Disposition']); response.close()
            self.assertEqual(client.get(code['read_url'] + '/preview').status_code, 400)
            self.assertEqual(client.get('/api/web-research/documents/nope/preview').status_code, 400)

    def test_background_import_idempotence_capacity_and_failure(self):
        started = threading.Event(); release = threading.Event()
        self.addCleanup(release.set)
        def work(*args, **kwargs):
            started.set(); release.wait(3); return {'title': 'saved.py'}
        with patch.object(workspace_jobs, '_jobs', {}), patch.object(workspace, 'import_material', side_effect=work) as download:
            first = workspace_jobs.start(self.root, 'a'*32, 'https://example.com/a.py', 'code')
            self.assertEqual(first['status'], 'running'); self.assertTrue(started.wait(1))
            workspace_jobs.start(self.root, 'a'*32, 'https://example.com/a.py', 'code')
            with self.assertRaises(ValueError): workspace_jobs.start(self.root, 'a'*32, 'https://example.com/b.py', 'code')
            workspace_jobs.start(self.root, 'b'*32, 'https://example.com/b.py', 'code')
            with self.assertRaises(RuntimeError): workspace_jobs.start(self.root, 'c'*32, 'https://example.com/c.py', 'code')
            release.set()
            deadline = time.monotonic()+3
            while any(job['status'] == 'running' for job in workspace_jobs._jobs.values()) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(workspace_jobs.get('a'*32)['status'], 'complete')
            self.assertEqual(download.call_count, 2)
        with patch.object(workspace_jobs, '_jobs', {}), patch.object(workspace, 'import_material', side_effect=wr.ResearchError('HTTP 404')):
            workspace_jobs.start(self.root, 'd'*32, 'https://example.com/d.py', 'code')
            deadline = time.monotonic()+3
            while workspace_jobs.get('d'*32)['status'] == 'running' and time.monotonic() < deadline: time.sleep(.01)
            self.assertEqual(workspace_jobs.get('d'*32)['error'], 'HTTP 404')


if __name__ == '__main__':
    unittest.main()
