import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from email.message import Message

from skills import web_research as wr
from skills import research_documents as docs


HTML = b'<html><title>Research article</title><script>steal_secret()</script><p>Gradient boosting regularization improves model complexity control. This is public article body.</p></html>'
ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1603.02754</id><title>XGBoost</title><summary>Abstract only.</summary></entry></feed>'


def pdf_fixture(pages=2):
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               ('<< /Type /Pages /Count %d /Kids [%s] >>' % (pages, ' '.join(f'{4+2*i} 0 R' for i in range(pages)))).encode(),
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    for index in range(pages):
        objects.append(('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>' % (5+2*index)).encode())
        stream = f'BT /F1 12 Tf 40 700 Td (Page {index+1}: Gradient boosting regularization research with reproducible results.) Tj ET'.encode()
        objects.append(b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream')
    data = b'%PDF-1.4\n'; offsets = [0]
    for index, value in enumerate(objects, 1):
        offsets.append(len(data)); data += str(index).encode() + b' 0 obj\n' + value + b'\nendobj\n'
    xref = len(data)
    data += f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode()
    data += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    return data + f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()


class DocumentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); self.cfg = wr.validate_config({'read_full_text': True})

    def store(self, data=HTML, content_type='text/html', **config):
        return docs.store_document(self.root, 'https://example.com/article', data, content_type, 'utf-8',
                                   {**self.cfg, **config}, time.monotonic()+20, ['regularization'])

    def test_article_download_extraction_pagination_and_safe_paths(self):
        document = self.store()
        self.assertEqual(document['extraction_status'], 'ok')
        self.assertNotIn('steal_secret', document['excerpt'])
        self.assertEqual(docs.document_file(self.root, document['document_id']).read_bytes(), HTML)
        first = docs.document_page(self.root, document['document_id'], limit=40)
        second = docs.document_page(self.root, document['document_id'], offset=first['next_offset'])
        self.assertEqual(first['text'] + second['text'], docs.document_file(self.root, document['document_id'], text=True).read_text())
        for invalid in ('../secret', 'a'*63, '/etc/passwd'):
            with self.assertRaises(ValueError): docs.document_metadata(self.root, invalid)

    @unittest.skipUnless(shutil.which('pdftotext'), 'pdftotext unavailable')
    def test_real_pdf_extraction_page_cap_and_download(self):
        data = pdf_fixture(3)
        document = self.store(data, 'application/pdf', max_pdf_pages=2)
        if not sys.platform.startswith('linux'):
            self.assertEqual(document['extraction_status'], 'unavailable')
            self.assertIn('Linux', document['extraction_error'])
            self.assertEqual(docs.document_file(self.root, document['document_id']).read_bytes(), data)
            return
        self.assertEqual(document['extraction_status'], 'ok', document)
        self.assertEqual(document['pages_read'], 2)
        self.assertTrue(document['truncated'])
        text = docs.document_page(self.root, document['document_id'])['text']
        self.assertIn('Page 2:', text); self.assertNotIn('Page 3:', text)
        self.assertEqual(docs.document_file(self.root, document['document_id']).read_bytes(), data)

    def test_parse_failure_preserves_download_and_does_not_claim_read(self):
        with patch.object(docs, '_pdf_text', side_effect=wr.ResearchError('扫描件需要 OCR')):
            document = self.store(pdf_fixture(), 'application/pdf')
        self.assertEqual(document['extraction_status'], 'unavailable')
        self.assertEqual(document['text_chars'], 0)
        self.assertEqual(docs.document_page(self.root, document['document_id'])['text'], '')
        self.assertTrue(docs.document_file(self.root, document['document_id']).exists())
        with self.assertRaises(wr.ResearchError): self.store(b'not a pdf', 'application/pdf')

    def test_fetcher_binary_limits_and_redirect_private_rejection(self):
        fetcher = wr.PublicFetcher(self.cfg)
        response = Mock(); response.status = 200; response.headers = Message()
        response.headers['Content-Type'] = 'application/pdf'
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        response.read = io.BytesIO(pdf_fixture()).read
        fetcher.opener = Mock(); fetcher.opener.open.return_value = response
        with patch.object(wr, 'validate_url'):
            data, _, mime, _ = fetcher.get_bytes('https://example.com/a.pdf', time.monotonic()+5, document=True)
            self.assertTrue(data.startswith(b'%PDF')); self.assertEqual(mime, 'application/pdf')
            response.headers['Content-Length'] = '999999999'
            with self.assertRaisesRegex(wr.ResearchError, '大小'):
                fetcher.get_bytes('https://example.com/a.pdf', time.monotonic()+5, document=True)

    def test_query_cache_separation_explicit_urls_and_full_text_prompt_context(self):
        class Fetcher:
            def get(self, url, deadline): return ATOM, url
            def get_bytes(self, url, deadline, **kwargs): return HTML, url, 'text/html', 'utf-8'
        fetcher = Fetcher()
        a = wr.collect(self.root, query='boosting', config=self.cfg, fetcher=fetcher)
        b = wr.collect(self.root, query='encoding', config=self.cfg, fetcher=fetcher)
        self.assertFalse(b['cached'])
        self.assertTrue(wr.collect(self.root, query='boosting', config=self.cfg, fetcher=fetcher)['cached'])
        self.assertIn('all:"boosting"', a['searches'][0]['query'])
        self.assertEqual(a['sources'][0]['abstract'], 'Abstract only.')
        self.assertIn('regularization', wr.prompt_context(a)['sources'][0]['excerpt'])
        self.assertEqual(a['sources'][0]['document']['extraction_status'], 'ok')
        explicit = wr.collect(self.root, urls=['https://example.com/paper'], config=self.cfg, fetcher=fetcher)
        self.assertEqual(explicit['searches'], [])
        self.assertEqual(explicit['sources'][0]['discovery'], 'user_url')

    def test_download_api_and_paged_reader(self):
        from webui import app as module
        document = self.store()
        client = module.app.test_client(); base = '/api/web-research/documents/' + document['document_id']
        with patch.object(module, '_ROOT', self.root):
            self.assertEqual(client.get(base).get_json()['extraction_status'], 'ok')
            response = client.get(base+'/download')
            self.assertEqual(response.data, HTML)
            self.assertIn('attachment', response.headers['Content-Disposition'])
            self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
            response.close()
            self.assertEqual(client.get(base+'?offset=-1').status_code, 400)
            self.assertEqual(client.get('/api/web-research/documents/not-an-id/download').status_code, 400)

    def test_bing_results_are_real_links_and_success_skips_fallback(self):
        fetcher = Mock()
        fetcher.get.return_value = ('<rss><channel><item><title>Public article</title><link>https://example.com/article</link></item></channel></rss>', 'https://www.bing.com/search')
        fetcher.get_bytes.return_value = HTML, 'https://example.com/article', 'text/html', 'utf-8'
        report = wr.collect(self.root, query='XGBoost', config={**self.cfg, 'providers': ['bing', 'duckduckgo']}, fetcher=fetcher)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual([s['provider'] for s in report['searches']], ['bing'])
        self.assertEqual(report['sources'][0]['discovery'], 'web_search')
        self.assertEqual(report['sources'][0]['document']['extraction_status'], 'ok')

    def test_interrupted_pdf_is_retried_and_not_saved_as_complete(self):
        fetcher = wr.PublicFetcher(self.cfg)
        response = Mock(); response.status = 200; response.headers = Message()
        response.headers['Content-Type'] = 'application/pdf'
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        responses = []
        for data in (b'%PDF-incomplete', pdf_fixture()):
            item = Mock(); item.status = 200; item.headers = response.headers
            item.__enter__ = Mock(return_value=item); item.__exit__ = Mock(return_value=False)
            item.read = io.BytesIO(data).read; responses.append(item)
        fetcher.opener = Mock(); fetcher.opener.open.side_effect = responses
        with patch.object(wr, 'validate_url'):
            data, _, _, _ = fetcher.get_bytes('https://example.com/a.pdf', time.monotonic()+5, document=True)
        self.assertTrue(data.endswith(b'%%EOF\n'))
        self.assertEqual(fetcher.opener.open.call_count, 2)
