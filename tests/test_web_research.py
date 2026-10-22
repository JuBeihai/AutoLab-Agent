import copy
import io
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from email.message import Message

from skills import web_research as wr

HTML = '<html><title>XGBoost Parameters</title><nav>ignore navigation</nav><script>steal secrets()</script><p>' + 'min_child_weight controls tree splits; max_depth controls complexity. ' * 20 + '</p></html>'
ATOM = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2601.12345v1</id><title>Tabular methods</title><published>2024-01-01T00:00:00Z</published><updated>2024-02-01T00:00:00Z</updated><summary>Study of gradient boosting. This is only an abstract.</summary></entry></feed>'''

class FakeFetcher:
    def __init__(self): self.calls = []
    def get(self, url, deadline):
        self.calls.append(url)
        return (ATOM if 'export.arxiv.org' in url else HTML), url

class WebResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_real_search_provenance_and_reference_are_distinct(self):
        fetch = FakeFetcher(); r = wr.collect(self.root, fetcher=fetch)
        self.assertEqual(r['status'], 'ok')
        self.assertEqual(r['sources'][0]['discovery'], 'official_reference')
        self.assertEqual(r['sources'][1]['discovery'], 'arxiv_search')
        self.assertEqual(r['sources'][1]['kind'], 'paper_abstract')
        self.assertEqual(r['sources'][1]['url'], 'https://arxiv.org/abs/2601.12345v1')
        self.assertFalse(r['model_synthesis'])
        self.assertNotIn('steal secrets', r['sources'][0]['excerpt'])
        self.assertNotIn('ignore navigation', r['sources'][0]['excerpt'])
        self.assertEqual(len(list((self.root/'artifacts/web_research/runs').glob('*.json'))), 1)
        self.assertTrue(all('ETS_DP_KEY' not in u for u in fetch.calls))

    def test_cache_and_force_refresh(self):
        fetch = FakeFetcher(); wr.collect(self.root, fetcher=fetch)
        cached = wr.collect(self.root, fetcher=fetch)
        self.assertTrue(cached['cached']); self.assertEqual(len(fetch.calls), 2)
        wr.collect(self.root, refresh=True, fetcher=fetch)
        self.assertEqual(len(fetch.calls), 4)
        wr.collect(self.root, config={'excerpt_chars': 300}, fetcher=fetch)
        self.assertEqual(len(fetch.calls), 6)

    def test_eta_uses_phrase_query_and_excludes_dictionary_results_with_fallback(self):
        fetch = Mock()
        def get(url, deadline):
            if 'export.arxiv.org' in url:
                return ATOM.replace('Tabular methods', 'Travel time estimation'), url
            if 'bing.com' in url:
                return '<rss><channel><item><title>Travel dictionary</title><link>https://dictionary.example.com/travel</link></item></channel></rss>', url
            if 'duckduckgo' in url:
                return '<a class="result__a" href="https://example.com/paper">Travel time estimation research</a>', url
            return HTML, url
        fetch.get.side_effect = get
        report = wr.collect(self.root, topic='travel_time', config={'providers': ['arxiv', 'bing', 'duckduckgo']}, fetcher=fetch)
        self.assertIn('ti:"travel time estimation" OR', report['searches'][0]['query'])
        self.assertEqual(report['searches'][1]['status'], 'failed')
        self.assertEqual(report['searches'][2]['status'], 'ok')
        self.assertFalse(any('dictionary.example.com' in call.args[0] for call in fetch.get.call_args_list))
        self.assertTrue(any(source['url'] == 'https://example.com/paper' for source in report['sources']))

    def test_failed_search_is_partial_not_fabricated_search(self):
        fetch = FakeFetcher(); old = fetch.get
        def get(url, deadline):
            if 'arxiv' in url: raise wr.ResearchError('HTTP 503')
            return old(url, deadline)
        fetch.get = get
        r = wr.collect(self.root, fetcher=fetch)
        self.assertEqual(r['status'], 'partial')
        self.assertEqual(r['searches'][0]['status'], 'failed')
        self.assertEqual(len(r['sources']), 1)
        self.assertEqual(r['sources'][0]['discovery'], 'official_reference')

    def test_all_network_failures_and_no_failed_cache_reuse(self):
        fetch = Mock(); fetch.get.side_effect = TimeoutError()
        for _ in range(2):
            r = wr.collect(self.root, fetcher=fetch)
            self.assertEqual(r['status'], 'unavailable'); self.assertEqual(r['sources'], [])
        self.assertEqual(fetch.get.call_count, 4)

    def test_captcha_is_failure_and_not_saved_as_document(self):
        fetch = FakeFetcher(); fetch.get = lambda url, deadline: ('<form id="challenge-form">verify</form>', url) if 'duckduckgo' in url else (HTML,url)
        r = wr.collect(self.root, config={'providers':['duckduckgo']}, fetcher=fetch)
        self.assertEqual(r['status'], 'partial')
        self.assertIn('验证',r['searches'][0]['error'])

    def test_url_and_private_dns_restrictions(self):
        for url in ['http://arxiv.org/abs/1','file:///etc/passwd',
                    'https://user:secret@arxiv.org/', 'https://127.0.0.1/', 'https://arxiv.org:444/']:
            with self.assertRaises((ValueError,wr.ResearchError)):
                wr.validate_url(url,wr.DEFAULT_CONFIG['allowed_domains'])
        with patch.object(wr.socket,'getaddrinfo',return_value=[(socket.AF_INET,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(wr.ResearchError): wr.validate_url('https://arxiv.org/',wr.DEFAULT_CONFIG['allowed_domains'])
        for host in ['www.baidu.com', 'www.google.com', 'raw.githubusercontent.com']:
            wr.validate_url('https://' + host + '/', None, resolve=False)
        with self.assertRaises(wr.ResearchError):
            wr.validate_url('https://arxiv.org.evil.com/', ['arxiv.org'], resolve=False)

    def test_redirects_revalidate_destination(self):
        fetch = wr.PublicFetcher(wr.validate_config({}))
        headers = Message(); headers['Location']='https://127.0.0.1/secret'
        fetch.opener = Mock(); fetch.opener.open.side_effect=HTTPError('https://arxiv.org/',302,'redirect',headers,io.BytesIO())
        with patch.object(wr.socket,'getaddrinfo',return_value=[(socket.AF_INET,1,6,'',('8.8.8.8',443))]):
            with self.assertRaises(wr.ResearchError): fetch.get('https://arxiv.org/',time.monotonic()+5)
        self.assertEqual(fetch.opener.open.call_count,1)

    def test_response_size_bound_and_status(self):
        fetch=wr.PublicFetcher(wr.validate_config({'max_bytes':1000}))
        response=Mock(); response.status=200; response.headers=Message(); response.headers['Content-Type']='text/html'
        response.read=io.BytesIO(b'x'*1001).read
        response.__enter__=Mock(return_value=response); response.__exit__=Mock(return_value=False)
        fetch.opener=Mock(); fetch.opener.open.return_value=response
        with patch.object(wr,'validate_url'):
            with self.assertRaisesRegex(wr.ResearchError,'大小'): fetch.get('https://arxiv.org/',time.monotonic()+5)
            response.status=202
            with self.assertRaisesRegex(wr.ResearchError,'202'): fetch.get('https://arxiv.org/',time.monotonic()+5)

    def test_config_and_xml_rejection(self):
        for cfg in [{'max_sources':True},{'budget_seconds':999},{'providers':['unknown']},{'allowed_domains':['localhost']},{'xgboost_docs_version':'../../secret'}]:
            with self.assertRaises(ValueError): wr.validate_config(cfg)
        with self.assertRaises(wr.ResearchError): wr._paper_sources('<!DOCTYPE feed>'+ATOM,3,1000)
        with self.assertRaises(ValueError): wr.collect(self.root,topic='internal SQL and API key')

    def test_session_budget_and_direction_queries_without_private_context(self):
        report=wr.collect(self.root,fetcher=FakeFetcher())
        session=wr.ResearchSession(self.root,self.root/'session')
        with patch.object(wr,'collect',return_value=report) as collect:
            for i in range(1,18):
                session.update(i,{'recommended_direction':'training_strategy','no_improvement_steps':9,'secret':'never send me'})
        self.assertEqual(collect.call_count,3)
        self.assertTrue(all(len(c.args)==2 for c in collect.call_args_list))
        self.assertEqual(len(list((self.root/'session').glob('web_research_*.json'))),3)


class WebResearchAPITests(unittest.TestCase):
    def test_api_runs_without_model_and_validates_body(self):
        from webui.app import app
        client=app.test_client()
        topics=client.get('/api/web-research').get_json()
        self.assertFalse(topics['llm_required'])
        for body in [{'topic':[]},{'refresh':'true'},{'url':'https://127.0.0.1/'},{'query':[]},
                     {'urls':['https://127.0.0.1/']}, {'read_full_text':'yes'}]:
            self.assertEqual(client.post('/api/web-research',json=body).status_code,400)
        report={'sources':[], 'status':'unavailable'}
        with patch('webui.app.web_research.collect',return_value=report) as collect:
            response=client.post('/api/web-research',json={'topic':'imbalance'})
        self.assertEqual(response.status_code,200); self.assertFalse(response.get_json()['ok'])
        self.assertEqual(collect.call_args.args[1],'imbalance')
        with patch('webui.app.runner.start_orchestrator',return_value='job') as start:
            response=client.post('/api/run',json={'web_research':True})
            self.assertEqual(response.status_code,200)
            self.assertTrue(start.call_args.kwargs['web_research'])
        self.assertEqual(client.post('/api/run',json={'web_research':'true'}).status_code,400)
