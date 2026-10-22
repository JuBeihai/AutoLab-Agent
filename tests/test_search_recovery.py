import ssl
import tempfile
import time
import unittest
import urllib.error
from unittest.mock import Mock, patch
from skills import web_research as wr


class SearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fetch = wr.PublicFetcher(wr.validate_config({'socks_proxy': 'socks5h://127.0.0.1:17890'}))

    def test_waits_for_tunnel_and_retries_complete_request(self):
        refused = urllib.error.URLError(ConnectionRefusedError(111, 'Connection refused'))
        response = (b'article', 'https://example.com/article', 'text/plain', 'utf-8')
        with patch.object(self.fetch, '_get_bytes', side_effect=[refused, refused, response]) as get, \
             patch.object(wr.time, 'sleep') as sleep:
            self.assertEqual(self.fetch.get_bytes(response[1], time.monotonic()+30), response)
        self.assertEqual(get.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.5, 3])

    def test_deadline_and_certificate_errors_are_not_retried(self):
        for error in [urllib.error.URLError(ConnectionRefusedError(111, 'refused')),
                      urllib.error.URLError(ssl.SSLCertVerificationError('untrusted certificate'))]:
            with patch.object(self.fetch, '_get_bytes', side_effect=error) as get, \
                 patch.object(wr.time, 'sleep') as sleep:
                with self.assertRaises((wr.ResearchError, urllib.error.URLError)):
                    self.fetch.get_bytes('https://example.com/', time.monotonic()+.5)
                self.assertEqual(get.call_count, 1)
                sleep.assert_not_called()

    def test_no_readable_bing_documents_tries_alternative_search(self):
        rss = '<rss><channel><item><title>Unreadable</title><link>https://example.com/bad</link></item></channel></rss>'
        html = '<a class="result__a" href="https://example.com/good">Useful article</a>'
        fetch = Mock()
        def get(url, deadline):
            if 'bing.com' in url: return rss, url
            if 'duckduckgo.com' in url: return html, url
            if url.endswith('/bad'): raise wr.ResearchError('HTTP 403')
            return '<title>Useful article</title><p>' + 'Useful article body. '*20 + '</p>', url
        fetch.get.side_effect = get
        with tempfile.TemporaryDirectory() as root:
            report = wr.collect(root, query='public paper', config={'providers':['bing','duckduckgo']}, fetcher=fetch)
        self.assertEqual([s['provider'] for s in report['searches']], ['bing','duckduckgo'])
        self.assertEqual(report['sources'][0]['url'], 'https://example.com/good')
