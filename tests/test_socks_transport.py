import io
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from skills import socks_transport as st
from skills import web_research as wr


class SocksTransportTests(unittest.TestCase):
    def sock(self, reply=b'\x05\x00\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00'):
        sock = Mock()
        sock.recv = io.BytesIO(reply).read
        return sock

    def test_only_loopback_socks5h(self):
        self.assertEqual(st.proxy_address('socks5h://127.0.0.1:17890'), ('127.0.0.1',17890))
        for value in ['http://127.0.0.1:17890', 'socks5://127.0.0.1:17890',
                      'socks5h://example.com:17890','socks5h://u:p@127.0.0.1:17890',
                      'socks5h://127.0.0.1:22','socks5h://127.0.0.1:17890/path',True]:
            with self.assertRaises(ValueError): st.proxy_address(value)

    def test_destination_dns_delegated_to_mac(self):
        sock = self.sock()
        with patch.object(st.socket, 'create_connection', return_value=sock) as connect:
            self.assertIs(st.connect_socks(('127.0.0.1',17890),'arxiv.org',443,8), sock)
        connect.assert_called_once_with(('127.0.0.1',17890), timeout=8)
        self.assertEqual(sock.sendall.call_args_list[1].args[0], b'\x05\x01\x00\x03\x09arxiv.org\x01\xbb')

    def test_failure_closes_socket(self):
        for reply in [b'',b'\x05\xff',b'\x05\x00\x05\x05\x00\x01']:
            sock=self.sock(reply)
            with patch.object(st.socket,'create_connection',return_value=sock):
                with self.assertRaises(OSError): st.connect_socks(('127.0.0.1',17890),'arxiv.org',443,8)
            sock.close.assert_called_once()

    def test_tls_still_verifies_target(self):
        handler=st.SocksHTTPSHandler('socks5h://127.0.0.1:17890')
        def do_open(connection, req):
            conn=connection('arxiv.org',timeout=8)
            self.assertTrue(conn._context.check_hostname)
            self.assertEqual(conn._context.verify_mode,ssl.CERT_REQUIRED)
            conn._context=Mock()
            conn.connect()
            conn._context.wrap_socket.assert_called_once_with(raw,server_hostname='arxiv.org')
        raw=Mock()
        with patch.object(st,'connect_socks',return_value=raw),patch.object(handler,'do_open',side_effect=do_open):
            handler.https_open(Mock())

    def test_proxy_allows_other_domains_without_cloud_dns(self):
        fetch=wr.PublicFetcher(wr.validate_config({'socks_proxy':'socks5h://127.0.0.1:17890'}))
        fetch.opener=Mock(); fetch.opener.open.side_effect=OSError('test transport')
        with patch.object(wr.socket,'getaddrinfo',side_effect=AssertionError('cloud DNS must not run')):
            with self.assertRaisesRegex(OSError,'test transport'): fetch.get('https://arxiv.org/',time.monotonic()+5)
            with self.assertRaises(wr.ResearchError): fetch.get('https://127.0.0.1/',time.monotonic()+5)
            for host in ['www.baidu.com', 'www.google.com', 'raw.githubusercontent.com']:
                with self.assertRaisesRegex(OSError,'test transport'):
                    fetch.get('https://' + host + '/',time.monotonic()+5)
        fetch.cfg['allowed_domains'] = ['arxiv.org']
        with self.assertRaises(wr.ResearchError): fetch.get('https://example.com/',time.monotonic()+5)

    def test_environment_override_and_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'configs').mkdir()
            (root/'configs/web_research.json').write_text('{"socks_proxy":"socks5h://127.0.0.1:17890"}')
            with patch.dict(wr.os.environ,{'ETS_WEB_RESEARCH_SOCKS_PROXY':''}):
                self.assertEqual(wr.load_config(root)['socks_proxy'],'')
            with patch.dict(wr.os.environ,{'ETS_WEB_RESEARCH_SOCKS_PROXY':'socks5h://127.0.0.1:17891'}):
                self.assertEqual(wr.load_config(root)['socks_proxy'],'socks5h://127.0.0.1:17891')
