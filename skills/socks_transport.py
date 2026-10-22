"""Optional loopback SOCKS5 transport for public research; standard library only."""
import http.client
import socket
import struct
import urllib.parse
import urllib.request


def proxy_address(value):
    if not isinstance(value, str):
        raise ValueError('调研代理必须为 socks5h://127.0.0.1:端口')
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'socks5h' or parsed.hostname != '127.0.0.1'
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment
            or parsed.port is None or not 1024 <= parsed.port <= 65535):
        raise ValueError('调研代理必须为 socks5h://127.0.0.1:端口，端口范围1024–65535')
    return parsed.hostname, parsed.port


def receive(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise OSError('SSH SOCKS 隧道提前关闭')
        data.extend(part)
    return bytes(data)


def connect_socks(proxy, host, port, timeout):
    # The request hostname is sent to the Mac: no cloud-side destination DNS.
    name = host.encode('idna')
    if not 1 <= len(name) <= 255 or port != 443:
        raise ValueError('SOCKS 调研只支持 HTTPS 443')
    sock = socket.create_connection(proxy, timeout=timeout)
    try:
        sock.sendall(b'\x05\x01\x00')
        if receive(sock, 2) != b'\x05\x00':
            raise OSError('SOCKS 隧道未接受无认证连接')
        sock.sendall(b'\x05\x01\x00\x03' + bytes([len(name)]) + name + struct.pack('!H', port))
        ver, reply, reserved, atyp = receive(sock, 4)
        if ver != 5 or reply != 0 or reserved != 0:
            raise OSError('SOCKS 目标连接失败（响应码 %d）' % reply)
        if atyp == 1:
            receive(sock, 4)
        elif atyp == 4:
            receive(sock, 16)
        elif atyp == 3:
            receive(sock, receive(sock, 1)[0])
        else:
            raise OSError('非法 SOCKS 响应地址类型')
        receive(sock, 2)
        return sock
    except BaseException:
        sock.close()
        raise


class SocksHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, proxy_url):
        super().__init__()
        self.proxy = proxy_address(proxy_url)

    def https_open(self, req):
        proxy = self.proxy

        class Connection(http.client.HTTPSConnection):
            def connect(self):
                if self._tunnel_host:
                    raise OSError('不支持叠加其他 HTTP 代理')
                raw = connect_socks(proxy, self.host, self.port, self.timeout)
                try:
                    # Keep normal certificate verification and target-host SNI.
                    self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                except BaseException:
                    raw.close()
                    raise

        return self.do_open(Connection, req)
