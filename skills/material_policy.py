"""Conservative resource classification, independent of network and storage."""
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit, unquote

CODE_SUFFIXES = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.scala', '.go', '.rs',
                 '.c', '.h', '.cpp', '.hpp', '.cc', '.sh', '.sql', '.r', '.jl', '.ipynb',
                 '.yaml', '.yml', '.toml'}


def category(item):
    explicit = item.get('category')
    if explicit in {'pdf', 'article', 'code', 'archive'}:
        return explicit
    if item.get('format') == 'pdf':
        return 'pdf'
    path = unquote(urlsplit(item.get('url', '')).path)
    if item.get('format') in ('txt', 'code') and PurePosixPath(path).suffix.lower() in CODE_SUFFIXES:
        return 'code'
    return 'article'


def exclusion_reason(item):
    """Exclude browsing/navigation pages, not research topics or model families.

    This is a URL/title heuristic, not semantic proof that every retained page is
    an article. Detailed technical documentation and arbitrary article domains
    remain eligible. Explicit source archives are never unpacked here.
    """
    if category(item) in {'pdf', 'code', 'archive'}:
        return None
    parsed = urlsplit(item.get('url', ''))
    host, path = (parsed.hostname or '').lower(), unquote(parsed.path).lower()
    title = str(item.get('title') or '')
    if (host.startswith(('dictionary.', 'dict.', 'baike.')) or host in {'www.iciba.com', 'iciba.com'}
            or re.search(r'在线词典|英汉词典|剑桥词典|是什么意思|音标.*读音|^百度百科$', title)):
        return '词典或百科检索页，不是下载文章或代码'
    if (re.search(r'/(?:search|search-results|login|signin|sign-in)(?:/|$)', path)
            or host in {'www.google.com', 'www.bing.com', 'www.baidu.com', 'html.duckduckgo.com'}
            or re.search(r'^(?:登录|请登录|安全验证|just a moment|access denied|sign in)', title, re.I)):
        return '搜索、登录或验证页面，不收录到工作空间'
    if host in {'arxiv.org', 'export.arxiv.org'} and path.startswith('/abs/'):
        return '论文摘要索引页；工作空间只收录下载的正文文件'
    if item.get('format') == 'html' and (
            path.rstrip('/') in ('', '/index.html', '/index.htm', '/index.php')
            or re.fullmatch(r'/[a-z]{2}/(?:stable|latest|release_[^/]+|v?[\d.]+)/?(?:index.html)?', path)):
        return '网站首页或文档目录，不是文章正文'
    if item.get('extraction_status') == 'unavailable':
        return '未读取到文章正文'
    return None
