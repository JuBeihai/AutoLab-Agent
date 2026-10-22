"""Bounded public-source research. No LLM, project data, credentials or code execution."""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import ipaddress
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from skills.socks_transport import SocksHTTPSHandler, proxy_address

TOPICS = {
    'travel_time': {'label': 'ETA / 行程时间预测与开源实现',
        'query': '"travel time estimation" "LibCity"',
        'arxiv': 'ti:"travel time estimation" OR ti:DeepTTE OR (ti:LibCity AND all:traffic)',
        'terms': ['travel time', 'estimation', 'trajectory', 'LibCity', 'DeepTTE'],
        'questions': ['论文的预测对象、数据集和开源实现是否适用于当前 ETS？行程时间预测不等同于排队等待时长，需验证后再迁移。']},
    'regularization': {'label': 'XGBoost 正则化与树复杂度',
        'query': 'XGBoost regularization max_depth min_child_weight tabular classification',
        'arxiv': 'ti:xgboost OR (all:"gradient boosting" AND (ti:regularization OR ti:optimization))',
        'terms': ['min_child_weight', 'max_depth', 'gamma', 'regularization'],
        'questions': ['在相同样本与训练预算下，树深度和最小叶权重的组合是否优于当前冠军？']},
    'missing_encoding': {'label': '缺失值、类别编码与目标编码',
        'query': 'XGBoost missing values categorical target encoding tabular classification',
        'arxiv': 'all:tabular AND (all:"missing values" OR all:"target encoding")',
        'terms': ['missing', 'categorical', 'target encoding', 'sparse'],
        'questions': ['原生缺失处理、缺失指示和常量填补应分别做消融；目标编码是否只使用过去已成熟标签？']},
    'feature_selection': {'label': '特征选择与交叉',
        'query': 'gradient boosting tabular feature selection interaction importance',
        'arxiv': 'all:tabular AND (all:"feature selection" OR all:"feature interactions")',
        'terms': ['interaction', 'feature', 'importance', 'colsample'],
        'questions': ['低贡献且高相关的输入删除后能否保持准确率，并减少复杂度？']},
    'imbalance': {'label': '类别不平衡与低估代价',
        'query': 'gradient boosting multiclass imbalanced cost sensitive classification',
        'arxiv': 'all:"gradient boosting" AND (all:imbalanced OR all:"cost sensitive")',
        'terms': ['weight', 'imbalance', 'max_delta_step', 'multiclass'],
        'questions': ['类别权重变弱后的准确率收益是否以长等待严重低估增加为代价？不要把二分类参数直接套用多分类。']},
    'temporal_validation': {'label': '时间验证与数据泄漏',
        'query': 'temporal cross validation target leakage tabular gradient boosting',
        'arxiv': 'all:"data leakage" AND (all:tabular OR all:"time series")',
        'terms': ['time series', 'leakage', 'future', 'validation', 'gap'],
        'questions': ['如何另设未参与调优的未来时间窗口，验证搜索收益是否能泛化？现有测试集筛选成绩不等于独立验收。']},
    'tabular_methods': {'label': '表格学习的新方法',
        'query': 'tabular learning gradient boosted trees recent methods benchmark',
        'arxiv': 'all:tabular AND (all:"gradient boosting" OR all:TabPFN)',
        'terms': ['tabular', 'benchmark', 'categorical', 'boosting'],
        'questions': ['新方法适用的样本规模、版本和计算成本是什么？超出当前执行器支持范围的模型只形成调研建议。']},
}
DEFAULT_CONFIG = {
    'providers': ['arxiv'], 'allowed_domains': None,
    'timeout_seconds': 8, 'budget_seconds': 35, 'max_sources': 5,
    'max_papers': 3, 'max_bytes': 1000000, 'excerpt_chars': 1800,
    'cache_hours': 24, 'refresh_every_steps': 3, 'max_research_calls': 3,
    'xgboost_docs_version': 'release_2.1.0',
    'read_full_text': False, 'max_full_texts': 2, 'max_download_bytes': 12000000,
    'max_pdf_pages': 40, 'max_text_chars': 200000, 'pdf_timeout_seconds': 15,
}
DIRECTION_TOPICS = {'missing_encoding': 'missing_encoding', 'feature_selection': 'feature_selection',
    'feature_interaction': 'feature_selection', 'training_strategy': 'regularization',
    'history_coverage': 'temporal_validation', 'temporal_change': 'temporal_validation',
    'spatial_context': 'missing_encoding', 'supply_demand': 'feature_selection',
    'database_inspection': 'missing_encoding'}


def load_config(root):
    path = Path(root) / 'configs/web_research.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    if 'ETS_WEB_RESEARCH_SOCKS_PROXY' in os.environ:
        config['socks_proxy'] = os.environ['ETS_WEB_RESEARCH_SOCKS_PROXY']
    return validate_config(config)


def validate_config(config):
    cfg = {**DEFAULT_CONFIG, **config}
    if cfg.get('socks_proxy'):
        proxy_address(cfg['socks_proxy'])
    ranges = {'timeout_seconds': (1, 20), 'budget_seconds': (5, 90), 'max_sources': (1, 8),
              'max_papers': (1, 5), 'max_bytes': (1000, 2000000), 'excerpt_chars': (200, 2500),
              'cache_hours': (0, 168), 'refresh_every_steps': (1, 20), 'max_research_calls': (1, 5),
              'max_full_texts': (1, 5), 'max_download_bytes': (1000, 30000000),
              'max_pdf_pages': (1, 100), 'max_text_chars': (1000, 500000), 'pdf_timeout_seconds': (1, 30)}
    for name, (low, high) in ranges.items():
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f'{name} 必须为 {low}–{high} 的整数')
    if not isinstance(cfg['providers'], list) or not cfg['providers'] or any(p not in ('arxiv', 'bing', 'duckduckgo') for p in cfg['providers']):
        raise ValueError('providers 仅支持 arxiv / bing / duckduckgo')
    if type(cfg['read_full_text']) is not bool:
        raise ValueError('read_full_text 必须为布尔值')
    allowed = cfg['allowed_domains']
    if allowed is not None and (not isinstance(allowed, list) or not allowed or any(
            not isinstance(h, str) or not re.fullmatch(r'[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}', h) for h in allowed)):
        raise ValueError('allowed_domains 必须为 null（不限制域名）或明确的公网域名列表')
    if not re.fullmatch(r'release_\d+\.\d+\.\d+|stable|latest', cfg['xgboost_docs_version']):
        raise ValueError('非法文档版本')
    return cfg


class ResearchError(RuntimeError):
    pass


def error_message(error):
    if isinstance(error, urllib.error.URLError):
        return '网络连接失败：' + str(error.reason)[:140]
    if isinstance(error, (ResearchError, OSError, TimeoutError)):
        return str(error)[:180] or type(error).__name__
    return type(error).__name__


def validate_url(url, allowed, resolve=True):
    if not isinstance(url, str) or any(ord(c) < 33 for c in url):
        raise ResearchError('URL 格式无效')
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not parsed.hostname or (allowed is not None and parsed.hostname not in allowed) or len(url) > 2000):
        raise ResearchError('URL 不在允许的 HTTPS 公网来源内')
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if (literal is not None and not literal.is_global) or '.' not in parsed.hostname or parsed.hostname.endswith(('.localhost', '.local', '.internal')) or parsed.hostname == 'localhost':
        raise ResearchError('拒绝访问非公网地址')
    if resolve:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ResearchError('拒绝访问非公网地址')
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PublicFetcher:
    def __init__(self, config):
        self.cfg = config
        self.proxy = config.get('socks_proxy')
        if self.proxy:
            self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                SocksHTTPSHandler(self.proxy), _NoRedirect())
        else:
            self.opener = urllib.request.build_opener(_NoRedirect())

    def get(self, url, deadline):
        data, final_url, content_type, charset = self.get_bytes(url, deadline)
        if content_type == 'application/pdf':
            raise ResearchError('PDF 请使用正文下载读取入口')
        return data.decode(charset or 'utf-8', errors='replace'), final_url

    def get_bytes(self, url, deadline, *, document=False, artifact=False):
        attempts = 4 if self.proxy else 2
        for attempt in range(attempts):
            try:
                return self._get_bytes(url, deadline, document=document, artifact=artifact)
            except (urllib.error.URLError, OSError, http.client.IncompleteRead) as error:
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                delay = 1.5 * (attempt + 1) if self.proxy else 0
                if (attempt + 1 >= attempts or deadline - time.monotonic() <= delay + .1
                        or isinstance(reason, (ssl.SSLCertVerificationError, socket.gaierror))):
                    if self.proxy and isinstance(reason, ConnectionRefusedError):
                        raise ResearchError('SSH 调研代理暂不可用，已等待重连；请确认 Mac 隧道运行后重试') from error
                    raise
                if delay:
                    time.sleep(delay)
        raise ResearchError('下载失败')

    def _get_bytes(self, url, deadline, *, document=False, artifact=False):
        limit = self.cfg['max_download_bytes'] if document or artifact else self.cfg['max_bytes']
        for _ in range(4):
            validate_url(url, self.cfg['allowed_domains'], resolve=not self.proxy)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ResearchError('联网调研时间预算已用尽')
            req = urllib.request.Request(url, headers={'User-Agent': 'ETS-WebResearch/1.0 (public academic research)',
                                                      'Accept': 'text/html,application/pdf,application/atom+xml,application/xml,text/plain'})
            try:
                response = self.opener.open(req, timeout=min(self.cfg['timeout_seconds'], remaining))
            except urllib.error.HTTPError as error:
                if error.code in (301, 302, 303, 307, 308) and error.headers.get('Location'):
                    url = urllib.parse.urljoin(url, error.headers['Location'])
                    error.close()
                    continue  # Validate every redirect, never follow internal/private destinations.
                code = error.code
                error.close()
                raise ResearchError(f'HTTP {code}') from None
            with response:
                if response.status != 200:
                    raise ResearchError(f'HTTP {response.status}（可能需要验证）')
                content_type = response.headers.get_content_type()
                accepted = ('text/html', 'text/plain', 'application/atom+xml', 'application/xml', 'text/xml')
                if document:
                    accepted += ('application/pdf', 'application/octet-stream')
                if artifact:
                    accepted += ('application/zip', 'application/x-zip-compressed', 'application/octet-stream',
                                 'application/json', 'application/javascript', 'application/x-python-code')
                if content_type not in accepted and not (artifact and content_type.startswith('text/')):
                    raise ResearchError('不支持的文档类型；仅支持网页、文本和 PDF')
                declared = response.headers.get('Content-Length')
                if declared and int(declared) > limit:
                    raise ResearchError('来源超过读取大小上限')
                chunks, size = [], 0
                while True:
                    if time.monotonic() > deadline:
                        raise ResearchError('联网调研时间预算已用尽')
                    chunk = response.read(min(65536, limit + 1 - size))
                    if not chunk: break
                    chunks.append(chunk); size += len(chunk)
                    if size > limit:
                        raise ResearchError('来源超过读取大小上限')
                payload = b''.join(chunks)
                if declared and size != int(declared):
                    raise OSError('下载中断：实际字节数与 Content-Length 不一致')
                if payload.startswith(b'%PDF-') and b'%%EOF' not in payload[-2048:]:
                    raise OSError('PDF 下载不完整，未收到文件结束标记')
                if content_type == 'application/octet-stream' and not artifact and not payload.startswith(b'%PDF-'):
                    raise ResearchError('拒绝非 PDF 二进制文件')
                return payload, url, content_type, response.headers.get_content_charset() or 'utf-8'
        raise ResearchError('来源重定向次数过多')


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored = []; self.parts = []; self.title = []; self.in_title = False
        self.links = []; self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style', 'nav', 'footer', 'header', 'noscript', 'iframe', 'object'):
            self.ignored.append(tag)
        if tag == 'title': self.in_title = True
        if tag in ('p', 'h1', 'h2', 'h3', 'li', 'br', 'div'): self.parts.append('\n')
        if tag == 'a' and 'result__a' in attrs.get('class', '').split():
            self.anchor = {'url': attrs.get('href', ''), 'title': ''}

    def handle_endtag(self, tag):
        if self.ignored and tag == self.ignored[-1]: self.ignored.pop()
        if tag == 'title': self.in_title = False
        if tag == 'a' and self.anchor:
            self.links.append(self.anchor); self.anchor = None

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)
            if self.in_title: self.title.append(data)
            if self.anchor: self.anchor['title'] += data


def excerpt(text, terms, limit):
    # Extract (never fabricate/summarize with an unavailable LLM) locally relevant passages.
    paragraphs = [' '.join(p.split()) for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = [p[i:i+600] for p in paragraphs for i in range(0, len(p), 600)]
    ranked = sorted(enumerate(chunks), key=lambda pair: (-sum(t.lower() in pair[1].lower() for t in terms), pair[0]))
    chosen = sorted(ranked[:4])
    return '\n…\n'.join(p for _, p in chosen)[:limit]


def _paper_sources(xml, limit, excerpt_chars):
    if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():
        raise ResearchError('拒绝含实体声明的 XML')
    root = ET.fromstring(xml); ns = {'a': 'http://www.w3.org/2005/Atom'}
    result = []
    for entry in root.findall('a:entry', ns)[:limit]:
        def value(name): return ' '.join((entry.findtext('a:' + name, default='', namespaces=ns)).split())
        url = value('id').replace('http://arxiv.org/', 'https://arxiv.org/')
        if not url.startswith('https://arxiv.org/abs/'):
            continue
        result.append({'title': value('title'), 'url': url, 'published_at': value('published'),
                       'updated_at': value('updated'), 'kind': 'paper_abstract', 'discovery': 'arxiv_search',
                       'pdf_url': url.replace('/abs/', '/pdf/'),
                       'excerpt': value('summary')[:excerpt_chars]})
    return result


def _web_results(xml):
    if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():
        raise ResearchError('拒绝含实体声明的 XML')
    root = ET.fromstring(xml)
    return [{'url': item.findtext('link', ''), 'title': item.findtext('title', '')}
            for item in root.findall('./channel/item')][:10]


def _doc_urls(topic, version):
    if topic == 'travel_time':
        return ['https://raw.githubusercontent.com/LibCity/Bigscity-LibCity/master/libcity/model/eta/DeepTTE.py']
    xgb = f'https://xgboost.readthedocs.io/en/{version}/'
    urls = [xgb + ('tutorials/categorical.html' if topic == 'missing_encoding' else 'parameter.html')]
    if topic == 'temporal_validation':
        urls = ['https://scikit-learn.org/stable/modules/cross_validation.html',
                'https://scikit-learn.org/stable/common_pitfalls.html']
    elif topic == 'feature_selection':
        urls.append(xgb + 'tutorials/feature_interaction_constraint.html')
    return urls


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def collect(root, topic='regularization', *, refresh=False, config=None, fetcher=None,
            query=None, urls=None, read_full_text=None):
    if not isinstance(topic, str) or topic not in TOPICS: raise ValueError('未知联网调研主题')
    cfg = validate_config(config) if config is not None else load_config(root)
    if query is not None and (not isinstance(query, str) or not 1 <= len(query.strip()) <= 300 or any(ord(c) < 32 for c in query)):
        raise ValueError('搜索词须为 1–300 字符的公开关键词')
    query = query.strip() if query else None
    if urls is not None and (not isinstance(urls, list) or len(urls) > 5 or any(not isinstance(u, str) for u in urls)):
        raise ValueError('最多接受 5 个 HTTPS 资料网址')
    urls = list(dict.fromkeys(urls or []))
    for url in urls:
        validate_url(url, cfg['allowed_domains'], resolve=False)
    if read_full_text is not None:
        if type(read_full_text) is not bool:
            raise ValueError('read_full_text 必须为布尔值')
        cfg = {**cfg, 'read_full_text': read_full_text}
    request_key = hashlib.sha256(json.dumps([topic, query, urls], ensure_ascii=False).encode()).hexdigest()[:20]
    cache = Path(root) / 'artifacts/web_research' / ((request_key if query or urls else topic) + '.json')
    signature = json.dumps({'version': 6, 'config': cfg, 'query': query, 'urls': urls}, sort_keys=True)
    if not refresh and cache.exists() and cache.stat().st_size < 200000:
        try:
            report = json.loads(cache.read_text())
            age = time.time() - report['collected_epoch']
            if report.get('config_signature') == signature and 0 <= age < cfg['cache_hours'] * 3600 and report.get('sources'):
                return {**report, 'cached': True}
        except (ValueError, KeyError, TypeError):
            pass
    spec = TOPICS[topic]; now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    report = {'schema_version': 2, 'topic': topic, 'query': query, 'requested_urls': urls,
              'label': query or ('指定资料阅读' if urls else spec['label']), 'collected_at': now,
              'collected_epoch': time.time(), 'config_signature': signature, 'cached': False,
              'status': 'unavailable', 'sources': [], 'searches': [], 'errors': [],
              'research_questions': spec['questions'], 'model_synthesis': False,
              'note': '外部资料仅提供假设，未验证 ETS 收益。每条来源区分摘要、正文片段和未能提取的下载文件；正文提取范围与截断明确标记。模型仅接收所列片段，不代表逐字阅读全部原文。'}
    fetcher = fetcher or PublicFetcher(cfg)
    deadline = time.monotonic() + cfg['budget_seconds']
    # Fetch official references first, then search current papers. Neither receives internal context.
    candidates = ([(url, 'user_url') for url in urls] if urls else [] if query else
                  [(url, 'official_reference') for url in _doc_urls(topic, cfg['xgboost_docs_version'])])
    terms = query.split() if query else spec['terms']

    def attach_document(source, document):
        source['document'] = document
        if document['extraction_status'] == 'ok':
            if source.get('kind') == 'paper_abstract':
                source['abstract'] = source['excerpt']
            source.update(kind='paper_excerpt' if document['format'] == 'pdf' else 'article_excerpt',
                          excerpt=document['excerpt'])
        else:
            report['errors'].append({'source': source['url'], 'error': document['extraction_error']})

    def add_document(url, discovery):
        validate_url(url, cfg['allowed_domains'], resolve=False)
        if any(s['url'] == url for s in report['sources']): return
        if len(report['sources']) >= cfg['max_sources']: return
        if cfg['read_full_text']:
            from skills.research_documents import read_document
            target = url
            if urllib.parse.urlsplit(url).hostname == 'arxiv.org' and '/abs/' in url:
                target = url.replace('/abs/', '/pdf/')
            document = read_document(root, target, fetcher, cfg, deadline, terms)
            source = {'title': document['title'], 'url': url, 'published_at': None,
                      'kind': 'downloaded_document', 'discovery': discovery, 'excerpt': ''}
            attach_document(source, document)
            report['sources'].append(source)
            return
        html, final_url = fetcher.get(url, deadline)
        parser = PageText(); parser.feed(html)
        text = ''.join(parser.parts)
        if len(text.strip()) < 80: raise ResearchError('来源正文为空或过短')
        report['sources'].append({'title': ''.join(parser.title).strip()[:250] or final_url,
            'url': final_url, 'published_at': None, 'kind': 'document_excerpt', 'discovery': discovery,
            'excerpt': excerpt(text, terms, cfg['excerpt_chars'])})
    for url, discovery in candidates:
        try: add_document(url, discovery)
        except Exception as error:
            report['errors'].append({'source': url, 'error': error_message(error)})
    for provider in ([] if urls and not query else cfg['providers']):
        if provider in ('bing', 'duckduckgo') and any(s['provider'] in ('bing', 'duckduckgo') and s['status'] == 'ok' and s.get('documents_read', 0) for s in report['searches']):
            continue  # These are alternative web engines, not duplicate mandatory requests.
        arxiv_query = ' AND '.join('all:"' + word.replace('"', '') + '"' for word in query.split()) if query else spec['arxiv']
        search = {'provider': provider, 'query': arxiv_query if provider == 'arxiv' else query or spec['query'], 'status': 'failed'}
        try:
            if provider == 'arxiv':
                url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({
                    'search_query': arxiv_query, 'start': 0, 'max_results': cfg['max_papers'],
                    'sortBy': 'relevance', 'sortOrder': 'descending'})
                xml, _ = fetcher.get(url, deadline)
                papers = _paper_sources(xml, cfg['max_papers'], cfg['excerpt_chars'])
                for paper in papers:
                    validate_url(paper['url'], cfg['allowed_domains'], resolve=False)
                    if len(report['sources']) < cfg['max_sources'] and not any(s['url'] == paper['url'] for s in report['sources']):
                        report['sources'].append(paper)
                search.update(status='ok', results=len(papers))
            else:
                if provider == 'bing':
                    xml, _ = fetcher.get('https://www.bing.com/search?' + urllib.parse.urlencode({'q': query or spec['query'], 'format': 'rss'}), deadline)
                    links = _web_results(xml)
                else:
                    html, _ = fetcher.get('https://html.duckduckgo.com/html/?' + urllib.parse.urlencode({'q': query or spec['query']}), deadline)
                    if 'anomaly.js' in html or 'challenge-form' in html:
                        raise ResearchError('搜索引擎要求验证；未绕过验证')
                    parser = PageText(); parser.feed(html)
                    links = parser.links
                if not links: raise ResearchError('没有可解析的搜索结果')
                if topic == 'travel_time' and not query:
                    links = [item for item in links if re.search(
                        r'travel[\s_-]*time|arrival[\s_-]*time|deeptte|libcity|trajectory|行程时间|到达时间|\beta\b',
                        item.get('title', '') + ' ' + item.get('url', ''), re.I)]
                    if not links:
                        raise ResearchError('搜索结果与行程时间预测不匹配，未引用；尝试其他来源')
                count = 0
                for item in links[:10]:
                    if len(report['sources']) >= cfg['max_sources']: break
                    url = urllib.parse.urljoin('https://html.duckduckgo.com', item['url'])
                    url = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get('uddg', [url])[0]
                    try: add_document(url, 'web_search'); count += 1
                    except Exception as error:
                        report['errors'].append({'source': url, 'error': error_message(error)})
                search.update(status='ok', results=len(links), documents_read=count)
        except Exception as error:
            search['error'] = error_message(error)
        report['searches'].append(search)
    if cfg['read_full_text']:
        from skills.research_documents import read_document
        papers = [source for source in report['sources'] if source['kind'] == 'paper_abstract']
        for source in papers[:cfg['max_full_texts']]:
            try:
                document = read_document(root, source['pdf_url'], fetcher, cfg, deadline, terms)
                attach_document(source, document)
            except Exception as error:
                source['full_text_error'] = error_message(error)
                report['errors'].append({'source': source['pdf_url'], 'error': source['full_text_error']})
    for i, source in enumerate(report['sources'], start=1):
        source.update(source_id=f'W{i}', retrieved_at=now, trust='untrusted_external_reference')
    report['status'] = ('ok' if report['sources'] and not report['errors'] and all(s['status'] == 'ok' for s in report['searches'])
                        else 'partial' if report['sources'] else 'unavailable')
    _save(cache, report)
    _save(cache.parent / 'runs' / (now.replace(':', '-') + '_' + uuid.uuid4().hex[:8] + '.json'), report)
    return report


def prompt_context(report):
    return {k: report[k] for k in ('topic', 'status', 'collected_at', 'cached', 'sources', 'searches', 'research_questions', 'note')}


class ResearchSession:
    """At most a few refreshes; public topics derived from direction IDs, never private metrics."""
    def __init__(self, root, directory):
        self.root, self.directory = Path(root), Path(directory)
        self.config = load_config(root); self.calls = 0; self.last_round = None; self.report = None

    def update(self, round_id, state=None):
        state = state or {}
        due = self.last_round is None or (round_id - self.last_round >= self.config['refresh_every_steps'] and state.get('no_improvement_steps', 2) >= 2)
        if not due or self.calls >= self.config['max_research_calls']:
            return prompt_context(self.report) if self.report else None
        topic = DIRECTION_TOPICS.get(state.get('recommended_direction'), 'regularization')
        if self.report and topic == self.report['topic']:
            topic = 'imbalance' if topic != 'imbalance' else 'tabular_methods'
        self.calls += 1; self.last_round = round_id
        self.report = collect(self.root, topic)
        _save(self.directory / f'web_research_{self.calls:02d}.json', self.report)
        return prompt_context(self.report)
