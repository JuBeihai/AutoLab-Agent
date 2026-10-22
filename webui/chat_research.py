"""Explicit chat search/read intent; only public request terms leave the server."""
import re
from datetime import datetime, timezone
from pathlib import Path
import time

from skills import web_research


def plan(question, history=None):
    # A negative request takes precedence; ordinary experiment consultation stays offline.
    if re.search(r'(不要|不用|无需|别).{0,6}(搜索|联网|检索|上网)|\b(?:do not|don.t)\s+(?:search|browse)', question, re.I):
        return None
    urls = re.findall(r'https://[^\s<>"\u3000]+', question)
    urls = [url.rstrip('，。；、）).,;') for url in urls][:5]
    if not urls and re.search(r'(?:之前|刚才|上次).*(?:搜|查|检索).*(?:什么|哪些|结果|内容)|(?:搜索|查询|检索|搜|查)(?:到|到了|出|出了)(?:了)?(?:什么|哪些)', question):
        return {'action': 'recall'}
    if not urls and re.search(r'联网|上网|搜索|检索', question) and re.fullmatch(
            r'(?:你|现在|目前|当前|这个会话|在|怎么|为什么|又|还|到底|是否|是不是|突然|就|已经|'
            r'还能|不能|能不能|能否|能|可以|不可以|无法|没法|支持|不支持|的|联网|网络|上网|'
            r'搜索|检索|功能|不行|不可用|可用|失败|了吗|吗|了|么|呢|[？?，,\s])+', question):
        return {'action': 'status'}
    if not urls and re.fullmatch(r'(?:那|请|你|再|重新|继续|帮我|一下|试试|搜索|检索|查询|查|搜|[吧。？?，,\s])+', question) and re.search(r'搜|查|检索', question):
        for message in reversed(history or []):
            if message.get('role') != 'user': continue
            previous = plan(message.get('content', ''))
            if previous and 'action' not in previous:
                return {**previous, 'refresh': True}
        return {'action': 'need_query'}
    intent = re.search(r'搜索|检索|查找|查询|搜一下|查一下|联网|上网|search|browse|look up', question, re.I)
    if not urls and re.search(r'实验|会话|冠军|S[0-9A-Z]{7}|E[0-9A-Z]{7}', question) and not re.search(r'联网|网络|上网|论文|文章|开源|公开', question):
        return None
    reading = urls and re.search(r'读|阅读|下载|看一下|分析|总结|read|download|summari', question, re.I)
    if not intent and not reading:
        return None
    if urls:
        return {'urls': urls, 'read_full_text': True}
    if re.search(r'(?<![a-z])(?:ets|eta)(?![a-z])|行程时间|到达时间|等待时长', question, re.I):
        return {'topic': 'travel_time', 'read_full_text': True}
    for key, pattern in [('missing_encoding', r'缺失|编码'), ('feature_selection', r'特征选择|特征交叉'),
                         ('regularization', r'正则|xgboost'), ('imbalance', r'不平衡|类别权重')]:
        if re.search(pattern, question, re.I):
            return {'topic': key, 'read_full_text': True}
    query = re.sub(r'^.*?(?:搜索|检索|查找|查询|搜一下|查一下|search for|search|look up)\s*(?:一下)?[：: ]*', '', question, count=1, flags=re.I).strip()
    if query.strip('？?。！! ') in ('', '了吗', '了么', '什么', '一下'):
        return {'action': 'need_query'}
    if not query or len(query) > 300 or re.search(r'密钥|订单号|手机号|内网|凭证|api.?key|token\s*=|secret|\b\d{11,}\b|\b(?:gdm|ods|dwd|app)\.', query, re.I):
        raise ValueError('请单独提供不含内部数据的公开搜索关键词，或公开论文网址')
    return {'query': query, 'read_full_text': True}


def collect(root, request):
    if request.get('action') == 'status':
        report = {'operation': 'status', 'status': 'unavailable', 'sources': [],
                  'checked_at': datetime.now(timezone.utc).isoformat(), 'errors': [], 'searches': []}
        try:
            cfg = web_research.load_config(root)
            fetcher = web_research.PublicFetcher(cfg)
            xml, _ = fetcher.get('https://www.bing.com/search?format=rss&q=XGBoost', time.monotonic() + 20)
            count = len(web_research._web_results(xml))
            if not count:
                raise web_research.ResearchError('搜索服务未返回可解析结果')
            report.update(status='ok', note='本次已实测搜索服务可用。应用会在收到明确的公开搜索请求后自动检索和读取，再将资料交给模型。模型本身没有独立工具调用接口，但这不等于应用不能联网。此次仅做连通性检查，没有检索用户所问论文，也不代表已读其正文。',
                          searches=[{'provider': 'bing', 'query': 'XGBoost', 'purpose': 'connectivity_check', 'status': 'ok', 'results': count}])
        except Exception as error:
            report['note'] = '应用支持联网检索，但本次搜索连通性检查失败：' + web_research.error_message(error)
            report['errors'] = [{'error': web_research.error_message(error)}]
        return report
    if request.get('action') == 'need_query':
        return {'operation': 'clarification', 'status': 'unavailable', 'sources': [],
                'note': '没有识别到明确的公开搜索主题，也没有可沿用的历史主题；未发起搜索，请用户补充关键词或网址。'}
    report = web_research.collect(root, **request)
    return {**web_research.prompt_context(report), 'operation': 'search',
            'query': report.get('query'), 'requested_urls': report.get('requested_urls', [])}


def recall(root, session_id):
    from webui import chat_store
    turns = chat_store.transcript(Path(root) / 'artifacts', session_id, limit=100)['turns']
    for turn in reversed(turns):
        report = (turn.get('result') or {}).get('web_research')
        if not report or turn.get('status') != 'complete': continue
        if report.get('operation') in ('status', 'recall', 'clarification'): continue
        if report.get('mode') == 'autonomous_tools':
            if not report.get('sources'): continue
        else:
            # Legacy reports need filtering for the old malformed-query bug.
            try:
                original = plan(turn.get('question') or '')
            except ValueError:
                continue
            if not original or 'action' in original: continue
        return {**report, 'operation': 'recall', 'reused_from_request_id': turn['request_id'],
                'note': '以下为同一会话此前实际检索记录，本次没有重新联网。' + report.get('note', '')}
    return {'operation': 'recall', 'status': 'unavailable', 'sources': [],
            'note': '同一会话最近100条记录中没有找到可核验的有效检索结果；未将历史回复中的自述当作检索证据，本次没有发起新的搜索。'}
