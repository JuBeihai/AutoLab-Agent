"""Project-wide public material library. Downloaded code is never executed/extracted."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit, unquote

from skills import research_documents as docs
from skills import web_research as wr
from skills import material_policy

STORAGE = 'artifacts/web_research/documents'
CATEGORIES = {'pdf', 'article', 'code', 'archive'}


def category(item):
    return material_policy.category(item)

def display_name(item, titles=None):
    """Human-readable names without changing the original archived files."""
    title = (titles or {}).get(item.get('document_id')) or item.get('filename') or item.get('title', '')
    if not title or str(title).startswith(('https://', 'http://')):
        title = unquote(urlsplit(item.get('url', '')).path.rstrip('/').rsplit('/', 1)[-1]) or '未命名资料'
    title = re.sub(r'[\x00-\x1f/\\\\]', '_', str(title)).strip()[:180] or '未命名资料'
    suffix = {'pdf': '.pdf', 'archive': '.zip'}.get(category(item), '')
    return title if not suffix or title.lower().endswith(suffix) else title + suffix

def _paper_titles(root):
    titles = {}
    # Current research reports already hold verified paper titles; no network required.
    for path in (Path(root) / 'artifacts/web_research').glob('*.json'):
        try:
            if path.is_symlink() or path.stat().st_size > 200000:
                continue
            report = json.loads(path.read_text())
            for source in report.get('sources', []):
                doc_id = source.get('document', {}).get('document_id')
                title = source.get('title')
                if doc_id and isinstance(title, str) and not title.startswith(('https://', 'http://')):
                    titles[doc_id] = title
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return titles


def list_materials(root, query='', kind='all', offset=0, limit=40):
    if kind not in CATEGORIES | {'all'} or not isinstance(query, str) or len(query) > 300:
        raise ValueError('资料类型或搜索词无效')
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('分页参数无效')
    items, skipped, excluded = [], 0, 0
    titles = _paper_titles(root)
    base = Path(root) / STORAGE
    for folder in base.iterdir() if base.exists() else []:
        if not re.fullmatch(r'[a-f0-9]{64}', folder.name):
            continue
        try:
            _, item = docs.document_metadata(root, folder.name)
            if not isinstance(item, dict) or item.get('document_id') != folder.name:
                raise ValueError('元数据不一致')
            original = docs.document_file(root, folder.name)
            if material_policy.exclusion_reason(item):
                excluded += 1
                continue
            item = {**item, 'category': category(item), 'storage_path': f'{STORAGE}/{folder.name}'}
            item['display_name'] = display_name(item, titles)
            if item['category'] == 'code':
                item['editor_path'] = str(original.resolve())
            # API URLs are derived from the validated ID, not trusted metadata.
            item.update(read_url=f'/api/web-research/documents/{folder.name}',
                        download_url=f'/api/web-research/documents/{folder.name}/download')
            items.append(item)
        except (OSError, ValueError, KeyError, TypeError):
            skipped += 1
    # Prefer successfully extracted originals and the explicit code import when
    # legacy generic-text copies have exactly the same bytes. Nothing is deleted.
    items.sort(key=lambda item: (item.get('extraction_status') == 'ok',
               item.get('format') == 'code', str(item.get('retrieved_at', ''))), reverse=True)
    retained, seen = [], set()
    successful_urls = {item.get('url') for item in items if item.get('extraction_status') == 'ok'}
    for item in items:
        key = (item['category'], item.get('sha256') or item['document_id'])
        if key in seen or (item.get('extraction_status') != 'ok' and item.get('url') in successful_urls):
            excluded += 1
            continue
        seen.add(key)
        retained.append(item)
    items = retained
    counts = {key: sum(item['category'] == key for item in items) for key in sorted(CATEGORIES)}
    total_bytes = sum(item.get('download_bytes', 0) for item in items if isinstance(item.get('download_bytes', 0), int))
    needle = query.strip().casefold()
    matched = [item for item in items if (kind == 'all' or item['category'] == kind)
               and (not needle or needle in ' '.join(str(item.get(k, '')) for k in ('title', 'url', 'filename', 'display_name')).casefold())]
    matched.sort(key=lambda item: (str(item.get('retrieved_at', '')), item['document_id']), reverse=True)
    stop = offset + limit
    return {'items': matched[offset:stop], 'total': len(matched), 'all_total': len(items), 'counts': counts,
            'total_bytes': total_bytes, 'offset': offset, 'next_offset': stop if stop < len(matched) else None,
            'skipped': skipped, 'excluded': excluded, 'storage_path': STORAGE, 'scope': 'project'}


def validate_import(url, kind, cfg):
    if kind not in ('document', 'code'):
        raise ValueError('请选择文章 / 论文或代码 / ZIP')
    wr.validate_url(url, cfg['allowed_domains'], resolve=False)


def code_url(url):
    """Support GitHub file links and repository / branch ZIPs without cloning."""
    parsed = urlsplit(url)
    parts = parsed.path.strip('/').split('/')
    if parsed.hostname == 'github.com' and len(parts) >= 2:
        owner, repo = parts[:2]
        if len(parts) == 2:
            return f'https://codeload.github.com/{owner}/{repo.removesuffix(".git")}/zip/HEAD'
        if len(parts) >= 5 and parts[2] == 'blob':
            return urlunsplit(('https', 'raw.githubusercontent.com', '/' + '/'.join(parts[:2] + parts[3:]), '', ''))
        if len(parts) >= 4 and parts[2] == 'tree':
            raise ValueError('分支或子目录链接请改用 GitHub 的 Download ZIP 地址，或具体文件链接')
    return url


def import_material(root, url, kind='document', *, config=None, fetcher=None):
    cfg = wr.validate_config(config) if config is not None else wr.load_config(root)
    validate_import(url, kind, cfg)
    fetcher = fetcher or wr.PublicFetcher(cfg)
    deadline = time.monotonic() + min(cfg['budget_seconds'], 45)
    if kind == 'document':
        target = url.replace('/abs/', '/pdf/', 1) if urlsplit(url).hostname == 'arxiv.org' else url
        return docs.read_document(root, target, fetcher, cfg, deadline)
    target = code_url(url)
    wr.validate_url(target, cfg['allowed_domains'], resolve=False)
    data, final_url, content_type, charset = fetcher.get_bytes(target, deadline, artifact=True)
    if len(data) > cfg['max_download_bytes']:
        raise wr.ResearchError('来源超过下载大小上限')
    is_zip = data.startswith((b'PK\x03\x04', b'PK\x05\x06'))
    if is_zip:
        # Never enumerate/extract the archive: no zip bombs or archive path traversal.
        text = ''
    else:
        if content_type in ('application/zip', 'application/x-zip-compressed', 'text/html') or b'\x00' in data:
            raise wr.ResearchError('不是可预览的源码文本或 ZIP；网页请使用文章入口，源码请提供文件链接')
        try:
            text = data.decode(charset or 'utf-8', errors='strict')
        except (UnicodeError, LookupError):
            raise wr.ResearchError('源码编码无法识别，请提供 UTF-8 文本或 ZIP') from None
        if any(ord(ch) < 32 and ch not in '\n\r\t\f' for ch in text):
            raise wr.ResearchError('拒绝非文本二进制文件')
    digest = hashlib.sha256(data).hexdigest()
    doc_id = hashlib.sha256(json.dumps(['code-v1', final_url, digest, cfg['max_text_chars']]).encode()).hexdigest()
    folder = Path(root) / STORAGE / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError('非法资料路径')
    extension = 'zip' if is_zip else 'code'
    filename = unquote(urlsplit(final_url).path.rsplit('/', 1)[-1])[:180] or 'source'
    if is_zip and not filename.lower().endswith('.zip'):
        filename = 'repository.zip'
    original = folder / ('original.' + extension)
    if original.is_symlink() or (folder / 'text.txt').is_symlink():
        raise ValueError('非法资料路径')
    if not original.exists():
        original.write_bytes(data)
    metadata = {'document_id': doc_id, 'url': final_url, 'requested_url': url, 'title': filename,
                'filename': filename, 'sha256': digest, 'format': extension, 'original_file': original.name,
                'category': 'archive' if is_zip else 'code', 'download_bytes': len(data),
                'retrieved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'extraction_status': 'unavailable' if is_zip else 'ok',
                'text_chars': min(len(text), cfg['max_text_chars']), 'truncated': len(text) > cfg['max_text_chars'],
                'download_url': f'/api/web-research/documents/{doc_id}/download',
                'read_url': f'/api/web-research/documents/{doc_id}'}
    if is_zip:
        metadata['extraction_error'] = 'ZIP 仅保存，未解压、未执行，也未验证代码内容'
    else:
        (folder / 'text.txt').write_text(text[:cfg['max_text_chars']], encoding='utf-8')
    wr._save(folder / 'metadata.json', metadata)
    return metadata
