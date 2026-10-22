"""Download public documents and persist bounded, attributable extracted text."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from skills import web_research as wr
from skills import material_policy


def _pdf_text(data, cfg, deadline):
    if not sys.platform.startswith("linux"):
        raise wr.ResearchError("受资源限制的 PDF 提取当前仅支持 Linux，原文件已保留")
    executable = shutil.which('pdftotext')
    if not executable:
        raise wr.ResearchError('未安装 pdftotext，PDF 已保存，但尚未提取正文')
    timeout = min(cfg['pdf_timeout_seconds'], deadline - time.monotonic())
    if timeout <= 0:
        raise wr.ResearchError('正文提取时间预算已用尽')
    with tempfile.TemporaryDirectory(prefix='ets-paper-') as directory:
        source, target = Path(directory) / 'source.pdf', Path(directory) / 'text.txt'
        source.write_bytes(data)
        worker = Path(__file__).with_name('pdf_text_worker.py')
        try:
            completed = subprocess.run([sys.executable, str(worker), executable, str(source), str(target),
                                        str(cfg['max_pdf_pages'] + 1)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise wr.ResearchError('PDF 正文提取超时，原文件已保留') from None
        if completed.returncode or not target.exists():
            raise wr.ResearchError('PDF 解析失败：可能是加密、损坏或超出解析资源限制')
        raw = target.read_text(encoding='utf-8', errors='replace')
    pages = raw.split('\f')
    if pages and not pages[-1].strip():
        pages.pop()
    truncated = len(pages) > cfg['max_pdf_pages']
    retained = pages[:cfg['max_pdf_pages']]
    text = '\n\n'.join(f'[第 {i} 页]\n{page.strip()}' for i, page in enumerate(retained, 1))
    if sum(len(page.strip()) for page in retained) < 40:
        raise wr.ResearchError('PDF 无可提取文本，可能需要 OCR；未将扫描图片当作已读正文')
    return text, {'pages_read': len(retained), 'page_limit': cfg['max_pdf_pages'], 'truncated': truncated}


def _article_text(data, content_type, charset):
    decoded = data.decode(charset or 'utf-8', errors='replace')
    if content_type == 'text/plain':
        return decoded, ''
    parser = wr.PageText()
    parser.feed(decoded)
    text = '\n'.join(' '.join(p.split()) for p in ''.join(parser.parts).splitlines() if p.strip())
    return text, ''.join(parser.title).strip()[:250]


def store_document(root, url, data, content_type, charset, cfg, deadline, terms=()):
    """Persist the original even when extraction fails; do not pretend it was read."""
    is_pdf = data.startswith(b'%PDF-')
    if content_type in ('application/pdf', 'application/octet-stream') and not is_pdf:
        raise wr.ResearchError('文件不是有效 PDF，未保存为论文')
    if not is_pdf and content_type not in ('text/html', 'text/plain'):
        raise wr.ResearchError('正文读取仅支持 HTML、纯文本或 PDF')
    extension = 'pdf' if is_pdf else 'html' if content_type == 'text/html' else 'txt'
    article_text, article_title = ('', '') if is_pdf else _article_text(data, content_type, charset)
    descriptor = {'url': url, 'title': article_title, 'format': extension}
    reason = material_policy.exclusion_reason(descriptor)
    if reason:
        raise wr.ResearchError('未保存：' + reason)
    # Failed browsing responses do not create archive folders. PDFs remain saved
    # even when text extraction fails, so scanned/encrypted originals are retained.
    if not is_pdf and len(article_text.strip()) < 40 and material_policy.category(descriptor) != 'code':
        raise wr.ResearchError('未保存：网页正文为空、过短或需要浏览器执行脚本')
    digest = hashlib.sha256(data).hexdigest()
    identity = json.dumps([2, url, digest, cfg['max_pdf_pages'], cfg['max_text_chars']], ensure_ascii=False)
    doc_id = hashlib.sha256(identity.encode()).hexdigest()
    folder = Path(root) / 'artifacts/web_research/documents' / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / ('original.' + extension)
    if not original.exists():
        original.write_bytes(data)
    metadata = {'document_id': doc_id, 'url': url, 'sha256': digest, 'download_bytes': len(data),
                'format': extension, 'category': material_policy.category(descriptor),
                'original_file': original.name, 'title': url,
                'retrieved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'extraction_status': 'unavailable', 'text_chars': 0, 'truncated': False,
                'download_url': f'/api/web-research/documents/{doc_id}/download',
                'text_download_url': f'/api/web-research/documents/{doc_id}/download?format=text',
                'read_url': f'/api/web-research/documents/{doc_id}'}
    try:
        if is_pdf:
            text, details = _pdf_text(data, cfg, deadline)
            metadata.update(details)
        else:
            text, title = article_text, article_title
            metadata['title'] = title or url
        if len(text.strip()) < 40 and metadata['category'] != 'code':
            raise wr.ResearchError('网页正文为空、过短或需要浏览器执行脚本')
        metadata['truncated'] |= len(text) > cfg['max_text_chars']
        text = text[:cfg['max_text_chars']]
        (folder / 'text.txt').write_text(text, encoding='utf-8')
        metadata.update(extraction_status='ok', text_chars=len(text),
                        excerpt=wr.excerpt(text, terms, cfg['excerpt_chars']))
    except wr.ResearchError as error:
        metadata['extraction_error'] = str(error)
    wr._save(folder / 'metadata.json', metadata)
    return metadata


def read_document(root, url, fetcher, cfg, deadline, terms=()):
    wr.validate_url(url, cfg['allowed_domains'], resolve=False)
    data, final_url, content_type, charset = fetcher.get_bytes(url, deadline, document=True)
    return store_document(root, final_url, data, content_type, charset, cfg, deadline, terms)


def document_metadata(root, doc_id):
    if not isinstance(doc_id, str) or not re.fullmatch(r'[0-9a-f]{64}', doc_id):
        raise ValueError('无效资料编号')
    base = (Path(root) / 'artifacts/web_research/documents').resolve()
    folder = base / doc_id
    if folder.is_symlink() or not folder.is_dir():
        raise FileNotFoundError('资料不存在')
    metadata_path = folder / 'metadata.json'
    if metadata_path.is_symlink():
        raise ValueError('非法资料路径')
    metadata = json.loads(metadata_path.read_text())
    return folder, metadata


def document_page(root, doc_id, offset=0, limit=12000):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20000:
        raise ValueError('正文分页范围无效')
    folder, metadata = document_metadata(root, doc_id)
    path = folder / 'text.txt'
    if path.is_symlink():
        raise ValueError('非法资料路径')
    text = path.read_text() if metadata['extraction_status'] == 'ok' else ''
    stop = min(len(text), offset + limit)
    return {**metadata, 'text': text[offset:stop], 'offset': offset,
            'next_offset': stop if stop < len(text) else None}


def document_file(root, doc_id, text=False):
    folder, metadata = document_metadata(root, doc_id)
    name = 'text.txt' if text else metadata['original_file']
    if name not in ('text.txt', 'original.pdf', 'original.html', 'original.txt', 'original.code', 'original.zip'):
        raise ValueError('非法资料文件')
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError('资料文件不存在')
    return path
