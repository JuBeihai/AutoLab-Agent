/* Untrusted downloaded content stays text-only. No code or HTML execution. */
(() => {
  const $ = id => document.getElementById(id);
  const list = $('workspace-list');
  const labels = {pdf: '论文 / PDF', article: '文章 / 文本', code: '源码文件', archive: '代码 ZIP'};
  let nextOffset = null, generation = 0, pending = null, polling = false;
  let selected = null, selectedButton = null, previewOffset = null, previewItem = null, previewGeneration = 0;
  let preferences = {pdf: 'download', host: ''};
  try {
    const saved = JSON.parse(localStorage.getItem('ets-workspace-open') || '{}');
    preferences = {pdf: saved.pdf === 'browser' ? 'browser' : 'download', host: /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/.test(saved.host || '') ? saved.host : ''};
  } catch (_) { /* Use safe defaults. */ }
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  function text(tag, value, parent) {
    const node = document.createElement(tag); node.textContent = value; parent.appendChild(node); return node;
  }
  function bytes(value) { return value >= 1048576 ? `${(value / 1048576).toFixed(1)} MB` : `${(value / 1024).toFixed(1)} KB`; }
  async function json(url, options = {}) {
    const response = await fetch(url, {...options, signal: AbortSignal.timeout(15000)});
    const data = await response.json();
    if (!response.ok) { const error = new Error(data.error || `请求失败 (${response.status})`); error.status = response.status; throw error; }
    return data;
  }
  const baseURL = item => `/api/web-research/documents/${item.document_id}`;
  const name = item => item.display_name || item.filename || item.title || '未命名资料';
  function openLabel(item) {
    return item.category === 'pdf' ? (preferences.pdf === 'browser' ? '打开 PDF' : '下载到 Mac 打开') : item.category === 'code' ? '在 VS Code 中打开' : item.category === 'archive' ? '下载 ZIP' : '快速查看';
  }
  function externalLink(url, newTab = false) {
    const link = document.createElement('a'); link.href = url;
    if (newTab) { link.target = '_blank'; link.rel = 'noopener noreferrer'; }
    document.body.appendChild(link); link.click(); link.remove();
  }
  function settings() {
    $('workspace-pdf-mode').value = preferences.pdf; $('workspace-ssh-host').value = preferences.host;
    $('workspace-settings-status').textContent = '';
    $('workspace-settings-dialog').showModal();
  }
  function openFile(item) {
    if (item.category === 'pdf') {
      externalLink(baseURL(item) + (preferences.pdf === 'browser' ? '/preview' : '/download'), preferences.pdf === 'browser');
    } else if (item.category === 'code') {
      if (!preferences.host) { settings(); $('workspace-settings-status').textContent = '先填写这台云服务器的 SSH Host 别名，保存后再打开文件。'; return; }
      if (!item.editor_path || !item.editor_path.startsWith('/') || /[\r\n]/.test(item.editor_path)) return;
      const remote = `ssh-remote+${preferences.host}`;
      const path = item.editor_path.split('/').map(encodeURIComponent).join('/');
      externalLink(`vscode://vscode-remote/${remote}${path}`);
      $('workspace-list-status').textContent = '已请求 VS Code 打开远程文件；请在 Chrome 中确认。若无响应，请检查打开方式中的 SSH 别名，或使用快速查看 / 下载。';
    } else if (item.category === 'archive') externalLink(baseURL(item) + '/download');
    else return preview(item);
  }
  async function readPreview() {
    if (previewOffset === null || !previewItem) return;
    const current = previewGeneration, item = previewItem;
    $('workspace-preview-more').disabled = true; $('workspace-preview-status').textContent = '正在读取…';
    try {
      const page = await json(`${baseURL(item)}?offset=${previewOffset}`);
      if (current !== previewGeneration) return;
      $('workspace-preview-text').textContent += page.text;
      previewOffset = page.next_offset;
      $('workspace-preview-status').textContent = previewOffset === null ? '已加载全部保存文本' : '';
    } catch (error) { if (current === previewGeneration) $('workspace-preview-status').textContent = error.message; }
    finally { if (current === previewGeneration) $('workspace-preview-more').disabled = previewOffset === null; }
  }
  function preview(item) {
    previewGeneration++; previewItem = item; previewOffset = item.extraction_status === 'ok' ? 0 : null;
    $('workspace-preview-title').textContent = name(item);
    $('workspace-preview-text').textContent = '';
    $('workspace-preview-info').textContent = item.truncated ? '保存文本已截断，请下载原文件查看完整内容。' : '纯文本快速查看 · 不执行代码或网页脚本';
    $('workspace-preview-status').textContent = item.extraction_status === 'ok' ? '' : item.extraction_error || '没有可提取文本，请下载原文件。';
    $('workspace-preview-more').disabled = previewOffset === null;
    $('workspace-preview-dialog').showModal();
    return readPreview();
  }
  function select(item, button) {
    if (selectedButton) selectedButton.setAttribute('aria-pressed', 'false');
    selected = item; selectedButton = button; button.setAttribute('aria-pressed', 'true');
    const panel = $('workspace-selection'); panel.hidden = false; panel.replaceChildren();
    text('strong', name(item), panel);
    text('p', `${labels[item.category] || '资料'} · ${bytes(item.download_bytes)} · ${new Date(item.retrieved_at).toLocaleString()}`, panel).className = 'hint';
    const actions = text('div', '', panel); actions.className = 'research-actions';
    const open = text('button', openLabel(item), actions); open.type = 'button'; open.addEventListener('click', () => openFile(item));
    const quick = text('button', '快速查看', actions); quick.type = 'button'; quick.addEventListener('click', () => preview(item));
    text('a', '下载原文件', actions).href = baseURL(item) + '/download';
    if (item.category === 'pdf') {
      const pdf = text('a', '浏览器预览 PDF', actions); pdf.href = baseURL(item) + '/preview'; pdf.target = '_blank'; pdf.rel = 'noopener noreferrer';
    }
    if (item.extraction_status === 'ok') text('a', '下载文本', actions).href = baseURL(item) + '/download?format=text';
    try {
      const url = new URL(item.requested_url || item.url);
      if (url.protocol === 'https:') { const source = text('a', '查看来源 ↗', actions); source.href = url.href; source.target = '_blank'; source.rel = 'noopener noreferrer'; source.title = url.href; }
    } catch (_) { /* Invalid external URL stays inert. */ }
  }
  function card(item) {
    if (!/^[a-f0-9]{64}$/.test(item.document_id)) return;
    const button = text('button', '', list); button.type = 'button'; button.className = 'workspace-file';
    button.title = name(item); button.setAttribute('aria-label', name(item)); button.setAttribute('aria-pressed', 'false');
    const kind = ['pdf', 'code', 'archive'].includes(item.category) ? item.category : 'article';
    const icon = text('span', '', button); icon.className = `workspace-file-icon file-${kind}`; icon.setAttribute('aria-hidden', 'true');
    text('span', {pdf: 'PDF', code: '</>', archive: 'ZIP', article: 'DOC'}[kind], icon);
    text('span', name(item), button).className = 'workspace-file-name';
    text('span', bytes(item.download_bytes), button).className = 'workspace-file-size';
    button.addEventListener('click', () => select(item, button));
    button.addEventListener('dblclick', () => openFile(item));
    button.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(item, button); return event.key === ' ' ? preview(item) : openFile(item); }
    });
  }
  async function load(append = false) {
    if (append && nextOffset === null) return;
    const current = ++generation;
    $('workspace-more').disabled = true;
    $('workspace-list-status').textContent = '正在读取资料库…';
    const params = new URLSearchParams({q: $('workspace-query').value.trim(), kind: $('workspace-kind').value, offset: append ? nextOffset : 0});
    try {
      const data = await json('/api/workspace?' + params);
      if (current !== generation) return;
      if (!append) { list.replaceChildren(); selected = null; selectedButton = null; $('workspace-selection').hidden = true; }
      data.items.forEach(card); nextOffset = data.next_offset;
      $('workspace-summary').textContent = `已保存 ${data.all_total} 份 · ${bytes(data.total_bytes)} · PDF ${data.counts.pdf} / 文章 ${data.counts.article} / 源码 ${data.counts.code} / ZIP ${data.counts.archive}`;
      $('workspace-list-status').textContent = data.total ? `匹配 ${data.total} 份资料${data.skipped ? `；${data.skipped} 条资料记录损坏或文件缺失，未展示` : ''}` : '暂无匹配资料。可粘贴网址保存，或使用顶部“资料调研”搜索下载。';
      $('workspace-more').hidden = nextOffset === null;
    } catch (error) { if (current === generation) $('workspace-list-status').textContent = `读取失败：${error.message}`; }
    finally { if (current === generation) $('workspace-more').disabled = false; }
  }
  function remember(job) {
    pending = job;
    try { if (job) sessionStorage.setItem('ets-workspace-import', JSON.stringify(job)); else sessionStorage.removeItem('ets-workspace-import'); } catch (_) { /* Storage may be disabled. */ }
  }
  async function poll(job, submit) {
    if (polling) return;
    polling = true;
    $('workspace-save').disabled = true;
    let accepted = !submit;
    const deadline = Date.now() + 180000;
    try {
      for (let attempt = 0; attempt < 120 && Date.now() < deadline; attempt++) {
        try {
          let status;
          if (!accepted) {
            // Retrying the same ID cannot create a second job in this server process.
            status = await json('/api/workspace/imports', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(job)});
            accepted = true;
          } else status = await json(`/api/workspace/imports/${job.request_id}`);
          if (status.status === 'failed') { remember(null); throw new Error(status.error || '下载失败'); }
          if (status.status === 'complete') {
            remember(null); $('workspace-import-status').textContent = `已保存：${status.item.title}。${status.item.extraction_status === 'ok' ? '可以在列表中阅读或下载。' : status.item.extraction_error || '原文件已保留。'}`;
            await load(); return;
          }
          $('workspace-import-status').textContent = '正在后台下载并保存资料，可切换页面；刷新后会继续查询任务。';
        } catch (error) {
          if (!pending || (error.status && error.status < 500)) {
            remember(null); throw error;
          }
          $('workspace-import-status').textContent = '连接暂时中断，正在查询同一下载任务…';
        }
        await sleep(1500);
      }
      throw new Error('等待超时，下载可能仍在进行。请刷新查询任务，不要重复提交。');
    } catch (error) { $('workspace-import-status').textContent = error.message; }
    finally { polling = false; $('workspace-save').disabled = Boolean(pending); }
  }
  $('tab-workspace').addEventListener('click', () => load());
  $('workspace-settings').addEventListener('click', settings);
  $('workspace-settings-close').addEventListener('click', () => $('workspace-settings-dialog').close());
  $('workspace-preview-close').addEventListener('click', () => $('workspace-preview-dialog').close());
  $('workspace-preview-dialog').addEventListener('close', () => { previewGeneration++; });
  $('workspace-preview-more').addEventListener('click', readPreview);
  $('workspace-settings-form').addEventListener('submit', event => {
    event.preventDefault(); const host = $('workspace-ssh-host').value.trim();
    if (host && !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/.test(host)) { $('workspace-settings-status').textContent = '连接名仅接受字母、数字、点、下划线和短横线；不接受命令、密码或完整 URL。'; return; }
    preferences = {host, pdf: $('workspace-pdf-mode').value === 'browser' ? 'browser' : 'download'};
    try { localStorage.setItem('ets-workspace-open', JSON.stringify(preferences)); }
    catch (_) { $('workspace-settings-status').textContent = '浏览器禁止保存设置；本次页面已生效，刷新后需重新填写。'; return; }
    $('workspace-settings-dialog').close();
    if (selected && selectedButton) select(selected, selectedButton);
  });
  $('workspace-refresh').addEventListener('click', () => { load(); if (pending && !polling) return poll(pending, false); });
  $('workspace-more').addEventListener('click', () => load(true));
  $('workspace-kind').addEventListener('change', () => load());
  $('workspace-filter').addEventListener('submit', event => { event.preventDefault(); return load(); });
  $('workspace-import').addEventListener('submit', event => {
    event.preventDefault(); if (pending) return;
    const url = $('workspace-url').value.trim();
    try { if (new URL(url).protocol !== 'https:') throw new Error(); }
    catch (_) { $('workspace-import-status').textContent = '请输入公开 HTTPS 网址'; return; }
    const job = {request_id: crypto.randomUUID().replaceAll('-', ''), url, kind: $('workspace-import-kind').value};
    remember(job); return poll(job, true);
  });
  try {
    const saved = JSON.parse(sessionStorage.getItem('ets-workspace-import') || 'null');
    if (saved && /^[a-f0-9]{32}$/.test(saved.request_id)) { remember(saved); poll(saved, false); }
  } catch (_) { /* No pending download to resume. */ }
})();
