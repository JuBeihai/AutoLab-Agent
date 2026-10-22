/* Source material is always rendered as text, never trusted HTML. */
(() => {
  const dialog = document.getElementById('web-research-dialog');
  const result = document.getElementById('web-research-result');
  const select = document.getElementById('web-research-topic');
  const button = document.getElementById('web-research-fetch');
  const labels = {ok: '资料获取成功', partial: '部分资料可用', unavailable: '暂未获取到资料'};
  function text(tag, value, parent = result) {
    const node = document.createElement(tag);
    node.textContent = value;
    parent.appendChild(node);
    return node;
  }
  function documentActions(source, card) {
    const doc = source.document;
    if (!doc || !/^[0-9a-f]{64}$/.test(doc.document_id)) return;
    const base = `/api/web-research/documents/${doc.document_id}`;
    const actions = text('div', '', card); actions.className = 'research-actions';
    const download = text('a', `下载原文（${doc.format.toUpperCase()}）`, actions);
    download.href = `${base}/download`;
    text('p', `已下载 ${(doc.download_bytes / 1024).toFixed(1)} KB；` +
      (doc.extraction_status === 'ok' ? `正文 ${doc.text_chars} 字符${doc.pages_read ? `，已提取 ${doc.pages_read} 页` : ''}${doc.truncated ? '（达到读取上限，正文不完整）' : ''}` : `未读到正文：${doc.extraction_error}`), card);
    if (doc.extraction_status !== 'ok') return;
    const downloadText = text('a', '下载提取文本', actions); downloadText.href = `${base}/download?format=text`;
    const read = text('button', '阅读正文', actions); read.type = 'button';
    const reader = text('pre', '', card); reader.className = 'research-reader'; reader.hidden = true;
    let offset = 0;
    read.addEventListener('click', async () => {
      read.disabled = true;
      try {
        const response = await fetch(`${base}?offset=${offset}`, {signal: AbortSignal.timeout(15000)});
        const page = await response.json();
        if (!response.ok) throw new Error(page.error || '正文读取失败');
        reader.hidden = false;
        reader.textContent += page.text;
        offset = page.next_offset;
        read.textContent = offset === null ? '正文已加载完' : '继续读取';
      } catch (error) { text('p', error.message, card); }
      finally { read.disabled = offset === null; }
    });
  }
  document.getElementById('web-research-close').addEventListener('click', () => dialog.close());
  document.getElementById('btn-web-research').addEventListener('click', async () => {
    dialog.showModal();
    if (select.options.length) return;
    try {
      const response = await fetch('/api/web-research');
      if (!response.ok) throw new Error('无法读取研究方向');
      const data = await response.json();
      for (const item of data.topics) {
        const option = document.createElement('option');
        option.value = item.id; option.textContent = item.label; select.appendChild(option);
      }
    } catch (error) { result.textContent = error.message; }
  });
  button.addEventListener('click', async () => {
    if (!select.value) return;
    button.disabled = true;
    result.textContent = '正在访问公开资料，请稍候…';
    try {
      const query = document.getElementById('web-research-query').value.trim();
      const urls = document.getElementById('web-research-urls').value.split(/\n/).map(u => u.trim()).filter(Boolean);
      if (urls.length > 5) throw new Error('最多填写5个网址');
      const response = await fetch('/api/web-research', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, signal: AbortSignal.timeout(120000),
        body: JSON.stringify({topic: select.value, refresh: document.getElementById('web-research-refresh').checked,
          ...(query ? {query} : {}), ...(urls.length ? {urls} : {}),
          read_full_text: document.getElementById('web-research-full-text').checked})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '调研请求失败');
      const report = data.report;
      result.replaceChildren();
      text('h3', `${labels[report.status] || report.status}${report.cached ? '（缓存）' : ''}`);
      text('p', `获取时间：${new Date(report.collected_at).toLocaleString()}；${report.note}`);
      for (const search of report.searches) {
        text('p', `${search.provider}：${search.query} — ${search.status === 'ok' ? `检索结果 ${search.results} 条` : `失败：${search.error}`}`);
      }
      for (const error of report.errors) text('p', `来源访问失败：${error.source}（${error.error}）`);
      for (const source of report.sources) {
        const card = document.createElement('article'); result.appendChild(card);
        const heading = text('h4', `[${source.source_id}] ${source.title}`, card);
        try {
          const url = new URL(source.url);
          if (url.protocol === 'https:') {
            const link = document.createElement('a'); link.href = url.href;
            link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = '查看原文';
            heading.append(' ', link);
          }
        } catch (_) { /* An invalid link remains inert text. */ }
        const kinds = {paper_abstract: '论文摘要', paper_excerpt: '论文正文片段', article_excerpt: '文章正文片段', downloaded_document: '已下载，未读到正文', document_excerpt: '文档片段'};
        text('p', `${kinds[source.kind] || source.kind} · ${source.discovery === 'official_reference' ? '官方参考入口' : source.discovery === 'user_url' ? '指定网址' : '搜索所得'}${source.published_at ? ` · 发表 ${source.published_at.slice(0,10)}` : ''}`, card);
        text('pre', source.excerpt, card);
        if (source.full_text_error) text('p', `全文未获取：${source.full_text_error}`, card);
        documentActions(source, card);
      }
      text('h4', '待验证的问题');
      for (const question of report.research_questions) text('p', question);
    } catch (error) {
      result.textContent = `未完成调研：${error.message}。不会自动启动训练或切换为虚构资料。`;
    } finally { button.disabled = false; }
  });
})();
