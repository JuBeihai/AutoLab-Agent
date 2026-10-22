const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.listeners = {}; this.value = ''; this.textContent = ''; }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren() { this.children = []; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  setAttribute(name, value) { this[name] = value; }
  showModal() { this.open = true; }
  close() { this.open = false; if (this.listeners.close) this.listeners.close(); }
  click() { if (this.listeners.click) return this.listeners.click(); }
  remove() {}
}
async function main() {
  const nodes = new Map(), calls = [], saved = new Map(), preferences = new Map(), opened = [];
  const node = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  node('workspace-kind').value = 'all'; node('workspace-import-kind').value = 'code';
  let firstPost = true, failedJob = false;
  const item = {document_id: 'a'.repeat(64), title: '<img src=x onerror=alert(1)>', category: 'code', url: 'https://example.com/code.py',
    editor_path: '/srv/project with spaces/original.code', download_bytes: 40, retrieved_at: '2024-10-09T08:00:00Z', extraction_status: 'ok'};
  const pdf = {...item, category: 'pdf', document_id: 'c'.repeat(64), display_name: 'Travel time estimation.pdf'};
  const context = vm.createContext({URL, URLSearchParams, AbortSignal, Date,
    crypto: {randomUUID: () => 'b'.repeat(32)}, setTimeout: fn => { queueMicrotask(fn); return 1; },
    document: {getElementById: node, createElement: tag => new Element(tag), body: {appendChild: link => { opened.push(link); }}},
    sessionStorage: {setItem: (k, v) => saved.set(k, v), getItem: k => saved.get(k), removeItem: k => saved.delete(k)},
    localStorage: {setItem: (k, v) => preferences.set(k, v), getItem: k => preferences.get(k)},
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      let result;
      if (options.method === 'POST') {
        if (firstPost) { firstPost = false; throw new TypeError('lost POST response'); }
        result = {status: 'running'};
      } else if (url.includes('/imports/')) result = failedJob ? {status: 'failed', error: 'HTTP 404'} : {status: 'complete', item};
      else if (url.includes('/documents/')) result = {text: url.endsWith('offset=0') ? '<script>plain text</script>' : '\nsecond page', next_offset: url.endsWith('offset=0') ? 27 : null};
      else result = {items: [item, pdf], all_total: 2, total: 2, total_bytes: 80, counts: {pdf: 1, article: 0, code: 1, archive: 0}, next_offset: null};
      return {ok: true, json: async () => result};
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../webui/static/workspace.js'), 'utf8'), context);
  await node('tab-workspace').listeners.click();
  assert.match(node('workspace-summary').textContent, /已保存 2 份/);
  let card = node('workspace-list').children[0];
  assert.equal(card.children.find(c => c.className === 'workspace-file-name').textContent, item.title);
  assert.equal(card.children.some(c => c.tag === 'pre'), false);
  await card.listeners.click();
  const actions = node('workspace-selection').children.find(c => c.className === 'research-actions');
  assert.equal(actions.children.find(c => c.textContent === '下载原文件').href, `/api/web-research/documents/${item.document_id}/download`);
  await actions.children.find(c => c.textContent === '快速查看').listeners.click();
  assert.equal(node('workspace-preview-dialog').open, true);
  assert.equal(node('workspace-preview-text').textContent, '<script>plain text</script>');
  await node('workspace-preview-more').listeners.click(); assert.equal(node('workspace-preview-more').disabled, true);
  node('workspace-preview-close').listeners.click();
  await card.listeners.dblclick();
  assert.equal(node('workspace-settings-dialog').open, true);
  assert.equal(opened.length, 0);
  node('workspace-ssh-host').value = 'server;rm';
  node('workspace-settings-form').listeners.submit({preventDefault() {}});
  assert.match(node('workspace-settings-status').textContent, /仅接受/);
  node('workspace-ssh-host').value = 'demo-host';
  node('workspace-pdf-mode').value = 'download';
  node('workspace-settings-form').listeners.submit({preventDefault() {}});
  assert.equal(node('workspace-settings-dialog').open, false);
  await card.listeners.dblclick();
  assert.equal(opened.at(-1).href, 'vscode://vscode-remote/ssh-remote+demo-host/srv/project%20with%20spaces/original.code');
  const pdfCard = node('workspace-list').children[1];
  pdfCard.listeners.dblclick();
  assert.equal(opened.at(-1).href, `/api/web-research/documents/${pdf.document_id}/download`);
  node('workspace-settings').listeners.click();
  node('workspace-pdf-mode').value = 'browser';
  node('workspace-settings-form').listeners.submit({preventDefault() {}});
  pdfCard.listeners.keydown({key: 'Enter', preventDefault() {}});
  assert.equal(opened.at(-1).href, `/api/web-research/documents/${pdf.document_id}/preview`);
  assert.equal(opened.at(-1).target, '_blank');
  await card.listeners.keydown({key: ' ', preventDefault() {}});
  assert.equal(node('workspace-preview-dialog').open, true);
  node('workspace-query').value = 'ETA code';
  await node('workspace-filter').listeners.submit({preventDefault() {}});
  assert.ok(calls.some(c => c.url.includes('q=ETA+code')));
  node('workspace-url').value = 'https://example.com/code.py';
  await node('workspace-import').listeners.submit({preventDefault() {}});
  const posts = calls.filter(c => c.options.method === 'POST');
  assert.equal(posts.length, 2); assert.equal(posts[0].options.body, posts[1].options.body);
  assert.match(node('workspace-import-status').textContent, /已保存/);
  assert.equal(saved.size, 0); assert.equal(node('workspace-save').disabled, false);
  failedJob = true;
  await node('workspace-import').listeners.submit({preventDefault() {}});
  assert.equal(node('workspace-import-status').textContent, 'HTTP 404');
  assert.equal(node('workspace-save').disabled, false);
  const html = fs.readFileSync(path.join(__dirname, '../webui/static/index.html'), 'utf8');
  assert.match(html, /data-tab="workspace"/); assert.match(html, /data-panel="workspace"/);
  assert.match(html, /src="workspace.js"/);
  console.log('Workspace file grid, selection, keyboard, safe preview, PDF modes, SSH URI, settings validation, filtering and import regression passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
