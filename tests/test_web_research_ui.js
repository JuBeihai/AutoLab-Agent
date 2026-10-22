const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.listeners = {}; this.value = '';
    this.checked = false; this.textContent = ''; this.disabled = false;
  }
  appendChild(child) { this.children.push(child); return child; }
  append(...items) { this.children.push(...items); }
  replaceChildren() { this.children = []; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  get options() { return this.children; }
  showModal() { this.open = true; }
  close() { this.open = false; }
}

async function main() {
  const nodes = new Map();
  const node = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  const calls = [];
  const id = 'a'.repeat(64);
  const report = {status: 'ok', cached: false, collected_at: '2024-10-09T00:00:00Z', note: '正文选段',
    searches: [{provider: 'bing', query: 'XGBoost', status: 'ok', results: 2}], errors: [], research_questions: [],
    sources: [{source_id: 'W1', title: '<script>not executed</script>', url: 'https://arxiv.org/pdf/1',
      kind: 'paper_excerpt', discovery: 'user_url', excerpt: 'A real passage', document: {
        document_id: id, format: 'pdf', download_bytes: 1000, text_chars: 40,
        pages_read: 2, truncated: true, extraction_status: 'ok'}}]};
  const context = vm.createContext({URL, AbortSignal, document: {getElementById: node, createElement: tag => new Element(tag)},
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      let result;
      if (options.method === 'POST') result = {report};
      else if (url.includes('/documents/')) result = {text: url.endsWith('offset=0') ? '<script>literal</script>' : 'second page',
                                                     next_offset: url.endsWith('offset=0') ? 23 : null};
      else result = {topics: [{id: 'regularization', label: '正则化'}]};
      return {ok: true, json: async () => result};
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../webui/static/web-research.js'), 'utf8'), context);
  await node('btn-web-research').listeners.click();
  assert.equal(node('web-research-dialog').open, true);
  node('web-research-topic').value = 'regularization';
  node('web-research-query').value = 'XGBoost';
  node('web-research-urls').value = 'https://arxiv.org/pdf/1';
  node('web-research-full-text').checked = true;
  await node('web-research-fetch').listeners.click();
  const body = JSON.parse(calls.find(call => call.options.method === 'POST').options.body);
  assert.equal(body.query, 'XGBoost'); assert.equal(body.read_full_text, true);
  assert.deepEqual(body.urls, ['https://arxiv.org/pdf/1']);
  const card = node('web-research-result').children.find(child => child.tag === 'article');
  assert.ok(card.children[0].textContent.includes('<script>'));
  const actions = card.children.find(child => child.className === 'research-actions');
  assert.equal(actions.children[0].href, `/api/web-research/documents/${id}/download`);
  const read = actions.children.find(child => child.tag === 'button');
  await read.listeners.click();
  assert.equal(read.textContent, '继续读取');
  const reader = card.children.find(child => child.className === 'research-reader');
  assert.equal(reader.textContent, '<script>literal</script>');
  await read.listeners.click();
  assert.equal(reader.textContent, '<script>literal</script>second page');
  assert.equal(read.disabled, true);
  node('web-research-close').listeners.click();
  assert.equal(node('web-research-dialog').open, false);
  console.log('Web research UI search, reader, download and text rendering checks passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
