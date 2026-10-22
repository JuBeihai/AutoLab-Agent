const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { detectRunIntent, buildRunConfig } = require('../webui/static/controls.js');
const { render } = require('../webui/static/markdown.js');

const page = fs.readFileSync(path.join(__dirname, '../webui/static/index.html'), 'utf8');
const header = page.match(/<header>([\s\S]*?)<\/header>/)[1];
const navigation = page.match(/<nav class="tabs">([\s\S]*?)<\/nav>/)[1];
assert.ok(!header.includes('id="session-select"'));
assert.equal((page.match(/id="session-select"/g) || []).length, 1);
assert.ok(navigation.includes('<label for="session-select">查看会话</label>'));
assert.ok(navigation.indexOf('id="session-select"') < navigation.indexOf('data-tab="history"'));
assert.ok(!page.includes('id="chat-mode"'));
assert.equal((page.match(/id="btn-add-hint"/g) || []).length, 1);
assert.ok(page.includes('指导全局共享，不绑定当前查看的会话'));
assert.ok(page.includes('id="btn-interrupt-agent"'));
assert.ok(page.includes('id="btn-new-chat"'));
assert.ok(!page.includes('<select id="session-select"'));
assert.ok(page.includes('id="continuous-search" type="checkbox" checked'));
assert.ok(page.includes('id="research-skills" type="checkbox" checked'));
assert.ok(page.includes('id="model-study" type="checkbox"'));
assert.ok(page.includes('id="chat-history-enabled" type="checkbox" checked'));

const table = render('| 分组 | **准确率** | 备注 |\n| :--- | ---: | :---: |\n| 示例区域A | 0.5000 | `a|b` |\n| LONG_WAIT | 206');
assert.ok(table.includes('<table><thead>'));
assert.ok(table.includes('class="md-align-right" scope="col"><strong>准确率</strong>'));
assert.ok(table.includes('<code>a|b</code>'));
assert.ok(table.includes('<td class="md-align-center"></td>'));
assert.ok(!table.includes('---:'));
const escapedTable = render('名称 | 值\n--- | ---\nA\\|B | <img src=x onerror=alert(1)>');
assert.ok(escapedTable.includes('A|B'));
assert.ok(escapedTable.includes('&lt;img'));
assert.ok(!escapedTable.includes('<img'));
assert.ok(render('a | b\n--- | ---\nx | y | 不丢失额外单元格').includes('y | 不丢失额外单元格'));
assert.ok(!render('a | b\n普通内容 | 不是分隔行').includes('<table>'));
assert.ok(!render('```\na | b\n--- | ---\n```').includes('<table>'));
assert.ok(render('a | b\n--- | ---\nx | y\n\n## 后续').includes('</table></div>\n<h2>后续</h2>'));

assert.equal(render('**会话 ID**：`mining_mock_123`'), '<p><strong>会话 ID</strong>：<code>mining_mock_123</code></p>');
assert.equal(render('__加粗__ 和 *斜体*'), '<p><strong>加粗</strong> 和 <em>斜体</em></p>');
assert.equal(render('**包含 `代码`**'), '<p><strong>包含 <code>代码</code></strong></p>');
assert.equal(render('`**不加粗**`'), '<p><code>**不加粗**</code></p>');
assert.equal(render('## 摘要\n\n- **状态**：完成\n- 测试\n\n3. 检查\n4. 发布'),
  '<h2>摘要</h2>\n<ul><li><strong>状态</strong>：完成</li><li>测试</li></ul>\n<ol start="3"><li>检查</li><li>发布</li></ol>');
assert.equal(render('- 第一行\n  补充内容'), '<ul><li>第一行<br>补充内容</li></ul>');
assert.equal(render('第一段\n换行\n\n第二段'), '<p>第一段<br>换行</p>\n<p>第二段</p>');
assert.equal(render('```python\nprint("<tag>")\n**原始代码**\n```'),
  '<pre><code>print(&quot;&lt;tag&gt;&quot;)\n**原始代码**</code></pre>');
assert.equal(render('```\n未关闭代码块'), '<pre><code>未关闭代码块</code></pre>');
assert.equal(render('> 引用\n\n---'), '<blockquote>引用</blockquote>\n<hr>');
assert.equal(render('<img src=x onerror=alert(1)>'), '<p>&lt;img src=x onerror=alert(1)&gt;</p>');
assert.equal(render('<script>alert(1)</script>'), '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
assert.ok(!render('[点击](javascript:alert) [点击](data:text/html,test)').includes('<a '));
assert.ok(!render('[点击](java&#115;cript:alert)').includes('<a '));
assert.equal(render('[文档](https://example.com?a=1&b=2)'),
  '<p><a href="https://example.com?a=1&amp;b=2" target="_blank" rel="noopener noreferrer">文档</a></p>');

for (const text of ['为什么启动失败？', '不要运行实验', '运行实验会怎样', '如果运行2轮', '怎么开始实验',
                    '停止实验', 'run not now', 'start training?', '这次结果怎么样？']) {
  assert.equal(detectRunIntent(text), null, text);
}
const intent = detectRunIntent('开始实验，使用真实数据，跑2轮');
assert.equal(intent.rounds, 2);
assert.equal(intent.mock, false);
assert.equal(detectRunIntent('运行2轮，每轮3个候选').maxProposals, 3);
assert.equal(detectRunIntent('只跑baseline').noAgent, true);
assert.equal(detectRunIntent('运行2轮，mock真实数据'), null);
assert.deepEqual(buildRunConfig(intent, { rounds: 1, max_proposals: 3, mock: true, no_agent: false, resume_recipes: false }),
  { rounds: 2, max_proposals: 3, mock: false, no_agent: false, resume_recipes: false });

class Element {
  constructor() {
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.style = {};
    this.dataset = {};
    this.children = [];
    this.listeners = {};
    this.classList = { add() {}, remove() {}, contains() { return false; } };
  }
  get parentNode() { return this.parent || null; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  replaceChildren(...children) { this.innerHTML = ""; for (const child of children) this.appendChild(child); }
  appendChild(child) { child.remove(); child.parent = this; this.children.push(child); return child; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this);
    this.parent = null;
  }
  set innerHTML(value) {
    this.html = value;
    for (const child of this.children) child.parent = null;
    this.children = [];
  }
  get innerHTML() { return this.html; }
  append() {}
  querySelector() { return new Element(); }
  querySelectorAll() { return []; }
  setAttribute() {}
  getBoundingClientRect() { return { width: 400 }; }
}

async function browser(confirmResult = false, initial = {}) {
  const elements = new Map();
  const calls = [];
  const streams = [];
  const timers = new Map();
  let timerId = 0;
  let chatResponse = null;
  let transcriptResponse = initial.transcriptResponse || (() => ({turns: [], history: [], total: 0, next_before: null}));
  const browserStorage = initial.storage || new Map();
  let currentSessionId = 'session_current';
  let jobs = [];
  const deleted = new Set();
  let deleteError = false;
  let holdout = null;
  const chats = [];
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, new Element());
    return elements.get(selector);
  };
  const session = { session_id: 'session_current', mock: true, status: 'complete', accepted_recipes: [] };
  const context = {
    console, URL, Set, Map, Date, JSON, AbortController,
    document: { querySelector: element, querySelectorAll: () => [], createElement: () => new Element(),
      getElementById: name => element('#' + name), addEventListener() {}, body: new Element() },
    window: { confirm: message => { calls.push({ confirm: message }); return confirmResult; }, innerWidth: 1400, addEventListener() {} },
    localStorage: { getItem: key => browserStorage.get(key) || null, setItem: (key, value) => browserStorage.set(key, value) },
    navigator: {},
    alert(message) { calls.push({ alert: message }); },
    setTimeout: () => 1, clearTimeout() {},
    setInterval(callback, delay) { timers.set(++timerId, { callback, delay }); return timerId; },
    clearInterval(identifier) { timers.delete(identifier); },
    EventSource: class {
      constructor(url) { this.url = url; this.readyState = 1; this.listeners = {}; streams.push(this); }
      get parentNode() { return this.parent || null; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
      close() { this.readyState = 2; }
    },
    async fetch(url, options = {}) {
      const endpoint = new URL(url, 'http://test').pathname;
      const body = options.body ? JSON.parse(options.body) : undefined;
      calls.push({ endpoint, method: options.method || 'GET', body });
      let data = {};
      if (endpoint === '/api/data-sources') data = { snapshots: initial.snapshots || [] };
      if (endpoint === '/api/sessions') {
        if (options.method === 'POST') {
          const created = { session_id: `S00000K${chats.length + 1}`, mode: 'chat', status: 'complete', mock: null };
          chats.push(created);
          data = { ok: true, session: created };
        } else data = { sessions: [...chats, session, { ...session, session_id: 'session_other' }].filter(item => !deleted.has(item.session_id)), current_session_id: currentSessionId };
      }
      if (endpoint.startsWith('/api/sessions/') && options.method === 'DELETE') {
        if (deleteError) return { ok: false, status: 409, text: async () => '实验正在运行' };
        const identifier = endpoint.split('/').pop();
        deleted.add(identifier);
        if (currentSessionId === identifier) currentSessionId = deleted.has('session_other') ? null : 'session_other';
        data = { ok: true, deleted_session_id: identifier };
      }
      if (/^\/api\/sessions\/[^/]+\/chat$/.test(endpoint)) data = await transcriptResponse(endpoint.split('/')[3], url);
      if (endpoint === '/api/champion') data = { champion: null, session };
      if (endpoint === '/api/discovery') data = { session, profile: null, sources: null, holdout };
      if (endpoint === '/api/experiments') data = { records: [] };
      if (endpoint === '/api/memory') data = { champion: null, recent_summaries: [], accepted_features: [], xgb_params_history: {} };
      if (endpoint === '/api/lineage') data = { lineages: [] };
      if (endpoint === '/api/user-hint') data = { hints: [] };
      if (endpoint === '/api/run') {
        if (options.method === 'POST') {
          jobs = [{ job_id: 'job_new', alive: true, session_id: session.session_id }];
          data = { ok: true, job_id: 'job_new' };
        } else data = { jobs };
      }
      if (endpoint === '/api/chat') data = chatResponse ? await chatResponse() : { ok: true, answer: '回答', session_id: session.session_id };
      return { ok: true, json: async () => data };
    },
  };
  vm.createContext(context);
  for (const name of ['controls.js', 'markdown.js', 'app.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../webui/static', name), 'utf8'), context);
  }
  await new Promise(resolve => setImmediate(resolve));
  element('#rounds').value = '1';
  element('#max-proposals').value = '3';
  return { context, element, calls, streams, timers, setJobs(value) { jobs = value; },
    storage: browserStorage, setTranscriptResponse(value) { transcriptResponse = value; },
    setHoldout(value) { holdout = value; },
    setDeleteError(value) { deleteError = value; },
    setSession(value) { Object.assign(session, value); },
    setChatResponse(callback) { chatResponse = callback; }, setCurrentSession(value) { currentSessionId = value; } };
}

async function main() {
  const snapshot = {id: 'raw_test', compatible: true, bytes: 49004946,
    window: {start_date: '2024-08-05', test_end: '2024-09-16'}};
  const offline = await browser(true, {snapshots: [snapshot], storage: new Map([['ets-data-source', 'offline']])});
  vm.runInContext('updateDataSourceControls()', offline.context);
  assert.equal(offline.element('#data-source').value, 'offline');
  assert.equal(offline.element('#snapshot-id').value, 'raw_test');
  assert.equal(offline.element('#snapshot-control').hidden, false);
  await vm.runInContext("startRun({...runConfig(), mock:false, snapshot_id:'raw_test'})", offline.context);
  const offlinePost = offline.calls.find(call => call.endpoint === '/api/run' && call.method === 'POST');
  assert.equal(offlinePost.body.data_source, 'offline');
  assert.equal(offlinePost.body.snapshot_id, 'raw_test');
  assert.equal(offline.storage.get('ets-snapshot-id'), 'raw_test');
  vm.runInContext('updateDataSourceControls()', offline.context);
  assert.equal(vm.runInContext('runConfig().mock', offline.context), true);
  assert.equal(vm.runInContext('runConfig().snapshot_id', offline.context), null);
  assert.equal(offline.element('#snapshot-control').hidden, false);
  const unavailable = await browser(true, {storage: new Map([['ets-data-source', 'offline']])});
  await vm.runInContext('startRun({...runConfig(), mock:false})', unavailable.context);
  assert.ok(!unavailable.calls.some(call => call.endpoint === '/api/run' && call.method === 'POST'));
  assert.ok(unavailable.calls.some(call => call.alert && call.alert.includes('快照')));
  const continuous = await browser(true);
  continuous.element('#continuous-search').checked = true;
  continuous.element('#research-skills').checked = true;
  continuous.element('#llm-profile').value = 'responses';
  continuous.element('#search-timeout').value = '60';
  continuous.element('#search-patience').value = '8';
  await vm.runInContext('startRun(runConfig())', continuous.context);
  const researchPost = continuous.calls.find(call => call.endpoint === '/api/run' && call.method === 'POST');
  assert.equal(researchPost.body.continuous, true);
  assert.equal(researchPost.body.research_skills, true);
  assert.equal(researchPost.body.llm_profile, 'responses');
  assert.equal(researchPost.body.search_timeout_minutes, 60);
  assert.equal(researchPost.body.search_patience, 8);
  const historicalChat = await browser();
  historicalChat.element('#chat-history-enabled').checked = true;
  historicalChat.element('#llm-profile').value = 'responses';
  historicalChat.setChatResponse(() => ({ ok: true, answer: '带来源的历史结果',
    history_sources: [{ session_id: 'S000000K', experiment_id: 'E000001G' }] }));
  historicalChat.element('#chat-text').value = '看看 S000000K';
  await vm.runInContext('handleChatSend()', historicalChat.context);
  assert.equal(historicalChat.calls.find(call => call.endpoint === '/api/chat').body.include_history, true);
  assert.equal(historicalChat.calls.find(call => call.endpoint === '/api/chat').body.llm_profile, 'responses');
  const collectText = element => [element.textContent || '', ...element.children.map(collectText)].join(' ');
  assert.ok(collectText(historicalChat.element('#log-view')).includes('S000000K / E000001G'));
  const interrupted = await browser();
  assert.equal(vm.runInContext('renderSearchDiagnostics({summary: {}})', interrupted.context), '');
  interrupted.context.diagnosticRecord = { summary: {
    selection_note: '通过门禁但未优于当前冠军',
    search_diagnostics: {
      errors: { errors: 10, queue_errors: 8, non_queue_errors: 2 },
      prediction_changes: { parent: { corrected: 3, introduced: 3, delta_pp: 0 } },
      features: { '<img src=x onerror=alert(1)>': {
        fitting: { queue: { rows: 100, missing_rate: 1, unique: 0 } },
        model_usage: { queue: { split_count: 0 }, no_queue: null },
        warnings: ['测试期存在整日缺失'], fully_missing_dates: ['2024-09-14']
      } }
    }
  } };
  const diagnosticHtml = vm.runInContext('renderSearchDiagnostics(diagnosticRecord)', interrupted.context);
  assert.ok(diagnosticHtml.includes('通过门禁但未优于当前冠军'));
  assert.ok(diagnosticHtml.includes('纠正 3 行，新增错误 3 行'));
  assert.ok(diagnosticHtml.includes('0.0000 pp'));
  assert.ok(diagnosticHtml.includes('2024-09-14'));
  assert.ok(diagnosticHtml.includes('0 次分裂'));
  assert.ok(diagnosticHtml.includes('&lt;img'));
  assert.ok(!diagnosticHtml.includes('<img'));
  interrupted.context.diagnosticRecord.summary.search_diagnostics.feature_selection = {
    note: '训练内重要性不代表因果收益', models: { queue: {
      fitting_rows: 100, permutation_rows: 50,
      ranking: [{ name: '<img src=x>', total_gain: 3, gain_share: .25, split_count: 2, missing_rate: .1 }],
      permutation: [{ name: '<img src=x>', mean_acc_drop_pp: -.02 }],
      ablation_candidates: ['<img src=x>'],
      high_correlation_pairs: [{ first: '<img src=x>', second: 'other', spearman: .99 }]
    } }
  };
  const importanceHtml = vm.runInContext('renderSearchDiagnostics(diagnosticRecord)', interrupted.context);
  assert.ok(importanceHtml.includes('全部输入的重要性与冗余'));
  assert.ok(importanceHtml.includes('25.00%'));
  assert.ok(importanceHtml.includes('-0.0200 pp'));
  assert.ok(!importanceHtml.includes('<img'));
  interrupted.context.inspectionProfile = { inspection_results: [{ result: {
    kind: 'table', table: '<img src=x>', fields: {
      supply: { rows: 10, unique: 3, missing_rate: .1, quantiles: {'0.5': 5} }
    }, correlations: [{ first: '<img src=x>', second: 'supply', spearman: .9 }]
  } }] };
  const inspectionHtml = vm.runInContext('renderInspectionResults(inspectionProfile)', interrupted.context);
  assert.ok(inspectionHtml.includes('Agent 自主只读探查'));
  assert.ok(inspectionHtml.includes('不代表全表分布'));
  assert.ok(inspectionHtml.includes('5.000'));
  assert.ok(!inspectionHtml.includes('<img'));
  for (const recipe of [
    {name: 'flag', kind: 'missing_indicator', column: '<img src=x>'},
    {name: 'filled', kind: 'impute', column: 'idle', strategy: 'median', value: null},
    {name: 'city', kind: 'one_hot', keys: ['city_code'], values: ['<img src=x>']},
    {name: 'target', kind: 'target_encoding', keys: ['city_code'], target: 'mean_bin', smoothing: 20}
  ]) {
    interrupted.context.newRecipe = recipe;
    const recipeHtml = vm.runInContext('_renderRecipe(newRecipe)', interrupted.context);
    assert.ok(!recipeHtml.includes('<img'));
    assert.ok(recipeHtml.includes(recipe.name));
  }
  let lateReply;
  interrupted.setChatResponse(() => new Promise(resolve => { lateReply = resolve; }));
  interrupted.element('#chat-text').value = '请分析当前实验';
  const inFlight = vm.runInContext('handleChatSend()', interrupted.context);
  const chatPost = interrupted.calls.find(call => call.endpoint === '/api/chat');
  assert.ok(chatPost.body.request_id.length >= 16);
  assert.equal(interrupted.element('#btn-interrupt-agent').disabled, false);
  await interrupted.element('#btn-interrupt-agent').listeners.click();
  await inFlight;
  assert.equal(interrupted.element('#btn-chat-send').disabled, false);
  assert.equal(interrupted.element('#btn-interrupt-agent').disabled, true);
  assert.ok(interrupted.calls.some(call => call.endpoint === `/api/chat/${chatPost.body.request_id}/cancel` && call.method === 'POST'));
  assert.ok(!interrupted.calls.some(call => call.endpoint.startsWith('/api/run/') && call.endpoint.endsWith('/stop')));
  assert.ok(![...interrupted.timers.values()].some(timer => timer.delay === 1000));
  interrupted.setChatResponse(() => ({ ok: true, answer: '新的回答' }));
  interrupted.element('#chat-text').value = '重新回答';
  await vm.runInContext('handleChatSend()', interrupted.context);
  lateReply({ ok: true, answer: '不应出现的晚到回复' });
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(!vm.runInContext('JSON.stringify(state.chatHistory)', interrupted.context).includes('不应出现'));
  assert.ok(vm.runInContext('JSON.stringify(state.chatHistory)', interrupted.context).includes('新的回答'));

  const newDialog = await browser(true);
  newDialog.element('#chat-text').value = '原会话草稿';
  await vm.runInContext('Promise.all([createConversation(), createConversation()])', newDialog.context);
  assert.equal(newDialog.calls.filter(call => call.endpoint === '/api/sessions' && call.method === 'POST').length, 1);
  assert.equal(vm.runInContext('state.sessionId', newDialog.context), 'S00000K1');
  assert.equal(newDialog.element('#chat-text').value, '');
  assert.equal(vm.runInContext('state.chatHistory.length', newDialog.context), 0);
  assert.ok(!newDialog.calls.some(call => call.endpoint === '/api/run' && call.method === 'POST'));
  const rows = newDialog.element('#session-list').children;
  assert.equal(rows.length, 3);
  assert.ok(rows.every(row => row.children[1].className === 'session-delete'));
  const otherRow = rows.find(row => row.children[0].textContent.includes('session_other'));
  await otherRow.children[1].listeners.click();
  assert.ok(newDialog.calls.some(call => call.endpoint === '/api/sessions/session_other' && call.method === 'DELETE'));
  assert.equal(vm.runInContext('state.sessionId', newDialog.context), 'S00000K1');
  const originalRow = newDialog.element('#session-list').children.find(row => row.children[0].textContent.includes('session_current'));
  await originalRow.children[0].listeners.click();
  assert.equal(newDialog.element('#chat-text').value, '原会话草稿');
  const ownRow = newDialog.element('#session-list').children.find(row => row.children[0].textContent.includes('S00000K1'));
  await ownRow.children[1].listeners.click();
  assert.equal(vm.runInContext('state.sessionId', newDialog.context), 'session_current');

  const switchedInterrupt = await browser();
  let replyAfterSwitch;
  switchedInterrupt.setChatResponse(() => new Promise(resolve => { replyAfterSwitch = resolve; }));
  switchedInterrupt.element('#chat-text').value = '会话 A 问题';
  const originalRequest = vm.runInContext('handleChatSend()', switchedInterrupt.context);
  await vm.runInContext('state.followCurrent = false; switchConversation("session_other")', switchedInterrupt.context);
  assert.equal(switchedInterrupt.element('#btn-interrupt-agent').disabled, true);
  await switchedInterrupt.element('#btn-interrupt-agent').listeners.click();
  assert.ok(!switchedInterrupt.calls.some(call => call.endpoint.endsWith('/cancel')));
  await vm.runInContext('switchConversation("session_current")', switchedInterrupt.context);
  await switchedInterrupt.element('#btn-interrupt-agent').listeners.click();
  await originalRequest;
  replyAfterSwitch({ ok: true, answer: '晚到' });

  for (const confirmed of [false, true]) {
    const stopExperiment = await browser(confirmed);
    await vm.runInContext('state.currentJobId = "job_stop"; taskView("job_stop", "session_current"); updateRunButtons()', stopExperiment.context);
    assert.equal(stopExperiment.element('#btn-interrupt-agent').textContent, '停止实验');
    await stopExperiment.element('#btn-interrupt-agent').listeners.click();
    assert.equal(stopExperiment.calls.filter(call => call.endpoint === '/api/run/job_stop/stop').length, confirmed ? 1 : 0);
    assert.ok(stopExperiment.calls.some(call => call.confirm && call.confirm.includes('整个实验')));
  }

  const statusRecovery = await browser();
  const originalFetch = statusRecovery.context.fetch;
  let statusTimeout;
  statusRecovery.context.setTimeout = (callback, delay) => {
    if (delay === 8000) statusTimeout = callback;
    return 1;
  };
  statusRecovery.context.fetch = (url, options) => url === '/api/run'
    ? new Promise(() => {}) : originalFetch(url, options);
  vm.runInContext('state.currentJobId = "finished_job"; updateRunButtons()', statusRecovery.context);
  const hungStatus = vm.runInContext('restoreRunningJob()', statusRecovery.context);
  statusTimeout();
  await hungStatus;
  assert.equal(vm.runInContext('state.restoring', statusRecovery.context), false);
  assert.ok(statusRecovery.element('#run-status').textContent.includes('请求超时'));
  assert.equal(statusRecovery.element('#btn-run').disabled, true);
  statusRecovery.context.fetch = originalFetch;
  statusRecovery.setJobs([{ job_id: 'finished_job', session_id: 'session_current', alive: false, returncode: 0 }]);
  await statusRecovery.element('#btn-sync-run').listeners.click();
  assert.equal(statusRecovery.element('#btn-run').disabled, false);
  assert.equal(statusRecovery.element('#btn-stop').disabled, true);
  assert.equal(vm.runInContext('state.currentJobId', statusRecovery.context), null);
  assert.ok(statusRecovery.element('#run-status').textContent.includes('已完成'));
  assert.ok(!statusRecovery.calls.some(call => call.endpoint === '/api/run' && call.method === 'POST'));
  await vm.runInContext('state.currentJobId = "render_failure"; const brokenView = taskView("render_failure", "session_current"); brokenView.tick = () => { throw new Error("simulated display failure"); }; finishJob({ job_id: "render_failure", returncode: 0 })', statusRecovery.context);
  assert.equal(statusRecovery.element('#btn-run').disabled, false);
  assert.equal(vm.runInContext('state.currentJobId', statusRecovery.context), null);

  const progressBrowser = await browser();
  await vm.runInContext('state.currentJobId = "job_progress"; switchConversation(null); startStreaming("job_progress")', progressBrowser.context);
  const progressStream = progressBrowser.streams[0];
  for (let index = 1; index <= 600; index++) {
    progressStream.onmessage({ data: `[stderr] Spark configuration ${index}`, lastEventId: String(index) });
  }
  const task = vm.runInContext('state.taskViews.get("job_progress")', progressBrowser.context);
  assert.equal(task.details.open, false);
  assert.equal(task.lineCount, 600);
  assert.equal(task.lines.length, 500);
  assert.equal(task.lines[0], '[stderr] Spark configuration 101');
  assert.equal(progressBrowser.element('#pinned-task').children.filter(node => node.className === 'task-progress').length, 1);
  assert.equal(progressBrowser.element('#pinned-task').hidden, false);
  await vm.runInContext('for (let i = 0; i < 30; i++) appendConversationNode(document.createElement("article"))', progressBrowser.context);
  assert.equal(task.card.parentNode, progressBrowser.element('#pinned-task'));
  assert.ok(!progressBrowser.element('#log-view').children.includes(task.card));
  assert.ok(!progressBrowser.element('#log-view').children.some(node => node.className === 'log-line stderr'));
  assert.ok(task.summary.textContent.includes('完整记录见服务日志'));
  task.details.open = true;
  task.details.listeners.toggle();
  assert.ok(task.body.textContent.includes('Spark configuration 600'));
  await progressStream.listeners.experiment({ lastEventId: '601', data: JSON.stringify({ type: 'session_updated', session_id: 'session_current', status: 'loading' }) });
  assert.equal(task.sessionId, 'session_current');
  assert.equal(vm.runInContext('conversation(null).nodes.includes(state.taskViews.get("job_progress").card)', progressBrowser.context), false);
  assert.ok(progressBrowser.element('#pinned-task').children.includes(task.card));
  progressBrowser.element('#session-select').listeners.change({ target: { value: 'session_other' } });
  await progressStream.listeners.experiment({ lastEventId: '602', data: JSON.stringify({ type: 'progress', session_id: 'session_current', stage: 'training', message: 'E0000013 · 训练 1000 行' }) });
  assert.equal(task.status.textContent, '训练中');
  assert.equal(task.message.textContent, 'E0000013 · 训练 1000 行');
  assert.ok(!progressBrowser.element('#log-view').children.includes(task.card));
  assert.equal(vm.runInContext('state.sessionId', progressBrowser.context), 'session_other');
  assert.equal(progressBrowser.element('#pinned-task').hidden, true);
  assert.ok(!progressBrowser.element('#pinned-task').children.includes(task.card));
  await vm.runInContext('switchConversation("session_current")', progressBrowser.context);
  assert.equal(task.card.parentNode, progressBrowser.element('#pinned-task'));
  assert.equal(task.details.open, true);
  progressBrowser.element('#btn-clear-log').listeners.click();
  assert.equal(task.card.parentNode, progressBrowser.element('#pinned-task'));
  assert.equal(progressBrowser.element('#pinned-task').hidden, false);
  await vm.runInContext('startStreaming("job_progress", "session_current")', progressBrowser.context);
  const reconnected = progressBrowser.streams[1];
  reconnected.onmessage({ data: '[stderr] duplicate', lastEventId: '600' });
  assert.equal(task.lineCount, 600);
  reconnected.onmessage({ data: '[stderr] new', lastEventId: '603' });
  assert.equal(task.lineCount, 601);
  progressStream.onmessage({ data: '[stderr] stale stream', lastEventId: '604' });
  assert.equal(task.lineCount, 601);
  reconnected.listeners.done({ lastEventId: '604', data: JSON.stringify({ job_id: 'job_progress', session_id: 'session_current', returncode: 1 }) });
  assert.ok(task.status.textContent.includes('失败'));
  assert.ok(task.message.textContent.includes('展开终端详细日志'));
  assert.equal(task.finished, true);
  assert.ok(!progressBrowser.timers.has(task.timer));
  await vm.runInContext('state.currentJobId = "oom_job"; taskView("oom_job", "session_current"); finishJob({job_id: "oom_job", returncode: -9, error_hint: "运行期间容器内存耗尽"})', progressBrowser.context);
  assert.equal(vm.runInContext('state.taskViews.get("oom_job").message.textContent', progressBrowser.context), '运行期间容器内存耗尽');

  const testEvaluation = await browser();
  testEvaluation.setSession({ evaluation_split: 'test', training_method: 'full_train' });
  const testMetrics = { xgb_44: { ALL: { total: 42437, recall_ratio: .5553, over_ratio: .138, under_ratio: .3067 } } };
  testEvaluation.setHoldout({ used_for_search: true, baseline: testMetrics, selected: testMetrics,
    selected_id: 'E0000013', gate: { pass: false, reasons: [] } });
  await vm.runInContext('refreshWorkspace()', testEvaluation.context);
  assert.ok(testEvaluation.element('#discovery-view').innerHTML.includes('统一测试集结果（参与候选筛选）'));
  assert.ok(testEvaluation.element('#holdout-evaluation').children[2].innerHTML.includes('55.5%'));
  assert.ok(testEvaluation.element('#holdout-evaluation').children[2].innerHTML.includes('42437'));
  const oldVersion = testEvaluation.element('#holdout-evaluation').children[0].children[1].children[0];
  oldVersion.value = 'h3_7';
  oldVersion.listeners.change();
  assert.ok(testEvaluation.element('#holdout-evaluation').children[2].innerHTML.includes('未保存'));
  assert.ok(!testEvaluation.element('#holdout-evaluation').children[2].innerHTML.includes('55.5%'));
  const viewRecord = {
    metrics_by_city: testMetrics,
    evaluation_views: { populations: {
      all: { raw: { metrics: testMetrics.xgb_44 } },
      answer_cancel: { h3_7: { metrics: { ALL: { total: 39000, recall_ratio: .587, over_ratio: .112, under_ratio: .301 } } } }
    } }
  };
  testEvaluation.context.viewRecord = viewRecord;
  await vm.runInContext('mountEvaluation(document.querySelector("#test-matrix"), viewRecord, null, "baseline")', testEvaluation.context);
  const matrix = testEvaluation.element('#test-matrix');
  assert.ok(!matrix.children[2].innerHTML.includes('challenger'));
  assert.ok(matrix.children[2].innerHTML.includes('<th>样本</th>'));
  const population = matrix.children[0].children[0].children[0];
  const version = matrix.children[0].children[1].children[0];
  population.value = 'answer_cancel';
  version.value = 'h3_7';
  version.listeners.change();
  assert.ok(matrix.children[2].innerHTML.includes('58.7%'));
  assert.ok(matrix.children[2].innerHTML.includes('39000'));
  assert.ok(matrix.children[1].textContent.includes('不在保序后重新过滤'));
  assert.ok(matrix.children[1].textContent.includes('人群可能不同'));
  assert.ok(matrix.children[1].textContent.includes('历史口径·全量保序前'));
  const h3Record = {
    experiment_id: 'new-h3-policy', config_diff: { selection_metric: 'all_h3_7' },
    metrics_by_city: { ...testMetrics, all_h3_7: { ALL: { total: 42437, recall_ratio: .61 } } },
    evaluation_views: { populations: { all: {
      raw: { metrics: testMetrics.xgb_44 },
      h3_7: { metrics: { ALL: { total: 42437, recall_ratio: .61, over_ratio: .1, under_ratio: .29 } } }
    } } }
  };
  testEvaluation.context.h3Record = h3Record;
  await vm.runInContext('mountEvaluation(document.querySelector("#h3-matrix"), h3Record)', testEvaluation.context);
  const h3Matrix = testEvaluation.element('#h3-matrix');
  const h3Version = h3Matrix.children[0].children[1].children[0];
  assert.equal(h3Version.value, 'h3_7');
  assert.ok(h3Matrix.children[2].innerHTML.includes('61.0%'));
  assert.ok(h3Matrix.children[1].textContent.includes('全量·H3-7保序后'));
  h3Version.value = 'raw';
  h3Version.listeners.change();
  assert.ok(h3Matrix.children[2].innerHTML.includes('55.5%'));
  assert.ok(h3Matrix.children[1].textContent.includes('全量·H3-7保序后'));
  assert.equal(vm.runInContext('selectionMetrics(h3Record).ALL.recall_ratio', testEvaluation.context), .61);
  const sixKey = 'answer_cancel_h3_7_6bin';
  const sixRows = { ALL: { total: 1000, recall_ratio: 500 / 1000, over_ratio: .2, under_ratio: .3 } };
  testEvaluation.context.sixRecord = {
    experiment_id: 'six-bucket-policy',
    config_diff: { selection_metric: sixKey, reference_baseline: { metrics: sixRows } },
    metrics_by_city: { [sixKey]: sixRows, all_h3_7: { ALL: { recall_ratio: .99 } } },
    evaluation_views: { populations: { answer_cancel: { h3_7: { metrics: sixRows } } } }
  };
  await vm.runInContext('mountEvaluation(document.querySelector("#six-matrix"), sixRecord)', testEvaluation.context);
  const sixMatrix = testEvaluation.element('#six-matrix');
  assert.equal(sixMatrix.children[0].children[0].children[0].value, 'answer_cancel');
  assert.equal(sixMatrix.children[0].children[1].children[0].value, 'h3_7');
  assert.ok(sixMatrix.children[1].textContent.includes('非排队强制回到第0桶'));
  assert.ok(sixMatrix.children[1].textContent.includes('示例阈值'));
  assert.ok(sixMatrix.children[2].innerHTML.includes('50.0%'));
  assert.equal(vm.runInContext('selectionMetrics(sixRecord).ALL.recall_ratio', testEvaluation.context), 500 / 1000);
  testEvaluation.context.unifiedRecord = {...testEvaluation.context.sixRecord, experiment_id: 'unified-policy',
    evaluation_views: {...testEvaluation.context.sixRecord.evaluation_views, non_queue_after_ordering: 'model'}};
  await vm.runInContext('mountEvaluation(document.querySelector("#unified-matrix"), unifiedRecord)', testEvaluation.context);
  assert.ok(testEvaluation.element('#unified-matrix').children[1].textContent.includes('保序后不强制归零'));
  assert.ok(!testEvaluation.element('#unified-matrix').children[1].textContent.includes('强制回到第0桶'));
  const comparison = vm.runInContext('evaluationTable({metrics: {ALL: {total: 20, recall_ratio: .5, over_ratio: .2, under_ratio: .3}}}, {metrics: {ALL: {total: 30, recall_ratio: .6, over_ratio: .2, under_ratio: .2}}}, "候选", "baseline", true)', testEvaluation.context);
  assert.ok(comparison.includes('baseline样本'));
  assert.ok(comparison.includes('候选样本'));
  assert.ok(comparison.includes('<td>30</td><td>20</td>'));
  assert.ok(testEvaluation.element('#history-table').innerHTML.includes('测试集（统一评估，参与候选筛选）'));
  assert.equal(vm.runInContext('evaluationLabel("train_internal_early_stop")', testEvaluation.context), '历史训练内早停子集（非测试集）');
  testEvaluation.setHoldout({ used_for_search: false, baseline: testMetrics, selected: testMetrics });
  await vm.runInContext('loadDiscovery()', testEvaluation.context);
  assert.ok(testEvaluation.element('#discovery-view').innerHTML.includes('历史独立测试集结果（未参与搜索）'));
  const unifiedHints = await browser();
  unifiedHints.element('#hint-input').value = '优先改善示例区域A样本效果';
  await unifiedHints.element('#btn-add-hint').listeners.click();
  const hintPosts = unifiedHints.calls.filter(call => call.endpoint === '/api/user-hint' && call.method === 'POST');
  assert.equal(hintPosts.length, 1);
  assert.equal(hintPosts[0].body.hint, '优先改善示例区域A样本效果');
  assert.equal(unifiedHints.element('#hint-input').value, '');
  assert.ok(!unifiedHints.calls.some(call => call.endpoint === '/api/chat'));
  unifiedHints.element('#chat-text').value = '示例区域A样本效果怎么样？';
  await vm.runInContext('handleChatSend()', unifiedHints.context);
  assert.equal(unifiedHints.calls.filter(call => call.endpoint === '/api/chat').length, 1);
  assert.equal(unifiedHints.calls.filter(call => call.endpoint === '/api/user-hint' && call.method === 'POST').length, 1);
  const cancelledDelete = await browser(false);
  await vm.runInContext('deleteSelectedSession()', cancelledDelete.context);
  assert.ok(cancelledDelete.calls.some(call => call.confirm && call.confirm.includes('session_current')));
  assert.ok(!cancelledDelete.calls.some(call => call.method === 'DELETE'));
  const deleting = await browser(true);
  await vm.runInContext('deleteSelectedSession()', deleting.context);
  assert.ok(deleting.calls.some(call => call.endpoint === '/api/sessions/session_current' && call.method === 'DELETE'));
  assert.equal(vm.runInContext('state.sessionId', deleting.context), 'session_other');
  assert.equal(vm.runInContext('state.conversations.has("session_current")', deleting.context), false);
  await vm.runInContext('deleteSelectedSession()', deleting.context);
  assert.equal(vm.runInContext('state.sessionId', deleting.context), null);
  assert.equal(deleting.element('#btn-delete-session').disabled, true);
  const failedDelete = await browser(true);
  failedDelete.setDeleteError(true);
  await vm.runInContext('deleteSelectedSession()', failedDelete.context);
  assert.equal(vm.runInContext('state.sessionId', failedDelete.context), 'session_current');
  assert.ok(failedDelete.calls.some(call => call.alert && call.alert.includes('删除失败')));
  assert.equal(failedDelete.element('#btn-delete-session').disabled, false);
  const activeDelete = await browser(true);
  await vm.runInContext('state.currentJobId = "active"; updateRunButtons(); deleteSelectedSession()', activeDelete.context);
  assert.equal(activeDelete.element('#btn-delete-session').disabled, true);
  assert.ok(!activeDelete.calls.some(call => call.method === 'DELETE'));
  const doubleDelete = await browser(true);
  await vm.runInContext('Promise.all([deleteSelectedSession(), deleteSelectedSession()])', doubleDelete.context);
  assert.equal(doubleDelete.calls.filter(call => call.method === 'DELETE').length, 1);
  const historicalDelete = await browser(true);
  await vm.runInContext('state.followCurrent = false; switchConversation("session_other"); deleteSelectedSession()', historicalDelete.context);
  assert.ok(historicalDelete.calls.some(call => call.endpoint === '/api/sessions/session_other' && call.method === 'DELETE'));
  assert.equal(vm.runInContext('state.sessionId', historicalDelete.context), 'session_current');
  const splitView = await browser();
  splitView.setSession({ window: { train_start: '2024-08-05', train_end: '2024-08-23',
    test_start: '2024-08-24', test_end: '2024-08-30' },
    selection_warning: '训练内部指标不是独立测试效果' });
  await vm.runInContext('loadDiscovery()', splitView.context);
  assert.ok(!splitView.element('#discovery-view').innerHTML.includes('历史验证'));
  assert.ok(splitView.element('#discovery-view').innerHTML.includes('训练内部指标不是独立测试效果'));
  splitView.setSession({ window: { train_start: '2024-08-05', train_end: '2024-08-21',
    validation_start: '2024-08-22', validation_end: '2024-08-23',
    test_start: '2024-08-24', test_end: '2024-08-30' } });
  await vm.runInContext('loadDiscovery()', splitView.context);
  assert.ok(splitView.element('#discovery-view').innerHTML.includes('历史验证'));
  for (const outcome of ['success', 'failure']) {
    const switched = await browser();
    const pendingReplies = [];
    switched.setChatResponse(() => new Promise((resolve, reject) => pendingReplies.push({ resolve, reject })));
    switched.element('#chat-text').value = '会话 A 的问题';
    const firstRequest = vm.runInContext('handleChatSend()', switched.context);
    const firstPending = switched.element('#log-view').children.find(node => node.className.includes('chat-message-pending'));
    switched.element('#chat-text').value = '会话 A 草稿';
    switched.element('#session-select').listeners.change({ target: { value: 'session_other' } });
    assert.equal(switched.element('#log-view').children.length, 1);
    assert.equal(switched.element('#log-view').children[0].className, 'chat-empty');
    assert.ok(!switched.element('#log-view').children.includes(firstPending));
    assert.equal(switched.element('#btn-chat-send').disabled, false);
    assert.equal(switched.element('#chat-text').value, '');
    assert.equal(vm.runInContext('state.chatHistory.length', switched.context), 0);
    await vm.runInContext('restoreConversation("session_other")', switched.context);
    switched.element('#chat-text').value = '会话 B 的问题';
    const secondRequest = vm.runInContext('handleChatSend()', switched.context);
    const secondPending = switched.element('#log-view').children.find(node => node.className.includes('chat-message-pending'));
    const requests = switched.calls.filter(call => call.endpoint === '/api/chat');
    assert.equal(requests[0].body.session_id, 'session_current');
    assert.equal(requests[1].body.session_id, 'session_other');
    assert.equal(requests[1].body.history.length, 0);
    switched.element('#session-select').listeners.change({ target: { value: 'session_current' } });
    assert.ok(switched.element('#log-view').children.includes(firstPending));
    assert.equal(switched.element('#chat-text').value, '会话 A 草稿');
    assert.equal(switched.element('#btn-chat-send').disabled, true);
    switched.element('#session-select').listeners.change({ target: { value: 'session_other' } });
    if (outcome === 'success') pendingReplies[0].resolve({ ok: true, answer: '会话 A 的回答', session_id: 'session_current', truncated: true });
    else pendingReplies[0].reject(new Error('仅会话 A 请求失败'));
    await firstRequest;
    assert.ok(switched.element('#log-view').children.includes(secondPending));
    assert.equal(switched.element('#log-view').children.length, 2);
    assert.equal(switched.element('#btn-chat-send').disabled, true);
    assert.equal(vm.runInContext('state.chatHistory.length', switched.context), 0);
    pendingReplies[1].resolve({ ok: true, answer: '会话 B 的回答', session_id: 'session_other' });
    await secondRequest;
    assert.equal(switched.element('#btn-chat-send').disabled, false);
    assert.equal(vm.runInContext('state.chatHistory.at(-1).content', switched.context), '会话 B 的回答');
    const secondAnswer = switched.element('#log-view').children.at(-1);
    switched.element('#session-select').listeners.change({ target: { value: 'session_current' } });
    assert.ok(!switched.element('#log-view').children.includes(secondAnswer));
    assert.ok(!switched.element('#log-view').children.includes(firstPending));
    const firstAnswer = switched.element('#log-view').children.at(-1);
    if (outcome === 'success') {
      assert.equal(firstAnswer.children[1].children[1].innerHTML, '<p>会话 A 的回答</p>');
      assert.equal(firstAnswer.children[1].children[1].children[0].className, 'chat-truncated-notice');
      assert.equal(vm.runInContext('state.chatHistory.at(-1).content', switched.context), '会话 A 的回答');
    } else {
      assert.equal(firstAnswer.textContent, '[error] 仅会话 A 请求失败');
      assert.equal(vm.runInContext('state.chatHistory.length', switched.context), 0);
    }
    switched.element('#btn-clear-log').listeners.click();
    switched.element('#session-select').listeners.change({ target: { value: 'session_other' } });
    assert.ok(switched.element('#log-view').children.includes(secondAnswer));
    assert.equal(vm.runInContext('state.chatHistory.at(-1).content', switched.context), '会话 B 的回答');
    switched.element('#session-select').listeners.change({ target: { value: 'session_current' } });
    assert.ok(!switched.element('#log-view').children.includes(firstAnswer));
    assert.ok(![...switched.timers.values()].some(timer => timer.delay === 1000));
    switched.setChatResponse(async () => ({ ok: true, answer: 'A 的第二次回答', session_id: 'session_current' }));
    switched.element('#chat-text').value = 'A 的后续问题';
    await vm.runInContext('handleChatSend()', switched.context);
    const followup = switched.calls.filter(call => call.endpoint === '/api/chat').at(-1);
    assert.equal(followup.body.history.length, outcome === 'success' ? 2 : 0);
    assert.ok(!followup.body.history.some(message => message.content.includes('会话 B')));
    switched.element('#session-select').listeners.change({ target: { value: 'session_other' } });
    switched.element('#session-select').listeners.change({ target: { value: '' } });
    await vm.runInContext('refreshSessions()', switched.context);
    assert.equal(vm.runInContext('state.sessionId', switched.context), 'session_current');
    assert.equal(vm.runInContext('state.chatHistory.at(-1).content', switched.context), 'A 的第二次回答');
    switched.setCurrentSession('session_other');
    await vm.runInContext('refreshSessions()', switched.context);
    assert.equal(vm.runInContext('state.sessionId', switched.context), 'session_other');
    assert.ok(switched.element('#log-view').children.includes(secondAnswer));
  }

  const continuation = await browser();
  const longAnswer = '完整内容'.repeat(1200) + '原始回答结尾';
  continuation.setChatResponse(async () => ({ ok: true, answer: longAnswer, session_id: 'session_current',
    finish_reason: 'length', truncated: true }));
  continuation.element('#chat-text').value = '详细说明指标';
  await vm.runInContext('handleChatSend()', continuation.context);
  const partial = continuation.element('#log-view').children.at(-1);
  const notice = partial.children[1].children[1].children.at(-1);
  assert.equal(notice.className, 'chat-truncated-notice');
  assert.equal(notice.children[1].textContent, '继续回答');
  assert.equal(vm.runInContext('state.chatHistory.at(-1).content', continuation.context), longAnswer);
  vm.runInContext('state.sessionId = "other_session"', continuation.context);
  await notice.children[1].listeners.click();
  assert.equal(continuation.calls.filter(call => call.endpoint === '/api/chat').length, 1);
  vm.runInContext('state.sessionId = "session_current"', continuation.context);
  continuation.setChatResponse(async () => { throw new Error('临时网络错误'); });
  await notice.children[1].listeners.click();
  assert.equal(notice.children[1].disabled, false);
  assert.equal(notice.children[1].textContent, '重试续答');
  continuation.setChatResponse(async () => ({ ok: true, answer: '剩余内容', session_id: 'session_current', finish_reason: 'stop' }));
  await notice.children[1].listeners.click();
  const continued = continuation.calls.filter(call => call.endpoint === '/api/chat').at(-1);
  assert.equal(continued.body.history.at(-1).content, longAnswer);
  assert.equal(continued.body.session_id, 'session_current');
  assert.equal(notice.children[1].textContent, '已续答');
  assert.equal(notice.children[1].disabled, true);
  assert.equal(vm.runInContext('state.chatHistory.at(-1).content', continuation.context), '剩余内容');
  assert.equal(continuation.calls.filter(call => call.endpoint === '/api/run' && call.method === 'POST').length, 0);

  for (const outcome of ['success', 'api-error', 'network-error']) {
    const waiting = await browser();
    let resolveResponse;
    let rejectResponse;
    waiting.setChatResponse(() => new Promise((resolve, reject) => {
      resolveResponse = resolve;
      rejectResponse = reject;
    }));
    waiting.element('#chat-text').value = '你好';
    const request = vm.runInContext('handleChatSend()', waiting.context);
    const pending = waiting.element('#log-view').children.find(child => child.className.includes('chat-message-pending'));
    assert.ok(pending);
    assert.equal(pending.children[1].children[1].children[0].textContent, '正在思考，请稍候…');
    assert.equal(waiting.element('#btn-chat-send').disabled, true);
    assert.equal(waiting.element('#btn-chat-send').textContent, '等待回复…');
    const tick = [...waiting.timers.values()].find(timer => timer.delay === 1000);
    assert.ok(tick);
    tick.callback();
    assert.match(pending.children[1].children[1].children[1].textContent, /已等待 \d+ 秒/);
    waiting.element('#chat-text').value = '再发一次';
    await vm.runInContext('handleChatSend()', waiting.context);
    assert.equal(waiting.calls.filter(call => call.endpoint === '/api/chat').length, 1);
    waiting.element('#btn-clear-log').listeners.click();
    assert.ok(waiting.element('#log-view').children.includes(pending));
    if (outcome === 'network-error') rejectResponse(new Error('网络中断'));
    else resolveResponse(outcome === 'success'
      ? { ok: true, answer: '完成', session_id: 'session_current' }
      : { ok: false, error: '接口暂不可用' });
    await request;
    assert.ok(!waiting.element('#log-view').children.includes(pending));
    assert.ok(![...waiting.timers.values()].some(timer => timer.delay === 1000));
    assert.equal(waiting.element('#btn-chat-send').disabled, false);
    assert.equal(waiting.element('#btn-chat-send').textContent, '发送');
    assert.equal(vm.runInContext('state.chatBusy', waiting.context), false);
    assert.equal(vm.runInContext('state.pendingChatRow', waiting.context), null);
  }

  const cancelled = await browser(false);
  cancelled.element('#chat-text').value = '开始实验，使用真实数据，跑2轮';
  await vm.runInContext('handleChatSend()', cancelled.context);
  assert.equal(cancelled.calls.filter(call => call.endpoint === '/api/run' && call.method === 'POST').length, 0);

  const confirmed = await browser(true);
  confirmed.element('#chat-text').value = '开始实验，使用真实数据，跑2轮';
  await vm.runInContext('handleChatSend()', confirmed.context);
  const started = confirmed.calls.find(call => call.endpoint === '/api/run' && call.method === 'POST');
  assert.equal(started.body.mock, false);
  assert.equal(started.body.rounds, 2);
  assert.equal(confirmed.element('#btn-run').disabled, true);
  assert.equal(confirmed.streams.length, 1);

  const consultation = await browser(true);
  consultation.element('#chat-text').value = '为什么启动失败？';
  await vm.runInContext('handleChatSend()', consultation.context);
  consultation.element('#chat-text').value = '那下一步呢？';
  await vm.runInContext('handleChatSend()', consultation.context);
  const chats = consultation.calls.filter(call => call.endpoint === '/api/chat');
  assert.equal(chats.length, 2);
  assert.equal(chats[1].body.session_id, 'session_current');
  assert.equal(chats[1].body.history.length, 2);
  assert.equal(consultation.calls.filter(call => call.endpoint === '/api/run' && call.method === 'POST').length, 0);
  const messages = consultation.element('#log-view').children.filter(child => (child.className || '').startsWith('chat-message '));
  assert.equal(messages[0].className, 'chat-message chat-message-user');
  assert.equal(messages[0].children[1].children[1].textContent, '为什么启动失败？');
  assert.equal(messages[1].className, 'chat-message chat-message-assistant');
  assert.equal(messages[1].children[1].children[1].innerHTML, '<p>回答</p>');
  assert.equal(messages[1].children[1].children[0].children[1].textContent, '会话 · session_current');
  vm.runInContext('appendLog("[assistant] **结果**：`test`\\n\\n- 完成", "chat-ai", {sessionId: "session_current"})', consultation.context);
  const formatted = consultation.element('#log-view').children.at(-1);
  assert.equal(formatted.children[1].children[1].innerHTML, '<p><strong>结果</strong>：<code>test</code></p>\n<ul><li>完成</li></ul>');
  vm.runInContext('appendLog("[user] <img src=x> **原文**", "chat-user")', consultation.context);
  const user = consultation.element('#log-view').children.at(-1);
  assert.equal(user.children[1].children[1].textContent, '<img src=x> **原文**');
  vm.runInContext('appendLog("[stdout] **原始日志**")', consultation.context);
  const log = consultation.element('#log-view').children.at(-1);
  assert.equal(log.className, 'log-line stdout');
  assert.equal(log.textContent, '[stdout] **原始日志**');

  const restored = await browser();
  restored.setJobs([{ job_id: 'job_running', alive: true, session_id: 'session_current' }]);
  await vm.runInContext('restoreRunningJob()', restored.context);
  assert.equal(restored.streams[0].url, '/api/run/job_running/stream');
  assert.equal(restored.element('#btn-stop').disabled, false);
  assert.equal(restored.element('#btn-run').disabled, true);
  restored.streams[0].listeners.done({ data: JSON.stringify({ job_id: 'job_running', returncode: 1 }) });
  assert.equal(restored.element('#btn-run').disabled, false);
  assert.equal(restored.element('#btn-stop').disabled, true);
  assert.equal(restored.streams[0].readyState, 2);
  const background = await browser();
  vm.runInContext(`
    globalThis.pollCalls = [];
    globalThis.pollStages = [];
    globalThis.pollReads = 0;
    api = async (url, options = {}) => {
      pollCalls.push({url, method: options.method});
      if (options.method === 'POST') throw new TypeError('Failed to fetch');
      pollReads += 1;
      if (pollReads === 1) return {status: 'running', stage: '正在搜索和读取公开资料'};
      if (pollReads === 2) throw new TypeError('Failed to fetch');
      return {status: 'complete', result: {ok: true, answer: '根据真实来源回答'}};
    };
    setTimeout = (fn) => { Promise.resolve().then(fn); return 1; };
  `, background.context);
  const recovered = await vm.runInContext(`pollChat({request_id: 'background_request_0001'}, new AbortController().signal, value => pollStages.push(value))`, background.context);
  assert.equal(recovered.answer, '根据真实来源回答');
  assert.equal(vm.runInContext('pollCalls.filter(c => c.method === "POST").length', background.context), 1);
  assert.ok(vm.runInContext('pollStages.some(s => s.includes("不会重复生成"))', background.context));

  // A genuinely fresh browser VM restores server records and selection without POSTing to the model.
  const savedTurn = {request_id: 'persist_browser_request_1', session_id: 'session_other', order: '1791540000000000000',
    created_at: '2024-10-09T10:00:00Z', question: '<img src=x> 原始问题', status: 'complete',
    result: {ok: true, answer: '**完整保存的回答**', llm: {model: 'saved-model'},
      web_research: {sources: [{source_id: 'W1', title: '已保存来源', url: 'https://example.com/paper'}]}}};
  const savedHistory = [{role: 'user', content: savedTurn.question}, {role: 'assistant', content: savedTurn.result.answer}];
  const archiveResponse = sid => sid === 'session_other' ? {turns: [savedTurn], history: savedHistory, total: 1, next_before: null} : {turns: [], history: [], total: 0, next_before: null};
  const storage = new Map([['ets_chat_session', JSON.stringify({sessionId: 'session_other', followCurrent: false})]]);
  const refreshed = await browser(false, {storage, transcriptResponse: archiveResponse});
  assert.equal(vm.runInContext('state.sessionId', refreshed.context), 'session_other');
  assert.equal(vm.runInContext('state.chatHistory.at(-1).content', refreshed.context), savedTurn.result.answer);
  const restoredArchive = vm.runInContext('conversation().archiveNode', refreshed.context);
  const savedRows = restoredArchive.children.filter(node => node.className === 'chat-saved-turn');
  assert.equal(savedRows.length, 1);
  assert.equal(savedRows[0].children[1].children[1].children[1].textContent, savedTurn.question);
  assert.equal(savedRows[0].children[2].children[1].children[1].innerHTML, '<p><strong>完整保存的回答</strong></p>');
  assert.equal(refreshed.calls.filter(call => call.endpoint === '/api/chat').length, 0);
  await refreshed.element('#btn-restore-chat').listeners.click();
  assert.equal(vm.runInContext('conversation().archiveNode.children.filter(n => n.className === "chat-saved-turn").length', refreshed.context), 1);
  refreshed.element('#btn-clear-log').listeners.click();
  assert.equal(refreshed.element('#log-view').children.length, 0);
  await refreshed.element('#btn-restore-chat').listeners.click();
  assert.equal(vm.runInContext('conversation().transcript.size', refreshed.context), 1);
  refreshed.element('#chat-text').value = '刷新后的追问';
  await vm.runInContext('handleChatSend()', refreshed.context);
  const restoredFollowup = refreshed.calls.find(call => call.endpoint === '/api/chat');
  assert.equal(restoredFollowup.body.history[1].content, savedTurn.result.answer);
  assert.equal(restoredFollowup.body.session_id, 'session_other');

  // Refreshing an active turn restores waiting, not another model invocation.
  const pendingTurn = {...savedTurn, result: null, status: 'running'};
  const rewaiting = await browser(false, {storage: new Map(storage), transcriptResponse: () => ({turns: [pendingTurn], history: [], total: 1, next_before: null})});
  assert.equal(vm.runInContext('state.chatBusy', rewaiting.context), true);
  assert.equal(rewaiting.element('#btn-chat-send').disabled, true);
  assert.equal(rewaiting.calls.filter(call => call.endpoint === '/api/chat').length, 0);
  rewaiting.setTranscriptResponse(archiveResponse);
  await vm.runInContext('restoreConversation()', rewaiting.context);
  assert.equal(vm.runInContext('state.chatBusy', rewaiting.context), false);
  assert.equal(vm.runInContext('conversation().transcript.size', rewaiting.context), 1);

  // A slow archive read for A must never paint into B.
  const race = await browser();
  let resolveOld;
  race.setTranscriptResponse(sid => sid === 'session_current' ? new Promise(resolve => { resolveOld = resolve; }) : archiveResponse(sid));
  const slow = vm.runInContext('restoreConversation("session_current")', race.context);
  vm.runInContext('switchConversation("session_other")', race.context);
  await vm.runInContext('restoreConversation("session_other")', race.context);
  resolveOld({turns: [{...savedTurn, request_id: 'persist_browser_request_2', session_id: 'session_current', question: '只属于 A'}], history: [], total: 1, next_before: null});
  await slow;
  assert.equal(vm.runInContext('state.sessionId', race.context), 'session_other');
  assert.equal(vm.runInContext('conversation().transcript.values().next().value.question', race.context), savedTurn.question);
  race.setTranscriptResponse(() => { throw new Error('读取失败'); });
  await vm.runInContext('restoreConversation()', race.context);
  assert.match(vm.runInContext('conversation().loadError', race.context), /不代表记录为空/);
  assert.equal(vm.runInContext('conversation().transcript.size', race.context), 1);
  console.log('Frontend regression checks passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
