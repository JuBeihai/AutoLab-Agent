// ETS Agent 控制台前端逻辑

const $ = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

const state = {
  currentJobId: null,
  eventSource: null,
  seenExpIdsInLog: new Set(),
  configCache: null,
  sessionId: null,
  followCurrent: true,
  chatHistory: [],
  chatBusy: false,
  pendingChatRow: null,
  starting: false,
  restoring: false,
  conversations: new Map(),
  sessions: [],
  deletingSession: false,
  deletedSessions: new Set(),
  taskViews: new Map(),
  evaluationSelections: new Map(),
  creatingChat: false,
  sessionListButtons: [],
  sessionsRequest: 0,
};

try {
  const saved = JSON.parse(localStorage.getItem('ets_chat_session') || 'null');
  if (saved && /^[A-Za-z0-9_-]+$/.test(saved.sessionId || '') && saved.followCurrent === false) {
    state.sessionId = saved.sessionId;
    state.followCurrent = false;
  }
} catch (_) { /* Browser storage is optional; transcripts live on the server. */ }

function conversation(sessionId = state.sessionId) {
  if (!state.conversations.has(sessionId)) {
    state.conversations.set(sessionId, { nodes: [], history: [], busy: false, pending: null,
      draft: '', scroll: 0, seen: new Set(), empty: null, transcript: new Map(), liveRequests: new Set(),
      archiveNode: null, loaded: false, loadPromise: null, loadError: null, nextBefore: null,
      restoreTimer: null, restoredRequestId: null });
  }
  return state.conversations.get(sessionId);
}

function selectedLLMProfile() {
  return $('#llm-profile').value || 'chat';
}

async function loadLLMProfiles() {
  try {
    const data = await api('/api/llm-models');
    if (!Array.isArray(data.profiles) || !data.profiles.length) return;
    const select = $('#llm-profile');
    const current = select.value;
    let saved = null;
    try { saved = localStorage.getItem('ets_llm_profile'); } catch (_) {}
    select.replaceChildren();
    for (const profile of data.profiles) {
      const option = document.createElement('option');
      option.value = profile.profile_id;
      option.textContent = profile.label || profile.model;
      select.appendChild(option);
    }
    select.value = [saved, current, data.default_profile].find(value => data.profiles.some(profile => profile.profile_id === value)) || data.profiles[0].profile_id;
  } catch (error) { console.warn('模型列表读取失败，保留当前选项', error); }
}

$('#llm-profile').addEventListener('change', () => {
  try { localStorage.setItem('ets_llm_profile', selectedLLMProfile()); } catch (_) {}
});
loadLLMProfiles();

function syncChatState() {
  const current = conversation();
  state.chatHistory = current.history;
  state.chatBusy = current.busy || Boolean(current.restoredRequestId);
  state.pendingChatRow = current.pending;
  state.seenExpIdsInLog = current.seen;
  $('#btn-chat-send').disabled = state.chatBusy;
  $('#btn-chat-send').textContent = state.chatBusy ? '等待回复…' : '发送';
  updateInterruptButton();
  updateDeleteSessionButton();
}

function refreshPinnedTask() {
  const host = $('#pinned-task');
  const views = [...state.taskViews.values()].filter(view => view.sessionId === state.sessionId && !state.deletedSessions.has(view.sessionId));
  const selected = views.filter(view => !view.finished).pop() || views[views.length - 1];
  // Keep earlier runs in their conversation; only the current run occupies the fixed area.
  for (const view of views) {
    if (view !== selected && view.card.parentNode === host) $('#log-view').appendChild(view.card);
  }
  host.hidden = !selected;
  if (!selected) host.replaceChildren();
  else if (selected.card.parentNode !== host) host.replaceChildren(selected.card);
}

function appendConversationNode(node, sessionId = state.sessionId) {
  const target = conversation(sessionId);
  if (target.empty) { target.empty.remove(); target.empty = null; }
  target.nodes.push(node);
  if (sessionId === state.sessionId) $('#log-view').appendChild(node);
}

function switchConversation(sessionId) {
  sessionId = sessionId || null;
  if (sessionId === state.sessionId) return;
  const previous = conversation();
  previous.draft = $('#chat-text').value;
  previous.scroll = $('#log-view').scrollTop;
  _closeMerge();
  state.sessionId = sessionId;
  const current = conversation();
  $('#log-view').innerHTML = '';
  for (const node of current.nodes) $('#log-view').appendChild(node);
  refreshPinnedTask();
  if (!current.nodes.length) {
    const empty = document.createElement('div');
    empty.className = 'chat-empty';
    empty.textContent = sessionId && !current.loaded ? `正在加载会话 ${sessionId} 的对话存档…` : `会话 ${sessionId || '暂无'} 暂无对话，可以在下方开始咨询。`;
    current.empty = empty;
    $('#log-view').appendChild(empty);
  }
  $('#chat-text').value = current.draft;
  $('#log-view').scrollTop = current.scroll;
  syncChatState();
  rememberChatSession();
  if (!current.loaded || current.restoredRequestId) restoreConversation(sessionId);
}

function rememberChatSession() {
  try { localStorage.setItem('ets_chat_session', JSON.stringify({sessionId: state.sessionId, followCurrent: state.followCurrent})); } catch (_) {}
}

function decorateChatAnswer(message, result, sessionId, history) {
  appendWebSources(message, result.web_research);
  if (result.llm && result.llm.model) {
    const label = document.createElement('div'); label.className = 'muted-tiny';
    label.textContent = `回答模型：${result.llm.model}`; message.bubble.appendChild(label);
  }
  if (result.history_sources && result.history_sources.length) {
    const references = document.createElement('details'), label = document.createElement('summary'), entries = document.createElement('div');
    label.textContent = `本次提供给助手的历史参考 · ${result.history_sources.length} 条`;
    entries.textContent = result.history_sources.map(source => `${source.session_id} / ${source.experiment_id}`).join('；');
    references.appendChild(label); references.appendChild(entries); message.bubble.appendChild(references);
  }
  if (result.persistence_warning) {
    const warning = document.createElement('p'); warning.className = 'error'; warning.textContent = result.persistence_warning;
    message.bubble.appendChild(warning);
  }
  if (result.truncated || result.finish_reason === 'length') addContinuation(message, sessionId, history);
}

function renderTranscript(sessionId) {
  const target = conversation(sessionId);
  if (!target.transcript.size && !target.loadError && !target.archiveNode) return;
  if (!target.archiveNode || !target.nodes.includes(target.archiveNode)) {
    target.archiveNode = document.createElement('section'); target.archiveNode.className = 'chat-archive';
    appendConversationNode(target.archiveNode, sessionId);
    target.nodes = [target.archiveNode, ...target.nodes.filter(node => node !== target.archiveNode)];
    if (sessionId === state.sessionId && $('#log-view').prepend) $('#log-view').prepend(target.archiveNode);
  }
  const archive = target.archiveNode; archive.innerHTML = '';
  const info = document.createElement('p'); info.className = 'muted-tiny'; info.setAttribute('role', 'status');
  info.textContent = target.loadError || `会话 ${sessionId} · 已保存 ${target.savedTotal || 0} 轮对话${target.skipped ? `（${target.skipped} 条记录损坏，未隐藏错误）` : ''}`;
  archive.appendChild(info);
  if (target.nextBefore) {
    const more = document.createElement('button'); more.type = 'button'; more.textContent = '加载更早的对话';
    more.addEventListener('click', () => restoreConversation(sessionId, target.nextBefore)); archive.appendChild(more);
  }
  const turns = [...target.transcript.values()].sort((a, b) => a.order.localeCompare(b.order));
  for (const turn of turns) {
    if (target.liveRequests.has(turn.request_id)) continue;
    const holder = document.createElement('div'); holder.className = 'chat-saved-turn'; archive.appendChild(holder);
    const time = document.createElement('div'); time.className = 'muted-tiny';
    time.textContent = `${turn.created_at ? new Date(turn.created_at).toLocaleString() : ''}${turn.recovered ? ' · 旧日志恢复（可能截断）' : ''}`;
    holder.appendChild(time);
    if (turn.question) appendChatMessage('user', turn.question, sessionId, holder);
    const result = turn.result || {};
    if (result.answer) {
      const message = appendChatMessage('assistant', result.answer, sessionId, holder);
      // Only offer continuation for the latest complete answer in this session.
      const latest = turns[turns.length - 1] === turn;
      decorateChatAnswer(message, latest ? result : {...result, truncated: false, finish_reason: 'stop'}, sessionId, target.history);
    }
    if (turn.recovered || turn.status === 'running' || !result.answer) {
      const note = document.createElement('p'); note.className = 'chat-truncated-notice';
      note.textContent = turn.note || (turn.status === 'running' ? '后台仍在回复，已恢复等待；无需重复发送。' : result.error || (turn.status === 'cancelled' ? '此回复已中断，提问已保存。' : '没有完整回复。'));
      holder.appendChild(note);
    }
  }
}

async function restoreConversation(sessionId = state.sessionId, before = null) {
  if (!sessionId || state.deletedSessions.has(sessionId)) return true;
  const target = conversation(sessionId);
  if (target.loadPromise) return target.loadPromise;
  clearTimeout(target.restoreTimer);
  target.loadPromise = (async () => {
    try {
      const data = await api(`/api/sessions/${encodeURIComponent(sessionId)}/chat${before ? `?before=${encodeURIComponent(before)}` : ''}`);
      if (!Array.isArray(data.turns)) throw new Error('聊天存档响应格式无效');
      if (state.deletedSessions.has(sessionId)) return false;
      for (const turn of data.turns) target.transcript.set(turn.request_id, turn);
      target.savedTotal = data.total; target.skipped = data.skipped;
      if (before || !target.loaded) target.nextBefore = data.next_before;
      target.loaded = true; target.loadError = null;
      if (!target.busy && !target.liveRequests.size) target.history = data.history || [];
      target.restoredRequestId = [...target.transcript.values()].find(turn => turn.status === 'running' && !target.liveRequests.has(turn.request_id))?.request_id || null;
      renderTranscript(sessionId);
      if (!target.transcript.size && target.empty) target.empty.textContent = `会话 ${sessionId} 暂无已存档对话，可以在下方开始咨询。`;
      if (sessionId === state.sessionId) syncChatState();
      if (target.restoredRequestId) target.restoreTimer = setTimeout(() => restoreConversation(sessionId), 2000);
      return true;
    } catch (error) {
      target.loadError = `聊天记录加载失败：${error.message}。请点“恢复对话”重试，不代表记录为空。`;
      renderTranscript(sessionId);
      if (target.restoredRequestId) target.restoreTimer = setTimeout(() => restoreConversation(sessionId), 5000);
      return false;
    } finally { target.loadPromise = null; }
  })();
  return target.loadPromise;
}

// ================================================================
// Utilities
// ================================================================

async function api(path, opts = {}) {
  const timeoutMs = path === '/api/chat' || path.startsWith('/api/chat/') ? 15000 : path === '/api/run' && opts.method !== 'POST' ? 8000 : 30000;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort();
    else opts.signal.addEventListener('abort', onAbort, {once: true});
  }
  let timer;
  if (state.sessionId && ["/api/champion", "/api/discovery", "/api/memory", "/api/experiments", "/api/lineage"].includes(path)) {
    path += `?session_id=${encodeURIComponent(state.sessionId)}`;
  }
  const request = async () => {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      cache: 'no-store',
      ...opts,
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${res.status} ${text}`);
    }
    return res.json();
  };
  try {
    if (!timeoutMs) return await request();
    return await Promise.race([request(), new Promise((resolve, reject) => {
      timer = setTimeout(() => {
        reject(new Error('请求超时，请检查连接后重试；超时不代表后台任务已停止'));
        controller.abort();
      }, timeoutMs);
    })]);
  } finally {
    clearTimeout(timer);
    if (opts.signal) opts.signal.removeEventListener('abort', onAbort);
  }
}

function pct(v) {
  if (v == null || isNaN(v)) return "—";
  return (v * 100).toFixed(1) + "%";
}
function pp(v) {
  if (v == null || isNaN(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(1)}pp`;
}
function fmtNum(v, d = 3) {
  if (v == null || isNaN(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return s + Number(v).toFixed(d);
}

function classifyLine(line) {
  if (/^\[runner\]/.test(line)) return "runner";
  if (/gate.*pass=True/i.test(line)) return "gate-pass";
  if (/失败|error|Traceback/i.test(line)) return "error";
  if (/^\[stderr\]/.test(line)) return "stderr";
  if (line.includes("[llm-think]") || line.includes("[sum-think]")) return "llm-think";
  if (line.includes("[llm-out]") || line.includes("[sum-out]")) return "llm-out";
  if (line.includes("[orch]")) return "orch";
  if (line.includes("[trainer]")) return "trainer";
  if (line.includes("[propose]")) return "propose";
  return "stdout";
}

// ---------- 合并块状态：相邻同类 llm-think/llm-out/sum-* 行拼进同一个 DOM ----------
let _mergeCtx = null;  // {kind: "llm-think"|"llm-out"|..., wrap: <details>, body: <div>}

const _MERGE_KINDS = new Set(["llm-think", "llm-out"]);
const _MERGE_LABEL = {
  "llm-think": "🧠 Agent 思考（点击折叠）",
  "llm-out":   "📤 Agent 输出",
};

function _closeMerge() { _mergeCtx = null; }

function _stripPrefix(line) {
  // 去掉 "[stdout] [llm-think] " / "[llm-think] " 这类装饰前缀，留纯正文
  return line
    .replace(/^\[stdout\]\s*/, "")
    .replace(/^\[stderr\]\s*/, "")
    .replace(/^\[llm-think\]\s*/, "")
    .replace(/^\[llm-out\]\s*/, "")
    .replace(/^\[sum-think\]\s*/, "")
    .replace(/^\[sum-out\]\s*/, "");
}

function _appendMergedChunk(kind, line) {
  if (!_mergeCtx || _mergeCtx.kind !== kind) {
    const wrap = document.createElement("details");
    wrap.className = "merge-block " + kind;
    wrap.open = true;
    const sum = document.createElement("summary");
    sum.textContent = _MERGE_LABEL[kind] || kind;
    const body = document.createElement("div");
    body.className = "merge-body";
    wrap.appendChild(sum);
    wrap.appendChild(body);
    appendConversationNode(wrap);
    _mergeCtx = {kind, wrap, body};
  }
  _mergeCtx.body.append(_stripPrefix(line) + " ");
  // 合并块内部也要滚到底
  if ($("#autoscroll").checked) {
    _mergeCtx.body.scrollTop = _mergeCtx.body.scrollHeight;
  }
}

function appendChatMessage(role, text, sessionId = state.sessionId, parent = null) {
  _closeMerge();
  const row = document.createElement('article');
  row.className = `chat-message chat-message-${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'chat-avatar';
  avatar.textContent = role === 'user' ? '我' : 'AI';
  avatar.setAttribute('aria-hidden', 'true');
  const content = document.createElement('div');
  content.className = 'chat-message-content';
  const header = document.createElement('div');
  header.className = 'chat-message-header';
  const label = document.createElement('span');
  label.textContent = role === 'user' ? '你' : 'Assistant';
  header.appendChild(label);
  if (role === 'assistant' && sessionId) {
    const session = document.createElement('span');
    session.className = 'chat-session';
    session.textContent = `会话 · ${sessionId}`;
    session.setAttribute('title', sessionId);
    header.appendChild(session);
  }
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble${role === 'assistant' ? ' markdown-body' : ''}`;
  if (role === 'assistant') bubble.innerHTML = ETSMarkdown.render(text);
  else bubble.textContent = text;
  content.appendChild(header);
  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  if (parent) parent.appendChild(row);
  else appendConversationNode(row, sessionId);
  return { row, bubble };
}

function showChatWaiting(sessionId) {
  const { row, bubble } = appendChatMessage('assistant', '', sessionId);
  row.className += ' chat-message-pending';
  const target = conversation(sessionId);
  target.pending = row;
  if (sessionId === state.sessionId) syncChatState();
  const status = document.createElement('span');
  status.className = 'chat-waiting-status';
  status.setAttribute('role', 'status');
  status.textContent = '正在思考，请稍候…';
  const elapsed = document.createElement('span');
  elapsed.className = 'chat-waiting-elapsed';
  elapsed.setAttribute('aria-hidden', 'true');
  bubble.appendChild(status);
  bubble.appendChild(elapsed);
  const startedAt = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    elapsed.textContent = seconds >= 30
      ? `已等待 ${seconds} 秒 · 回复耗时较长，请耐心等待`
      : `已等待 ${seconds} 秒`;
  }, 1000);
  if ($('#autoscroll').checked) {
    $('#log-view').scrollTop = $('#log-view').scrollHeight;
  }
  const cleanup = () => {
    clearInterval(timer);
    row.remove();
    target.nodes = target.nodes.filter(node => node !== row);
    if (target.pending === row) target.pending = null;
    if (sessionId === state.sessionId) syncChatState();
  };
  cleanup.setStatus = value => { status.textContent = value; };
  return cleanup;
}

function appendLog(line, cls, metadata = {}) {
  const kind = cls || classifyLine(line);
  const sessionId = Object.prototype.hasOwnProperty.call(metadata, 'sessionId') ? metadata.sessionId : state.sessionId;
  let message = null;

  // llm-think / llm-out / sum-think / sum-out 走合并块（sum-* 复用 llm-* 样式）
  const mergeKind = kind === "sum-think" ? "llm-think"
                  : kind === "sum-out"   ? "llm-out"
                  : kind;
  if (kind === 'chat-user' || kind === 'chat-ai') {
    const text = line.replace(/^\[(?:user|assistant|user-hint)\]\s?/, '');
    message = appendChatMessage(kind === 'chat-user' ? 'user' : 'assistant', text, sessionId);
  } else if (_MERGE_KINDS.has(mergeKind) && sessionId === state.sessionId) {
    _appendMergedChunk(mergeKind, line);
  } else {
    _closeMerge();  // 遇到别的类型，前一个合并块封口
    const el = document.createElement("div");
    el.className = "log-line " + kind;
    el.textContent = line;
    appendConversationNode(el, sessionId);
  }

  if (sessionId === state.sessionId && $("#autoscroll").checked) {
    $("#log-view").scrollTop = $("#log-view").scrollHeight;
  }

  // SSE 通道的行由后端 runner 落盘；前端产生的 UI 事件（chat / user-hint / summary-card）走 /api/log/write
  const skipDisk = /^\[stdout\]|^\[stderr\]|^\[runner\]/.test(line);
  if (!skipDisk) {
    try {
      fetch("/api/log/write", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({line: metadata.sessionId ? `[会话 ${metadata.sessionId}] ${line}` : line}),
        keepalive: true,
      }).catch(() => {});
    } catch (_) {}
  }
  return message;
}

// 缓存 baseline metrics（用于分城市对照）
async function _ensureReferenceRecord(record) {
  try {
    const parentId = (record.config_diff || {}).parent_experiment_id;
    if (parentId) {
      const { record: parent } = await api(`/api/experiments/${encodeURIComponent(parentId)}`);
      return parent;
    }
    const sessionId = (record.config_diff || {}).session_id;
    if (!sessionId) return null;
    const { baseline } = await api(`/api/champion?session_id=${encodeURIComponent(sessionId)}`);
    return baseline || null;
  } catch (_) {}
  return null;
}

function evaluationLabel(split) {
  if (split === 'validation') return '训练期内时间验证集（参与调参）';
  if (split === 'retrospective_test') return '历史测试回测（曾参与旧实验搜索）';
  if (split === 'test') return '测试集（统一评估，参与候选筛选）';
  if (split === 'train_internal_early_stop') return '历史训练内早停子集（非测试集）';
  return '历史验证集';
}

function selectionMetric(record) {
  return ((record || {}).config_diff || {}).selection_metric || 'xgb_44';
}

function selectionLabel(key) {
  if (key === 'answer_only_h3_7_6bin') return '6桶·固定已应答·H3-7保序后';
  if (key === 'answer_cancel_h3_7_6bin') return '6桶·应答＋取消过滤·H3-7保序后';
  return key === 'all_h3_7' ? '全量·H3-7保序后' : '历史口径·全量保序前';
}

function selectionMetrics(record) {
  return ((record || {}).metrics_by_city || {})[selectionMetric(record)] || {};
}

function selectedEvaluation(record, population, version) {
  if (!record) return { unavailable_reason: '未找到对照模型记录' };
  const saved = (((record.evaluation_views || {}).populations || {})[population] || {})[version];
  if (saved) return saved;
  const legacyKeys = { all: 'xgb_44', answer_only: 'notebook_answer_only', answer_cancel: 'notebook_answer_cancel' };
  const legacy = (record.metrics_by_city || {})[legacyKeys[population]];
  if (version === 'raw' && legacy) return { metrics: legacy };
  return { unavailable_reason: '该实验未保存此人群／保序版本的指标，需要使用新流程重新运行；不会用保序前数据代替' };
}

function evaluationTable(current, reference, currentLabel, referenceLabel, compare) {
  const cities = [['100001', '示例区域A'], ['100002', '示例区域B'], ['100003', '示例区域C'], ['ALL', '全量']];
  if (!current.metrics || (compare && !reference.metrics)) {
    return `<p class="hint">${escapeHTML(!current.metrics ? current.unavailable_reason : reference.unavailable_reason)}</p>`;
  }
  const rows = cities.map(([code, name]) => {
    const currentRow = current.metrics[code];
    const referenceRow = compare ? reference.metrics[code] : null;
    if (!currentRow && !referenceRow) return '';
    const value = (row, key) => row && row.total > 0 ? pct(row[key]) : '—';
    const metrics = ['recall_ratio', 'over_ratio', 'under_ratio'].map(key => {
      if (!compare) return `<td>${value(currentRow, key)}</td>`;
      const delta = currentRow && referenceRow && currentRow.total > 0 && referenceRow.total > 0
        ? pp(currentRow[key] - referenceRow[key]) : '—';
      return `<td>${value(referenceRow, key)}</td><td>${value(currentRow, key)}</td><td>${delta}</td>`;
    }).join('');
    return `<tr><td>${name}<br><span class="muted-tiny">${code}</span></td>` +
      (compare ? `<td>${referenceRow ? referenceRow.total : '—'}</td>` : '') +
      `<td>${currentRow ? currentRow.total : '—'}</td>${metrics}</tr>`;
  }).join('');
  const headings = ['准确率 (acc)', '高估 (over)', '低估 (under)'];
  const header = compare
    ? `<tr><th rowspan="2">城市</th><th rowspan="2">${escapeHTML(referenceLabel)}样本</th><th rowspan="2">${escapeHTML(currentLabel)}样本</th>${headings.map(label => `<th colspan="3">${label}</th>`).join('')}</tr><tr>${headings.map(() => `<th>${escapeHTML(referenceLabel)}</th><th>${escapeHTML(currentLabel)}</th><th>Δ</th>`).join('')}</tr>`
    : `<tr><th>城市</th><th>样本</th>${headings.map(label => `<th>${label}</th>`).join('')}</tr>`;
  return `<div class="city-compare-scroll"><table class="city-compare ${compare ? '' : 'single-model'}"><thead>${header}</thead><tbody>${rows || `<tr><td colspan="${compare ? 12 : 5}">该人群无可评估样本</td></tr>`}</tbody></table></div>`;
}

function mountEvaluation(container, record, reference = null, currentLabel = '本次模型', referenceLabel = '对照模型', compare = false) {
  const selectionKey = `${state.sessionId}:${record.experiment_id || 'test-summary'}:${compare}`;
  const fixedAnswer = selectionMetric(record) === 'answer_only_h3_7_6bin';
  const sixBucket = fixedAnswer || selectionMetric(record) === 'answer_cancel_h3_7_6bin';
  const savedSelection = state.evaluationSelections.get(selectionKey) || { population: fixedAnswer ? 'answer_only' : sixBucket ? 'answer_cancel' : 'all', version: sixBucket || selectionMetric(record) === 'all_h3_7' ? 'h3_7' : 'raw' };
  const controls = document.createElement('div');
  controls.className = 'evaluation-controls';
  const population = document.createElement('select');
  population.setAttribute('aria-label', '评估人群');
  population.innerHTML = '<option value="all">全量</option><option value="answer_only">仅应答</option><option value="answer_cancel">应答＋取消过滤</option>';
  population.value = savedSelection.population;
  const version = document.createElement('select');
  version.setAttribute('aria-label', '预测版本');
  version.innerHTML = '<option value="raw">保序前</option><option value="h3_6">H3-6</option><option value="h3_7">H3-7</option><option value="h3_8">H3-8</option>';
  version.value = savedSelection.version;
  for (const [title, select] of [['评估人群：', population], ['预测版本：', version]]) {
    const label = document.createElement('label');
    label.textContent = title;
    label.appendChild(select);
    controls.appendChild(label);
  }
  const note = document.createElement('p');
  note.className = 'hint';
  const table = document.createElement('div');
  const render = () => {
    state.evaluationSelections.set(selectionKey, { population: population.value, version: version.value });
    note.textContent = `${currentLabel}${compare ? ` vs ${referenceLabel}` : ''}。先按保序前预测筛选人群，再在人群内独立执行所选级别保序，不在保序后重新过滤。` +
      (population.value === 'answer_cancel' ? '取消过滤使用真实标签和各模型自身预测，两模型人群可能不同，Δ仅供描述性对照。' : '') +
      (version.value !== 'raw' ? (sixBucket ? ((record.evaluation_views || {}).non_queue_after_ordering === 'model' ? '统一模型预测全部订单，保序后不强制归零。' : '保序后非排队强制回到第0桶（≤1分钟）。') : '保序沿用 Notebook 离队时间逻辑，仅作离线对照；非排队预测也可能被保序抬高。') : '') +
      (sixBucket && (record.config_diff || {}).reference_baseline ? `示例阈值：${pct(record.config_diff.reference_baseline.metrics.ALL.recall_ratio)}（人工示例值）；本会话重训对照另行记录。` : '') +
      `切换只影响本表；本实验门禁与冠军口径：${selectionLabel(selectionMetric(record))}。` +
      (fixedAnswer ? '自主模型实验同时报告准确率最高方案和满足低估约束的推荐方案，最终结果以跨窗口复验汇总为准。' : '低估率仅供观察；误差切片仍是保序前诊断，晋级收益单独按门禁口径计算。');
    table.innerHTML = evaluationTable(selectedEvaluation(record, population.value, version.value),
      selectedEvaluation(reference, population.value, version.value), currentLabel, referenceLabel, compare);
  };
  population.addEventListener('change', render);
  version.addEventListener('change', render);
  container.appendChild(controls);
  container.appendChild(note);
  container.appendChild(table);
  render();
}

function renderSearchDiagnostics(record) {
  const summary = record.summary || {};
  const diagnostics = summary.search_diagnostics;
  if (!diagnostics) return '';
  const changes = diagnostics.prediction_changes || {};
  const comparisons = Object.entries(changes).map(([reference, value]) =>
    `<p>保序前诊断 · ${reference === 'parent' ? '相对父模型' : reference === 'champion' ? '相对比较时冠军' : '相对初始 baseline'}：纠正 ${escapeHTML(value.corrected)} 行，新增错误 ${escapeHTML(value.introduced)} 行；准确率变化 ${Number(value.delta_pp).toFixed(4)} pp</p>`).join('');
  const edit = (record.config_diff || {}).candidate_changes;
  const rows = Object.entries(diagnostics.features || {}).map(([name, value]) => {
    const fitting = Object.entries(value.fitting || {}).map(([tag, stats]) =>
      `${tag}: ${stats.rows} 行 / 缺失 ${stats.missing_rate == null ? '未知' : (stats.missing_rate * 100).toFixed(1) + '%'} / ${stats.unique} 种取值`).join('；');
    const usage = Object.entries(value.model_usage || {}).map(([tag, stats]) =>
      `${tag}: ${stats == null ? '不可用' : stats.split_count + ' 次分裂'}`).join('；');
    return `<tr><td>${escapeHTML(name)}</td><td>${escapeHTML(fitting)}</td>
      <td>${value.test && value.test.missing_rate != null ? (value.test.missing_rate * 100).toFixed(1) + '%' : '未记录'}</td>
      <td>${escapeHTML(usage)}</td><td>${escapeHTML((value.warnings || []).join('；') || '无')}
      ${value.fully_missing_dates && value.fully_missing_dates.length ? '<br>全缺失日期：' + escapeHTML(value.fully_missing_dates.join('、')) : ''}</td></tr>`;
  }).join('');
  const errors = diagnostics.errors || {};
  const selection = diagnostics.feature_selection || {};
  const importance = Object.entries(selection.models || {}).map(([tag, model]) => {
    const permutation = Object.fromEntries((model.permutation || []).map(item => [item.name, item]));
    const ranking = (model.ranking || []).map(item => `<tr><td>${escapeHTML(item.name)}</td>
      <td>${(Number(item.gain_share) * 100).toFixed(2)}%</td><td>${escapeHTML(item.split_count)}</td>
      <td>${item.missing_rate == null ? '—' : (item.missing_rate * 100).toFixed(2) + '%'}</td>
      <td>${permutation[item.name] ? Number(permutation[item.name].mean_acc_drop_pp).toFixed(4) + ' pp' : '未抽查'}</td></tr>`).join('');
    const redundant = (model.high_correlation_pairs || []).map(pair =>
      `${pair.first} / ${pair.second} (${Number(pair.spearman).toFixed(3)})`).join('；');
    return `<details><summary>${escapeHTML(tag)} · 全部输入的重要性与冗余</summary>
      <p>实际拟合 ${escapeHTML(model.fitting_rows)} 行；置换抽查 ${escapeHTML(model.permutation_rows)} 行。置换值是训练内准确率下降，不是独立泛化收益。</p>
      <p>待验证消融候选：${escapeHTML((model.ablation_candidates || []).join('、') || '无')}</p>
      <p>高相关特征对：${escapeHTML(redundant || '无')}</p>
      <table><thead><tr><th>特征</th><th>总 gain 占比</th><th>分裂数</th><th>训练缺失率</th><th>置换下降</th></tr></thead><tbody>${ranking}</tbody></table>
      <p class="hint">${escapeHTML(selection.note || '')}</p></details>`;
  }).join('');
  return `<details class="search-diagnostics"><summary>搜索诊断 · 错误变化与特征有效性</summary>
    ${importance}
    ${summary.research_plan && summary.research_plan.action ? `<p>研究动作：${escapeHTML(summary.research_plan.action)} / ${escapeHTML(summary.research_plan.direction)}<br>${escapeHTML(summary.research_plan.reason)}<br>后备方向：${escapeHTML(summary.research_plan.next_if_failed)}</p>` : ''}
    <p>${escapeHTML(summary.selection_note || '')}</p>${comparisons}
    ${edit && edit.feature_count != null ? `<p>本次模型输入：${escapeHTML(edit.feature_count)} 个；相对父实验新增：${escapeHTML((edit.added || []).join('、') || '无')}；删除：${escapeHTML((edit.removed || []).join('、') || '无')}<br>参数改动：${escapeHTML(JSON.stringify(edit.xgb_params || {}))}；类别权重：${escapeHTML(edit.class_weight_scheme)}</p>` : ''}
    <p>全量错误 ${escapeHTML(errors.errors ?? '-')} 行；排队错误 ${escapeHTML(errors.queue_errors ?? '-')} 行；非排队错误 ${escapeHTML(errors.non_queue_errors ?? '-')} 行。</p>
    <p class="hint">${escapeHTML(errors.note || '')} 分裂次数不代表因果收益。</p>
    ${rows ? '<table><thead><tr><th>新增特征</th><th>实际拟合人群</th><th>测试缺失率</th><th>模型使用</th><th>覆盖提示</th></tr></thead><tbody>' + rows + '</tbody></table>' : '<p class="hint">基线暂无新增特征诊断。</p>'}
    </details>`;
}

async function appendSummaryCard(rec) {
  const sessionId = state.sessionId;
  const s = rec.summary || {};
  const inn = s.innovation || {};
  const badgeClass = "badge-" + (s.verdict || "mixed");
  const box = document.createElement("div");
  box.className = "summary-card";

  const isBaseline = (rec.proposer === "baseline") || (s.verdict === "baseline");
  const challenger = selectionMetrics(rec);
  const referenceRecord = isBaseline ? null : await _ensureReferenceRecord(rec);
  const baseline = referenceRecord ? selectionMetrics(referenceRecord) : null;
  if (sessionId !== state.sessionId) return;

  // 门禁徽章
  const gate = rec.gate_pass;
  const gateHtml = gate == null ? "" :
    `<span class="sc-gate ${gate ? 'gate-ok' : 'gate-fail'}">${gate ? '✓ 通过门禁' : '✗ 未通过'}</span>`;
  const reasons = (rec.gate_reasons || []).length
    ? `<div class="sc-reasons">${(rec.gate_reasons || []).map(r => `· ${escapeHTML(r)}`).join("<br>")}</div>`
    : "";

  // 分城市对照表（baseline 记录只展示自己）
  box.innerHTML = `
    <div class="sc-title">
      <span class="sc-id">${escapeHTML(rec.experiment_id)}</span>
      · <span class="sc-inn">${escapeHTML(inn.title || "（无创新点）")}</span>
      <span class="sc-badge ${badgeClass}">${escapeHTML(s.verdict || "-")}</span>
      ${gateHtml}
    </div>
    <div class="sc-narr">${escapeHTML(s.narrative_zh || "（无总结）")}</div>
    <div class="hint">评估口径：${evaluationLabel((rec.config_diff || {}).evaluation_split)} · ${selectionLabel(selectionMetric(rec))}</div>
    ${reasons}
    ${renderSearchDiagnostics(rec)}
  `;
  const studyRecord = (rec.config_diff || {}).model_study;
  const sameFold = referenceRecord && (referenceRecord.config_diff || {}).fold_index === (rec.config_diff || {}).fold_index &&
    (referenceRecord.config_diff || {}).model_seed === (rec.config_diff || {}).model_seed;
  mountEvaluation(box, rec, referenceRecord, isBaseline ? 'baseline' : 'challenger', '对照模型', !isBaseline && (!studyRecord || sameFold));
  appendConversationNode(box);
  if ($("#autoscroll").checked) {
    $("#log-view").scrollTop = $("#log-view").scrollHeight;
  }
  // Summary 卡片也落盘：一行紧凑格式，便于命令行 grep
  try {
    const deltaLines = ["100001","100002","100003","ALL"].map(c => {
      const b = baseline ? (baseline[c] || baseline[String(c)]) : null;
      const ch = challenger ? (challenger[c] || challenger[String(c)]) : null;
      if (!b || !ch) return "";
      return `${c} acc ${(b.recall_ratio*100).toFixed(1)}→${(ch.recall_ratio*100).toFixed(1)}`;
    }).filter(Boolean).join(" · ");
    const flat = `[summary-card] ${rec.experiment_id} verdict=${s.verdict || "-"} ` +
                 `innovation="${(inn.title || "").replace(/\n/g, " ")}" ` +
                 `delta="${deltaLines}" narrative="${(s.narrative_zh || "").replace(/\n/g, " ")}"`;
    fetch("/api/log/write", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({line: flat}),
      keepalive: true,
    }).catch(() => {});
  } catch (_) {}
}

function escapeHTML(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g,
    ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function modelRatioLabel(record = {}) {
  const config = record.config_diff || record;
  const ensemble = config.ensemble;
  if (ensemble) {
    const weight = ensemble.mlp_weight;
    if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0 || weight > 1) return '融合（配比未记录）';
    if (weight === 0) return '纯 XGB（100%）';
    if (weight === 1) return '纯 MLP（100%）';
    const percent = value => Number((100 * value).toFixed(4)) + '%';
    return `MLP ${percent(weight)} ＋ XGB ${percent(1 - weight)}`;
  }
  const family = config.model_family || config.experiment_spec?.train_config?.model_family;
  if (family === 'xgb') return '纯 XGB（100%）';
  if (family === 'mlp') return '纯 MLP（100%）';
  return '未记录';
}

// ================================================================
// Tabs
// ================================================================
$$(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach(t => t.classList.remove("active"));
    $$(".tab-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    $(`.tab-panel[data-panel="${tab.dataset.tab}"]`).classList.add("active");
    if (tab.dataset.tab === "history") loadHistory();
    if (tab.dataset.tab === "lineage") loadLineage();
    if (tab.dataset.tab === "memory") loadMemory();
    if (tab.dataset.tab === "discovery") loadDiscovery();
    if (tab.dataset.tab === "models") loadModels();
    if (tab.dataset.tab === "tuning") loadTuning();
    if (tab.dataset.tab === "hints") loadHints();
  });
});

// ================================================================
// Champion 卡片
// ================================================================
async function refreshCityTraining(session) {
  const panel = $('#city-training-summary');
  if (!session || !session.city_group_id) { panel.hidden = true; panel.replaceChildren(); return; }
  const selectedSession = state.sessionId;
  const group = await api(`/api/city-training/${encodeURIComponent(session.city_group_id)}`);
  if (selectedSession !== state.sessionId) return;
  panel.hidden = false;
  const labels = {pending:'等待', running:'运行中', loading:'取数中', profiled:'画像完成', searching:'搜索中', complete:'完成', profile_complete:'画像完成', failed:'失败', cancelled:'已停止', archived:'已归档'};
  panel.innerHTML = `<strong>三城独立训练 · 各城市独立保留最优模型</strong><div class="table-wrap"><table><thead><tr><th>城市</th><th>状态</th><th>本次基线</th><th>最优模型</th><th>城市参考门槛</th><th>门禁 / 更新</th><th>会话</th></tr></thead><tbody>${Object.entries(group.cities).map(([city, entry]) => {
    const outcome = entry.updated ? (entry.gate && entry.gate.pass ? '通过 · 已更新' : '未通过') : entry.selected ? '保留本次基线' : '—';
    return `<tr><td>${escapeHTML(entry.city_name)}</td><td>${escapeHTML(labels[entry.status] || entry.status)}</td><td>${pct(entry.baseline?.recall_ratio)}</td><td title="${escapeHTML(entry.model_path || '')}">${pct(entry.selected?.recall_ratio)} ${escapeHTML(entry.champion_id || '')}</td><td>${pct(entry.reference?.recall_ratio)}</td><td>${escapeHTML(outcome)}</td><td>${entry.session_id ? `<button type="button" data-city-session="${escapeHTML(entry.session_id)}">查看${escapeHTML(entry.city_name)}</button>` : '—'}</td></tr>`;
  }).join('')}</tbody></table></div>`;
  panel.querySelectorAll('[data-city-session]').forEach(button => button.addEventListener('click', async () => {
    state.followCurrent = false; switchConversation(button.dataset.citySession); rememberChatSession(); await refreshWorkspace();
  }));
}

async function refreshChampion() {
  const sessionId = state.sessionId;
  try {
    const { champion, session, evaluation_split } = await api("/api/champion");
    if (sessionId !== state.sessionId) return;
    await refreshCityTraining(session);
    if (sessionId !== state.sessionId) return;
  } catch (e) {
    console.warn('城市训练摘要加载失败', e);
  }
}

// ================================================================
// 历史实验
// ================================================================
// 历史实验：全量数据缓存 + 前端搜索 + 分页
state.historyAll = [];
state.historyPage = 1;
state.historyPageSize = 20;
state.historyQuery = "";
state.historyPassOnly = false;

function _historyFiltered() {
  let list = state.historyAll;
  if (state.historyPassOnly) {
    list = list.filter(r => r.gate_pass === true);
  }
  const q = (state.historyQuery || "").toLowerCase().trim();
  if (!q) return list;
  return list.filter(r => {
    const hay = [
      r.experiment_id, r.proposer, r.verdict, r.innovation_title, r.hypothesis,
      modelRatioLabel(r),
      "r" + r.round_id,
      r.gate_pass === true ? "通过 pass ✓" : (r.gate_pass === false ? "未通过 fail ✗" : "baseline"),
    ].join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function _renderHistoryTable() {
  const filtered = _historyFiltered();
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / state.historyPageSize));
  if (state.historyPage > totalPages) state.historyPage = totalPages;
  const start = (state.historyPage - 1) * state.historyPageSize;
  const pageRows = filtered.slice(start, start + state.historyPageSize);

  const passCount = state.historyAll.filter(r => r.gate_pass === true).length;

  const rows = pageRows.map(r => {
    const gate = r.gate_pass === true ? '<td class="pass">✓</td>'
              : r.gate_pass === false ? '<td class="fail">✗</td>'
              : '<td class="baseline">—</td>';
    const verdictBadge = r.verdict
      ? `<span class="sc-badge badge-${r.verdict}">${escapeHTML(r.verdict)}</span>`
      : "";
    // ID 用 CSS 省略号截断（完整 ID 在 title 里）；同时兼容展示末尾 hash
    return `<tr class="clickable" data-exp="${escapeHTML(r.experiment_id)}">
      <td class="mono id-cell" title="${escapeHTML(r.experiment_id || '')}">
        <span class="id-text">${escapeHTML(r.experiment_id || '')}</span>
      </td>
      <td>r${r.round_id}</td>
      <td class="model-ratio">${escapeHTML(modelRatioLabel(r))}</td>
      <td>${escapeHTML(r.proposer || "")}</td>
      ${gate}
      <td title="${selectionLabel(r.selection_metric)}">${pct(r.all_acc)}<br><span class="muted-tiny">${selectionLabel(r.selection_metric)}</span></td>
      <td>${verdictBadge}</td>
    </tr>`;
  }).join("");

  const pageInfo = total === 0
    ? `<span class="muted-tiny">无匹配记录</span>`
    : `<span class="muted-tiny">${start + 1}-${Math.min(start + pageRows.length, total)} / 共 ${total} 条</span>`;

  $("#history-table").innerHTML = `
    <p class="hint">评估口径：${evaluationLabel((state.sessions.find(session => session.session_id === state.sessionId) || {}).evaluation_split)}</p>
    <div class="table-search-row">
      <input id="history-search" type="search" placeholder="搜索：实验 ID / 模型配比 / 提议方 / verdict / 创新点 / 假设..." value="${escapeHTML(state.historyQuery)}">
      <label class="chk history-filter-chk" title="只显示 gate_pass=True 的实验">
        <input id="history-pass-only" type="checkbox" ${state.historyPassOnly ? 'checked' : ''}>
        <span>仅门禁通过</span>
        <span class="muted-tiny">(${passCount})</span>
      </label>
    </div>
    <div class="city-compare-scroll"><table class="history-table-tight">
      <thead><tr>
        <th style="min-width:200px">ID</th><th>轮次</th><th title="每个订单预测概率的融合权重">模型配比</th><th>提议方</th><th>门禁</th>
        <th title="按各实验记录的门禁口径">ALL acc（门禁）</th><th>判定</th>
      </tr></thead>
      <tbody>${rows || `<tr><td colspan="7" class="muted-tiny" style="text-align:center;padding:20px">无匹配记录</td></tr>`}</tbody>
    </table></div>
    <div class="pagination">
      ${pageInfo}
      <div class="pagination-ctrls">
        <button id="hist-prev" ${state.historyPage <= 1 ? "disabled" : ""}>‹ 上一页</button>
        <span class="mono">${state.historyPage} / ${totalPages}</span>
        <button id="hist-next" ${state.historyPage >= totalPages ? "disabled" : ""}>下一页 ›</button>
        <select id="hist-page-size">
          <option value="10" ${state.historyPageSize===10?'selected':''}>10 条/页</option>
          <option value="20" ${state.historyPageSize===20?'selected':''}>20 条/页</option>
          <option value="50" ${state.historyPageSize===50?'selected':''}>50 条/页</option>
          <option value="100" ${state.historyPageSize===100?'selected':''}>100 条/页</option>
        </select>
      </div>
    </div>`;

  // 绑事件
  const searchInput = $("#history-search");
  if (searchInput) {
    // 保持焦点
    searchInput.focus();
    searchInput.setSelectionRange(searchInput.value.length, searchInput.value.length);
    let debTimer = null;
    searchInput.addEventListener("input", () => {
      clearTimeout(debTimer);
      debTimer = setTimeout(() => {
        state.historyQuery = searchInput.value;
        state.historyPage = 1;
        _renderHistoryTable();
      }, 180);
    });
  }
  const passOnly = $("#history-pass-only");
  if (passOnly) {
    passOnly.addEventListener("change", () => {
      state.historyPassOnly = passOnly.checked;
      state.historyPage = 1;
      _renderHistoryTable();
    });
  }
  const prev = $("#hist-prev"), next = $("#hist-next"), sz = $("#hist-page-size");
  if (prev) prev.addEventListener("click", () => { state.historyPage--; _renderHistoryTable(); });
  if (next) next.addEventListener("click", () => { state.historyPage++; _renderHistoryTable(); });
  if (sz) sz.addEventListener("change", () => {
    state.historyPageSize = Number(sz.value);
    state.historyPage = 1;
    _renderHistoryTable();
  });
  $$("#history-table tr.clickable").forEach(tr => {
    tr.addEventListener("click", () => showExperimentDetail(tr.dataset.exp));
  });
}

async function loadHistory() {
  const sessionId = state.sessionId;
  try {
    const { records } = await api("/api/experiments");
    if (sessionId !== state.sessionId) return;
    // 最新在前
    state.historyAll = (records || []).slice().reverse();
    _renderHistoryTable();
  } catch (e) {
    if (sessionId !== state.sessionId) return;
    $("#history-table").textContent = "加载失败: " + e.message;
  }
}

function renderReproduction(info) {
  if (!info || info.status === 'unavailable') return `<h4>代码与复现</h4><p>${escapeHTML(info?.reason || '尚未生成复现档案')}</p>`;
  const cfg = info.effective_train_config || {};
  const family = cfg.model_family || 'xgb';
  const params = family === 'mlp' ? {...cfg.mlp, activation:'relu', solver:'adam', shuffle:true,
    random_state:cfg.model_seed ?? cfg.split?.seed} : {...cfg.xgboost, num_class:6,
    random_state: cfg.model_seed ?? cfg.xgboost?.random_state};
  const row = (key, value) => `<tr><td>${escapeHTML(key)}</td><td><code>${escapeHTML(typeof value === 'object' ? JSON.stringify(value) : value)}</code></td></tr>`;
  const verification = ['evaluate', 'retrain'].map(mode => {
    const result = info.verification?.[mode];
    const label = mode === 'evaluate' ? '原模型回读评估' : '从头重训复现';
    if (!result) return `${label}：${mode === 'evaluate' ? '未单独执行' : '尚未验证'}`;
    const status = {matched:'一致', mismatch:'不一致', failed:'失败', running:'运行中'}[result.status] || result.status;
    return `${label}：${status}${result.finished_at ? '（' + new Date(result.finished_at).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai'}) + ' 示例区域A时间）' : ''}${result.error ? ' · ' + result.error : ''}${result.platform_changed ? ' · 跨平台' : ''}${result.environment_differences?.length ? ' · 依赖环境有差异' : ''}`;
  });
  return `<section class="reproduction-details"><h4>代码与一键复现</h4>
    <p>${escapeHTML(info.scope)}</p>
    <p>${info.capture_origin === 'historical_backfill' ? '历史实验事后补档：冻结的是补档时的代码；是否能复现以重跑校验结果为准。' : '代码和输入已在本次训练前冻结。'}</p>
    ${verification.map(text => `<p><b>${escapeHTML(text)}</b></p>`).join('')}
    <p>激活匹配的 Python 环境后，在当前部署服务器运行（默认从头训练）：</p>
    <pre>${escapeHTML(info.command)}</pre><button type="button" id="copy-reproduction-command">复制复现命令</button>
    <p class="hint">也可通过 ETS_PYTHON 指定解释器。加 --mode evaluate 只回读原模型评估。命令会校验文件、依赖版本、逐单预测和历史指标；不一致时退出码非零，原模型不会被覆盖。</p>
    <p>迁移服务器：保留该会话目录中的实验文件夹及共享的 _reproduction 文件夹。在新项目根目录运行：</p>
    <pre>${escapeHTML(info.portable_command)}</pre>
    <table><tbody>${row('复现入口', info.entrypoint)}${row('冻结的对应代码目录', info.source_directory)}
      ${Object.entries(info.source_files || {}).map(([key,value]) => row(key, value)).join('')}
      ${row('实验档案', info.manifest_path)}${row('依赖版本文件', info.requirements_path)}
      ${row('特征配方与预处理记录', info.feature_manifest_path)}
      ${row('Python', info.environment?.python)}${row('捕获平台', info.environment?.platform)}
      ${row('原始数据快照', info.snapshot_id || '在线/合成数据；复现使用已冻结特征输入')}
      ${row('时间窗口', info.window)}${row('训练划分方式', cfg.training_method)}
      ${row('模型随机种子', cfg.model_seed ?? cfg.split?.seed ?? cfg.xgboost?.random_state)}
      ${row('拆分参数', cfg.training_method === 'temporal' ? '按上述时间窗口划分；随机比例和拆分种子不参与本次划分' : cfg.split)}${row('类别权重', cfg.class_weight)}
      ${row('训练人群/路由', cfg.training_population || cfg.non_queue_strategy)}
      ${row('保序后非排队策略', cfg.non_queue_after_ordering)}</tbody></table>
    <h4>具体超参 · ${escapeHTML(family.toUpperCase())}</h4>
    <table><tbody>${Object.entries(params || {}).map(([key, value]) => row(key, value)).join('')}</tbody></table>
    <h4>实际训练情况（原模型）</h4><pre>${escapeHTML(JSON.stringify(info.training_details, null, 2))}</pre>
    <details><summary>完整生效配置与特征顺序（${info.feature_columns?.length || 0} 个特征）</summary><pre>${escapeHTML(JSON.stringify({train_config:cfg, feature_columns:info.feature_columns}, null, 2))}</pre></details>
    <h4>模型文件</h4><table><thead><tr><th>文件位置</th><th>大小</th><th>SHA-256</th></tr></thead><tbody>${(info.models || []).map(file => `<tr><td><code>${escapeHTML(file.path)}</code></td><td>${Number(file.bytes).toLocaleString('zh-CN')} B</td><td><code>${escapeHTML(file.sha256)}</code></td></tr>`).join('')}</tbody></table>
    <details><summary>冻结的数据输入与校验值</summary><pre>${escapeHTML(JSON.stringify(info.inputs, null, 2))}</pre></details>
    </section>`;
}

async function showExperimentDetail(exp_id) {
  try {
    const { record, slices_top, reproduction } = await api(`/api/experiments/${encodeURIComponent(exp_id)}`);
    const s = record.summary || {};
    const inn = s.innovation || {};
    const delta = ((s.delta_vs_baseline || {}).xgb_44) || {};
    const cities = ["100001", "100002", "100003", "ALL"];
    const cityNames = { "100001": "示例区域A", "100002": "示例区域B", "100003": "示例区域C", "ALL": "全量" };
    const deltaRows = cities.map(c => {
      const row = delta[c];
      if (!row) return "";
      return `<tr><td>${cityNames[c]}</td><td>${pp(row.acc)}</td><td>${pp(row.under)}</td>
              <td>${pp(row.over)}</td><td>${fmtNum(row.mae)}</td><td>${pp(row.severe_under)}</td></tr>`;
    }).join("");

    const slicesRows = (slices_top || []).map(row =>
      `<tr>
        <td>${escapeHTML(row.slice_col)}=${escapeHTML(row.slice_value)}</td>
        <td>${row.n}</td>
        <td>${Number(row.acc || 0).toFixed(3)}</td>
        <td>${Number(row.under || 0).toFixed(3)}</td>
        <td>${row.errors == null ? '未记录' : escapeHTML(row.errors)}</td>
        <td>${row.error_contribution_pp == null ? '未记录' : Number(row.error_contribution_pp).toFixed(3) + ' pp'}</td>
      </tr>`
    ).join("");

    $("#detail-content").innerHTML = `
      <h3>${escapeHTML(exp_id)}</h3>
      <p><b>模型配比</b>: ${escapeHTML(modelRatioLabel(record))}</p>
      <p><b>假设</b>: ${escapeHTML(record.hypothesis || "")}</p>
      <p><b>创新点</b>: ${escapeHTML(inn.title || "")}
         <span class="sc-badge badge-${s.verdict || 'mixed'}">${escapeHTML(s.verdict || '-')}</span></p>
      <p><b>门禁</b>: ${record.gate_pass === true ? "通过" : record.gate_pass === false ? "未通过" : "N/A"}
         · ${selectionLabel(selectionMetric(record))}
         ${record.gate_reasons && record.gate_reasons.length ? " · 原因: " + record.gate_reasons.join("; ") : ""}</p>
      ${s.selection_delta_pp ? `<p>门禁准确率变化：${Object.entries(s.selection_delta_pp).map(([key, value]) => `${({parent:'相对父模型', baseline:'相对基线', champion:'相对比较时冠军'})[key] || key} ${Number(value).toFixed(4)} pp`).join('；')}</p>` : ''}
      <p><b>中文总结</b>: ${escapeHTML(s.narrative_zh || "（无）")}</p>
      ${renderReproduction(reproduction)}
      ${renderSearchDiagnostics(record)}
      <p><b>评估数据</b>: ${evaluationLabel((record.config_diff || {}).evaluation_split)}
        · ${(record.config_diff || {}).mock ? "合成数据" : "请结合实验来源确认"}</p>
      ${(record.config_diff || {}).feature_recipes ? `<h4>特征计算配方</h4><pre>${escapeHTML(JSON.stringify(record.config_diff.feature_recipes, null, 2))}</pre>` : ""}
      <h4>Delta vs baseline（固定：全量·保序前）</h4>
      <table><thead><tr><th>城市</th><th>acc</th><th>under</th><th>over</th><th>mae</th><th>severe_under</th></tr></thead>
      <tbody>${deltaRows || '<tr><td colspan="6">（无对比）</td></tr>'}</tbody></table>
      <h4>本次指标：按人群／预测版本查看</h4>
      <div id="detail-evaluation"></div>
      <h4>Top 误差切片（固定：全量·保序前${s.search_diagnostics ? '，按错误行数排序' : ''}）</h4>
      <table><thead><tr><th>slice</th><th>n</th><th>acc</th><th>under</th><th>错误行数</th><th>全量错误贡献</th></tr></thead>
      <tbody>${slicesRows || '<tr><td colspan="6">（无）</td></tr>'}</tbody></table>
    `;
    mountEvaluation($('#detail-evaluation'), record, null, record.proposer === 'baseline' ? 'baseline' : '本次模型');
    const copyReplay = $('#copy-reproduction-command');
    if (copyReplay) copyReplay.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(reproduction.command); copyReplay.textContent = '已复制'; }
      catch (_) { copyReplay.textContent = '请选中上方命令复制'; }
    });
    $("#detail-modal").classList.remove("hidden");
  } catch (e) {
    alert("加载详情失败: " + e.message);
  }
}
$("#btn-modal-close").addEventListener("click", () => {
  $("#detail-modal").classList.add("hidden");
});

$("#btn-refresh-history").addEventListener("click", loadHistory);

// ================================================================
// 记忆
// ================================================================
// 记忆：全量总结 + 搜索 + 分页
state.memoryAll = null;   // 缓存整个 /api/memory 响应
state.memoryQuery = "";
state.memoryPage = 1;
state.memoryPageSize = 10;

function _memorySummariesFiltered() {
  const all = (state.memoryAll && state.memoryAll.recent_summaries) || [];
  const q = (state.memoryQuery || "").toLowerCase().trim();
  if (!q) return all;
  return all.filter(r => {
    const hay = [r.experiment_id, r.verdict, r.innovation_title, r.narrative, r.timestamp].join(" ").toLowerCase();
    return hay.includes(q);
  });
}

function _renderMemoryPanel() {
  const mem = state.memoryAll;
  if (!mem) return;
  const champion = mem.champion;
  const cham_all = selectionMetrics(champion).ALL;

  // 分页 summaries
  const filtered = _memorySummariesFiltered();
  const total = filtered.length;
  const totalPages = Math.max(1, Math.ceil(total / state.memoryPageSize));
  if (state.memoryPage > totalPages) state.memoryPage = totalPages;
  const start = (state.memoryPage - 1) * state.memoryPageSize;
  const pageRows = filtered.slice(start, start + state.memoryPageSize);

  const summariesHTML = pageRows.map(r => {
    const shortId = String(r.experiment_id || "");
    return `<div class="memory-block" style="margin-bottom:8px">
      <h4 title="${escapeHTML(r.experiment_id || '')}">${escapeHTML(shortId)} · <span class="muted-tiny">${escapeHTML(r.timestamp || "")}</span></h4>
      <div>${escapeHTML(r.innovation_title || "")}
        <span class="sc-badge badge-${r.verdict || 'mixed'}">${escapeHTML(r.verdict || '-')}</span>
      </div>
      <div class="sc-narr">${escapeHTML(r.narrative || "")}</div>
    </div>`;
  }).join("");

  const pageInfo = total === 0
    ? `<span class="muted-tiny">${state.memoryQuery ? '无匹配总结' : '（暂无总结）'}</span>`
    : `<span class="muted-tiny">${start + 1}-${Math.min(start + pageRows.length, total)} / 共 ${total} 条</span>`;

  $("#memory-view").innerHTML = `
    <div class="memory-block">
      <h4>Champion · ${evaluationLabel(((champion || {}).config_diff || {}).evaluation_split)} · ${selectionLabel(selectionMetric(champion))}</h4>
      ${champion ? `<div><code title="${escapeHTML(champion.experiment_id || '')}">${escapeHTML(champion.experiment_id || '')}</code>
        · ALL acc <b>${pct(cham_all && cham_all.recall_ratio)}</b>
        · under <b>${pct(cham_all && cham_all.under_ratio)}</b></div>` : "（无）"}
    </div>
    <div class="memory-block">
      <h4>累计</h4>
      <div>共 ${mem.total_experiments} 次实验，${mem.gate_passed} 次通过门禁</div>
    </div>
    <div class="memory-block">
      <h4>已接纳特征（${(mem.accepted_features || []).length}）</h4>
      <input id="mem-features-search" type="search" class="catalog-search" placeholder="过滤特征名..." style="margin-bottom:6px">
      <div id="mem-features-chips">${(mem.accepted_features || []).map(f => `<span class="chip" data-name="${escapeHTML(f)}">${escapeHTML(f)}</span>`).join("") || "（无）"}</div>
    </div>
    <div class="memory-block">
      <h4>XGB 参数变更历史</h4>
      ${mem.champion_spec ? `<details><summary>当前冠军完整输入与训练配置</summary><pre>${escapeHTML(JSON.stringify(mem.champion_spec, null, 2))}</pre></details>` : ''}
      <div class="mono">${escapeHTML(JSON.stringify(mem.xgb_params_history || {}, null, 2))}</div>
    </div>
    <div class="memory-block">
      <h4>历次实验总结</h4>
      <input id="mem-summary-search" type="search" class="catalog-search" placeholder="搜索：ID / verdict / 创新点 / narrative..." value="${escapeHTML(state.memoryQuery)}" style="margin-bottom:8px">
      ${summariesHTML || `<div class='empty-hint'>${state.memoryQuery ? '无匹配总结' : '暂无实验总结'}</div>`}
      <div class="pagination">
        ${pageInfo}
        <div class="pagination-ctrls">
          <button id="mem-prev" ${state.memoryPage <= 1 ? "disabled" : ""}>‹ 上一页</button>
          <span class="mono">${state.memoryPage} / ${totalPages}</span>
          <button id="mem-next" ${state.memoryPage >= totalPages ? "disabled" : ""}>下一页 ›</button>
          <select id="mem-page-size">
            <option value="5" ${state.memoryPageSize===5?'selected':''}>5 条/页</option>
            <option value="10" ${state.memoryPageSize===10?'selected':''}>10 条/页</option>
            <option value="20" ${state.memoryPageSize===20?'selected':''}>20 条/页</option>
            <option value="50" ${state.memoryPageSize===50?'selected':''}>50 条/页</option>
          </select>
        </div>
      </div>
    </div>
  `;

  // 事件绑定
  const featSearch = $("#mem-features-search");
  if (featSearch) {
    featSearch.addEventListener("input", () => {
      const kw = featSearch.value.toLowerCase().trim();
      document.querySelectorAll("#mem-features-chips .chip").forEach(el => {
        const name = (el.dataset.name || "").toLowerCase();
        el.style.display = !kw || name.includes(kw) ? "" : "none";
      });
    });
  }
  const sumSearch = $("#mem-summary-search");
  if (sumSearch) {
    sumSearch.focus();
    sumSearch.setSelectionRange(sumSearch.value.length, sumSearch.value.length);
    let debTimer = null;
    sumSearch.addEventListener("input", () => {
      clearTimeout(debTimer);
      debTimer = setTimeout(() => {
        state.memoryQuery = sumSearch.value;
        state.memoryPage = 1;
        _renderMemoryPanel();
      }, 180);
    });
  }
  const prev = $("#mem-prev"), next = $("#mem-next"), sz = $("#mem-page-size");
  if (prev) prev.addEventListener("click", () => { state.memoryPage--; _renderMemoryPanel(); });
  if (next) next.addEventListener("click", () => { state.memoryPage++; _renderMemoryPanel(); });
  if (sz) sz.addEventListener("change", () => {
    state.memoryPageSize = Number(sz.value);
    state.memoryPage = 1;
    _renderMemoryPanel();
  });
}

async function loadMemory() {
  const sessionId = state.sessionId;
  try {
    const data = await api("/api/memory");
    if (sessionId !== state.sessionId) return;
    state.memoryAll = data;
    _renderMemoryPanel();
  } catch (e) {
    if (sessionId !== state.sessionId) return;
    $("#memory-view").textContent = "加载失败: " + e.message;
  }
}
$("#btn-refresh-memory").addEventListener("click", loadMemory);

// ================================================================
// 血缘图 lineage
// ================================================================
function _lineageNodeColor(n) {
  if (n.proposer === "baseline") return "node-baseline";
  if (n.gate_pass === true) return "node-pass";
  if (n.gate_pass === false) return "node-fail";
  return "node-neutral";
}

function _renderLineageSVG(session) {
  const nodes = session.nodes || [];
  if (!nodes.length) return `<div class="empty-hint">本会话没有实验记录</div>`;

  // 布局：baseline 在最左；其它按 (round_id, seq) 一列一列排开
  // 简化：baseline col=0；round1 col=1（按 gate 分上下：pass 中间、fail 上下岔开）
  const baseline = nodes.find(n => n.proposer === "baseline");
  const others = nodes.filter(n => n.proposer !== "baseline");

  // 每 round 一列
  const byRound = {};
  others.forEach(n => {
    (byRound[n.round_id] = byRound[n.round_id] || []).push(n);
  });
  const rounds = Object.keys(byRound).map(Number).sort((a, b) => a - b);

  const NODE_W = 150, NODE_H = 62, X_GAP = 210, Y_GAP = 90;
  const positions = {};   // exp_id -> {x, y}

  // 计算总行数：每 round 里最多的节点数决定行高
  let maxRows = 1;
  rounds.forEach(r => { if (byRound[r].length > maxRows) maxRows = byRound[r].length; });
  const height = Math.max(NODE_H + 40, maxRows * Y_GAP + 40);
  const width = (rounds.length + 1) * X_GAP + NODE_W + 40;

  // baseline 放在垂直中央
  if (baseline) {
    positions[baseline.id] = { x: 20, y: height / 2 - NODE_H / 2 };
  }

  rounds.forEach((r, colIdx) => {
    const arr = byRound[r];
    // pass 排前面（视觉靠中）；fail 挤到边缘
    arr.sort((a, b) => {
      const aScore = a.gate_pass === true ? 0 : (a.gate_pass === false ? 1 : 2);
      const bScore = b.gate_pass === true ? 0 : (b.gate_pass === false ? 1 : 2);
      return aScore - bScore;
    });
    arr.forEach((n, rowIdx) => {
      const total = arr.length;
      const yCenter = height / 2;
      // 中心对称排布：0 在中央；1 在下；2 在上；...
      const offset = rowIdx % 2 === 0 ? rowIdx / 2 : -Math.ceil(rowIdx / 2);
      positions[n.id] = {
        x: 20 + (colIdx + 1) * X_GAP,
        y: yCenter + offset * Y_GAP - NODE_H / 2,
      };
    });
  });

  // 画连线
  let edges = "";
  nodes.forEach(n => {
    if (!n.parent_id || !positions[n.parent_id]) return;
    const p = positions[n.parent_id], c = positions[n.id];
    const x1 = p.x + NODE_W, y1 = p.y + NODE_H / 2;
    const x2 = c.x, y2 = c.y + NODE_H / 2;
    const midX = (x1 + x2) / 2;
    const passing = n.gate_pass === true;
    const cls = passing ? "edge-pass" : (n.gate_pass === false ? "edge-fail" : "edge-neutral");
    edges += `<path class="${cls}" d="M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}" fill="none"/>`;
  });

  // 画节点
  let nodesSvg = "";
  nodes.forEach(n => {
    const p = positions[n.id];
    if (!p) return;
    const cls = _lineageNodeColor(n);
    const gateIcon = n.gate_pass === true ? "✓" : (n.gate_pass === false ? "✗" : "•");
    const accTxt = n.all_acc != null ? `${(n.all_acc * 100).toFixed(2)}%` : "-";
    const inn = (n.innovation_title || (n.proposer === "baseline" ? "baseline" : "")).slice(0, 22);
    nodesSvg += `
      <g class="lineage-node ${cls}" transform="translate(${p.x},${p.y})" data-exp-id="${escapeHTML(n.id)}">
        <title>${selectionLabel(n.selection_metric)}</title>
        <rect width="${NODE_W}" height="${NODE_H}" rx="6" ry="6"/>
        <text class="ln-label" x="10" y="18">${escapeHTML(n.short_label)} ${gateIcon}</text>
        <text class="ln-inn"   x="10" y="34">${escapeHTML(inn)}</text>
        <text class="ln-acc"   x="10" y="52">ALL acc ${accTxt}${n.acc_delta_vs_parent != null ? `  ${n.acc_delta_vs_parent >= 0 ? '+' : ''}${(n.acc_delta_vs_parent*100).toFixed(2)}pp` : ''}</text>
      </g>`;
  });

  return `<svg class="lineage-svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
    <defs></defs>
    ${edges}
    ${nodesSvg}
  </svg>`;
}

async function loadLineage() {
  const sessionId = state.sessionId;
  const container = $("#lineage-view");
  container.innerHTML = "<div class='empty-hint'>加载中…</div>";
  try {
    const data = await api("/api/lineage");
    if (sessionId !== state.sessionId) return;
    const lineages = data.lineages || [];
    if (!lineages.length) {
      container.innerHTML = "<div class='empty-hint'>还没有实验记录。点击右上角启动跑一次。</div>";
      return;
    }
    container.innerHTML = lineages.map(s => `
      <div class="lineage-session">
        <div class="ls-header">
          <span class="ls-title mono">${escapeHTML(s.session_id)}</span>
          ${s.status ? `<span class="status-badge st-${s.status === 'complete' ? 'ok' : (s.status === 'failed' ? 'err' : 'info')}">${escapeHTML(s.status)}</span>` : ""}
          ${s.mock ? '<span class="tag tag-mock">合成</span>' : '<span class="tag tag-real">真实</span>'}
          <span class="muted-tiny">节点 ${s.nodes.length}, 通过 ${s.nodes.filter(n=>n.gate_pass===true).length}</span>
        </div>
        <div class="ls-canvas">${_renderLineageSVG(s)}</div>
      </div>
    `).join("");

    // 绑定点击 → 复用 history tab 里的 detail modal
    document.querySelectorAll(".lineage-node").forEach(g => {
      g.addEventListener("click", () => {
        const id = g.dataset.expId;
        if (id) showExperimentDetail(id);
      });
    });
  } catch (e) {
    if (sessionId !== state.sessionId) return;
    container.innerHTML = `<div class='empty-hint'>加载失败：${escapeHTML(e.message)}</div>`;
  }
}
$("#btn-refresh-lineage").addEventListener("click", loadLineage);

// ================================================================
// 超参调整
// ================================================================
let tuningRequest = 0;
async function loadTuning() {
  const requestId = ++tuningRequest;
  const family = $("#model-family").value || "xgb";
  $("#btn-save-tuning").disabled = true;
  try {
    const cfg = await api("/api/config");
    if (requestId !== tuningRequest) return;
    state.configCache = cfg;
    const wl = cfg.xgb_whitelist.num;
    const current = (cfg.train.xgboost) || {};
    const win = (cfg.data && cfg.data.window) || {};

    const winRow = (label, key) => `
      <div class="tuning-item">
        <label>${label}</label>
        <input data-window-key="${key}" type="date" value="${escapeHTML(win[key] || '')}" />
        <div class="range">${escapeHTML(key)}</div>
      </div>`;

    const windowBlock = `
      <div class="tuning-section">
        <h4>训练/测试窗口</h4>
        <div class="tuning-grid">
          ${winRow('数据起', 'start_date')}
          ${winRow('数据止', 'end_date')}
          ${winRow('训练起', 'train_start')}
          ${winRow('训练止', 'train_end')}
          ${winRow('测试起', 'test_start')}
          ${winRow('测试止', 'test_end')}
        </div>
        <div class="tuning-hint">按日期划分训练和测试窗口；当前 Notebook 随机拆分模式从训练期内划出早停集。测试集用于评估和候选筛选，不是独立泛化评估。</div>
      </div>`;

    const items = Object.entries(wl).map(([k, [lo, hi]]) => {
      const cur = current[k] != null ? current[k] : "";
      return `<div class="tuning-item">
        <label>${k}</label>
        <input data-key="${k}" type="number" step="any" value="${cur}" />
        <div class="range">范围 [${lo}, ${hi}]</div>
      </div>`;
    }).join("");
    const xgbBlock = `
      <div class="tuning-section">
        <h4>XGBoost 基线超参（用于下一次启动的实验）</h4>
        <div class="tuning-grid">${items}</div>
      </div>`;

    const mlp = { ...(cfg.mlp_defaults || {}), ...(cfg.train.mlp || {}) };
    const labels = {max_epochs:'最大训练轮数', patience:'早停耐心轮数', batch_size:'批大小', learning_rate:'学习率', alpha:'L2 正则强度'};
    const mlpItems = Object.entries(cfg.mlp_bounds || {}).map(([key, [lo, hi]]) =>
      `<div class="tuning-item"><label>${labels[key]}（${key}）</label><input data-key="${key}" type="number" min="${lo}" max="${hi}" step="${['max_epochs','patience','batch_size'].includes(key) ? '1' : 'any'}" value="${escapeHTML(String(mlp[key]))}" /><div class="range">范围 [${lo}, ${hi}]</div></div>`).join('');
    const mlpBlock = `<div class="tuning-section"><h4>MLP 基线超参</h4><div class="tuning-grid"><div class="tuning-item"><label>隐藏层（hidden_layers）</label><input id="mlp-hidden-layers" type="text" value="${escapeHTML((mlp.hidden_layers || [128,64]).join(', '))}" /><div class="range">逗号分隔，1–4 层，每层 1–1024 个神经元</div></div>${mlpItems}</div><div class="tuning-hint">ReLU + Adam；填补和标准化仅拟合训练子集。保存后用于下一次启动的 MLP 实验，不改变正在运行的实验。</div></div>`;
    const populationBlock = `<div class="tuning-section"><h4>基线训练人群</h4><label>训练策略 <select id="training-population"><option value="unified">统一训练</option><option value="queue_only">只训练排队</option></select></label><p id="training-population-note" class="tuning-hint"></p></div>`;
    $("#tuning-form").innerHTML = populationBlock + windowBlock + (family === 'mlp' ? mlpBlock : xgbBlock);
    $('#training-population').value = cfg.train.training_population || (cfg.train.non_queue_strategy === 'unified' ? 'unified' : 'queue_only');
    const updatePopulationNote = () => {
      $('#training-population-note').textContent = $('#training-population').value === 'unified'
        ? '排队与非排队的已应答订单共同训练；全部订单由同一模型预测，保序前后均不强制归零。适用于 XGB 和 MLP，保存后下次启动生效。'
        : '仅排队且已应答订单训练；非排队保序前后均固定为第 0 桶。保存后下次启动生效。';
    };
    $('#training-population').addEventListener('change', updatePopulationNote);
    updatePopulationNote();
    $("#tuning-form").dataset.modelFamily = family;
    $("#tuning-msg").textContent = '';
    $("#btn-save-tuning").disabled = false;
  } catch (e) {
    $("#tuning-form").textContent = "加载失败: " + e.message;
  }
}
$("#btn-save-tuning").addEventListener("click", async () => {
  const family = $('#tuning-form').dataset.modelFamily || 'xgb';
  if (family !== ($('#model-family').value || 'xgb')) return;
  const msgs = [];
  // 1) 保存 window（如果有改动就发）
  const winInputs = $$("#tuning-form input[data-window-key]");
  const winPatch = {};
  winInputs.forEach(inp => {
    const k = inp.dataset.windowKey;
    if (inp.value) winPatch[k] = inp.value;
  });
  const requiredKeys = ["start_date","end_date","train_start","train_end","test_start","test_end"];
  if (requiredKeys.every(k => winPatch[k])) {
    try {
      const r = await api("/api/config/data-window", { method: "POST", body: JSON.stringify(winPatch) });
      msgs.push("窗口已保存: " + JSON.stringify(r.window));
      await loadDataSources();
    } catch (e) {
      $("#tuning-msg").textContent = "窗口保存失败: " + e.message;
      $("#tuning-msg").className = "msg err"; return;
    }
  }

  // 2) 保存 xgb 超参
  const inputs = $$("#tuning-form input[data-key]");
  const patch = {};
  inputs.forEach(inp => {
    const k = inp.dataset.key;
    const v = inp.value;
    if (v !== "" && !isNaN(Number(v))) patch[k] = Number(v);
  });
  if (family === 'mlp') patch.hidden_layers = $('#mlp-hidden-layers').value.split(/[,，]/).map(value => Number(value.trim()));
  if (Object.keys(patch).length) {
    try {
      const r = await api("/api/config/train", { method: "POST", body: JSON.stringify({training_population: $('#training-population').value, ...(family === 'mlp' ? {mlp_params: patch} : {xgb_params: patch})}) });
      msgs.push((family === 'mlp' ? 'MLP' : 'XGB') + " 已保存: " + JSON.stringify(family === 'mlp' ? r.mlp : r.xgboost));
    } catch (e) {
      $("#tuning-msg").textContent = (family === "mlp" ? "MLP" : "XGB") + " 保存失败: " + e.message;
      $("#tuning-msg").className = "msg err"; return;
    }
  }
  if (!msgs.length) {
    $("#tuning-msg").textContent = "没有可保存的字段"; $("#tuning-msg").className = "msg err"; return;
  }
  $("#tuning-msg").textContent = msgs.join(" · "); $("#tuning-msg").className = "msg ok";
});

// ================================================================
// 用户指导
// ================================================================
async function loadHints() {
  try {
    const { hints } = await api("/api/user-hint");
    const rows = hints.slice().reverse().map(h => `<div class="memory-block">
      <div class="mono muted-tiny">${escapeHTML(h.ts || "")} · ${h.consumed ? "已消费" : "<b style='color:var(--warn)'>待生效</b>"}</div>
      <div>${escapeHTML(h.hint)}</div>
    </div>`).join("");
    $("#hints-list").innerHTML = rows || "<div class='empty-hint'>（暂无指导）</div>";
  } catch (e) {
    $("#hints-list").textContent = "加载失败: " + e.message;
  }
}
$("#btn-add-hint").addEventListener("click", async () => {
  const hint = $("#hint-input").value.trim();
  if (!hint) return;
  try {
    await api("/api/user-hint", {
      method: "POST", body: JSON.stringify({ hint }),
    });
    $("#hint-input").value = "";
    loadHints();
    appendLog(`[user-hint] 已追加: ${hint}`, "chat-user");
  } catch (e) {
    alert("追加失败: " + e.message);
  }
});

$("#btn-view-prompt").addEventListener("click", async () => {
  try {
    const p = await api("/api/prompt");
    const activePrompt = p.prompts && p.prompts.feature_miner ? p.prompts.feature_miner : "(未找到 feature_miner.txt)";
    const hintBlock = p.hint_section_preview || "";
    const html = `
      <h3 style="margin:0 0 12px 0">当前 LLM Prompt 预览</h3>
      <p class="muted" style="font-size:12px">
        agent 每次 propose 时会把下面的 <b>system prompt</b> 送给 LLM。<br>
        用户指导（如有待消费的）会拼进 <b>user prompt</b> 的开头。
      </p>
      <div class="disc-section">
        <h4>System Prompt（${escapeHTML(p.active_prompt_file || '')}）</h4>
        <pre class="prompt-view">${escapeHTML(activePrompt)}</pre>
      </div>
      ${$('#continuous-search').checked && p.prompts && p.prompts.research_planner ? `<div class="disc-section"><h4>持续探索追加协议</h4><pre class="prompt-view">${escapeHTML(p.prompts.research_planner)}</pre></div>` : ''}
      ${$('#research-skills').checked && p.prompts && p.prompts.research_skills ? `<div class="disc-section"><h4>扩展研究能力协议</h4><pre class="prompt-view">${escapeHTML(p.prompts.research_skills)}</pre></div>` : ''}
      <div class="disc-section">
        <h4>下一轮将拼入的用户指导</h4>
        ${p.pending_hint
          ? `<pre class="prompt-view">${escapeHTML(hintBlock)}</pre>`
          : `<div class="empty-hint">${escapeHTML(hintBlock)}</div>`}
      </div>`;
    $("#detail-content").innerHTML = html;
    $("#detail-modal").classList.remove("hidden");
  } catch (e) {
    alert("加载 prompt 失败: " + e.message);
  }
});

// ================================================================
// 运行 orchestrator
// ================================================================
let dataSnapshots = [];

function updateDataSourceControls() {
  const offline = $('#data-source').value === 'offline';
  $('#data-source').disabled = false;
  $('#snapshot-control').hidden = !offline;
  $('#btn-refresh-snapshots').hidden = !offline;
  $('#snapshot-id').disabled = !offline;

}

function saveDataSourcePreference() {
  try {
    localStorage.setItem('ets-data-source', $('#data-source').value);
    localStorage.setItem('ets-snapshot-id', $('#snapshot-id').value);
  } catch (_) { /* Browser storage may be disabled. */ }
}

async function loadDataSources() {
  const select = $('#snapshot-id');
  let preferred = select.value;
  try { preferred = preferred || localStorage.getItem('ets-snapshot-id') || ''; } catch (_) {}
  try {
    const result = await api('/api/data-sources');
    dataSnapshots = result.snapshots || [];
    select.innerHTML = '<option value="">请选择本地快照</option>' + dataSnapshots.map(item =>
      `<option value="${escapeHTML(item.id)}" ${item.compatible ? '' : 'disabled'}>${escapeHTML(item.id)}${item.compatible ? '' : '（不可用或配置不匹配）'}</option>`).join('');
    const preferredEntry = dataSnapshots.find(item => item.id === preferred && item.compatible);
    select.value = preferred ? (preferredEntry ? preferred : '') : (dataSnapshots.find(item => item.compatible) || {}).id || '';
    updateDataSourceControls();
  } catch (error) {
    dataSnapshots = [];
    select.innerHTML = '<option value="">快照列表加载失败，请重试</option>';
    updateDataSourceControls();
  }
}

try {
  const saved = localStorage.getItem('ets-data-source');
  if (['online', 'offline'].includes(saved)) $('#data-source').value = saved;
} catch (_) {}
$('#data-source').addEventListener('change', () => { saveDataSourcePreference(); updateDataSourceControls(); loadDataSources(); });
$('#snapshot-id').addEventListener('change', () => { saveDataSourcePreference(); updateDataSourceControls(); });
$('#btn-refresh-snapshots').addEventListener('click', loadDataSources);
updateDataSourceControls();
loadDataSources();

function trainingDataLabel(config) {
  return config.mock ? '合成数据（mock）' : config.data_source === 'offline'
    ? `离线 Parquet · ${config.snapshot_id || '未选择快照'}` : '在线 Spark/Hive';
}

function updateTrainingMode() {
  try { localStorage.setItem('ets-training-mode', $('#training-mode').value); } catch (_) {}
}
try {
  const savedMode = localStorage.getItem('ets-training-mode');
  if (['pooled', 'by_city'].includes(savedMode)) $('#training-mode').value = savedMode;
} catch (_) {}
$('#training-mode').addEventListener('change', updateTrainingMode);
updateTrainingMode();

function runConfig() {
  return {
    llm_profile: selectedLLMProfile(),
    rounds: Number($("#rounds").value),
    max_proposals: Number($("#max-proposals").value),
    mock: true,
    data_source: $('#data-source').value,
    training_mode: $('#training-mode').value,
    model_family: $('#model-family').value || 'xgb',
    snapshot_id: null,
    no_agent: $("#no-agent").checked,
    resume_recipes: $("#resume-recipes").checked,
    sample_rows: 0,
    profile_only: $('#profile-only').checked,
    continuous: $('#continuous-search').checked,
    research_skills: $('#research-skills').checked,
    model_study: $('#model-study').checked,
    web_research: $('#web-research-enabled').checked,
    search_timeout_minutes: Number($('#search-timeout').value || 120),
    search_patience: Number($('#search-patience').value || 9),
  };
}

function updateRunButtons() {
  $("#btn-run").disabled = state.starting || !!state.currentJobId;
  $("#btn-stop").disabled = !state.currentJobId;
  updateInterruptButton();
  updateDeleteSessionButton();
}

async function startRun(body) {
  if (state.starting || state.currentJobId) {
    appendLog("[ui] 已有实验正在运行，请先停止或等待完成", "runner");
    return;
  }
  if (!Number.isInteger(body.rounds) || body.rounds < 0 || body.rounds > 20 ||
      !Number.isInteger(body.max_proposals) || body.max_proposals < 1 || body.max_proposals > 8) {
    alert("轮数需为 0–20 的整数，候选数需为 1–8 的整数");
    return;
  }
  if (!body.mock && body.sample_rows !== 0) {
    alert('Notebook 一致模式不允许抽样；请缩短日期窗口');
    return;
  }
  if (body.model_study && (body.model_family !== 'mlp' || body.no_agent || body.profile_only || body.rounds * body.max_proposals < 4)) {
    alert('自主模型实验请选择MLP，至少4个候选预算，并取消“只跑baseline / 只做画像”');
    return;
  }
  if (!body.mock && body.data_source === 'offline' && !body.snapshot_id) {
    alert('请先选择与当前配置匹配的完整本地快照');
    return;
  }
  if (body.continuous && (!Number.isInteger(body.search_timeout_minutes) || body.search_timeout_minutes < 1 || body.search_timeout_minutes > 1440 ||
      !Number.isInteger(body.search_patience) || body.search_patience < 1 || body.search_patience > 160)) {
    alert('搜索时间需为1–1440分钟，连续无提升预算需为1–160次');
    return;
  }
  state.starting = true;
  saveDataSourcePreference();
  updateRunButtons();
  try {
    const r = await api("/api/run", { method: "POST", body: JSON.stringify(body) });
    state.followCurrent = true;
    state.currentJobId = r.job_id;
    switchConversation(null);
    startStreaming(r.job_id);
    appendTaskLog(state.taskViews.get(r.job_id), `[ui] 启动 job=${r.job_id} 配置=${JSON.stringify(body)}`);
  } catch (e) {
    alert("启动失败: " + e.message);
    await restoreRunningJob();
  } finally {
    state.starting = false;
    updateRunButtons();
  }
}

$("#btn-run").addEventListener("click", () => startRun(runConfig()));
$("#btn-stop").addEventListener("click", async () => {
  if (!state.currentJobId) return;
  try {
    const result = await api(`/api/run/${state.currentJobId}/stop`, { method: "POST" });
    appendLog(result.ok ? `[ui] 已请求停止 ${state.currentJobId}` : "[ui] 任务已结束或无法停止，正在同步状态", "runner");
    await restoreRunningJob();
  } catch (e) {
    alert("停止失败: " + e.message);
  }
});

function attachTask(view, sessionId) {
  if (state.deletedSessions.has(sessionId)) return;
  if (view.sessionId !== sessionId) {
    const previous = state.conversations.get(view.sessionId);
    if (previous) previous.nodes = previous.nodes.filter(node => node !== view.card);
    view.card.remove();
    view.sessionId = sessionId;
  }
  if (!conversation(sessionId).nodes.includes(view.card)) appendConversationNode(view.card, sessionId);
  refreshPinnedTask();
}

function taskView(jobId, sessionId = null) {
  if (state.taskViews.has(jobId)) return state.taskViews.get(jobId);
  const card = document.createElement('section');
  card.className = 'task-progress';
  const title = document.createElement('div');
  title.className = 'task-title';
  title.textContent = '实验运行进度';
  const status = document.createElement('div');
  status.className = 'task-status';
  status.setAttribute('role', 'status');
  const message = document.createElement('div');
  message.className = 'task-message';
  message.textContent = '正在启动进程和初始化环境，请稍候…';
  const elapsed = document.createElement('div');
  elapsed.className = 'muted-tiny';
  const details = document.createElement('details');
  details.className = 'task-terminal';
  details.open = false;
  const summary = document.createElement('summary');
  summary.textContent = '终端详细日志 · 0 行（点击展开）';
  const body = document.createElement('pre');
  body.className = 'task-terminal-body';
  details.appendChild(summary);
  details.appendChild(body);
  card.appendChild(title);
  card.appendChild(status);
  card.appendChild(message);
  card.appendChild(elapsed);
  card.appendChild(details);
  const view = { card, title, status, message, elapsed, details, summary, body, sessionId,
    lines: [], lineCount: 0, startedAt: Date.now(), lastLogAt: null, finished: false, timer: null,
    lastEventId: 0, dirty: false };
  const tick = () => {
    const seconds = Math.max(0, Math.floor((Date.now() - view.startedAt) / 1000));
    elapsed.textContent = `已耗时 ${Math.floor(seconds / 60)}分${seconds % 60}秒` +
      (view.finished ? '' : ' · 阶段耗时不确定，不显示估算百分比') +
      (view.lastLogAt ? ` · 最近日志 ${Math.max(0, Math.floor((Date.now() - view.lastLogAt) / 1000))} 秒前` : '');
    renderTaskLog(view);
  };
  details.addEventListener('toggle', () => renderTaskLog(view));
  view.tick = tick;
  view.timer = setInterval(tick, 1000);
  status.textContent = '启动中';
  tick();
  state.taskViews.set(jobId, view);
  attachTask(view, sessionId);
  return view;
}

function renderTaskLog(view) {
  if (!view.details.open || !view.dirty) return;
  view.body.textContent = view.lines.join('\n');
  view.dirty = false;
  if ($('#autoscroll').checked) view.body.scrollTop = view.body.scrollHeight;
}

function appendTaskLog(view, line) {
  if (state.deletedSessions.has(view.sessionId)) return;
  attachTask(view, view.sessionId);
  view.lineCount += 1;
  view.lastLogAt = Date.now();
  view.lines.push(line.length > 8000 ? line.slice(0, 8000) + '…（完整内容见服务日志）' : line);
  if (view.lines.length > 500) view.lines.shift();
  view.dirty = true;
  view.summary.textContent = `终端详细日志 · ${view.lineCount} 行（点击展开）` +
    (view.lineCount > 500 ? ' · 显示最近 500 行，完整记录见服务日志' : '');
}

function updateTaskProgress(view, event) {
  if (state.deletedSessions.has(event.session_id)) return;
  attachTask(view, event.session_id);
  view.title.textContent = `实验运行进度 · ${event.session_id}`;
  if (event.type === 'progress') {
    const labels = { loading: '取数中', profiling: '生成画像', features: '准备特征', training: '训练中',
      evaluating: '测试集评估中', evaluated: '实验评估完成', proposing: 'Agent 生成方案', planning: 'Agent 规划下一项实验',
      rejected: '候选未完成', finishing: '汇总结果' };
    view.status.textContent = labels[event.stage] || '运行中';
    view.message.textContent = event.message || '正在处理…';
  } else if (event.type === 'session_updated') {
    const labels = { loading: '取数中', profiled: '画像完成', searching: '候选搜索中',
      complete: '实验完成，正在收尾', profile_complete: '画像完成，正在收尾',
      failed: '实验失败，正在收尾', cancelled: '实验已停止，正在收尾' };
    view.status.textContent = labels[event.status] || event.status || '运行中';
    if (event.status === 'loading') view.message.textContent = '正在准备数据源并读取数据，请稍候…';
    if (['complete', 'profile_complete'].includes(event.status)) view.message.textContent = '结果已保存，等待进程退出。';
    if (['failed', 'cancelled'].includes(event.status)) view.message.textContent = '请展开终端详细日志查看原因。';
  }
}

function startStreaming(job_id, knownSessionId = null, startedAt = null) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/api/run/${job_id}/stream`);
  state.eventSource = es;
  const view = taskView(job_id, knownSessionId);
  if (startedAt) view.startedAt = startedAt * 1000;
  if (knownSessionId) attachTask(view, knownSessionId);
  const accept = ev => {
    if (state.eventSource !== es) return false;
    const identifier = Number(ev.lastEventId || 0);
    if (identifier && identifier <= view.lastEventId) return false;
    if (identifier) view.lastEventId = identifier;
    return true;
  };
  es.onmessage = ev => { if (accept(ev)) appendTaskLog(view, ev.data); };
  es.addEventListener("experiment", async ev => {
    if (!accept(ev)) return;
    try {
      const event = JSON.parse(ev.data);
      const firstSession = !view.sessionId;
      updateTaskProgress(view, event);
      if (event.type === 'progress' && !firstSession) return;
      await refreshSessions();
      if (event.session_id !== state.sessionId) return;
      if (event.type === "experiment_completed") await pullSummary(event.experiment_id);
      scheduleWorkspaceRefresh();
    } catch (error) {
      appendLog(`[ui] 实验事件同步失败：${error.message}`, "error", { sessionId: view.sessionId });
    }
  });
  es.addEventListener("done", ev => { if (accept(ev)) finishJob(JSON.parse(ev.data)); });
  es.onerror = () => {
    if (state.eventSource !== es) return;
    view.status.textContent = '连接中断，正在重连';
    restoreRunningJob();
  };
}

function finishJob(job) {
  if (job.job_id !== state.currentJobId) return;
  const stream = state.eventSource;
  state.eventSource = null;
  state.currentJobId = null;
  updateRunButtons();
  if (stream) stream.close();
  const outcome = job.stop_requested ? "已停止" : job.returncode === 0 ? "已完成" : `失败（退出码 ${job.returncode}）`;
  const view = state.taskViews.get(job.job_id);
  if (view) {
    view.finished = true;
    clearInterval(view.timer);
    view.status.textContent = outcome;
    view.message.textContent = job.stop_requested ? '任务已停止，已生成的记录保留。' :
      job.returncode === 0 ? '结果已保存，可在左侧查看实验记录和评估结果。' :
      job.error_hint || (job.returncode === -9 ? '进程被强制结束，可能是内存不足。请检查闲置 Notebook 和 Spark 的内存占用。' :
        '任务未正常完成，请展开终端详细日志检查原因。');
    view.card.className = 'task-progress ' + (job.stop_requested ? 'task-stopped' : job.returncode === 0 ? 'task-complete' : 'task-failed');
    try { view.tick(); } catch (error) { console.warn('Task display update failed', error); }
  }
  $('#run-status').textContent = `${job.session_id || job.job_id} ${outcome}`;
  refreshWorkspace().then(() => {
    if (!job.session_id || job.session_id === state.sessionId) return pullLatestSummary();
  });
}

async function restoreRunningJob() {
  if (state.restoring) return;
  state.restoring = true;
  try {
    const { jobs } = await api("/api/run");
    const active = jobs.find(job => job.alive);
    $('#run-status').textContent = active ? `运行中：${active.session_id || active.job_id}` : '空闲 · 可以启动';
    if (active) {
      if (state.currentJobId !== active.job_id || !state.eventSource || state.eventSource.readyState === 2) {
        state.currentJobId = active.job_id;
        if (!active.session_id && state.followCurrent) switchConversation(null);
        startStreaming(active.job_id, active.session_id, active.started_at);
        appendTaskLog(state.taskViews.get(active.job_id), `[ui] 已恢复运行中的任务 ${active.job_id}`);
        scheduleWorkspaceRefresh();
      }
    } else if (state.currentJobId) {
      const finished = jobs.find(job => job.job_id === state.currentJobId);
      finishJob(finished || { job_id: state.currentJobId, returncode: "未知，服务可能已重启" });
    }
    updateRunButtons();
  } catch (error) {
    $('#run-status').textContent = `状态同步失败：${error.message}`;
    console.warn("restoreRunningJob", error.message);
  } finally {
    state.restoring = false;
  }
}

async function pullSummary(exp_id) {
  if (state.seenExpIdsInLog.has(exp_id)) return;
  state.seenExpIdsInLog.add(exp_id);
  try {
    const { record } = await api(`/api/experiments/${encodeURIComponent(exp_id)}`);
    if (record && record.summary && (record.config_diff || {}).session_id === state.sessionId) await appendSummaryCard(record);
  } catch (e) {
    state.seenExpIdsInLog.delete(exp_id);
    console.warn("pullSummary", e.message);
  }
}
async function pullLatestSummary() {
  try {
    const { records } = await api("/api/experiments");
    if (records && records.length) {
      const last = records[records.length - 1];
      if (last.experiment_id && !state.seenExpIdsInLog.has(last.experiment_id)) {
        pullSummary(last.experiment_id);
      }
    }
  } catch (e) {}
}

// ================================================================
// 聊天 / 指导 输入
// ================================================================
$("#btn-chat-send").addEventListener("click", handleChatSend);
$("#chat-text").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); handleChatSend(); }
});
async function handleChatSend() {
  if (state.chatBusy || state.deletingSession) return;
  const text = $("#chat-text").value.trim();
  if (!text) return;
  if (!state.sessionId) {
    await createConversation();
    if (!state.sessionId) return;
  }
  const sessionId = state.sessionId;
  const loaded = conversation(sessionId).loaded || await restoreConversation(sessionId);
  if (!loaded || state.sessionId !== sessionId || state.chatBusy) return;
  $("#chat-text").value = "";
  appendLog(`[user] ${text}`, "chat-user");

  const intent = detectRunIntent(text);
  if (intent) {
    const config = ETSControls.buildRunConfig(intent, runConfig());
    // Chat intents may override mock; do not carry an offline snapshot into mock runs.
    if (config.mock) config.snapshot_id = null;
    else if (config.data_source === 'offline') config.snapshot_id = $('#snapshot-id').value;
    const description = `${trainingDataLabel(config)}；抽样=${config.sample_rows || '全窗口'}；轮数=${config.rounds}；候选数=${config.max_proposals}；${config.profile_only ? '只做画像' : config.no_agent ? "只跑 baseline" : "自动特征发现"}；${config.resume_recipes ? "继承上次特征" : "不继承特征"}` +
      (config.continuous ? `；逐次规划，最多${config.rounds * config.max_proposals}次尝试，搜索${config.search_timeout_minutes}分钟，连续${config.search_patience}次无提升停止` : '；批量提案模式') +
      (config.research_skills ? '；允许特征删改、分支和有界调参，新数据与代码需人工审批' : '；仅新增特征、固定训练参数') + `；模型=${config.llm_profile}`;
    if (!window.confirm(`确认启动实验？\n${description}`)) {
      appendLog("[ui] 已取消启动，未提交实验", "runner");
      return;
    }
    $("#rounds").value = config.rounds;
    $("#max-proposals").value = config.max_proposals;
    updateDataSourceControls();
    $("#no-agent").checked = config.no_agent;
    await startRun(config);
    return;
  }

  await requestChat(text, state.sessionId, state.chatHistory.slice(-10));
}

function addContinuation(message, sessionId, history) {
  const notice = document.createElement('div');
  notice.className = 'chat-truncated-notice';
  const label = document.createElement('span');
  label.textContent = '回复达到输出上限，内容尚未完成。';
  label.setAttribute('role', 'status');
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = '继续回答';
  button.addEventListener('click', async () => {
    if (state.chatBusy || button.disabled) return;
    if (state.sessionId !== sessionId) {
      appendLog('[ui] 请先切回这条回复所属的会话，再继续回答。', 'runner');
      return;
    }
    button.disabled = true;
    const question = '上一条回复因输出长度限制被截断。请接着未完成的部分继续回答，不重复完整答案。如果停在表格或代码块中，请在本条回复中补上表头或代码围栏，并完整重写被截断的最后一行，再继续。';
    const success = await requestChat(question, sessionId, history.slice(-12));
    button.disabled = success;
    button.textContent = success ? '已续答' : '重试续答';
  });
  notice.appendChild(label);
  notice.appendChild(button);
  message.bubble.appendChild(notice);
  if (sessionId === state.sessionId && $('#autoscroll').checked) $('#log-view').scrollTop = $('#log-view').scrollHeight;
}

async function pollChat(body, signal, setStatus) {
  const url = `/api/chat/${encodeURIComponent(body.request_id)}`;
  const submit = () => api('/api/chat', {method: 'POST', signal, body: JSON.stringify({...body, background: true})});
  let job;
  try { job = await submit(); }
  catch (error) {
    if (signal.aborted || /^4\d\d /.test(error.message)) throw error;
    setStatus('连接短暂中断，正在确认后台是否已接收…');
    try { job = await api(url, {signal}); }
    catch (checkError) {
      if (!checkError.message.startsWith('404 ')) throw checkError;
      job = await submit(); // Same request ID: server guarantees no repeated work.
    }
    if (!job.status && !job.pending) throw error;
  }
  if (!job.pending && !job.status) return job; // Older synchronous server compatibility.
  const deadline = Date.now() + 30 * 60 * 1000;
  while (!signal.aborted && Date.now() < deadline) {
    if (job.status === 'complete' || job.status === 'cancelled') return job.result;
    setStatus(job.stage || '正在等待后台回复…');
    await new Promise(resolve => {
      const done = () => { clearTimeout(timer); signal.removeEventListener('abort', done); resolve(); };
      const timer = setTimeout(done, 1500);
      signal.addEventListener('abort', done, {once: true});
    });
    if (signal.aborted) return {cancelled: true};
    try { job = await api(url, {signal}); }
    catch (error) {
      if (signal.aborted) return {cancelled: true};
      if (/^4\d\d /.test(error.message)) throw error;
      job = {...job, stage: '连接中断，正在重连并查询原请求；不会重复生成…'};
    }
  }
  if (signal.aborted) return {cancelled: true};
  throw new Error('等待超过30分钟；后台结果未确认，请检查连接。请求编号：' + body.request_id);
}

function codeDownloadLink(file, parent) {
  if (!file || typeof file.download_url !== 'string' ||
      !/^\/api\/code-workspace\/[A-Za-z0-9_-]+\/download\?path=/.test(file.download_url)) return;
  const row = document.createElement('p');
  const link = document.createElement('a');
  link.href = file.download_url; link.textContent = file.path;
  row.appendChild(link); parent.appendChild(row);
}

$('#btn-code-workspace').addEventListener('click', async () => {
  if (!state.sessionId) return;
  const sessionId = state.sessionId;
  const dialog = document.createElement('dialog'); dialog.className = 'workspace-dialog';
  const title = document.createElement('h2'); title.textContent = `代码工作区 · ${sessionId}`;
  const close = document.createElement('button'); close.textContent = '关闭'; close.type = 'button';
  close.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => dialog.remove());
  const content = document.createElement('div'); content.textContent = '正在加载…';
  dialog.append(title, close, content); document.body.appendChild(dialog); dialog.showModal();
  try {
    const data = await api(`/api/code-workspace/${encodeURIComponent(sessionId)}`);
    content.replaceChildren();
    const location = document.createElement('p'); location.textContent = data.directory;
    content.appendChild(location);
    const note = document.createElement('p'); note.textContent = data.files.length ? data.note : '暂无文件。在右侧对话中描述要编写和验证的程序，Agent会按需创建并运行。';
    content.appendChild(note);
    for (const file of data.files) codeDownloadLink(file, content);
    if (data.truncated) { const tip = document.createElement('p'); tip.textContent = '仅显示前300项，完整文件保留在服务器目录。'; content.appendChild(tip); }
  } catch (error) { content.textContent = error.message; }
});

function appendCodeResults(message, report) {
  if (!report.workspace && !(report.code_runs || []).length && !(report.generated_files || []).length) return;
  const panel = document.createElement('details'); panel.open = true;
  const heading = document.createElement('summary');
  heading.textContent = `文件与运行结果 · ${(report.generated_files || []).length} 个源文件 · ${(report.code_runs || []).length} 次运行`;
  panel.appendChild(heading);
  const directory = document.createElement('p'); directory.textContent = report.workspace || ''; panel.appendChild(directory);
  for (const file of report.generated_files || []) codeDownloadLink(file, panel);
  const labels = {success: '执行成功', failed: '执行失败', timeout: '运行超时', output_limit: '输出超限', disk_limit: '产物超限'};
  for (const run of report.code_runs || []) {
    const title = document.createElement('p');
    title.textContent = `${run.path} · ${labels[run.status] || run.status} · 退出码 ${run.exit_code} · ${run.seconds} 秒`;
    panel.appendChild(title);
    for (const key of ['stdout', 'stderr']) {
      if (!run[key]) continue;
      const output = document.createElement('pre'); output.className = 'research-reader';
      output.textContent = `${key}\n${run[key]}`; panel.appendChild(output);
    }
    for (const file of run.files || []) codeDownloadLink(file, panel);
    const note = document.createElement('p'); note.textContent = run.note || ''; panel.appendChild(note);
  }
  message.bubble.appendChild(panel);
}

function appendWebSources(message, report) {
  if (!report) return;
  appendCodeResults(message, report);
  const references = document.createElement('details');
  if (Array.isArray(report.trace) && report.trace.length) {
    const steps = document.createElement('details');
    const title = document.createElement('summary');
    title.textContent = `自主研究过程 · ${report.trace.length} 次工具调用`;
    steps.appendChild(title);
    for (const step of report.trace) {
      const row = document.createElement('p');
      row.textContent = `${step.step}. ${step.tool} · ${step.status === 'ok' ? '完成' : '失败'} · ${JSON.stringify(step.arguments || {})}` + (step.error ? ` · ${step.error}` : '');
      steps.appendChild(row);
    }
    message.bubble.appendChild(steps);
  }
  const label = document.createElement('summary');
  label.textContent = report.operation === 'status'
    ? `联网能力检查 · ${report.status === 'ok' ? '搜索服务可用' : '连接检查未通过'}`
    : report.operation === 'clarification' ? '联网检索 · 等待明确主题'
    : `${report.operation === 'recall' ? '此前联网资料' : '本次联网资料'} · ${report.sources.length} 条 · ${!report.sources.length ? '本轮未获取联网资料' : report.status === 'ok' ? '已获取' : report.status === 'partial' ? '部分可用' : '未获取成功'}`;
  references.appendChild(label);
  const note = document.createElement('p'); note.textContent = report.note || ''; references.appendChild(note);
  for (const source of report.sources) {
    const row = document.createElement('p');
    row.textContent = `[${source.source_id}] ${source.discovery === 'session_history' ? '历史资料 · ' : ''}${source.title} · ${source.kind === 'paper_abstract' ? '摘要' : source.kind === 'downloaded_document' ? '已下载，正文未提取' : '正文片段'} `;
    try {
      const url = new URL(source.url);
      if (url.protocol === 'https:') {
        const link = document.createElement('a'); link.href = url.href; link.textContent = '原文';
        link.target = '_blank'; link.rel = 'noopener noreferrer'; row.appendChild(link);
      }
    } catch (_) {}
    const doc = source.document;
    if (doc && /^[0-9a-f]{64}$/.test(doc.document_id)) {
      const link = document.createElement('a'); link.href = `/api/web-research/documents/${doc.document_id}/download`;
      link.textContent = ' · 下载原文'; row.appendChild(link);
      if (doc.extraction_status === 'ok') {
        const content = document.createElement('a'); content.href = `/api/web-research/documents/${doc.document_id}/download?format=text`;
        content.textContent = ' · 下载正文'; row.appendChild(content);
      }
    }
    references.appendChild(row);
  }
  message.bubble.appendChild(references);
}

async function requestChat(text, sessionId, history) {
    const target = conversation(sessionId);
    if (!target.loaded && sessionId && !await restoreConversation(sessionId)) return false;
    if (target.busy || target.restoredRequestId) return false;
    const requestId = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() :
      `chat_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}_${Math.random().toString(36).slice(2)}`;
    target.liveRequests.add(requestId);
    const controller = new AbortController();
    let interrupt;
    const cancelled = new Promise(resolve => { interrupt = () => resolve({ cancelled: true }); });
    const pending = { requestId, controller, interrupt, cancelled: false };
    target.request = pending;
    target.busy = true;
    if (sessionId === state.sessionId) syncChatState();
    const clearWaiting = showChatWaiting(sessionId);
    try {
      const r = await Promise.race([pollChat({ question: text, session_id: sessionId, history, request_id: requestId,
          llm_profile: selectedLLMProfile(),
          include_history: $('#chat-history-enabled').checked }, controller.signal, clearWaiting.setStatus), cancelled]);
      if (state.deletedSessions.has(sessionId) || pending.cancelled || r.cancelled) return false;
      if (r.ok) {
        const message = appendLog(`[assistant] ${r.answer}`, "chat-ai", { sessionId });
        const nextHistory = [...history, { role: 'user', content: text },
          { role: 'assistant', content: r.answer }].slice(-12);
        decorateChatAnswer(message, r, sessionId, nextHistory);
        target.history = nextHistory;
        if (state.sessionId === sessionId) syncChatState();
        return true;
      }
      else appendLog(`[error] ${r.error}`, "error", { sessionId });
    } catch (e) {
      if (!state.deletedSessions.has(sessionId) && !pending.cancelled) appendLog(`[error] ${e.message}`, "error", { sessionId });
    } finally {
      clearWaiting();
      if (target.request === pending) {
        target.busy = false;
        target.request = null;
      }
      if (state.sessionId === sessionId) syncChatState();
      else updateDeleteSessionButton();
    }
    return false;
}

function displayedJob() {
  const view = state.taskViews.get(state.currentJobId);
  return state.currentJobId && view && view.sessionId === state.sessionId ? state.currentJobId : null;
}

function updateInterruptButton() {
  const button = $('#btn-interrupt-agent');
  const chat = state.conversations.get(state.sessionId);
  const generating = chat && ((chat.busy && chat.request && !chat.request.cancelled) || chat.restoredRequestId);
  button.disabled = !generating && !displayedJob();
  button.textContent = generating || !displayedJob() ? '中断生成' : '停止实验';
  button.title = generating ? '中断当前对话回复，不影响训练任务' : '停止当前会话整个实验（包含取数、训练和 Agent 提案）';
}

async function interruptAgent() {
  const sessionId = state.sessionId;
  const target = state.conversations.get(sessionId);
  if (target && target.restoredRequestId && !target.request) {
    try {
      await api(`/api/chat/${encodeURIComponent(target.restoredRequestId)}/cancel`, {method: 'POST'});
      await restoreConversation(sessionId);
    } catch (error) { alert(`取消失败：${error.message}`); }
    return;
  }
  if (target && target.busy && target.request) {
    const pending = target.request;
    if (pending.cancelled) return;
    pending.cancelled = true;
    pending.interrupt();
    pending.controller.abort();
    appendLog('[ui] 已中断本次回复，正在取消后台模型请求。', 'runner', { sessionId });
    updateInterruptButton();
    try {
      await api(`/api/chat/${encodeURIComponent(pending.requestId)}/cancel`, { method: 'POST' });
    } catch (error) {
      if (!state.deletedSessions.has(sessionId)) appendLog(`[ui] 回复已停止显示，但后台取消尚未确认：${error.message}`, 'error', { sessionId });
    }
    return;
  }
  const jobId = displayedJob();
  if (!jobId || !window.confirm('确认停止当前会话的整个实验？这将中断取数、训练或 Agent 提案，已保存结果保留。')) return;
  try {
    await api(`/api/run/${encodeURIComponent(jobId)}/stop`, { method: 'POST' });
    await restoreRunningJob();
  } catch (error) { alert(`停止失败：${error.message}`); }
}

$('#btn-interrupt-agent').addEventListener('click', interruptAgent);

const detectRunIntent = ETSControls.detectRunIntent;

$('#btn-restore-chat').addEventListener('click', () => {
  const target = conversation();
  // Live rows already on screen stay excluded, but after clear-screen they can
  // be restored from disk without reconstructing DOM from a browser cache.
  if (!target.nodes.length || target.nodes.every(node => node === target.pending)) target.liveRequests.clear();
  return restoreConversation();
});

$("#btn-clear-log").addEventListener("click", () => {
  const current = conversation();
  $("#log-view").innerHTML = "";
  current.nodes = current.pending ? [current.pending] : [];
  current.empty = null;
  if (current.pending) $("#log-view").appendChild(current.pending);
  refreshPinnedTask();
  _closeMerge();
});

// ================================================================
// 初始化
// ================================================================
let workspaceRefreshTimer = null;

function scheduleWorkspaceRefresh() {
  clearTimeout(workspaceRefreshTimer);
  workspaceRefreshTimer = setTimeout(refreshWorkspace, 150);
}

async function refreshSessions() {
  const requestVersion = ++state.sessionsRequest;
  const data = await api("/api/sessions");
  if (requestVersion !== state.sessionsRequest) return;
  data.sessions = data.sessions.filter(session => !state.deletedSessions.has(session.session_id));
  if (state.deletedSessions.has(data.current_session_id)) {
    data.current_session_id = data.sessions.length ? data.sessions[0].session_id : null;
  }
  state.sessions = data.sessions;
  if (!state.followCurrent && !data.sessions.some(session => session.session_id === state.sessionId)) {
    state.followCurrent = true;
  }
  const selected = state.followCurrent ? data.current_session_id : state.sessionId;
  if (selected !== state.sessionId) {
    switchConversation(selected);
  }
  else if (!conversation().loaded) await restoreConversation(selected);
  rememberChatSession();
  const select = $("#session-select");
  const current = data.sessions.find(session => session.session_id === state.sessionId);
  select.textContent = state.followCurrent ? '最新会话 ▾' : `${current && current.mode === 'chat' ? '对话 ' : current && current.city_name ? current.city_name + ' · ' : ''}${state.sessionId || '选择会话'} ▾`;
  select.value = state.followCurrent ? "" : state.sessionId || "";
  renderSessionList();
  updateDeleteSessionButton();
}

function updateDeleteSessionButton() {
  const selected = state.sessions.find(session => session.session_id === state.sessionId);
  const finished = selected && ['complete', 'failed', 'profile_complete', 'cancelled'].includes(selected.status);
  const button = $('#btn-delete-session');
  button.disabled = !finished || state.deletingSession || state.starting || !!state.currentJobId || state.chatBusy;
  button.title = state.deletingSession ? '正在删除…' : state.chatBusy ? '请等待当前回复结束' :
    state.currentJobId || state.starting ? '实验运行期间不能删除会话' : !finished ? '请选择已结束的会话' : `删除会话 ${state.sessionId}`;
  for (const item of state.sessionListButtons) item.button.disabled = !canDeleteSession(item.sessionId);
  $('#btn-new-chat').disabled = state.creatingChat || state.deletingSession;
}

function canDeleteSession(sessionId) {
  const session = state.sessions.find(item => item.session_id === sessionId);
  const chat = state.conversations.get(sessionId);
  return session && ['complete', 'failed', 'profile_complete', 'cancelled'].includes(session.status) &&
    !state.deletingSession && !state.creatingChat && !state.starting && !state.currentJobId && !(chat && (chat.busy || chat.restoredRequestId));
}

async function deleteSelectedSession(sessionId = state.sessionId) {
  if (!canDeleteSession(sessionId)) return;
  if (!sessionId || !window.confirm(`确认删除会话 ${sessionId}？\n该会话的实验记录、模型产物及本页对话将从界面移除，磁盘文件移入回收备份。其他会话不受影响。`)) return;
  state.deletingSession = true;
  $('#session-select').disabled = true;
  updateDeleteSessionButton();
  try {
    await api(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    state.deletedSessions.add(sessionId);
    if (state.sessionId === sessionId) {
      state.followCurrent = true;
      switchConversation(null);
    }
    clearTimeout(state.conversations.get(sessionId)?.restoreTimer);
    state.conversations.delete(sessionId);
    await refreshWorkspace();
  } catch (error) {
    alert(`删除失败：${error.message}`);
  } finally {
    state.deletingSession = false;
    $('#session-select').disabled = false;
    updateDeleteSessionButton();
  }
}

$('#btn-delete-session').addEventListener('click', () => deleteSelectedSession());

function closeSessionMenu() {
  $('#session-menu').hidden = true;
  $('#session-select').setAttribute('aria-expanded', 'false');
}

const citySessionGroupOpen = new Map();

function renderSessionList() {
  const list = $('#session-list');
  list.innerHTML = '';
  state.sessionListButtons = [];
  const names = { complete: '已完成', profile_complete: '画像完成', failed: '失败', cancelled: '已停止', loading: '取数中', searching: '搜索中' };
  const groups = new Map();
  const entries = [];
  for (const session of state.sessions) {
    if (session.city_group_id) {
      if (!groups.has(session.city_group_id)) {
        const entry = {id: session.city_group_id, sessions: []};
        groups.set(entry.id, entry); entries.push(entry);
      }
      groups.get(session.city_group_id).sessions.push(session);
    } else entries.push({sessions: [session]});
  }
  for (const entry of entries) {
    let container = list;
    if (entry.id) {
      const group = document.createElement('details');
      group.className = 'session-city-group';
      const selected = entry.sessions.some(session => session.session_id === state.sessionId);
      group.open = citySessionGroupOpen.has(entry.id) ? citySessionGroupOpen.get(entry.id) : selected;
      if (selected) group.className += ' selected';
      const summary = document.createElement('summary');
      const date = (entry.sessions.map(session => session.created_at || '').sort()[0] || '').replace('T', ' ').slice(0, 16);
      summary.textContent = `分城市训练 · ${entry.sessions[0]?.model_family === 'mlp' ? 'MLP' : 'XGB'} · ${entry.sessions.some(session => session.mock) ? '合成' : '真实'}${date ? ' · ' + date : ''}`;
      group.appendChild(summary);
      group.addEventListener('toggle', () => citySessionGroupOpen.set(entry.id, group.open));
      container = document.createElement('div');
      container.className = 'session-city-children';
      group.appendChild(container);
      list.appendChild(group);
      const cityOrder = ['100001', '100002', '100003'];
      entry.sessions.sort((a, b) => cityOrder.indexOf(a.training_city) - cityOrder.indexOf(b.training_city));
    }
    for (const session of entry.sessions) {
    const row = document.createElement('div');
    row.className = 'session-row' + (session.session_id === state.sessionId ? ' selected' : '');
    const choice = document.createElement('button');
    choice.type = 'button';
    choice.className = 'session-row-choice';
    choice.textContent = session.mode === 'chat' ? `对话 · ${session.session_id}` : session.training_city ? `${session.city_name} · ${session.session_id} · ${names[session.status] || session.status || '—'}` :
      `${session.mock ? 'mock' : '真实'} · ${session.session_id} · ${names[session.status] || session.status || '—'}`;
    choice.setAttribute('aria-current', session.session_id === state.sessionId ? 'true' : 'false');
    choice.addEventListener('click', async () => {
      state.followCurrent = false;
      switchConversation(session.session_id);
      closeSessionMenu();
      await refreshWorkspace();
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'session-delete';
    remove.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></svg>';
    remove.setAttribute('aria-label', `删除 ${session.mode === 'chat' ? '对话' : '实验会话'} ${session.session_id}`);
    remove.title = `删除 ${session.session_id}（运行或回复期间不可删除）`;
    remove.disabled = !canDeleteSession(session.session_id);
    remove.addEventListener('click', () => deleteSelectedSession(session.session_id));
    state.sessionListButtons.push({ sessionId: session.session_id, button: remove });
    row.appendChild(choice);
    row.appendChild(remove);
    container.appendChild(row);
    }
  }
}

async function createConversation() {
  if (state.creatingChat || state.deletingSession) return;
  state.creatingChat = true;
  updateDeleteSessionButton();
  try {
    const result = await api('/api/sessions', { method: 'POST' });
    state.sessionsRequest += 1;
    state.sessions.unshift(result.session);
    state.followCurrent = false;
    switchConversation(result.session.session_id);
    closeSessionMenu();
    await refreshWorkspace();
  } catch (error) { alert(`新建对话失败：${error.message}`); }
  finally { state.creatingChat = false; updateDeleteSessionButton(); }
}

$('#btn-new-chat').addEventListener('click', createConversation);
$('#session-select').addEventListener('click', () => {
  const menu = $('#session-menu');
  if (!menu.hidden) { closeSessionMenu(); return; }
  const rect = $('#session-select').getBoundingClientRect();
  menu.style.left = `${Math.max(12, Math.min(rect.left, window.innerWidth - 434))}px`;
  menu.style.top = `${Math.min(rect.bottom + 6, window.innerHeight - 180)}px`;
  menu.style.maxHeight = `${Math.max(140, window.innerHeight - Math.min(rect.bottom + 6, window.innerHeight - 180) - 16)}px`;
  menu.hidden = false;
  $('#session-select').setAttribute('aria-expanded', 'true');
  $('#btn-new-chat').focus?.();
  refreshSessions().catch(error => alert(`加载会话失败：${error.message}`));
});
$('#btn-close-sessions').addEventListener('click', closeSessionMenu);
window.addEventListener('resize', closeSessionMenu);
$('#btn-follow-session').addEventListener('click', () => {
  state.followCurrent = true;
  closeSessionMenu();
  refreshWorkspace();
});
document.addEventListener('click', event => {
  if (!$('#session-menu').hidden && !$('#session-menu').contains(event.target) && event.target !== $('#session-select')) closeSessionMenu();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('#session-menu').hidden) {
    closeSessionMenu();
    $('#session-select').focus?.();
  }
});

async function refreshWorkspace() {
  try {
    await refreshSessions();
    await Promise.all([refreshChampion(), loadHistory(), loadDiscovery(), loadMemory(), loadLineage(), loadHints()]);
  } catch (error) {
    appendLog(`[ui] 会话同步失败：${error.message}`, "error");
  }
}

$("#session-select").addEventListener("change", event => {
  state.followCurrent = !event.target.value;
  if (event.target.value) switchConversation(event.target.value);
  scheduleWorkspaceRefresh();
});

// 拖动分隔条：左右宽度可调
(function initSplitter() {
  const splitter = document.getElementById("splitter");
  const leftPane = document.querySelector(".left-pane");
  if (!splitter || !leftPane) return;

  // 恢复上次保存的宽度
  const saved = parseInt(localStorage.getItem("left_pane_width") || "0", 10);
  if (saved >= 300 && saved <= 900) {
    leftPane.style.width = saved + "px";
  }

  let dragging = false;
  let startX = 0;
  let startW = 0;

  splitter.addEventListener("mousedown", (e) => {
    dragging = true;
    startX = e.clientX;
    startW = leftPane.getBoundingClientRect().width;
    splitter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    let w = startW + dx;
    // clamp
    w = Math.max(300, Math.min(window.innerWidth - 400, w));
    leftPane.style.width = w + "px";
  });
  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    const w = leftPane.getBoundingClientRect().width;
    localStorage.setItem("left_pane_width", String(Math.round(w)));
  });
})();

// 拉一下日志路径显示到 header
fetch("/api/log/latest").then(r => r.json()).then(d => {
  if (!d || !d.session_log) return;
  const el = document.getElementById("log-path-hint");
  if (!el) return;
  const valueEl = el.querySelector(".lph-value");
  const path = d.session_log;
  // 展示时只取文件名（完整路径太长），完整路径进 title 悬浮
  const filename = path.split(/[\\/]/).pop();
  if (valueEl) valueEl.textContent = filename;
  el.setAttribute("title", "点击复制完整路径：\n" + path);
  el.addEventListener("click", () => {
    navigator.clipboard?.writeText(path);
    if (valueEl) {
      const orig = valueEl.textContent;
      valueEl.textContent = "✓ 已复制";
      setTimeout(() => { valueEl.textContent = orig; }, 1500);
    }
  });
}).catch(() => {});

// ---------- Discovery（v2 特征挖掘：数据画像 + 已接受配方 + holdout） ----------
function _renderRecipe(r) {
  // 把 DSL recipe 转成人类可读的一行
  if (!r || typeof r !== "object") return escapeHTML(String(r));
  const name = escapeHTML(r.name || "");
  if (r.kind === 'missing_indicator') {
    return `<code>${name}</code> = 缺失指示 <code>${escapeHTML(r.column)}</code>`;
  }
  if (r.kind === 'impute') {
    return `<code>${name}</code> = <code>${escapeHTML(r.column)}</code> · ${r.strategy === 'median' ? '实际训练行中位数填补' : '常数填补 ' + escapeHTML(r.value)}`;
  }
  if (r.kind === 'one_hot') {
    const categories = (r.keys || []).map((key, index) => `${key}=${(r.values || [])[index]}`).join(' × ');
    return `<code>${name}</code> = 独热/类别交叉 <code>${escapeHTML(categories)}</code>`;
  }
  if (r.kind === 'target_encoding') {
    return `<code>${name}</code> = 前日成熟训练标签编码 <code>${escapeHTML((r.keys || []).join(' × '))}</code> · ${r.target === 'mean_bin' ? '平均档位' : '档位 ' + escapeHTML(r.target) + ' 概率'} · 平滑 ${escapeHTML(r.smoothing)}`;
  }
  if (r.kind === "expression") {
    const stringify = (n) => {
      if (n == null) return "?";
      if ("col" in n) return n.col;
      if ("const" in n) return String(n.const);
      const op = n.op || "?";
      const args = (n.args || []).map(stringify);
      const bin = {add:"+", sub:"-", mul:"*", div:"/"}[op];
      if (bin && args.length === 2) return `(${args[0]} ${bin} ${args[1]})`;
      return `${op}(${args.join(", ")})`;
    };
    return `<code>${name}</code> = <code>${escapeHTML(stringify(r.expression))}</code> <span class="tag">expr</span>`;
  }
  if (r.kind === "history") {
    return `<code>${name}</code> = <code>${escapeHTML(r.agg)}(${r.value})</code> by <code>${escapeHTML((r.keys||[]).join(", "))}</code> · 前 ${r.days} 天 <span class="tag tag-hist">history</span>`;
  }
  if (r.kind === "rolling") {
    return `<code>${name}</code> = <code>${escapeHTML(r.agg)}(${r.value})</code> by <code>${escapeHTML((r.keys||[]).join(", "))}</code> · 过去 ${r.minutes} 分钟 <span class="tag tag-roll">rolling</span>`;
  }
  return `<code>${name}</code> ${escapeHTML(JSON.stringify(r))}`;
}

function _renderGate(gate) {
  if (!gate) return "";
  const cls = gate.pass ? "gate-ok" : "gate-fail";
  const label = gate.pass ? "通过" : "未通过";
  const reasons = (gate.reasons || []).map(r => `<li>${escapeHTML(r)}</li>`).join("");
  return `<span class="gate-badge ${cls}">${label}</span>${reasons ? `<ul class="gate-reasons">${reasons}</ul>` : ""}`;
}

function _statusBadge(status) {
  const map = {
    loading: ["加载中", "st-neutral"],
    profiled: ["已探查", "st-info"],
    searching: ["搜索中", "st-info"],
    complete: ["完成", "st-ok"],
    failed: ["失败", "st-err"],
    cancelled: ["已停止", "st-neutral"],
    profile_complete: ["仅画像", "st-info"],
  };
  const [label, cls] = map[status] || [status || "-", "st-neutral"];
  return `<span class="status-badge ${cls}">${escapeHTML(label)}</span>`;
}

function renderInspectionResults(profile) {
  const results = (profile || {}).inspection_results || [];
  if (!results.length) return '';
  const entries = results.map((entry, index) => {
    const result = entry.result || {};
    const rows = Object.entries(result.fields || {}).map(([name, stats]) => {
      const quantiles = stats.quantiles || {};
      const values = ['0.05', '0.5', '0.95'].map(key => quantiles[key] == null ? '—' : Number(quantiles[key]).toFixed(3)).join(' / ');
      return `<tr><td>${escapeHTML(name)}</td><td>${escapeHTML(stats.rows)}</td><td>${pct(stats.missing_rate)}</td><td>${escapeHTML(stats.unique)}</td><td>${values}</td></tr>`;
    }).join('');
    const correlations = (result.correlations || []).map(pair =>
      `${pair.first} / ${pair.second}: ${Number(pair.spearman).toFixed(3)}`).join('；');
    return `<details><summary>第 ${index + 1} 次 · ${escapeHTML(result.table || '已加载训练期字段')}</summary>
      <p>${result.kind === 'table' ? '训练日期内有界前缀样本，不代表全表分布；未自动关联入模。' : '仅统计已加载训练日期内的行，不含测试期。'}</p>
      <table class="sources-table"><thead><tr><th>字段</th><th>统计行数</th><th>缺失率</th><th>不同值</th><th>5% / 50% / 95%</th></tr></thead><tbody>${rows}</tbody></table>
      <p>相关性：${escapeHTML(correlations || '无可用字段对')}</p></details>`;
  }).join('');
  return `<div class="disc-section"><h4>Agent 自主只读探查（${results.length} 次）</h4>${entries}</div>`;
}

function renderModelStudy(study) {
  if (!study) return '<p>自主模型实验正在准备时间验证数据。</p>';
  const confirmed = Object.entries(study.confirmation || {});
  const entries = confirmed.length ? confirmed.map(([name, result]) => [name, result.aggregate])
    : Object.entries(study.screening || {}).map(([name, result]) => [name,
      Object.fromEntries(Object.entries(result.scores).map(([key, value]) => [key, {mean: value, std: 0}]))]);
  const rows = entries.map(([name, result]) => `<tr><td>${escapeHTML(name)}${name === study.recommended ? ' · 推荐' : ''}</td>
    <td class="model-ratio">${escapeHTML(modelRatioLabel({ensemble: study.confirmation?.[name]?.ensemble,
      experiment_spec: study.screening?.[name]?.spec}))}</td>
    <td>${pct(result.accuracy.mean)}${confirmed.length ? ' ± ' + (100 * result.accuracy.std).toFixed(2) + 'pp' : ''}</td>
    <td>${pct(result.severe_under.mean)}</td><td>${pct(result.long_wait_severe_under.mean)}</td>
    <td>${pct(result.legacy_accuracy.mean)}</td></tr>`).join('');
  return `<div class="disc-section"><h4>自主模型实验 · ${study.status === 'complete' ? '已完成' : study.status === 'budget_exhausted' ? '预算结束，复验未完成' : '进行中'}</h4>
    <p>推荐：${escapeHTML(study.recommended || '等待复验')}；准确率最高：${escapeHTML(study.accuracy_leader || '等待复验')}</p>
    <p>${escapeHTML(study.policy || '')}</p>
    <p>候选 ${Object.keys(study.screening || {}).length} / ${study.candidate_budget}；${confirmed.length ? '两个时间窗口 × 三个初始化种子；±为六次结果标准差，不是置信区间' : '当前展示单窗口初筛结果'}</p>
    ${study.stop_reason ? `<p>${escapeHTML(study.stop_reason)}</p>` : ''}
    <div class="city-compare-scroll"><table class="city-compare"><thead><tr><th>方案</th><th title="每个订单预测概率的融合权重">模型配比</th><th>固定已应答准确率</th><th>已应答严重低估</th><th>长等待严重低估（保序前）</th><th>取消过滤指标</th></tr></thead><tbody>${rows}</tbody></table></div>
    <details><summary>自主实验决策记录</summary>${(study.decisions || []).map(d => `<p>${escapeHTML(d.step)}. ${escapeHTML(d.reason)}</p>`).join('')}</details>
    <p class="hint">${escapeHTML(study.test_warning || '')}</p></div>`;
}

async function loadDiscovery() {
  const sessionId = state.sessionId;
  const container = $('#discovery-view');
  try {
    const data = await api('/api/discovery');
    if (sessionId !== state.sessionId) return;
    const session = data.session;
    if (!session) {
      container.innerHTML = `<p class="empty-hint">尚未运行过特征发现。点击右上角 <b>启动</b> 后，agent 会自动完成：<br>
        1) 探查 Hive 表结构 & SDS/queue JSON 字段 →
        2) 生成字段画像 →
        3) 评估 baseline，结合历史经验提出 DSL 配方 →
        4) 训练并在固定测试窗口比较 →
        5) 准确率优先、持平优先更少特征，选择 champion</p>`;
      return;
    }
    if (session.mode === 'chat') {
      container.innerHTML = '<p class="empty-hint">这是独立对话，尚未绑定实验数据。可在右侧直接咨询；查看具体模型结果请切换到对应实验会话。</p>';
      return;
    }

    const catalog = (data.profile || {}).catalog || {};
    const accepted = session.accepted_recipes || [];
    const sources = (data.sources || {}).sources || (Array.isArray(data.sources) ? data.sources : []);

    // 顶部会话卡片
    const window = session.window || {};
    const sessionCard = `
      <div class="discovery-header">
        <div class="dh-title">${escapeHTML(session.session_id || '-')} ${_statusBadge(session.status)}
          <span class="tag ${session.mock ? 'tag-mock' : 'tag-real'}">${escapeHTML(trainingDataLabel(session))}</span>
          ${!session.mock ? `<span class="tag">${session.num_bins === 6 ? 'Notebook 43特征·6桶' : session.data_contract === 'notebook_xgb44_v1' ? 'Notebook 历史11桶口径' : '旧取数口径'}</span>` : ''}
          ${session.smoke ? '<span class="tag tag-smoke">smoke</span>' : ''}
        </div>
        <div class="dh-meta">
          训练 <code>${window.train_start || '?'} → ${window.train_end || '?'}</code>
          ${window.validation_start ? `· 历史验证 <code>${escapeHTML(window.validation_start)} → ${escapeHTML(window.validation_end || '?')}</code>` : ''}
          · 测试 <code>${window.test_start || '?'} → ${window.test_end || '?'}</code>
        </div>
        <div class="dh-stats">
          ${session.model_study ? `<div><span class="stat-num">${session.study_trials_completed || 0}</span><span class="stat-label">已完成训练与评估</span></div>
          <div><span class="stat-num">${(session.study_budget || {}).candidates || '-'}</span><span class="stat-label">候选预算</span></div>` : `
          <div><span class="stat-num">${Object.keys(catalog).length}</span><span class="stat-label">可用字段</span></div>
          <div><span class="stat-num">${accepted.length}</span><span class="stat-label">已接受配方</span></div>
          <div><span class="stat-num">${session.rows ?? '-'}</span><span class="stat-label">总行数</span></div>
          <div><span class="stat-num">${session.profile_columns ?? Object.keys(catalog).length}</span><span class="stat-label">画像列</span></div>`}
        </div>
        ${session.error_hint ? `<div class="error-hint">⚠ ${escapeHTML(session.error_hint)}</div>` : ''}
        ${session.selection_warning ? `<div class="hint">${escapeHTML(session.selection_population || '')}。${escapeHTML(session.selection_warning)}</div>` : ''}
        ${session.reference_baseline && !session.model_study ? `<div class="hint">示例阈值：<b>${(100 * session.reference_baseline.metrics.ALL.recall_ratio).toFixed(4)}%</b>（${session.reference_baseline.metrics.ALL.recall} / ${session.reference_baseline.metrics.ALL.total}，6桶·H3-7保序后）。人工示例值，不是本会话重训结果。${escapeHTML(session.reference_baseline.warning || '')}</div>` : ''}
      </div>`;

    // 已接受的配方
    const acceptedBlock = accepted.length ? `
      <div class="disc-section">
        <h4>✓ 已接受配方（累计入 champion） · ${accepted.length}</h4>
        <ul class="recipe-list">${accepted.map(r => `<li>${_renderRecipe(r)}</li>`).join("")}</ul>
      </div>` : `
      <div class="disc-section muted">
        <h4>已接受配方</h4>
        <p class="empty-hint">尚未接受特征。新实验使用6桶·应答＋取消过滤·H3-7保序后整体召回，须达到示例阈值 且不低于当前对照；测试源样本须一致，过滤后人数可不同。城市和低估指标仅供观察。门禁通过不等于替换冠军，持平时优先更简单模型。历史实验保留原口径。</p>
      </div>`;
    const searchBlock = session.champion_policy ? `
      <div class="disc-section">
        <h4>持续搜索状态</h4>
        ${session.llm ? `<p>本次实验模型：${escapeHTML(session.llm.model)} · ${escapeHTML(session.llm.protocol)}</p>` : ''}
        <p>加载历史经验 ${escapeHTML(session.memory_entries_loaded ?? 0)} 条；保留探索分支 ${escapeHTML((session.exploration_branches || []).length)} 个。</p>
        ${session.research_state ? `<p>持续探索：已尝试 ${escapeHTML(session.research_state.attempted_steps)} / ${escapeHTML(session.research_state.max_steps)}；连续无提升 ${escapeHTML(session.research_state.no_improvement_steps)} 次。${escapeHTML(session.research_state.stop_reason || '每次实验后重新规划')}</p>` : ''}
        ${session.research_skills ? `<p>扩展研究能力已启用：特征编辑／分支／有界调参；新数据与代码需审批。</p>` : ''}
        ${(data.extension_requests || session.extension_requests || []).length ? `<details><summary>扩展申请（不会自动审批或合入）</summary>${(data.extension_requests || session.extension_requests).map(entry => `<p>${escapeHTML(entry.id)} · ${escapeHTML(entry.kind)} · ${escapeHTML(entry.status)}<br>${escapeHTML(entry.reason)}</p>`).join('')}</details>` : ''}
        <p class="hint">准确率持平时优先更少特征；执行失败可重试。同数据、父模型、完整特征及训练配置的已完成方案不重复训练。</p>
      </div>` : '';

    // Holdout 结果
    const holdoutBlock = data.holdout ? `
      <div class="disc-section">
        <h4>${data.holdout.historically_exposed ? '历史测试回测（曾参与旧实验搜索，不是独立验收）' : data.holdout.used_for_search ? '统一测试集结果（参与候选筛选）' : '历史独立测试集结果（未参与搜索）'}</h4>
        <div id="holdout-evaluation"></div>
        ${_renderGate(data.holdout.gate)}
        <p class="muted-tiny">冠军 id: <code>${escapeHTML(data.holdout.selected_id || '-')}</code> ·
        production_approved: ${data.holdout.production_approved ? '✓' : '✗'}</p>
      </div>` : '';

    // 数据源探查
    const sourcesBlock = sources && sources.length ? `
      <div class="disc-section">
        <h4>数据源探查（${sources.length} 张表）</h4>
        <table class="sources-table">
          <thead><tr><th>表</th><th>状态</th><th>字段数</th><th>用途</th></tr></thead>
          <tbody>${sources.map(s => `<tr>
            <td><code>${escapeHTML(s.table || '')}</code></td>
            <td>${s.status === 'schema_read' ? '✓ 已读' : (s.status === 'unavailable' ? '✗ 不可用' : escapeHTML(s.status || ''))}</td>
            <td>${(s.fields || []).length}</td>
            <td class="muted-tiny">${escapeHTML((s.materialization || '').replace('active','入模').replace('schema_only_time_join_unverified','仅探查 schema'))}</td>
          </tr>`).join("")}</tbody>
        </table>
      </div>` : '';

    // 字段画像（可搜索、按 spearman 排序）
    const catalogRows = Object.entries(catalog)
      .sort(([, a], [, b]) => Math.abs(b.train_label_spearman || 0) - Math.abs(a.train_label_spearman || 0))
      .map(([name, f]) => {
        const sp = f.train_label_spearman;
        const spCls = sp == null ? '' : (Math.abs(sp) >= 0.15 ? 'sp-strong' : (Math.abs(sp) >= 0.05 ? 'sp-mid' : 'sp-weak'));
        const q = f.quantiles || {};
        return `<tr class="catalog-row" data-name="${escapeHTML(name)}">
          <td><code>${escapeHTML(name)}</code></td>
          <td class="muted-tiny">${escapeHTML(f.source || '')}</td>
          <td>${f.role === 'group' ? '<span class="tag tag-group">group</span>' : (f.groupable ? '<span class="tag tag-group">可分组</span>' : '')}</td>
          <td>${pct(f.missing_rate)}</td>
          <td>${f.unique ?? '-'}</td>
          <td class="${spCls}">${sp == null ? '-' : sp.toFixed(3)}</td>
          <td class="muted-tiny">${q['0.05'] != null ? `${Number(q['0.05']).toFixed(2)} / ${Number(q['0.5']).toFixed(2)} / ${Number(q['0.95']).toFixed(2)}` : '-'}</td>
        </tr>`;
      }).join("");

    const catalogBlock = `
      <div class="disc-section">
        <h4>字段画像（${Object.keys(catalog).length} 项，按与 true_bin 的 |spearman| 排序）</h4>
        <input id="catalog-search" placeholder="过滤字段名…" class="catalog-search">
        <div class="catalog-legend"><span class="sp-strong">|ρ|≥0.15 强相关</span> · <span class="sp-mid">|ρ|≥0.05 中相关</span> · <span class="sp-weak">|ρ|&lt;0.05 弱相关</span></div>
        <div class="catalog-table-wrap"><table class="catalog-table">
          <thead><tr><th>字段</th><th>来源</th><th>类型</th><th>缺失率</th><th>不同值</th><th>ρ(true_bin)</th><th>5% / 50% / 95%</th></tr></thead>
          <tbody id="catalog-tbody">${catalogRows}</tbody>
        </table></div>
      </div>`;

    container.innerHTML = sessionCard + (session.model_study ? renderModelStudy(data.model_study) : searchBlock + acceptedBlock) + holdoutBlock + sourcesBlock + renderInspectionResults(data.profile) + (session.model_study ? '' : catalogBlock);
    if (data.holdout) mountEvaluation($('#holdout-evaluation'),
      { metrics_by_city: data.holdout.selected, evaluation_views: data.holdout.selected_views, config_diff: { selection_metric: data.holdout.selection_metric } },
      { metrics_by_city: data.holdout.baseline, evaluation_views: data.holdout.baseline_views, config_diff: { selection_metric: data.holdout.selection_metric } },
      '最终模型', 'baseline', true);

    // 字段过滤
    const search = $('#catalog-search');
    if (search) {
      search.addEventListener('input', () => {
        const kw = search.value.toLowerCase().trim();
        document.querySelectorAll('.catalog-row').forEach(row => {
          const name = (row.dataset.name || '').toLowerCase();
          row.style.display = !kw || name.includes(kw) ? '' : 'none';
        });
      });
    }
  } catch (e) {
    if (sessionId !== state.sessionId) return;
    container.textContent = '加载失败: ' + e.message;
  }
}
$('#btn-refresh-discovery').addEventListener('click', loadDiscovery);
$('#btn-sync-run').addEventListener('click', async () => {
  $('#run-status').textContent = '正在核对后台任务状态…';
  await restoreRunningJob();
  scheduleWorkspaceRefresh();
});
window.addEventListener('focus', () => restoreRunningJob());
refreshWorkspace();
restoreRunningJob();
setInterval(restoreRunningJob, 3000);

async function loadModels() {
  const data = await api('/api/models');
  const catalog = $('#model-catalog'); catalog.replaceChildren();
  for (const item of data.catalog) {
    const p = document.createElement('p');
    p.textContent = `${item.name}：${item.description} `;
    const button = document.createElement('button'); button.textContent = '选择训练';
    button.onclick = () => { $('#model-family').value = item.id; loadTuning(); };
    p.appendChild(button); catalog.appendChild(p);
  }
  const target = $('#saved-models'); target.replaceChildren();
  const note = document.createElement('p');
  note.textContent = '已保存模型（最近100个）；自主模型实验为时间验证成绩，其余为对应会话筛选成绩，不可跨口径直接比较。最终组合模型的复验与回测见特征发现页。';
  target.appendChild(note);
  const table = document.createElement('table');
  const head = document.createElement('tr');
  for (const label of ['模型', '训练策略', '范围', '会话 / 实验', '召回', '会话最优']) {
    const th = document.createElement('th'); th.textContent = label; head.appendChild(th);
  }
  table.appendChild(head);
  for (const row of data.models) {
    const tr = document.createElement('tr');
    for (const value of [row.model_name, row.training_population === 'unified' ? '统一训练' : '只训练排队', (row.city_name || '三城合并') + (row.mock ? ' · mock' : ' · 真实'),
      `${row.session_id} / ${row.experiment_id}`, row.recall_ratio == null ? '—' : `${(row.recall_ratio*100).toFixed(2)}%`, row.selected ? '是' : '否']) {
      const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  target.appendChild(table);
}
$('#refresh-models').addEventListener('click', loadModels);

$('#model-family').addEventListener('change', loadTuning);

$("#btn-run-settings").addEventListener("click", () => $('.tab[data-tab="run-settings"]').click());
