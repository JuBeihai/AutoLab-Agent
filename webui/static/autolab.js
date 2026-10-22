/* The workbench renders persisted experiment evidence, never fabricated demo results. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const labels = {queued:'排队中',running:'运行中',complete:'已完成',failed:'失败',cancelled:'已停止',interrupted:'已中断'};
  const models = {xgb:'XGBoost',random_forest:'Random Forest',linear:'Logistic / Ridge',mlp:'MLP'};
  let selected = null, state = null, selectedRecord = null, pollBusy = false, generation = 0;
  const active = value => ['queued','running'].includes(value?.status);
  const fmt = value => value == null ? '—' : Number(value).toFixed(4);
  const empty = (title, text, symbol='◇') => `<div class="empty-state"><div class="empty-symbol">${symbol}</div><h3>${escape(title)}</h3><p>${escape(text)}</p></div>`;
  async function api(path, options={}) {
    const response = await fetch('/api/lab'+path, {headers:{'Content-Type':'application/json'},...options});
    const body = await response.json(); if (!response.ok) throw new Error(body.error || `请求失败 ${response.status}`); return body;
  }
  function tab(name) {
    document.querySelectorAll('[data-tab]').forEach(button => button.classList.toggle('active',button.dataset.tab===name));
    ['experiments','lineage','attribution','research'].forEach(key => $('view-'+key).hidden=key!==name);
  }
  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click',() => tab(button.dataset.tab)));
  function formState() {
    $('csv-fields').hidden=$('dataset').value!=='local_csv';
    $('time-field').hidden=$('split').value!=='time';
    $('llm-field').hidden=$('planner').value!=='llm';
    $('privacy-note').hidden=$('planner').value!=='llm'&&!$('web-research').checked;
    $('budget-hint').textContent=`1 次基线 + 最多 ${Number($('rounds').value)*Number($('trials').value)} 次候选尝试。实验轮数与模型内部训练迭代次数不同。`;
  }
  ['dataset','split','planner','web-research','rounds','trials'].forEach(id=>$(id).addEventListener('change',formState));
  $('task-kind').addEventListener('change',()=> {
    $('task-metric').innerHTML=$('task-kind').value==='classification'?'<option value="accuracy">Accuracy · 越高越好</option><option value="f1_macro">Macro F1 · 越高越好</option>':'<option value="rmse">RMSE · 越低越好</option><option value="mae">MAE · 越低越好</option>';
  });
  async function runs() {
    const {runs:list}=await api('/runs');
    $('run-list').innerHTML=list.length?list.map(run=>`<button class="${run.id===selected?'active':''}" data-run="${escape(run.id)}"><span>${run.config.task.kind==='regression'?'回归':'分类'}实验 · ${escape(run.id.slice(-6))}</span><small>${escape(labels[run.status]||run.status)} · ${run.attempts||0} / ${run.budget||0} 次尝试</small></button>`).join(''):'<p class="side-empty">还没有实验记录。<br>从合成数据开始，体验完整闭环。</p>';
    $('run-list').querySelectorAll('[data-run]').forEach(button=>button.addEventListener('click',()=>select(button.dataset.run)));
  }
  async function select(id) {selected=id;selectedRecord=null;generation++;await refresh();await runs();}
  async function refresh() {
    if (!selected) return;
    const id=selected, token=generation;
    const result=await api('/runs/'+encodeURIComponent(id));
    if (id!==selected||token!==generation) return;
    state=result;render();
  }
  $('refresh-runs').addEventListener('click',()=>runs().catch(error=>$('form-error').textContent=error.message));
  $('new-run').addEventListener('click',()=> {selected=null;state=null;selectedRecord=null;generation++;render();runs();$('task-kind').focus();});
  $('experiment-form').addEventListener('submit',async event=> {
    event.preventDefault();$('form-error').textContent='';$('start').disabled=true;
    const csv=$('dataset').value==='local_csv';
    const config={task:{kind:$('task-kind').value,dataset:$('dataset').value,metric:$('task-metric').value,target:csv?$('target-column').value:'target',file:csv?$('dataset-file').value:'',split:csv?$('split').value:'random',time_column:csv&&$('split').value==='time'?$('time-column').value:''},models:[...document.querySelectorAll('input[name=model]:checked')].map(input=>input.value),rounds:Number($('rounds').value),trials_per_round:Number($('trials').value),minutes:Number($('minutes').value),patience:Number($('patience').value),planner:$('planner').value,llm_profile:$('llm-profile').value,web_research:$('web-research').checked};
    try {const result=await api('/runs',{method:'POST',body:JSON.stringify(config)});await select(result.id);tab('experiments');}
    catch(error){$('form-error').textContent=error.message;}
    finally{$('start').disabled=active(state);}
  });
  $('stop').addEventListener('click',async()=> {
    if (!selected) return;
    try {const result=await api('/runs/'+selected+'/stop',{method:'POST',body:'{}'});$('stage').textContent=result.message||result.status;$('stop').disabled=true;}
    catch(error){$('form-error').textContent=error.message;}
  });
  function render() {
    const records=state?.records||[], metric=state?.config?.task.metric;
    const champion=records.find(row=>row.id===state?.champion_id);
    $('run-id').textContent=state?.id||'EXPERIMENT WORKSPACE';
    $('run-title').textContent=state?`${state.config.task.kind==='regression'?'回归':'分类'}任务 · ${metric} ${state.config.task.kind==='regression'?'最小化':'最大化'}`:'新实验，等待你的第一个问题。';
    $('stat-attempts').textContent=state?`${state.attempts||0} / ${state.budget||0}`:'—';
    $('stat-round').textContent=state?`第 ${state.round||0} / ${state.config.rounds} 轮 · 基线独立计数`:'等待开始';
    $('stat-score').textContent=fmt(champion?.validation?.[metric]);
    $('stat-model').textContent=champion?`${champion.id} · ${models[champion.candidate.model]}`:'尚未建立基线';
    $('stat-test').textContent=fmt(state?.test_metrics?.[metric]);
    $('status-tag').textContent=state?(labels[state.status]||state.status):'就绪';
    $('stage').textContent=state?.stage||'选择任务与预算，开始一组可追溯的实验。';
    $('status-dot').className='status-dot '+(active(state)?'running':state?.status==='failed'?'failed':'');
    $('progress').max=Math.max(1,state?.budget||1);$('progress').value=state?.attempts||0;
    $('stop').hidden=!active(state);$('stop').disabled=!!state?.stop_requested;$('start').disabled=active(state);
    $('record-count').textContent=records.length;
    $('source-count').textContent=(state?.research||[]).reduce((sum,r)=>sum+(r.sources||[]).length,0);
    $('planner-label').textContent=state?(state.config.planner==='llm'?'LLM AGENT':'内置策略 · 非 LLM'):'等待执行';
    $('timeline').innerHTML=state?.events?.length?state.events.slice(-25).reverse().map(event=>`<div class="timeline-row"><time>${escape(new Date(event.time).toLocaleTimeString('zh-CN',{hour12:false}))}</time><span>${escape(event.message)}</span></div>`).join(''):'<p class="empty-inline">每一个提案、训练和评估步骤，都会留下记录。</p>';
    const data=state?.dataset;
    const prefix=data?`<div class="data-note">${data.rows.toLocaleString()} 行 · ${Object.keys(data.feature_names).length} 个特征 · 训练 ${data.split_sizes.train} / 验证 ${data.split_sizes.validation} / 测试 ${data.split_sizes.test}</div>`:'';
    $('view-experiments').innerHTML=prefix+(records.length?`<table class="experiment-table"><thead><tr><th>实验 / 父节点</th><th>模型</th><th>验证 ${escape(metric)}</th><th>输入</th><th>结果</th></tr></thead><tbody>${records.map(row=>`<tr class="${row.id===state.champion_id?'winner':''}" data-record="${row.id}"><td class="id">${row.id}<br><small>${row.parent_id?'↳ '+row.parent_id:'BASELINE'}</small></td><td>${escape(models[row.candidate.model])}</td><td>${fmt(row.validation?.[metric])}</td><td>${row.candidate.features.length}</td><td><span class="pill ${row.status==='failed'?'red':row.id===state.champion_id?'green':''}">${row.status==='running'?'训练中':row.status==='failed'?'失败':row.id===state.champion_id?'当前最优':row.promoted?'曾晋升':'对照'}</span></td></tr>`).join('')}</tbody></table><p class="table-help">点击实验查看假设、改动与归因；验证指标用于选择，最终测试不会反馈给规划器。</p>`:empty('实验记录会在这里逐步出现','Agent 会先建立基线，再在固定数据划分上验证每一个候选方案。','⊞'));
    $('view-experiments').querySelectorAll('[data-record]').forEach(row=>row.addEventListener('click',()=>{selectedRecord=row.dataset.record;attribution();tab('attribution');}));
    lineage();attribution();researchView();
  }
  function lineage() {
    const records=state?.records||[];
    if (!records.length){$('view-lineage').innerHTML=empty('每次实验都有来处','连线来自真实 parent_id；可以追踪模型切换、参数调整和特征消融的分支。','⌘');return;}
    const positions={};records.forEach((row,i)=>positions[row.id]={x:i===0?280:(i%2?145:415),y:i===0?46:46+Math.ceil(i/2)*105});
    const edges=records.filter(row=>positions[row.parent_id]).map(row=>{const p=positions[row.parent_id],c=positions[row.id];return `<path d="M ${p.x} ${p.y+25} C ${p.x} ${c.y-40},${c.x} ${c.y-40},${c.x} ${c.y-26}" fill="none" stroke="#c6d9d8" stroke-width="1.3" marker-end="url(#arrow)"/>`;}).join('');
    const nodes=records.map(row=>{const p=positions[row.id],winner=row.id===state.champion_id;return `<g class="node" data-record="${row.id}" tabindex="0" role="button" aria-label="查看实验 ${row.id}"><rect x="${p.x-106}" y="${p.y-25}" width="212" height="56" rx="8" fill="${winner?'#ecf7f1':'#fff'}" stroke="${row.status==='failed'?'#d39a89':winner?'#7dbda8':'#d6e2e8'}"/><text x="${p.x-92}" y="${p.y-5}" font-size="10" fill="#3f6671">${row.id} · ${escape(models[row.candidate.model])}</text><text x="${p.x-92}" y="${p.y+13}" font-size="9" fill="#8a9fa8">${escape(state.config.task.metric)} ${fmt(row.validation?.[state.config.task.metric])} ${winner?' · 当前最优':''}</text></g>`;}).join('');
    $('view-lineage').innerHTML=`<div class="lineage-key"><b>● 当前最优</b><span>○ 基线 / 对照</span><span>连线 = 实际比较的父实验</span></div><svg class="lineage-svg" viewBox="0 0 560 ${100+Math.ceil((records.length-1)/2)*105}" role="img" aria-label="实验血缘图"><defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#b2cac6"/></marker></defs>${edges}${nodes}</svg>`;
    $('view-lineage').querySelectorAll('[data-record]').forEach(node=>{const show=()=>{selectedRecord=node.dataset.record;attribution();tab('attribution');};node.addEventListener('click',show);node.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' ')show();});});
  }
  function attribution() {
    const records=state?.records||[];
    const row=records.find(r=>r.id===selectedRecord)||records.find(r=>r.id===state?.champion_id)||records.at(-1);
    if(!row){$('view-attribution').innerHTML=empty('用对照解释变化','查看实验假设、参数差异、特征置换重要性和误差切片，为下一轮规划提供证据。','≋');return;}
    const info=row.diagnostics||{}, imp=info.importance||[], delta=row.attribution?.utility_delta;
    const max=Math.max(1e-8,...imp.map(x=>Math.abs(x.importance)));
    const names=state.dataset?.feature_names||{};
    $('view-attribution').innerHTML=`<div class="attribution-top"><h3>实验归因</h3><select id="record-selector" aria-label="选择归因实验">${records.map(r=>`<option value="${r.id}" ${r.id===row.id?'selected':''}>${r.id} · ${escape(models[r.candidate.model])}</option>`).join('')}</select></div><p class="hypothesis">假设：${escape(row.candidate.hypothesis)}</p>${row.error?`<div class="error-box">${escape(row.error)}</div>`:''}<div class="attribution-summary">${escape(row.attribution?.summary||'训练与诊断尚未完成。')}<br>${row.parent_id?`相对 ${row.parent_id} 的方向统一收益：<strong>${delta==null?'—':(delta>=0?'+':'')+fmt(delta)}</strong>（正值表示更好）`:'基线：后续候选的统一参照。'}<br>${escape((row.attribution?.changes||[]).join(' / '))}${row.attribution?.controlled_ablation?' · 同模型同参数的特征消融对照':''}</div><p class="view-note">有效参数：${escape(JSON.stringify(row.candidate.params))}</p><h3 class="hypothesis">验证集置换重要性</h3><div class="bars">${imp.slice(0,10).map(item=>`<div class="bar-row"><span title="${escape(names[item.feature]||item.feature)}">${escape(names[item.feature]||item.feature)}</span><span class="bar-track"><i style="width:${Math.min(100,Math.abs(item.importance)/max*100)}%;${item.importance<0?'background:#bd9a88':''}"></i></span><span class="bar-value">${fmt(item.importance)}</span></div>`).join('')||'<p class="empty-inline">诊断完成后显示。</p>'}</div><p class="view-note">打乱单个特征后验证指标的损失；负值表示打乱后改善。使用有界样本估计，不是因果贡献。</p>${info.error_slices?.length?`<table class="experiment-table"><thead><tr><th>误差切片</th><th>样本数</th><th>${state.config.task.kind==='classification'?'错误率':'MAE'}</th></tr></thead><tbody>${info.error_slices.map(s=>`<tr><td>${escape(s.slice)}</td><td>${s.n}</td><td>${fmt(s.error_rate??s.mae)}</td></tr>`).join('')}</tbody></table>`:''}`;
    $('record-selector').addEventListener('change',event=>{selectedRecord=event.target.value;attribution();});
  }
  function researchView() {
    const reports=state?.research||[];
    if(!reports.length){$('view-research').innerHTML=empty('让公开研究参与实验规划','开启联网研究后，按诊断方向检索公开方法资料。LLM 规划器可将来源用于下一次假设；内置策略仍按固定规则执行。','↗');return;}
    $('view-research').innerHTML=reports.map(report=>`<section><p class="view-note">第 ${report.round} 轮 · ${report.cached?'复用缓存来源':'本轮检索'} · ${(report.sources||[]).length} 个来源</p>${report.summary?`<p class="hypothesis research-summary">${escape(report.summary)}</p>`:''}${report.error?`<div class="error-box">${escape(report.error)}</div>`:''}${(report.errors||[]).map(error=>`<p class="view-note">${escape(error.source||'研究工具')}：${escape(error.error||error)}</p>`).join('')}${(report.trace||report.searches||[]).length?`<details class="research-trace"><summary>检索与阅读记录</summary>${(report.trace||report.searches).map(step=>`<p class="view-note">${escape(step.tool||step.provider)} · ${escape(step.status)}<br>${escape(step.query||JSON.stringify(step.arguments||{}))}${step.error?'<br>'+escape(step.error):''}</p>`).join('')}</details>`:''}${(report.sources||[]).map(source=>{let url='';try{const u=new URL(source.url);if(u.protocol==='https:')url=u.href;}catch{}return `<article class="research-card">${url?`<a href="${escape(url)}" target="_blank" rel="noreferrer">${escape(source.title||source.url)} ↗</a>`:`<strong>${escape(source.title||'来源')}</strong>`}<p>${escape(source.excerpt||'已保存来源；正文或摘要请查看来源链接。')}</p><small>${escape(source.kind||'公开资料')}</small></article>`;}).join('')||'<p class="empty-inline">本轮未获取可用来源；不将检索失败当作研究证据。</p>'}</section>`).join('');
  }
  setInterval(async()=>{if(pollBusy||!selected||!active(state))return;pollBusy=true;try{await refresh();if(!active(state))await runs();}catch(error){$('stage').textContent='连接暂时中断，正在重试：'+error.message;}finally{pollBusy=false;}},1500);
  render();formState();runs().catch(error=>$('form-error').textContent=error.message);
  fetch('/api/llm-models').then(r=>r.json()).then(data=>{if(data.profiles?.length){$('llm-profile').innerHTML=data.profiles.map(m=>`<option value="${escape(m.profile_id)}">${escape(m.label||m.model||m.profile_id)}</option>`).join('');$('llm-profile').value=data.default_profile;}}).catch(()=>{});
})();
