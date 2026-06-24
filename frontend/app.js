const API = 'http://127.0.0.1:8000';
const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  models: [],
  compatibility: [],
  hardware: null,
  datasetValidation: null,
  selectedModel: null,
  exportCapabilities: null,
  trainingStatus: { status: 'idle', running: false }
};

function toast(msg){ const t=$('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2600); }
async function api(path, opts={}){
  const res = await fetch(`${API}${path}`, { headers: {'Content-Type':'application/json'}, ...opts });
  if(!res.ok){ let text = await res.text(); try{text=JSON.parse(text).detail || text}catch{} throw new Error(text); }
  return res.json();
}
function setApiState(ok){ $('apiDot').className = `dot ${ok?'ok':'bad'}`; $('apiState').textContent = ok ? 'Connected' : 'Offline'; }
function esc(v){ return String(v ?? '').replace(/[&<>'"]/g, s=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[s])); }
function metric(k,v){ return `<div class="metric"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`; }
function notice(text,type='warn'){ return `<div class="notice ${type}">${esc(text)}</div>`; }

function syncInputsToConfig(){
  if(!state.config) return;
  const cfg = state.config;
  cfg.device.mode = $('deviceToggle').checked ? 'gpu' : 'cpu';
  cfg.device.cuda = cfg.device.mode === 'gpu';
  cfg.dataset.path = $('datasetPath').value.trim();
  cfg.dataset.format = $('datasetFormat').value;
  cfg.dataset.yaml = $('datasetYaml').value.trim() || 'dataset.yaml';
  cfg.training.epochs = Number($('epochs').value || 1);
  cfg.training.batch_size = Number($('batchSize').value || 1);
  cfg.training.image_size = Number($('imageSize').value || 224);
  cfg.training.learning_rate = Number($('learningRate').value || 0.001);
  cfg.training.optimizer = $('optimizer').value;
  cfg.training.workers = Number($('workers').value || 0);
  cfg.training.seed = Number($('seed').value || 42);
  cfg.training.precision = $('precision').value;
  cfg.scheduler.name = $('scheduler').value;
  cfg.output.experiment_name = $('experimentName').value.trim() || 'exp001';
  if(state.selectedModel){
    cfg.model = {
      id: state.selectedModel.id,
      name: state.selectedModel.name,
      family: state.selectedModel.family,
      weights: state.selectedModel.weights,
      trainer: state.selectedModel.trainer,
      dataset_format: state.selectedModel.dataset_format
    };
    cfg.task = state.selectedModel.task;
  }
  renderConfig();
}
function fillInputsFromConfig(){
  const c=state.config; if(!c) return;
  $('deviceToggle').checked = c.device?.mode === 'gpu';
  $('datasetPath').value = c.dataset?.path || '';
  $('datasetFormat').value = c.dataset?.format || 'yolo';
  $('datasetYaml').value = c.dataset?.yaml || 'dataset.yaml';
  $('epochs').value = c.training?.epochs ?? 50;
  $('batchSize').value = c.training?.batch_size ?? 8;
  $('imageSize').value = c.training?.image_size ?? 640;
  $('learningRate').value = c.training?.learning_rate ?? 0.001;
  $('optimizer').value = c.training?.optimizer ?? 'auto';
  $('workers').value = c.training?.workers ?? 4;
  $('seed').value = c.training?.seed ?? 42;
  $('precision').value = c.training?.precision ?? 'fp32';
  $('scheduler').value = c.scheduler?.name ?? 'cosine';
  $('experimentName').value = c.output?.experiment_name ?? 'exp001';
}
function renderConfig(){
  if(!state.config) return;
  $('jsonPreview').value = JSON.stringify(state.config, null, 2);
  $('trainCommand').textContent = 'python -m backend.trainer.train_router --config backend/configs/saved/current_config.json';
  $('configSummary').innerHTML = [
    ['Task', state.config.task], ['Model', state.config.model?.name], ['Trainer', state.config.model?.trainer], ['Dataset format', state.config.dataset?.format], ['Device', state.config.device?.mode], ['Epochs', state.config.training?.epochs], ['Batch', state.config.training?.batch_size], ['Precision', state.config.training?.precision]
  ].map(([k,v])=>`<div><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join('');
}
function activeDatasetFormat(){ return state.datasetValidation?.dataset_format || state.config?.dataset?.format || null; }

async function loadBase(){
  try{
    const [health, cfg, models, hardware, exports] = await Promise.all([
      api('/api/health'), api('/api/default-config'), api('/api/models'), api('/api/hardware-profile'), api('/api/export/capabilities')
    ]);
    setApiState(true);
    state.config = cfg; state.models = models.models || []; state.hardware = hardware; state.exportCapabilities = exports;
    state.selectedModel = state.models.find(m=>m.id===cfg.model?.id) || state.models.find(m=>m.status==='supported');
    fillTaskFilter(); fillInputsFromConfig(); renderHardware(); await refreshCompatibility(); renderExportCapabilities(); renderConfig(); updateOverview();
  }catch(err){ setApiState(false); toast(`Backend unavailable: ${err.message}`); }
}
function fillTaskFilter(){
  const tasks = [...new Set(state.models.map(m=>m.task))].sort();
  $('taskFilter').innerHTML = '<option value="all">All tasks</option>' + tasks.map(t=>`<option value="${t}">${t}</option>`).join('');
}
async function refreshCompatibility(){
  syncInputsToConfig();
  const mode = state.config?.device?.mode || 'cpu';
  const fmt = activeDatasetFormat() || state.config?.dataset?.format || '';
  try{
    const data = await api(`/api/model-compatibility?device_mode=${encodeURIComponent(mode)}&dataset_format=${encodeURIComponent(fmt)}`);
    state.compatibility = data.models || [];
    state.hardware = data.hardware_profile || state.hardware;
    if(state.selectedModel){ state.selectedModel = state.compatibility.find(m=>m.id===state.selectedModel.id) || state.selectedModel; }
    renderModels(); renderHardware(); updateGate(); updateOverview();
  }catch(err){ toast(`Compatibility check failed: ${err.message}`); }
}
function renderHardware(){
  const hw=state.hardware; if(!hw) return;
  const cpu=hw.cpu||{}, gpu=hw.gpu||{};
  $('readinessScore').textContent = hw.readiness_score ?? '--';
  $('cpuMetrics').innerHTML = [metric('CPU', cpu.name), metric('Physical cores', cpu.physical_cores), metric('Logical cores', cpu.logical_cores), metric('CPU usage', `${cpu.usage_percent ?? '--'}%`), metric('RAM total', `${cpu.ram_total_gb ?? '--'} GB`), metric('RAM used', `${cpu.ram_used_gb ?? '--'} GB`), metric('RAM usage', `${cpu.ram_usage_percent ?? '--'}%`), metric('Recommended workers', cpu.recommended_workers)].join('');
  const dev = gpu.selected_device || {};
  $('gpuMetrics').innerHTML = [metric('CUDA available', gpu.cuda_available), metric('GPU count', gpu.gpu_count), metric('GPU name', dev.name || 'None'), metric('VRAM total', dev.vram_total_gb ? `${dev.vram_total_gb} GB` : '0 GB'), metric('VRAM allocated', dev.vram_allocated_gb ? `${dev.vram_allocated_gb} GB` : '0 GB'), metric('CUDA version', gpu.cuda_version || 'None'), metric('PyTorch', gpu.torch_version || 'Unknown'), metric('FP16 supported', gpu.fp16_supported)].join('');
  $('cpuModeLabel').textContent = state.config?.device?.mode === 'cpu' ? 'selected' : 'available';
  $('gpuModeLabel').textContent = state.config?.device?.mode === 'gpu' ? 'selected' : (gpu.cuda_available ? 'available' : 'unavailable');
  const warnings=[];
  if(!gpu.cuda_available) warnings.push(notice('CUDA is not available. GPU mode and TensorRT export will be blocked.', 'warn'));
  if(!gpu.tensorrt_available) warnings.push(notice('TensorRT not detected. Engine export is disabled until NVIDIA TensorRT is installed.', 'warn'));
  if(!hw.packages?.ultralytics) warnings.push(notice('Ultralytics package is not installed. YOLO / RT-DETR cards are disabled.', 'bad'));
  $('hardwareWarnings').innerHTML = warnings.join('') || notice('Hardware profile is ready. Capability checks are based on installed packages and PyTorch.', 'good');
  $('readinessList').innerHTML = (hw.readiness || []).map(x=>`<div class="check"><b>${esc(x.name)}</b><span>${x.ok?'OK':'Missing'} · ${esc(x.detail)}</span></div>`).join('');
  $('checklistState').textContent = `${hw.readiness_score ?? 0}%`;
}
function renderModels(){
  const task=$('taskFilter').value, status=$('statusFilter').value;
  const list = state.compatibility.filter(m => (task==='all'||m.task===task) && (status==='all'||m.status===status));
  $('modelCount').textContent = `${list.length} visible`;
  $('modelGrid').innerHTML = list.map(m=>{
    const disabled = !m.enabled;
    const selected = state.selectedModel?.id === m.id;
    const badge = m.status === 'supported' ? (m.enabled ? 'good' : 'warn') : 'bad';
    return `<article class="model-card ${disabled?'disabled':''} ${selected?'selected':''}" data-model="${esc(m.id)}">
      <h4>${esc(m.name)}</h4>
      <div class="tags">
        <span class="tag ${badge}">${esc(m.status)}</span><span class="tag">${esc(m.task)}</span><span class="tag">${esc(m.trainer)}</span>
        <span class="tag ${m.supports_cpu?'good':'bad'}">CPU ${m.supports_cpu?'yes':'no'}</span><span class="tag ${m.supports_gpu?'good':'bad'}">GPU ${m.supports_gpu?'yes':'no'}</span>
      </div>
      <div class="model-reason">RAM ${m.min_ram_gb}/${m.recommended_ram_gb} GB · VRAM ${m.min_vram_gb}/${m.recommended_vram_gb} GB</div>
      <div class="model-reason"><b>${disabled?'Disabled':'Enabled'}:</b> ${esc(m.reason || 'Ready')}</div>
      ${(m.warnings||[]).map(w=>`<div class="model-reason">⚠ ${esc(w)}</div>`).join('')}
    </article>`;
  }).join('') || notice('No models match the current filters.', 'warn');
  document.querySelectorAll('.model-card').forEach(card=>card.addEventListener('click',()=>{
    const model = state.compatibility.find(m=>m.id===card.dataset.model);
    if(!model || !model.enabled){ toast(model?.reason || 'Model disabled'); return; }
    state.selectedModel = model;
    state.config.model = {id:model.id,name:model.name,family:model.family,weights:model.weights,trainer:model.trainer,dataset_format:model.dataset_format};
    state.config.task = model.task;
    state.config.dataset.format = model.dataset_format;
    $('datasetFormat').value = model.dataset_format;
    $('imageSize').value = model.recommended_image_size || $('imageSize').value;
    $('batchSize').value = model.recommended_batch_size || $('batchSize').value;
    $('precision').value = model.recommended_precision || 'fp32';
    syncInputsToConfig(); renderModels(); updateGate(); toast(`Selected ${model.name}`);
  }));
}
async function validateDataset(){
  syncInputsToConfig();
  try{
    const res = await api('/api/validate-dataset', {method:'POST', body:JSON.stringify(state.config)});
    state.datasetValidation = res;
    $('datasetValidState').textContent = res.valid ? 'valid' : 'invalid';
    $('datasetMetrics').innerHTML = [metric('Total images', res.total_images), metric('Train images', res.train_image_count), metric('Val images', res.val_image_count), metric('Labels/masks', res.label_count), metric('Classes', res.class_count ?? '--'), metric('Format', res.dataset_format)].join('');
    const msgs = [];
    (res.errors||[]).forEach(e=>msgs.push(notice(e,'bad'))); (res.warnings||[]).forEach(w=>msgs.push(notice(w,'warn'))); (res.recommended_fixes||[]).forEach(f=>msgs.push(notice(`Fix: ${f}`,'warn')));
    if(res.valid) msgs.push(notice('Dataset validation passed. Training can proceed after model compatibility passes.','good'));
    $('datasetMessages').innerHTML = msgs.join('');
    await refreshCompatibility();
  }catch(err){ toast(`Validation failed: ${err.message}`); }
}
function updateGate(){
  syncInputsToConfig();
  const blockers=[];
  const datasetOk = !!state.datasetValidation?.valid;
  const model = state.selectedModel;
  if(!datasetOk) blockers.push('Dataset validation has not passed.');
  if(!model) blockers.push('No model selected.');
  if(model && !model.enabled) blockers.push(model.reason || 'Model is disabled by compatibility check.');
  if(model && state.datasetValidation && model.dataset_format !== state.datasetValidation.dataset_format) blockers.push(`Dataset format mismatch. Model requires ${model.dataset_format}.`);
  if(state.config.device.mode==='gpu' && !state.hardware?.gpu?.cuda_available) blockers.push('GPU selected but CUDA is not available.');
  if(model && model.status !== 'supported') blockers.push('Selected model is future integration, not supported.');
  const ready = blockers.length===0;
  state.config.status.dataset_valid = datasetOk;
  state.config.status.model_enabled = !!model?.enabled;
  state.config.status.ready_to_train = ready;
  $('startBtn').disabled = !ready; $('startBtn2').disabled = !ready;
  $('trainGate').textContent = ready ? 'ready' : 'blocked';
  $('trainBlockers').innerHTML = ready ? notice('Ready to launch a real backend trainer.', 'good') : blockers.map(b=>notice(b,'bad')).join('');
  renderConfig(); updateOverview();
}
function updateOverview(){
  $('datasetBadge').textContent = state.datasetValidation?.valid ? 'Valid' : 'Not ready';
  $('datasetMini').textContent = state.datasetValidation ? `${state.datasetValidation.total_images} images · ${state.datasetValidation.dataset_format}` : 'Validate dataset first';
  $('modelBadge').textContent = state.selectedModel?.name || 'No model';
  $('modelMini').textContent = state.selectedModel ? (state.selectedModel.enabled ? 'Compatible' : state.selectedModel.reason) : 'Select compatible model';
  $('deviceBadge').textContent = state.config?.device?.mode?.toUpperCase() || 'CPU';
  $('deviceMini').textContent = state.config?.device?.mode === 'gpu' ? (state.hardware?.gpu?.cuda_available ? 'CUDA available' : 'CUDA unavailable') : 'CPU safe mode';
  $('statusBadge').textContent = state.trainingStatus?.status || 'idle';
  $('trainingStatusHero').textContent = state.trainingStatus?.status || 'idle';
  $('statusMini').textContent = state.trainingStatus?.running ? 'Trainer running' : 'No active run';
}
async function startTraining(){
  updateGate(); if($('startBtn').disabled){ toast('Training blocked. Check blockers.'); return; }
  try{ const res = await api('/api/start-training', {method:'POST', body:JSON.stringify(state.config)}); toast(res.message || 'Training started'); await pollStatus(); }
  catch(err){ toast(`Start failed: ${err.message}`); }
}
async function stopTraining(){ try{ const res=await api('/api/stop-training',{method:'POST',body:'{}'}); toast(res.message || 'Stop requested'); }catch(err){ toast(err.message); } }
async function pollStatus(){
  try{ state.trainingStatus = await api('/api/training-status'); updateOverview(); }
  catch{ setApiState(false); }
}
async function pollLogs(){
  try{
    const data = await api('/api/training-logs');
    const text = (data.logs||[]).join('\n'); $('liveLogs').textContent = text; $('overviewLogs').textContent = text.split('\n').slice(-12).join('\n');
    $('liveLogs').scrollTop = $('liveLogs').scrollHeight;
  }catch{}
}
async function loadExperiments(){
  try{
    const data=await api('/api/experiments'); const list=data.experiments||[];
    $('experimentTimeline').innerHTML = list.length ? list.map(e=>`<div class="timeline-item"><strong>${esc(e.experiment_name || e.name || 'experiment')}</strong><span>${esc(e.status)} · ${esc(e.started_at || e.created_at || '')}</span><small>${esc(JSON.stringify(e.metrics || {}))}</small></div>`).join('') : notice('No experiments recorded yet.','warn');
  }catch(err){ $('experimentTimeline').innerHTML = notice(err.message,'bad'); }
}
function renderExportCapabilities(){
  const c=state.exportCapabilities||{};
  const tensorrt = c.tensorrt || {};
  $('exportCapabilityBadge').textContent = tensorrt.enabled ? 'TensorRT ready' : 'TensorRT disabled';
  const fmt = $('exportFormat').value;
  const warnings=[];
  if(fmt==='tensorrt' && !tensorrt.enabled) warnings.push(notice(tensorrt.reason || 'TensorRT not detected.', 'bad'));
  if($('exportPrecision').value==='fp16' && !c.fp16?.enabled) warnings.push(notice(c.fp16?.reason || 'FP16 not supported on current GPU.', 'warn'));
  if(fmt==='openvino' && $('exportFamily').value !== 'ultralytics') warnings.push(notice('OpenVINO export is exposed only through Ultralytics in this build.', 'warn'));
  $('exportWarnings').innerHTML = warnings.join('') || notice('Export settings are available. The backend will still validate checkpoint and dependencies before running.', 'good');
}
function exportPayload(){ return { checkpoint:$('exportCheckpoint').value.trim(), model_family:$('exportFamily').value, format:$('exportFormat').value, precision:$('exportPrecision').value, opset:Number($('exportOpset').value||17), image_size:Number($('exportImageSize').value||640), dynamic_batch:$('exportDynamic').checked, config:state.config }; }
function updateExportCommand(){ const p=exportPayload(); $('exportCommand').textContent = `python -m backend.exporter.export_router --checkpoint "${p.checkpoint || 'backend/runs/exp001/best.pt'}" --format ${p.format} --precision ${p.precision} --imgsz ${p.image_size} --opset ${p.opset}`; renderExportCapabilities(); }
async function runExport(){
  const p=exportPayload(); if(!p.checkpoint){ toast('Select a checkpoint first.'); return; }
  const path = p.format === 'tensorrt' ? '/api/export/tensorrt' : p.format === 'torchscript' ? '/api/export/torchscript' : p.format === 'openvino' ? '/api/export/openvino' : '/api/export/onnx';
  try{ const res=await api(path,{method:'POST',body:JSON.stringify(p)}); $('exportStatus').textContent=res.status; toast(res.status==='failed'?`Export failed: ${res.error}`:'Export completed'); await pollExport(); }
  catch(err){ toast(`Export failed: ${err.message}`); }
}
async function convertPrecision(){ const p=exportPayload(); try{ const res=await api('/api/export/precision',{method:'POST',body:JSON.stringify(p)}); $('exportStatus').textContent=res.status; toast(res.status==='failed'?res.error:'Precision conversion completed'); await pollExport(); }catch(err){ toast(err.message); } }
async function pollExport(){ try{ const [s,l]=await Promise.all([api('/api/export/status'), api('/api/export/logs')]); $('exportStatus').textContent=s.status; $('exportLogs').textContent=(l.logs||[]).join('\n'); }catch{} }
async function saveConfig(){ syncInputsToConfig(); try{ const res=await api('/api/save-config',{method:'POST',body:JSON.stringify(state.config)}); toast(`Config saved: ${res.current_path || res.path}`); }catch(err){ toast(err.message); } }
function exportJson(){ syncInputsToConfig(); const blob=new Blob([JSON.stringify(state.config,null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='visiontrain_config.json'; a.click(); URL.revokeObjectURL(a.href); }
function applyJson(){ try{ state.config=JSON.parse($('jsonPreview').value); fillInputsFromConfig(); refreshCompatibility(); toast('JSON applied'); }catch(err){ toast(`Invalid JSON: ${err.message}`); } }
function showPage(page, updateHash=true){
  if(!$(`page-${page}`)) page = 'overview';
  document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.page===page));
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  $(`page-${page}`).classList.add('active');
  $('pageTitle').textContent = page.charAt(0).toUpperCase()+page.slice(1);
  if(updateHash && window.location.hash !== `#${page}`) history.replaceState(null, '', `#${page}`);
  if(page==='experiments') loadExperiments();
}
function bind(){
  document.querySelectorAll('.nav button,[data-page-shortcut]').forEach(b=>b.addEventListener('click',()=>showPage(b.dataset.page||b.dataset.pageShortcut)));
  window.addEventListener('hashchange',()=>showPage((window.location.hash || '#overview').slice(1), false));
  ['deviceToggle','datasetFormat','epochs','batchSize','imageSize','learningRate','optimizer','scheduler','workers','seed','precision','experimentName','datasetPath','datasetYaml'].forEach(id=>$(id).addEventListener('input',()=>{ syncInputsToConfig(); refreshCompatibility(); }));
  $('refreshBtn').onclick=loadBase; $('validateBtn').onclick=validateDataset; $('startBtn').onclick=startTraining; $('startBtn2').onclick=startTraining; $('stopBtn').onclick=stopTraining; $('saveConfigBtn').onclick=saveConfig;
  $('taskFilter').onchange=renderModels; $('statusFilter').onchange=renderModels; $('reloadExperiments').onclick=loadExperiments;
  $('copyTrainCmd').onclick=()=>navigator.clipboard.writeText($('trainCommand').textContent).then(()=>toast('Training command copied'));
  $('copyExportCmd').onclick=()=>navigator.clipboard.writeText($('exportCommand').textContent).then(()=>toast('Export command copied'));
  ['exportCheckpoint','exportFamily','exportFormat','exportPrecision','exportOpset','exportImageSize','exportDynamic'].forEach(id=>$(id).addEventListener('input',updateExportCommand));
  $('exportBtn').onclick=runExport; $('precisionBtn').onclick=convertPrecision;
  $('applyJsonBtn').onclick=applyJson; $('exportJsonBtn').onclick=exportJson; $('resetJsonBtn').onclick=loadBase; $('importJsonBtn').onclick=()=>$('importJsonFile').click();
  $('importJsonFile').addEventListener('change',async(e)=>{ const f=e.target.files[0]; if(!f) return; const txt=await f.text(); $('jsonPreview').value=txt; applyJson(); });
}
bind(); showPage((window.location.hash || '#overview').slice(1), false); loadBase(); setInterval(pollStatus,1500); setInterval(pollLogs,1000); setInterval(pollExport,2000);
