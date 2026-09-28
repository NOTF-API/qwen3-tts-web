const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Exercise the shipped queue with controlled HTTP completions, including an
// out-of-order response. No browser framework or frontend dependency is needed.
const html = fs.readFileSync(path.join(__dirname, '../src/qwen3_tts_web/web/index.html'), 'utf8');
const queueSource = html.slice(html.indexOf('    function enqueueJob('), html.indexOf('    async function runPlaylist('));

function setup(t, count, mac = false, automatic = false) {
  const requests = [];
  const storage = new Map();
  const clips = Array.from({length: count}, (_, id) => ({id, serverId: `clip-${id}`, text: `text-${id}`, queued: true}));
  const element = () => ({textContent: '', disabled: false, classList: {add() {}, remove() {}, toggle() {}}});
  const context = vm.createContext({
    clips, setTimeout, clearTimeout, console, performance,
    apiRequest: async (url, options) => {
      if (url === '/api/inference/status') return {json: async () => ({active_clip_ids: []})};
      assert.equal(url, '/api/generate-batch');
      return new Promise(resolve => requests.push({items: JSON.parse(options.body).items, resolve}));
    },
    applyRecord: (clip, record) => Object.assign(clip, record),
    setStatus() {}, setGenStatus() {}, loadCapabilities() {}, updateSelection() {},
    sessionStorage: {getItem: key => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)},
    listEl: {querySelector() { return null; }},
    statClips: element(), statQueue: element(), statDuration: element(), statReady: element(),
    footQueue: element(), footClips: element(), footReady: element(), queueIndicator: element(),
    playBtn: element(), stopBtn: element(), clearBtn: element(),
    genBtn: element(), startBatchBtn: element(), queueHint: element(), autoGenerate: {checked: automatic},
    genRole: {value:'test'}, genEmotion: {value:''}, genLanguage: {value:'Chinese'}, genText: {value:''},
    designDownloadConsent: async () => false,
  });
  vm.runInContext(`
    const jobQueue = [];
    const activeJobs = new Map();
    let activeBatches = 0, batchSubmissionSize = ${mac ? 1 : 32}, parallelRequestLimit = ${mac ? 1 : 2};
    let queueTimer = null, inferencePollTimer = null, queueWorkerRunning = false;
    const PENDING_JOBS_KEY = 'test-queue';
    let queuePersistenceAvailable = true;
    let libraryLoaded = true, generatorMode = 'clone', roleList = ['test'];
    let playlistRunning = false, singlePlayingId = null;
    function renderList() { updateStats(); }
    function updateGeneratorMode() { updateQueueControls(); }
    ${queueSource}
  `, context);
  const run = code => vm.runInContext(code, context);
  t.after(() => run('clearTimeout(queueTimer); clearTimeout(inferencePollTimer);'));
  const enqueue = id => run(`enqueueJob({clipId:${id},text:'text-${id}',language:'Chinese',role:'voice-${id}',synthesis_mode:'clone'})`);
  const finish = (request, failedId = '') => request.resolve({json: async () => ({items: request.items.map(item =>
    item.clip_id === failedId ? {status: 'error', detail: '显存不足'} :
      {status: 'ok', result: {clip: {url: `/audio/${item.clip_id}`, generatedText: item.text}}})})});
  return {requests, clips, run, enqueue, finish, storage};
}

async function until(predicate) {
  const end = Date.now() + 3000;
  while (!predicate()) {
    if (Date.now() > end) assert.fail('queue did not reach expected state');
    await new Promise(resolve => setTimeout(resolve, 5));
  }
}

test('two bounded requests, third waits, out-of-order results preserve clip mapping', async t => {
  const q = setup(t, 70);
  for (let id = 0; id < 70; id++) q.enqueue(id);
  q.run('startBatch(); startBatch();');
  await until(() => q.requests.length === 2);
  assert.deepEqual(q.requests.map(r => r.items.length), [32, 32]);
  assert.equal(q.run('jobQueue.length'), 6);
  assert.equal(q.run('queueWorkerRunning'), true);
  q.finish(q.requests[1], 'clip-40');
  await until(() => q.requests.length === 3);
  assert.equal(q.clips[35].url, '/audio/clip-35');
  assert.equal(q.clips[35].generatedText, 'text-35');
  assert.equal(q.clips[0].url, undefined);
  assert.equal(q.clips[40].error, '显存不足');
  assert.equal(q.requests[2].items.length, 6);
  q.finish(q.requests[0]);
  q.finish(q.requests[2]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.run('activeJobs.size'), 0);
  assert.equal(q.clips.filter(c => c.url).length, 69);
  assert.equal(q.clips.some(c => c.queued || c.generating), false);
});

test('failed pending save is isolated and duplicate submission is ignored', async t => {
  const q = setup(t, 3);
  q.clips[1].saving = Promise.reject(new Error('保存失败'));
  q.clips[1].saving.catch(() => {});
  q.enqueue(0); q.enqueue(0); q.enqueue(1); q.enqueue(2);
  q.run('startBatch()');
  await until(() => q.requests.length === 1);
  assert.deepEqual(q.requests[0].items.map(i => i.clip_id), ['clip-0', 'clip-2']);
  q.finish(q.requests[0]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.clips[1].error, '保存失败');
  assert.equal(q.clips[2].url, '/audio/clip-2');
});

test('Mac capabilities keep one submitted job at a time', async t => {
  const q = setup(t, 2, true);
  q.enqueue(0); q.enqueue(1);
  q.run('startBatch()');
  await until(() => q.requests.length === 1);
  assert.equal(q.requests[0].items.length, 1);
  assert.equal(q.run('jobQueue.length'), 1);
  q.finish(q.requests[0]);
  await until(() => q.requests.length === 2);
  q.finish(q.requests[1]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.clips.filter(c => c.url).length, 2);
});

test('manual staging waits for a click, and later additions wait for the next click', async t => {
  const q = setup(t, 3);
  q.enqueue(0); q.enqueue(1);
  await new Promise(resolve => setTimeout(resolve, 120));
  assert.equal(q.requests.length, 0);
  assert.equal(q.run('queueTimer'), null);
  assert.equal(q.run('startBatchBtn.textContent'), '开始批量生成（2）');
  assert.equal(q.run('queueIndicator.textContent'), '暂存 2');
  q.run('startBatch()');
  // Even additions within the 80 ms collection window must stay staged.
  q.enqueue(2);
  await until(() => q.requests.length === 1);
  assert.deepEqual(q.requests[0].items.map(i => i.clip_id), ['clip-0', 'clip-1']);
  assert.equal(q.run('startBatchBtn.textContent'), '开始批量生成（1）');
  q.finish(q.requests[0]);
  await until(() => q.run('activeJobs.size') === 0);
  await new Promise(resolve => setTimeout(resolve, 120));
  assert.equal(q.requests.length, 1);
  assert.equal(q.clips[2].queued, true);
  q.run('startBatch()');
  await until(() => q.requests.length === 2);
  q.finish(q.requests[1]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.run('startBatchBtn.disabled'), true);
});

test('automatic mode releases existing stages, while turning it off only holds new tasks', async t => {
  const q = setup(t, 4);
  q.enqueue(0);
  q.run('autoGenerate.checked = true; changeAutoGenerate();');
  q.enqueue(1);
  q.run('autoGenerate.checked = false; changeAutoGenerate();');
  q.enqueue(2);
  await until(() => q.requests.length === 1);
  assert.deepEqual(q.requests[0].items.map(i => i.clip_id), ['clip-0', 'clip-1']);
  q.finish(q.requests[0]);
  await until(() => q.run('activeJobs.size') === 0);
  q.run('autoGenerate.checked = true; changeAutoGenerate();');
  q.enqueue(3);
  await until(() => q.requests.length === 2);
  assert.deepEqual(q.requests[1].items.map(i => i.clip_id), ['clip-2', 'clip-3']);
  q.finish(q.requests[1]);
  await until(() => !q.run('queueWorkerRunning'));
});

test('cancel staging keeps the draft editable and allows re-adding it', async t => {
  const q = setup(t, 2);
  q.enqueue(0); q.enqueue(1);
  q.run('cancelStagedJob(0)');
  assert.equal(q.clips[0].queued, false);
  assert.equal(q.clips[0].text, 'text-0');
  assert.equal(q.run('jobQueue.length'), 1);
  q.enqueue(0);
  q.run('startBatch(); cancelStagedJob(0);');
  await until(() => q.requests.length === 1);
  assert.deepEqual(q.requests[0].items.map(i => i.clip_id), ['clip-1', 'clip-0']);
  q.finish(q.requests[0]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.storage.size, 0);
});

test('refresh restores unsubmitted drafts as staged and never resubmits active jobs', async t => {
  const q = setup(t, 3, true);
  q.enqueue(0); q.enqueue(1);
  q.run('startBatch()');
  q.enqueue(2);
  await until(() => q.requests.length === 1);
  const saved = q.storage.get('test-queue');
  assert.deepEqual(JSON.parse(saved).map(item => item.serverId), ['clip-1', 'clip-2']);
  const reloaded = setup(t, 3, true);
  // Local IDs are assigned anew when the library is loaded.
  reloaded.clips.forEach(clip => { clip.id += 10; clip.queued = false; });
  reloaded.storage.set('test-queue', saved);
  reloaded.run('restorePendingJobs(); restorePendingJobs(); renderList();');
  assert.equal(reloaded.run('jobQueue.length'), 2);
  assert.equal(reloaded.run('jobQueue.every(job => !job.released)'), true);
  assert.equal(reloaded.clips[0].queued, false);
  assert.equal(reloaded.clips[1].queued, true);
  assert.equal(reloaded.clips[2].queued, true);
  assert.equal(reloaded.run('queueTimer'), null);
  assert.equal(reloaded.requests.length, 0);
  // Cancel the restored jobs; complete the original requests independently.
  reloaded.run('cancelStagedJob(11); cancelStagedJob(12);');
  q.finish(q.requests[0]);
  await until(() => q.requests.length === 2);
  q.finish(q.requests[1]);
  await until(() => q.run('activeJobs.size') === 0);
  q.run('cancelStagedJob(2)');
});

test('unavailable browser storage does not prevent staging or generation', async t => {
  const q = setup(t, 1);
  q.run("sessionStorage.setItem = () => { throw new Error('storage blocked'); };");
  q.enqueue(0);
  assert.match(q.run('queueHint.textContent'), /请勿刷新页面/);
  q.run('startBatch()');
  await until(() => q.requests.length === 1);
  q.finish(q.requests[0]);
  await until(() => !q.run('queueWorkerRunning'));
  assert.equal(q.clips[0].url, '/audio/clip-0');
});

test('status distinguishes calibration, waiting and actual native batch generation', t => {
  const q = setup(t, 0);
  assert.equal(q.run(`inferenceLabel({active_clip_ids:['a'],queued_clip_ids:['b'],active_phase:'calibrating',active:1}, 'a')`), '单条生成 · 测量显存');
  assert.equal(q.run(`inferenceLabel({active_clip_ids:['a'],queued_clip_ids:['b'],active_phase:'calibrating',active:1}, 'b')`), '等待显存测量');
  assert.equal(q.run(`inferenceLabel({active_clip_ids:['a','b'],active_phase:'generating',active:2,active_elapsed_seconds:69.25}, 'b')`), '同批 2 条 · 生成中 69s');
  assert.equal(q.run(`generationTimeLabel({duration:37.28,generation_stats:{batch_size:2,batch_seconds:69.25}})`), '同批 2 条推理 69.25s');
  assert.equal(q.run(`generationTimeLabel({duration:34.24})`), '');
});
