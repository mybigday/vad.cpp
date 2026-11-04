#!/usr/bin/env node
// Runs examples/web/test.html in a separate headless Chrome with WebGPU enabled
// (temporary profile, the user's browser is untouched) and prints the results.
// usage: node scripts/web-test.mjs [build-dir] [port]
//   needs: a server on the repository root (python3 -m http.server PORT), google-chrome
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const build = process.argv[2] || 'build-web';
const port = process.argv[3] || '8765';
const page = process.argv[4] || `test.html?build=${build}`;   // e.g. "debug.html?build=build-web&model=..."
const chrome = process.env.CHROME || 'google-chrome';
const profile = mkdtempSync(join(tmpdir(), 'vad-chrome-'));
const debugPort = 9300 + Math.floor(Math.random() * 500);

const proc = spawn(chrome, [
  '--headless=new', `--user-data-dir=${profile}`, `--remote-debugging-port=${debugPort}`,
  '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan', '--ignore-gpu-blocklist',
  '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ok = false;
try {
  let target;
  for (let i = 0; i < 50 && !target; ++i) {
    try {
      const url = `http://127.0.0.1:${port}/examples/web/${page}`;
      target = await (await fetch(`http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent(url)}`, { method: 'PUT' })).json();
    } catch { await sleep(200); }
  }
  if (!target) throw new Error('chrome did not start');
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener('open', r, { once: true }));
  let id = 0;
  const pending = new Map();
  ws.addEventListener('message', (e) => {
    const m = JSON.parse(e.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
    if (m.method === 'Runtime.exceptionThrown') console.error('page exception:', m.params.exceptionDetails.exception?.description);
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') console.error('console:', m.params.args.map((a) => a.value).join(' '));
  });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Runtime.enable');
  const gpu = await send('Runtime.evaluate', {
    expression: 'navigator.gpu ? navigator.gpu.requestAdapter().then(a => a ? `${a.info.vendor} ${a.info.architecture}` : "no adapter") : "no navigator.gpu"',
    awaitPromise: true });
  console.log('WebGPU adapter:', gpu.result.value);
  let text = '';
  for (let i = 0; i < 600 && !text.includes('done'); ++i) {
    await sleep(500);
    const r = await send('Runtime.evaluate', { expression: 'document.getElementById("out")?.textContent || ""' });
    text = r.result.value;
  }
  console.log(text);
  ok = text.includes('done') && !text.includes('FAILED');
  ws.close();
} finally {
  proc.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
process.exit(ok ? 0 : 1);
