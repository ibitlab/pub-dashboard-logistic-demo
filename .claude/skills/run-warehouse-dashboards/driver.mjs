#!/usr/bin/env node
/**
 * Driver for the Warehouse Activity Intelligence dashboards.
 *
 * Both apps (Feature-MediaPipe :8000, Feature-RTMPose :8001) ship the same
 * frontend shell, so one driver handles both; --app selects port + whether the
 * pose stage is multi-person.
 *
 * Usage:
 *   node driver.mjs shots    --app rtmpose   --out /tmp/shots [--video <file in data/demo>] [--project <name>]
 *   node driver.mjs preview  --app rtmpose   --from 463 --to 610 --step 4 --out /tmp/frames [--keep-empty 1]
 *   node driver.mjs annotate --app mediapipe --out /tmp/shots
 *   node driver.mjs reid     --out /tmp/shots [--collection <name>]   (rtmpose only)
 *
 * --video picks the demo clip (default: the first one the server lists).
 * --project / --collection select a Training project / Workers collection by
 * name first, so the screenshot shows that context instead of "No Project".
 *
 * Requires: `npm install` in this directory, and Google Chrome installed
 * (we drive the system Chrome via channel:'chrome' - see SKILL.md Gotchas).
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const APPS = {
  mediapipe: { port: 8000, multi: false },
  rtmpose: { port: 8001, multi: true },
};

// ---------- args ----------
const [, , cmd, ...rest] = process.argv;
const args = {};
for (let i = 0; i < rest.length; i += 2) args[rest[i].replace(/^--/, '')] = rest[i + 1];

const appName = args.app || 'rtmpose';
const app = APPS[appName];
if (!cmd || !app) {
  console.error(`usage: driver.mjs <shots|preview|annotate|reid> --app <mediapipe|rtmpose> --out <dir>`);
  process.exit(2);
}
const PORT = Number(args.port || app.port);
const OUT = args.out || './driver-out';
fs.mkdirSync(OUT, { recursive: true });

const DEMO = args.video
  ? `/demo/${args.video}`
  : (await (await fetch(`http://localhost:${PORT}/api/demo/videos`)).json()).videos[0]?.url;
if (!DEMO) {
  console.error('no demo videos: put a clip into data/demo/ or pass --video');
  process.exit(2);
}

// ---------- helpers ----------
async function newPage(browser, height = 1000) {
  const page = await browser.newPage({ viewport: { width: 1600, height }, deviceScaleFactor: 2 });
  page.on('pageerror', (e) => console.log('  [pageerror]', e.message.slice(0, 160)));
  return page;
}

async function open(page) {
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);
}

const tab = (page, name) => page.click(`.tab-btn[data-tab="${name}"]`);

/**
 * Screenshots end up in a public repo: replace wall-clock dates, times and
 * timestamp-based version ids on the page with placeholders before each shot.
 */
async function maskTimestamps(page) {
  await page.evaluate(() => {
    const rules = [
      [/\b\d{8}_\d{6}\b/g, 'YYYYMMDD_hhmmss'],
      [/\b\d{4}-\d{2}-\d{2}\b/g, 'YYYY-MM-DD'],
      [/\b\d{1,2}:\d{2}:\d{2}\b/g, 'hh:mm:ss'],
    ];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      let t = n.nodeValue;
      for (const [re, rep] of rules) t = t.replace(re, rep);
      if (t !== n.nodeValue) n.nodeValue = t;
    }
  });
}

/** Select a <select> option whose label contains `name`, as a user would. */
async function chooseByName(page, selectId, name) {
  if (!name) return;
  const ok = await page.evaluate(([id, name]) => {
    const s = document.getElementById(id);
    const o = s && [...s.options].find((o) => o.textContent.includes(name));
    if (!o) return false;
    s.value = o.value;
    s.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, [selectId, name]);
  if (!ok) console.log(`  [warn] no option containing "${name}" in #${selectId}`);
  await page.waitForTimeout(1500);
}

/** Training tab -> Demo source -> DEMO clip, and wait for real video metadata. */
async function openPosePreview(page) {
  await tab(page, 'training');
  await page.waitForTimeout(1200);
  await chooseByName(page, 'projectSelect', args.project);
  const demoBtn = await page.$('.upload-tab[data-source="current"]');
  if (demoBtn) { await demoBtn.click(); await page.waitForTimeout(400); }

  await page.selectOption('#trainingDemoSelect', DEMO);
  await page.waitForFunction(() => {
    const v = document.getElementById('previewVideo');
    return v && v.duration > 0 && v.videoWidth > 0;
  }, { timeout: 45000 });
  await page.waitForTimeout(900);

  await page.evaluate(() => {
    const cb = document.getElementById('poseAutoAnalyze');
    if (cb && !cb.checked) cb.click();
  });
  return page.evaluate(() => Number(document.getElementById('frameSlider').max));
}

/**
 * Seek to a frame and wait for THAT frame's inference to return.
 * Sleeping instead is flaky and ~3x slower - the analysis is debounced 150ms
 * then round-trips to /api/predict/landmarks.
 */
async function seekAndAnalyze(page, frameIdx) {
  const wait = page
    .waitForResponse((r) => r.url().includes('/api/predict/landmarks') && r.status() === 200, { timeout: 25000 })
    .catch(() => null);
  await page.evaluate((idx) => {
    const s = document.getElementById('frameSlider');
    s.value = String(idx);
    // Both events required: 'input' fires on drag, 'change' on release.
    s.dispatchEvent(new Event('input', { bubbles: true }));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  }, frameIdx);
  await wait;
  await page.waitForTimeout(280); // canvas paints the skeleton after the response
}

async function readPose(page, multi) {
  return page.evaluate((multi) => {
    const badge = document.getElementById('poseBadge');
    const visible = badge && getComputedStyle(badge).display !== 'none';
    const text = badge ? badge.innerText.replace(/\s+/g, ' ').trim() : '';
    const feats = document.getElementById('poseFeatures');
    if (multi) {
      const cards = feats ? [...feats.children] : [];
      const names = cards.map((c) => (c.textContent.match(/#\d+\s+(.+?)\s+\(\d+%\)/) || [])[1]).filter(Boolean);
      const confs = cards.map((c) => Number((c.textContent.match(/\((\d+)%\)/) || [])[1] || 0));
      return { persons: cards.length, names, avg: confs.length ? confs.reduce((a, b) => a + b) / confs.length : 0, text };
    }
    const noPerson = /no.?person/i.test(text);
    return {
      persons: visible && !noPerson ? 1 : 0,
      names: [],
      avg: Number((text.match(/(\d+)%/) || [])[1] || 0),
      text,
    };
  }, multi);
}

/**
 * The demo clips are portrait (720x1280) inside a wide dark box, so a naive
 * element screenshot is mostly black. Compute the real letterboxed content rect.
 */
async function contentClip(page) {
  return page.evaluate(() => {
    const v = document.getElementById('previewVideo');
    const r = v.getBoundingClientRect();
    const ar = v.videoWidth / v.videoHeight;
    let w, h;
    if (r.width / r.height > ar) { h = r.height; w = h * ar; } else { w = r.width; h = w / ar; }
    return {
      x: Math.round(r.x + (r.width - w) / 2),
      y: Math.round(r.y + (r.height - h) / 2),
      width: Math.round(w),
      height: Math.round(h),
    };
  });
}

const scrollPreviewIntoView = (page) =>
  page.evaluate(() => {
    const p = document.getElementById('posePreviewPanel');
    if (p) p.scrollIntoView({ block: 'center', behavior: 'instant' });
  });

// ---------- commands ----------
async function cmdShots(browser) {
  const page = await newPage(browser);
  await open(page);
  const saved = [];
  const snap = async (name, note) => {
    const f = path.join(OUT, `${appName}-${name}.png`);
    await maskTimestamps(page);
    await page.screenshot({ path: f });
    saved.push({ name, note, file: f });
    console.log(`  saved ${path.basename(f)} - ${note}`);
  };

  // Live Analysis, actually streaming (an idle dashboard screenshots as all zeros).
  await tab(page, 'live');
  await page.waitForTimeout(1000);
  await page.selectOption('#demoVideoSelect', DEMO);
  await page
    .waitForFunction(() => {
      const a = document.getElementById('kpiCurrentActivity');
      return a && a.textContent.trim() && a.textContent.trim() !== '--';
    }, { timeout: 60000 })
    .catch(() => console.log('  [warn] live KPIs never populated'));
  await page.waitForTimeout(18000); // let KPIs / timeline / events accumulate
  const live = await page.evaluate(() => ({
    activity: document.getElementById('kpiCurrentActivity')?.textContent.trim(),
    conf: document.getElementById('kpiConfidence')?.textContent.trim(),
    conn: document.getElementById('connectionLabel')?.textContent.trim(),
  }));
  await snap('live', `Live Analysis - ${live.conn}, ${live.activity} ${live.conf}`);

  // Pose Preview (lives on the Training tab).
  await openPosePreview(page);
  const maxFrame = await page.evaluate(() => Number(document.getElementById('frameSlider').max));
  await seekAndAnalyze(page, Math.floor(maxFrame * 0.3));
  const st = await readPose(page, app.multi);
  await scrollPreviewIntoView(page); // MediaPipe puts this panel below the fold
  await page.waitForTimeout(3500); // let the "Loaded N labels" toast fade out of the shot
  await snap('pose-preview', `Pose Preview - ${st.text} (${st.persons} person(s))`);

  await tab(page, 'events');
  await page.waitForTimeout(2500);
  await page.evaluate(() => window.scrollTo(0, 0));
  const rows = await page.evaluate(() => document.querySelectorAll('table tbody tr').length);
  await snap('events', `Events Log - ${rows} rows`);

  console.log(JSON.stringify({ app: appName, live, pose: st, saved: saved.length }, null, 2));
}

async function cmdPreview(browser) {
  const from = Number(args.from ?? 2);
  const step = Number(args.step ?? 4);
  const page = await newPage(browser, 1100);
  await open(page);
  const maxFrame = await openPosePreview(page);
  const to = Number(args.to ?? maxFrame - 2);
  await scrollPreviewIntoView(page);
  await page.waitForTimeout(600);

  const clip = await contentClip(page);
  console.log('  clip', JSON.stringify(clip), 'maxFrame', maxFrame);

  const kept = [];
  for (let f = from; f <= Math.min(to, maxFrame - 2); f += step) {
    await seekAndAnalyze(page, f);
    const st = await readPose(page, app.multi);
    // --keep-empty 1 keeps frames where nothing was detected: a comparison GIF
    // that silently drops misses would flatter the weaker engine.
    if (st.persons === 0 && !args['keep-empty']) { console.log(`  skip f=${f} (no person)`); continue; }
    const file = path.join(OUT, `f${String(kept.length).padStart(3, '0')}.png`);
    await page.screenshot({ path: file, clip });
    kept.push({ f, ...st });
    console.log(`  f=${f} persons=${st.persons} ${st.text} [${st.names.join(',')}]`);
  }
  fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify(kept, null, 2));
  console.log(`captured ${kept.length} frames -> ${OUT}`);
}

async function cmdAnnotate(browser) {
  const page = await newPage(browser);
  await open(page);
  await tab(page, 'annotate');
  await page.waitForTimeout(1500);

  // Saved projects store ABSOLUTE video_paths; any project created before the
  // repo moved cannot load frames. Always make a fresh one from the demo.
  await page.selectOption('#annotDemoSelect', { index: 1 });
  await page.waitForTimeout(400);
  await page.click('#btnNewProjectDemo');
  await page.waitForFunction(() => {
    const ws = document.getElementById('annotWorkspace');
    return ws && getComputedStyle(ws).display !== 'none';
  }, { timeout: 60000 });
  await page.waitForFunction(() => {
    const i = document.getElementById('annotFrameImg');
    return i && i.complete && i.naturalWidth > 0;
  }, { timeout: 60000 });

  await page.evaluate(() => {
    const s = document.getElementById('annotFrameSlider');
    s.value = String(Math.floor(Number(s.max) * 0.3));
    s.dispatchEvent(new Event('input', { bubbles: true }));
    s.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.waitForTimeout(3000);

  const det = page.waitForResponse((r) => /\/detect\//.test(r.url()) && r.status() === 200, { timeout: 60000 }).catch(() => null);
  await page.click('#btnAutoDetect');
  await det;
  await page.waitForTimeout(2500);

  const info = await page.evaluate(() => ({
    frame: document.getElementById('annotFrameInfo')?.textContent,
    keypoints: document.getElementById('annotKpList')?.children.length,
  }));
  const f = path.join(OUT, `${appName}-annotate.png`);
  await page.evaluate(() => window.scrollTo(0, 0));
  await maskTimestamps(page);
  await page.screenshot({ path: f });
  console.log(`  saved ${path.basename(f)} - ${info.keypoints} keypoints, ${info.frame}`);
}

async function cmdReid(browser) {
  const page = await newPage(browser, 1400);
  await open(page);
  await tab(page, 'training');
  await page.waitForTimeout(2500);
  await chooseByName(page, 'projectSelect', args.project);
  await chooseByName(page, 'collectionSelect', args.collection);
  await page.waitForFunction(() => {
    const w = document.getElementById('reidWorkers');
    return w && w.children.length > 0;
  }, { timeout: 30000 });

  // The gallery is a 200px scroll box; expand it so every identity fits one shot.
  await page.evaluate(() => {
    const g = document.getElementById('reidGallery');
    if (g) { g.style.maxHeight = 'none'; g.style.overflowY = 'visible'; }
    document.getElementById('reidPanel')?.scrollIntoView({ block: 'center', behavior: 'instant' });
  });
  await page.waitForTimeout(1200);
  await page
    .waitForFunction(() => {
      const imgs = document.querySelectorAll('#reidWorkers img');
      return imgs.length > 0 && [...imgs].every((i) => i.complete);
    }, { timeout: 30000 })
    .catch(() => console.log('  [warn] some thumbnails never completed'));

  const info = await page.evaluate(() => {
    const imgs = [...document.querySelectorAll('#reidWorkers img')];
    return {
      badge: document.getElementById('reidBadge')?.textContent.trim(),
      workers: document.getElementById('reidWorkers').children.length,
      thumbs: imgs.length,
      broken: imgs.filter((i) => !i.naturalWidth).length,
    };
  });
  const f = path.join(OUT, 'rtmpose-workers-reid.png');
  await maskTimestamps(page);
  await (await page.$('#reidPanel')).screenshot({ path: f });
  console.log(`  saved ${path.basename(f)} -`, JSON.stringify(info));
  if (info.broken > 0) console.log('  [warn] broken thumbnails - check /api/reid/thumb (see SKILL.md Gotchas)');
}

// ---------- main ----------
const browser = await chromium.launch({
  channel: 'chrome', // bundled Chromium may not match the cached build, and lacks H.264
  args: ['--autoplay-policy=no-user-gesture-required'],
});
try {
  if (cmd === 'shots') await cmdShots(browser);
  else if (cmd === 'preview') await cmdPreview(browser);
  else if (cmd === 'annotate') await cmdAnnotate(browser);
  else if (cmd === 'reid') await cmdReid(browser);
  else { console.error(`unknown command: ${cmd}`); process.exit(2); }
} finally {
  await browser.close();
}
