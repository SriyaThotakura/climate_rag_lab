const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// ── 10 primary map HTML files across Folders 1–4 ──
const maps = [
  // Folder 1 — SegmentationCharts
  { id: 'exhibit_index',           file: '1.SegmentationCharts/exhibit/index.html',                       label: 'Exhibit — Evidence Corpus',       folder: 1 },
  { id: 'evidence_corpus_grid',    file: '1.SegmentationCharts/exhibition_prints/01_evidence_corpus_grid.html', label: 'Evidence Corpus Grid',       folder: 1 },

  // Folder 2 — Maps
  { id: 'trench_combined',         file: '2.Maps/TRENCH/index.html',                                     label: 'Combined CBX 3D Dashboard',       folder: 2 },
  { id: 'scroll_main',             file: '2.Maps/Scroll/index.html',                                     label: 'Scroll — Forensic Analysis',      folder: 2 },
  { id: 'community_atlas',         file: '2.Maps/CommunityAtlas/index.html',                             label: 'Community Atlas',                 folder: 2 },
  { id: 'idling_atlas',            file: '2.Maps/Final_GIS/WEB/index.html',                              label: 'Idling Atlas — GIS',              folder: 2 },
  { id: 'cbx_trench_visual',      file: '2.Maps/TRENCH/CBX_Trench_Visual.html',                         label: 'CBX Trench 3D Terrain',           folder: 2 },
  { id: 'trauma_map',              file: '2.Maps/TRENCH/trauma_map.html',                                label: 'Trauma / 311 Map',                folder: 2 },

  // Folder 4 — CBX Corpus & RAG
  { id: 'cv_map',                  file: '4.CBX_Corpus_and_RAG/cbx_corpus/CV_map.html',                  label: 'CV Segmentation Map',             folder: 4 },
  { id: 'accurate_terrain',        file: '4.CBX_Corpus_and_RAG/cbx_corpus/accurate_terrain_v2.html',     label: 'Accurate Terrain DEM',            folder: 4 },
];

const BASE = __dirname;
const OUT  = path.join(BASE, 'dist', 'thumbnails');

(async () => {
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-web-security', '--allow-file-access-from-files']
  });

  for (const m of maps) {
    const filePath = path.join(BASE, m.file);
    if (!fs.existsSync(filePath)) {
      console.log(`  SKIP (not found): ${m.file}`);
      // Create a dark placeholder image
      const page = await browser.newPage();
      await page.setViewport({ width: 800, height: 800 });
      await page.setContent(`
        <html><body style="margin:0;background:#0a0a0f;display:flex;align-items:center;justify-content:center;width:800px;height:800px">
          <div style="text-align:center;font-family:monospace;color:#444;font-size:14px">
            <div style="font-size:40px;margin-bottom:12px">&#9744;</div>
            ${m.label}<br><span style="color:#222;font-size:10px">${m.file}</span>
          </div>
        </body></html>`);
      await page.screenshot({ path: path.join(OUT, `${m.id}.png`), type: 'png' });
      await page.close();
      continue;
    }

    console.log(`  Capturing: ${m.id} — ${m.file}`);
    const page = await browser.newPage();
    await page.setViewport({ width: 800, height: 800 });

    try {
      const fileUrl = 'file:///' + filePath.replace(/\\/g, '/');
      await page.goto(fileUrl, { waitUntil: 'networkidle2', timeout: 15000 });
    } catch (e) {
      // networkidle2 can time out on Mapbox tiles; that's fine — the map is rendered
      console.log(`    (timeout on networkidle2 — capturing anyway)`);
    }

    // Wait 3s for map/D3 transitions to settle
    await new Promise(r => setTimeout(r, 3000));

    await page.screenshot({ path: path.join(OUT, `${m.id}.png`), type: 'png' });
    console.log(`    → saved ${m.id}.png`);
    await page.close();
  }

  await browser.close();
  console.log(`\nDone. ${maps.length} thumbnails saved to ${OUT}`);
})();
