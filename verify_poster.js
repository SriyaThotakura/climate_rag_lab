const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-web-security', '--allow-file-access-from-files']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  const filePath = path.join(__dirname, 'Radial_Poster_Index.html');
  const fileUrl = 'file:///' + filePath.replace(/\\/g, '/');

  try {
    await page.goto(fileUrl, { waitUntil: 'networkidle2', timeout: 15000 });
  } catch(e) {
    console.log('(timeout on networkidle2 — capturing anyway)');
  }

  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: path.join(__dirname, 'dist', 'poster_preview.png'), type: 'png', fullPage: false });
  console.log('Preview saved to dist/poster_preview.png');

  await browser.close();
})();
