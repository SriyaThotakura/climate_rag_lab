const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-web-security', '--allow-file-access-from-files']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 2560, height: 1440, deviceScaleFactor: 2 });

  const filePath = path.join(__dirname, 'Radial_Poster_Index.html');
  const fileUrl = 'file:///' + filePath.replace(/\\/g, '/');

  try {
    await page.goto(fileUrl, { waitUntil: 'networkidle2', timeout: 15000 });
  } catch(e) {}

  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: path.join(__dirname, 'dist', 'poster_detail.png'), type: 'png', fullPage: false });
  console.log('Detail saved to dist/poster_detail.png');

  await browser.close();
})();
