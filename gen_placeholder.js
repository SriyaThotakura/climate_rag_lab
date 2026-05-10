const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 800, height: 800 });
  await page.setContent(`
    <html><body style="margin:0;background:#0a0a0f;display:flex;align-items:center;justify-content:center;width:800px;height:800px">
      <div style="text-align:center;font-family:monospace">
        <div style="font-size:80px;margin-bottom:16px;color:#ff007f">&#916;</div>
        <div style="color:#ff007f;font-size:16px;letter-spacing:2px;text-transform:uppercase">CV vs Model Delta</div>
        <div style="color:#333;font-size:11px;margin-top:8px">Observed SVF − Predicted SVF</div>
        <div style="color:#222;font-size:10px;margin-top:4px">28 ensemble zones · signed residual</div>
      </div>
    </body></html>`);
  await page.screenshot({ path: path.join(__dirname, 'dist', 'thumbnails', 'cv_model_delta.png'), type: 'png' });
  console.log('Saved cv_model_delta.png');
  await browser.close();
})();
