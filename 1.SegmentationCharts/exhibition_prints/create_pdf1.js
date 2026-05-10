const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function createLabloidPDF1() {
    const browser = await puppeteer.launch({
        headless: true,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--single-process',
            '--disable-gpu'
        ]
    });

    try {
        const page = await browser.newPage();
        
        // Set viewport to match the print dimensions
        await page.setViewport({
            width: 3456,  // 36 inches at 96 DPI
            height: 4608, // 48 inches at 96 DPI
            deviceScaleFactor: 1
        });

        // Load the HTML file
        const htmlPath = path.resolve(__dirname, '01_evidence_corpus_grid.html');
        const htmlContent = fs.readFileSync(htmlPath, 'utf8');
        
        await page.setContent(htmlContent, {
            waitUntil: ['networkidle0', 'domcontentloaded']
        });

        // Wait for fonts to load
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Generate PDF with print settings
        const pdfBuffer = await page.pdf({
            path: 'labloid_evidence_corpus_part1.pdf',
            format: undefined, // We'll use custom size
            width: '36in',
            height: '48in',
            margin: {
                top: '0.5in',
                right: '0.5in',
                bottom: '0.5in',
                left: '0.5in'
            },
            printBackground: true,
            preferCSSPageSize: false,
            scale: 1
        });

        console.log('First labloid PDF created successfully: labloid_evidence_corpus_part1.pdf');
        
    } catch (error) {
        console.error('Error creating PDF:', error);
    } finally {
        await browser.close();
    }
}

createLabloidPDF1();
