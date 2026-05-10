import fitz
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import os

# Set tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# Poppler is required by pdf2image on Windows
# Download from: https://github.com/oschwartz10612/
# poppler-windows/releases
# Extract and set path below:
POPPLER_PATH = r'C:\Users\ReiChiquita\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin'

FAILED_PDFS = [
    '20250529163737.pdf',
    '20251001092329.pdf',
    'BXCB4-General-Board-Meeting-agenda-3242026.pdf',
    'BXCB4-General-Board-Meeting-Agenda-with-Public-Hearing-Notic.pdf',
    'December-2025-General-Board-Meeting-minutes.pdf',
    'EJScreen.pdf',
    'General Board Meeting Minutes - April 24, 2019.pdf',
    'General Board Meeting Minutes - December 11, 2019.pdf',
    'General Board Meeting Minutes - February 27, 2019.pdf',
    'General Board Meeting Minutes - January 23rd, 2019.pdf',
    'General Board Meeting Minutes - June 26, 2019.pdf',
    'General Board Meeting Minutes - March 27, 2019.pdf',
    'General Board Meeting Minutes - May 22, 2019.pdf',
    'General Board Meeting Minutes - November 13, 2019.pdf',
    'General Board Meeting Minutes - October 23rd, 2019.pdf',
    'General Board Meeting Minutes - September 25, 2019.pdf',
    'General Board Minutes - January 22, 2020.pdf',
    'May-2025-General-Board-meeting-minutes.pdf',
    'minutes-20220614.pdf',
]

corpus_dir  = Path('./cbx_corpus')
output_dir  = Path('./cbx_corpus')

success = []
failed  = []

for pdf_name in FAILED_PDFS:
    pdf_path = corpus_dir / pdf_name
    if not pdf_path.exists():
        print(f"❌ Not found: {pdf_name}")
        failed.append(pdf_name)
        continue

    print(f"OCR processing: {pdf_name[:50]}...")
    try:
        # Convert PDF pages to images
        pages = convert_from_path(
            str(pdf_path),
            dpi=200,
            poppler_path=POPPLER_PATH
        )

        full_text = []
        for i, page_img in enumerate(pages):
            text = pytesseract.image_to_string(
                page_img,
                lang='eng',
                config='--psm 6'
            )
            if text.strip():
                full_text.append(
                    f"--- Page {i+1} ---\n{text}"
                )

        if full_text:
            # Save as .txt alongside the PDF
            out_name = pdf_path.stem + '_ocr.txt'
            out_path = output_dir / out_name
            with open(out_path, 'w', 
                      encoding='utf-8') as f:
                f.write(f"Source: {pdf_name}\n")
                f.write("="*50 + "\n\n")
                f.write("\n\n".join(full_text))

            char_count = sum(len(t) for t in full_text)
            print(f"  ✅ {len(pages)} pages, "
                  f"{char_count:,} chars → {out_name}")
            success.append(out_name)
        else:
            print(f"  ⚠️  No text extracted from "
                  f"{pdf_name}")
            failed.append(pdf_name)

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        failed.append(pdf_name)

print(f"\n{'='*50}")
print(f"✅ OCR complete: {len(success)} files")
print(f"❌ Still failed: {len(failed)} files")
if failed:
    print("Still failed:")
    for f in failed:
        print(f"  {f}")
print(f"\nNew .txt files ready for RAG ingestion.")