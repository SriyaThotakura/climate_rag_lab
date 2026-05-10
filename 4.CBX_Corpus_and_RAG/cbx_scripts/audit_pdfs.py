import fitz  # PyMuPDF
import os
from pathlib import Path

corpus_dir = Path('./cbx_corpus')
pdfs = list(corpus_dir.glob('*.pdf'))

text_pdfs = []
scanned_pdfs = []
empty_pdfs = []

for pdf_path in sorted(pdfs):
    try:
        doc = fitz.open(pdf_path)
        total_chars = sum(
            len(doc[i].get_text()) 
            for i in range(min(3, len(doc)))
        )
        if total_chars > 500:
            text_pdfs.append((pdf_path.name, 
                            total_chars, len(doc)))
        elif total_chars > 0:
            scanned_pdfs.append((pdf_path.name, 
                               total_chars, len(doc)))
        else:
            empty_pdfs.append(pdf_path.name)
        doc.close()
    except Exception as e:
        empty_pdfs.append(f"{pdf_path.name} — ERROR: {e}")

print(f"TEXT-EXTRACTABLE ({len(text_pdfs)}):")
for name, chars, pages in text_pdfs[:10]:
    print(f"  ✅ {name[:50]} — {pages}pp, {chars:,} chars")
if len(text_pdfs) > 10:
    print(f"  ... and {len(text_pdfs)-10} more")

print(f"\nSCANNED/LOW TEXT ({len(scanned_pdfs)}):")
for name, chars, pages in scanned_pdfs:
    print(f"  ⚠️  {name[:50]} — {chars} chars only")

print(f"\nEMPTY/FAILED ({len(empty_pdfs)}):")
for name in empty_pdfs:
    print(f"  ❌ {name[:60]}")

print(f"\nSUMMARY:")
print(f"  Ready for RAG:     {len(text_pdfs)} PDFs")
print(f"  Need OCR:          {len(scanned_pdfs)} PDFs")  
print(f"  Failed/empty:      {len(empty_pdfs)} PDFs")
print(f"  CSVs:              9")
print(f"  TXTs:              9")