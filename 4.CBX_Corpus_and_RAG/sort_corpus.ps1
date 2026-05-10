$base = 'C:\Users\ReiChiquita\Desktop\spring\CQ\climate_rag_lab\4.CBX_Corpus_and_RAG\cbx_corpus'

# 1. CB Minutes — PDFs with board/meeting/minutes keywords (exclude academic)
Get-ChildItem $base -File -Filter '*.pdf' |
  Where-Object { $_.Name -match '(board|minutes|meeting|general|Public-Hearing|cb4|cb5)' -and $_.Name -notmatch '(AJPH|Jonnes)' } |
  Move-Item -Destination "$base\cb_minutes" -Force
Write-Host "Moved CB minutes PDFs"

# 2. CB Minutes — OCR text files that are meeting minutes
Get-ChildItem $base -File -Filter '*_ocr.txt' |
  Where-Object { $_.Name -match '(Board|Minutes|meeting|General)' } |
  Move-Item -Destination "$base\cb_minutes" -Force
Write-Host "Moved CB minutes OCR"

# 3. Health data — NYC EH Data Portal CSVs
Get-ChildItem $base -File -Filter '*.csv' |
  Where-Object { $_.Name -match '(Asthma|NYC EH)' } |
  Move-Item -Destination "$base\health_data" -Force
Write-Host "Moved health data"

# 4. 311 Complaints
Get-ChildItem $base -File -Filter '311_*' |
  Move-Item -Destination "$base\complaints" -Force
Write-Host "Moved 311 complaints"

# 5. Academic — Jill Jonnes + AJPH
Get-ChildItem $base -File |
  Where-Object { $_.Name -match '(Jonnes|AJPH)' } |
  Move-Item -Destination "$base\academic" -Force
Write-Host "Moved academic papers"

# 6. Journalism — Streetsblog + Gothamist
Get-ChildItem $base -File |
  Where-Object { $_.Name -match '(streetsblog|gothamist)' } |
  Move-Item -Destination "$base\journalism" -Force
Write-Host "Moved journalism"

# 7. EJScreen
Get-ChildItem $base -File |
  Where-Object { $_.Name -match '(ejscreen|EJScreen)' } |
  Move-Item -Destination "$base\ejscreen" -Force
Write-Host "Moved EJScreen data"

# 8. CV / Segmentation data
Get-ChildItem $base -File -Filter '*.csv' |
  Where-Object { $_.Name -match '(segmentation|Segmentation)' } |
  Move-Item -Destination "$base\cv_data" -Force
Write-Host "Moved CV data"

# 9. Remaining unclassified OCR files → cb_minutes (likely scanned board docs)
Get-ChildItem $base -File -Filter '*_ocr.txt' |
  Move-Item -Destination "$base\cb_minutes" -Force
Write-Host "Moved remaining OCR"

# Show results
Write-Host "`n=== CORPUS SORTED ==="
Get-ChildItem $base -Directory | ForEach-Object {
  $count = (Get-ChildItem $_.FullName -File).Count
  Write-Host "$($_.Name): $count files"
}
$remaining = (Get-ChildItem $base -File).Count
Write-Host "root (unsorted): $remaining files"
