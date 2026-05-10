# CBX PROJECT HANDOFF — Sriya Thotakura
# Columbia GSAPP M.S. Computational Design Practices 2026
# Cross Bronx Expressway Environmental Justice Research

---

## WHO YOU ARE
- Sriya Thotakura, Columbia GSAPP '26
- M.S. Computational Design Practices
- Email: st3827@columbia.edu
- GitHub: github.com/SriyaThotakura
- Portfolio: sriyathotakura.com
- NYC-based, OPT start date: July 1 2026
- Graduating: May 20, 2026

---

## PROJECT OVERVIEW — ONE SENTENCE
A computational proof that the Cross Bronx Expressway trench geometry is the cause of South Bronx environmental injustice — using RAG corpus extraction, CV segmentation, ensemble ML, and parametric C# design response.

## PROJECT THESIS (4-phase pipeline)
RAG (qualitative) → Mapillary/CV (spatial) → CFD/Ensemble Modeling (predictive) → C# (generative)

---

## WORKING DIRECTORY
`C:\Users\ReiChiquita\Desktop\spring\CQ\climate_rag_lab\`

### Folder structure:
```
climate_rag_lab/
├── app.py                          # existing Streamlit RAG UI (DO NOT TOUCH)
├── rag.py                          # existing RAG pipeline (DO NOT TOUCH)
├── evaluate.py                     # existing evaluation (DO NOT TOUCH)
├── prompts.py                      # existing prompts (DO NOT TOUCH)
├── ensemble_model.py               # NEW — built this session
├── cbx_ingest.py                   # NEW — built this session
├── cbx_extract.py                  # NEW — built this session
├── cbx_api_server.py               # NEW — built this session (FastAPI port 8765)
├── cbx_corpus/                     # 139 documents indexed
│   ├── 311_complaints_real.csv     # 199,980 rows, 12 zips, 2021-2024
│   ├── segmentation_metrics.csv    # 99 frames, real SegFormer output
│   ├── Secondary_Intervention_Residential.geojson     # 153 polygons
│   ├── Vulnerability_facilities.geojson               # 3,776 points
│   ├── Intervention_Zones_3D.geojson                  # 57 polygons (main units)
│   ├── Secondary_Intervention_Residential_TRIVARIATE.geojson  # 19 priority zones
│   ├── [101 PDFs — CB minutes, health data, academic papers]
│   ├── [18 OCR txt files from scanned PDFs]
│   ├── [9 CSVs — asthma, health data]
│   └── streetscape_images/         # 98 Mapillary frames already segmented
├── cbx_scripts/                    # Python data collection scripts moved here
│   └── CV_CBX.ipynb               # SegFormer notebook
├── chroma_cbx/                     # ChromaDB persistence for cbx_trauma collection
├── outputs/
│   ├── trauma_points.geojson       # 133 RAG features (noise/health/air/displacement)
│   ├── trauma_points_311.geojson   # 2,991 direct 311 features (spatially sampled)
│   ├── cv_frames.geojson           # 99 CV segmentation points
│   ├── ensemble_predictions.geojson # 57 zone vulnerability predictions
│   ├── feature_importance.csv      # 22 features ranked
│   ├── model_summary.txt           # Human-readable ensemble summary
│   ├── rf_decision_tree.png        # RF decision tree visualization
│   ├── feature_correlation_heatmap.png  # Feature correlation matrix
│   ├── trauma_map.html             # Main exhibition map (Screen 02)
│   ├── cv_map_encoded.html         # CV segmentation map (Folium)
│   └── segmentation_metrics.csv   # copy of CV data
└── venv/                           # Python virtual environment
```

---

## PHASE STATUS

### ✅ PHASE 1 — FORENSIC INPUT (COMPLETE)

**RAG Corpus — 139 documents indexed into ChromaDB collection "cbx_trauma"**

Sources collected:
- 311 complaints: 199,980 real NYC complaints, 12 zip codes (10451-10462, 10473-10474), 2021-2024
- Community Board 4 & 5 minutes: 2019-2025 (18 OCR txt files)
- NYC DOHMH asthma data: ED visits adults/children, hospitalizations, NTA + UHF42, 2012-2023
- EPA EJScreen: CBE corridor polygon (fetched via EJAM API as JSON + OCR)
- South Bronx Rising (Jonnes) — if downloaded from Columbia library
- Marshall Berman "All That Is Solid" CBX chapter — if downloaded
- Streetsblog/Gothamist articles as txt files
- Historical context document
- ejscreen_cbx_corridor.txt, ejscreen_cbx_data.json

**Key data files:**
- `311_complaints_real.csv`: 199,980 rows, lon range -73.931 to -73.826 (full corridor)
- `segmentation_metrics.csv`: 99 frames, columns: frame_id, lat, lon, sky_ratio, building_ratio, vegetation_ratio
  - All frames from lon -73.930 to -73.926 (Jerome Ave area only — narrow coverage)
  - sky_ratio mean: 0.402, range 0.000-0.987
  - building_ratio mean: 0.048, range 0.000-0.220

**RAG extraction results:**
- Total features: 133
- noise: 40, air_quality: 19, health: 35, displacement: 39
- geo_confidence: exact 28 (311 data), bronx_anchor 38, centroid_fallback 67
- Bbox verified: lat 40.8282-40.8548, lon -73.9298 to -73.8939

**311 direct extract:**
- 2,991 spatially sampled features (from 199,980)
- All exact coordinates
- Covers full corridor lon -73.931 to -73.826

---

### ✅ PHASE 2 — ENSEMBLE MODEL (COMPLETE)

**Model: GIS + RAG + Facilities + Distance → Composite Vulnerability Score**

Target variable (composite, NOT synthetic PM2.5):
```
vulnerability_score = 
  (Asthma_I_R_mean / 355.8) * 0.50 +
  (CVI_mean_normalised) * 0.30 +
  (trauma_max_intensity) * 0.20
```

**Feature matrix: 57 zones × 22 features**

Feature importance (RF):
1. Asthma_I_R_mean: 0.6163 (GIS)
2. CVI_mean: 0.2473 (GIS)
3. trauma_max_intensity: 0.1008 (RAG) ← #3 is notable
4. trivariate_canopy: 0.0066 (GIS)
5. trauma_exact_count: 0.0062 (RAG)
6-10: trauma_noise_density, nearest_hospital_m, dist_to_cbe_m, facilities_500m, nearest_school_m

**Model performance:**
- Random Forest: CV R² 0.566 ± 0.222, Test R² 0.908
- XGBoost: CV R² 0.552 ± 0.509, Test R² 0.972
- Gradient Boosting: CV R² 0.781 ± 0.086, Test R² 0.954
- Weighted Ensemble: Test R² 0.974 (GB weight 0.411 dominant)

**IMPORTANT NOTE on R²:** High R² is partly due to data leakage — target variable uses same features as inputs. State as: "ensemble confirmed zone prioritisation with high internal consistency" — do NOT claim 97% predictive accuracy.

**Top 5 priority intervention zones:**
- fid=41: Asthma 355.8, vulnerability 0.7447 (HIGHEST)
- fid=6: Asthma 355.8, CVI -0.08, vulnerability 0.7231
- fid=17: Asthma 287.9, CVI 1.30, vulnerability 0.6662
- fid=52: Asthma 266.2, CVI 2.25, vulnerability 0.6623
- fid=29: Asthma 199.5, vulnerability 0.5250

**Known limitations to state:**
1. 45/57 zones have zero trauma overlap (eastern corridor corpus gap)
2. 20/57 zones have null CVI filled with median 0.69
3. CV features show 0.0000 importance (only 3 zones got real CV, 54 median-filled)
4. Top 5 priority zones all show trauma_max=0.00 (eastern corpus gap)

**Novel finding check:** trauma total = 10.7% — below 20% threshold. trauma_max_intensity alone at #3 (9.81%) is notable given 79% of zones had zero trauma data.

---

### 🔄 PHASE 2b — CV SEGMENTATION (PARTIAL)

**What you have:**
- 99 real SegFormer frames from Jerome Ave (narrow western coverage)
- cv_frames.geojson: 99 points with sky_ratio, building_ratio, vegetation_ratio, trench_enclosure, canyon_ar
- cv_map_encoded.html: Folium map with colour-encoded CV data

**What is missing:**
- Full-corridor Mapillary coverage (eastern CBE = no data)
- Trench-floor imagery (Mapillary has NO coverage inside the CBE — it is a controlled-access highway)

**The correct CV approach for trench geometry:**
- Use overpass fence-line Google Street View images (fetch_gsv_overpass.py — written but not yet run — needs GSV API key)
- Use geometric canyon analysis (canyon_ar = (trench_depth + building_height) / trench_width)
  - Jerome-Webster section: AR = 0.64, SVF = 0.61
  - Webster-Southern: AR = 0.59
  - Southern-Bronx River: AR = 0.48
  - Bronx River-Soundview: AR = 0.41

**Code written but not yet run:**
- `fetch_gsv_overpass.py` — fetches Google Street View from 5 overpass positions
- `segment_trench_frames.py` — runs SegFormer on GSV frames
- `geometric_canyon_analysis.py` — computes AR from physical dimensions, produces canyon_geometry.geojson

---

### 🔄 PHASE 3 — CFD (NOT YET DONE)

**Decision:** Do NOT use Butterfly/OpenFOAM before exhibition (installation too risky, Rhino 8 compatibility not tested).

**Use instead — cfd_simplified.py (written, not yet run):**
- Python simplified wind flow model
- 2D cross-section at Jerome Ave overpass
- Gaussian plume + recirculation approximation
- Outputs: `cfd_cross_section.png` + `stagnation_zones.geojson`
- Run time: 2 minutes

**After exhibition:** Install Butterfly + OpenFOAM for proper 3D CFD validation.

---

### 🔄 PHASE 4 — C# GENERATIVE TRENCH CAPS (NOT YET DONE)

**Concept:**
- TrenchCapComponent.gha — adapt existing ShingleFacade.gha plugin
- Inputs: windScore (from CFD), traumaDensity (from RAG), capType (0/1/2)
- windScore drives porosity (higher wind = more open)
- traumaDensity drives access point count (more trauma = more community entry)
- 3 typologies: solid barrier, porous lattice, green armature
- Export as GeoJSON for Mapbox intervention simulator

**Existing C# foundation:**
- ShingleFacade.gha is compiled and working
- Located: C:\Users\ReiChiquita\Desktop\ShinglesFacade\vs\ShingleFacade\
- Installed: %AppData%\Grasshopper\Libraries\ShingleFacade.gha
- SHGC = 0.492 (18% reduction vs 0.60 baseline)
- Components: UVSamplerComponent, CellSelectorComponent, HingedPanelComponent, ColorPreviewComponent, LouverPanelComponent

---

## MAPS / VISUALIZATIONS BUILT

### trauma_map.html (MAIN EXHIBITION MAP — Screen 02)
File: `./outputs/trauma_map.html`
Mapbox GL JS 2.15.0, dark-v11 style
Centre: [-73.889, 40.840], zoom 13, pitch 45

**Sources (5):**
- trauma: trauma_points.geojson (133 RAG features)
- trauma-311: trauma_points_311.geojson (2,991 direct 311)
- cbe-line: 55 real coordinates from Freight_Routes.geojson
- bruckner-blvd: 52 points dashed cyan
- bruckner-expy: 61 points solid subtle cyan

**Layers (9):**
- cbe-glow, cbe-centreline (red dashed, real CBE path)
- bruckner-expy-line, bruckner-blvd-line (cyan)
- direct-311-halo, direct-311-dots (311 data)
- trauma-halo, trauma-fallback, trauma-exact (RAG data)

**Ensemble layer — TO ADD:**
- ensemble_predictions.geojson as choropleth
- Code written (in previous conversation), not yet applied to file

**TOKEN:** Replace 'YOUR_MAPBOX_TOKEN' with real token

**Category colours:**
- noise: #E8A23A
- air_quality: #C8F04A (lime)
- health: #E84A3A
- displacement: #4A9EE8

**Palette:**
- bg: #0C0D0F, surface: #131519, border: #252830
- text-1: #E8E4DC, text-2: #8A8880, accent: #C8F04A

---

## EXHIBITION PLAN (3 SCREENS)

### PHYSICAL TABLE (in front of Screen 01)
Printed documents:
1. CB4/5 minutes — noise complaint highlighted, label: "action taken: none"
2. EJScreen report — 99th percentile traffic circled, label: "action taken: none"
3. DOHMH table — Mott Haven 320.2/10k circled
4. 311 printout — Noise Vehicle, Burnside Ave
5. Interview transcript — 3 sentences highlighted by trauma colour

### SCREEN 01 — EVIDENCE (left, autonomous loop)
1. Interview video — subtitles only, looping 2-3 min clip (BUILD: 1.5 hrs)
2. GIS pre-analysis maps — 8s crossfade slideshow (BUILD: 45 min)
3. RAG corpus quotes — slow text scroll, 4-5 quotes from CB minutes (BUILD: 30 min)
4. ML truck speed heatmap — static image (ALREADY DONE)

### SCREEN 02 — ANALYSIS (centre, visitor-operated, you demo here)
1. Trauma atlas — trauma_map.html with Mapbox (BUILT — add token + ensemble layer)
2. CV segmentation evidence matrix — 3×3 grid (BUILD: 2 hrs Python matplotlib)
3. CFD before section — cfd_cross_section.png left panel (BUILD: 2 min run)
4. ~~RAG pipeline diagram~~ CUT

### SCREEN 03 — RESPONSE (right, visitor-operated)
1. Intervention simulator — Mapbox + scenario toggle + trauma density slider (BUILD: 2 hrs)
2. C# trench cap — 3 typologies, Rhino pen mode export (BUILD: 1 hr)
3. CFD after section — same cross-section with cap installed (BUILD: 1 hr composite)
4. Diffusion biophilic render — one hero image only (BUILD: 1 hr ComfyUI)
5. ~~SHGC metrics poster~~ MOVED TO PRESENTATION

---

## INTERVIEW ASSET
- Youth minister interview video
- Status: recorded, transcript pending
- Plan: transcribe → add to cbx_corpus → re-run cbx_ingest.py --reset
- Exhibition: loop on Screen 01 with burned-in subtitles, no audio
- Consent needed for public display of image/voice

---

## REMAINING WORK (priority order)

### MUST DO BEFORE EXHIBITION:
1. Run `cfd_simplified.py` → get cfd_cross_section.png + stagnation_zones.geojson
2. Add ensemble choropleth layer to trauma_map.html
3. Add Mapbox token to trauma_map.html
4. Build Screen 01 HTML (video loop + map slideshow + quote scroll)
5. Build intervention simulator for Screen 03
6. Transcribe interview → add to corpus → re-run RAG

### SHOULD DO:
7. Run geometric_canyon_analysis.py → canyon_geometry.geojson
8. Add canyon_geometry.geojson as trench layer to trauma_map.html
9. Run rag_cv_correlation.py → prove spatial overlap between trauma and enclosure
10. Build 3×3 evidence matrix (matplotlib)
11. C# TrenchCapComponent.gha for trench caps

### AFTER EXHIBITION:
12. Butterfly/OpenFOAM full CFD
13. Full-corridor Mapillary re-run with GSV overpass images
14. Stable Diffusion biophilic renders (ComfyUI + ControlNet)
15. Re-run ensemble with CV features once full-corridor segmentation done

---

## KEY CODE COMMANDS

```bash
# Activate venv
cd C:\Users\ReiChiquita\Desktop\spring\CQ\climate_rag_lab
./venv/Scripts/activate

# Re-index corpus (after adding new documents)
python cbx_ingest.py --reset

# Re-extract trauma points
python cbx_extract.py

# Start RAG UI server
python cbx_api_server.py
# Then open rag_ui.html in browser

# Run ensemble model
python ensemble_model.py
python ensemble_model.py --with-cv

# Run CFD (when ready)
python cfd_simplified.py

# Pull more 311 data
python cbx_scripts/pull_311_eastern.py
```

---

## REFERENCE PAPERS (grounding methodology)
1. SAGEPUB 10.1177/23998083221083677 — urban morphology + ML predicts PM2.5, RF best performer, canyon AR key feature, XGBoost 91-95% accuracy
2. Frontiers feart.2021.659296 — ensemble stacking for spatial susceptibility mapping, mean ensemble outperforms individual models

---

## GSAPP EVALUATION FRAMEWORK
Three axes GSAPP CDP evaluates on:
1. Methodological clarity (Methods as Practices colloquium)
2. Spatial argument (Explore Explain Propose colloquium)
3. Forward-looking implementation (Design in Action colloquium)

Your project maps: RAG → Methods, Ensemble → Explore/Explain, C# Caps → Design in Action

---

## OTHER ACTIVE PROJECTS (portfolio)
1. CommunityAtlas: sriyathotakura.github.io/CommunityAtlas — STURLA + Random Forest, PM2.5, 2,392 cells, dist_to_cbe = 100% importance
2. GIS Atlas: sriyathotakura.github.io/Final_GIS
3. ABM/Feral Governance: sriyathotakura.github.io/Propose
4. Web BIM: sriyathotakura.github.io/web_bim
5. Parametric Zoning: medium.com/generative-design-course/parametric-zoning-tool
6. Memory Gate UE5: virtualarchitecture.org/2026/03/09/final-review-memory-gate
7. Speculative Planetary: sriyathotakura.github.io/colloquium-i
8. Adaptive Facade C# — ShingleFacade.gha, SHGC 0.492
9. Wearable sensor — ESP32C + GPS, in progress

---

## PORTFOLIO WEBSITE
- URL: sriyathotakura.com
- Stack: vanilla HTML/CSS/JS
- Structure: 01 Sense / 02 Simulate / 03 Build / 04 Think
- Pipeline UI: causal phase model with "carries:" connectors between phases
- PROJECTS data object: defined in projects.js
- Status: Phase 1 shell built, Phase 2 project data wiring in progress

---

## CONTEXT FOR NEXT CONVERSATION
When starting a new conversation paste this entire document and say:
"This is my project handoff. I am Sriya Thotakura, Columbia GSAPP '26.
Continue where we left off. My next task is: [state what you want to do next]"

The most likely next tasks:
- "Run cfd_simplified.py and show me the output"
- "Add ensemble choropleth layer to trauma_map.html"  
- "Build Screen 01 autonomous loop HTML"
- "Build intervention simulator for Screen 03"
- "Transcribe interview and add to RAG corpus"
