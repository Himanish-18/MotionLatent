# MotionLatent dashboard — install into your existing repository

This is a React + Vite dashboard starter wired to the real Phase 5 benchmark artifacts.

## 1. Put this folder in your repository

Extract this ZIP into your existing MotionLatent repository root. It creates a `frontend/` folder.

Expected layout:

```text
MotionLatent/
├── reports/
│   ├── phase5_comparison.csv
│   └── phase5_summary.json
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── src/
    └── public/
        └── data/
```

## 2. Copy the real benchmark data

Open a terminal at the MotionLatent repository root (not inside frontend) and run in PowerShell:

```powershell
New-Item -ItemType Directory -Force frontend\public\data
Copy-Item reports\phase5_comparison.csv frontend\public\data\
Copy-Item reports\phase5_summary.json frontend\public\data\
```

These files are read at runtime by the React app. Do not rename their headers or keys.

## 3. Install and run

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL Vite prints, usually `http://localhost:5173`.

## 4. Build check

```powershell
npm run build
```

## 5. Git workflow

Make sure you are on your own `dashboard` branch before committing:

```bash
git branch --show-current
git status
git add frontend
git commit -m "Add MotionLatent research dashboard"
git push origin dashboard
```

Then open a Pull Request from your fork's `dashboard` branch to the original repository's `main` branch.

## Data and scientific notes

- Primary chart metric: `Bal Acc (8-sub Macro)` from the Phase 5 CSV.
- The raw baseline is displayed separately from reduced-dimension model curves.
- `phase5_summary.json` supplies the separate MVD deployment summary for PCA-32 C99 and DSC-AE-32 INT8.
- CSV DSC-AE-32 is the FP32 Phase 4 row. It is not the 10 KB INT8 deployment artifact.
- CSV `Latency (ms/win)` and summary `inference_latency_ms` are different execution measurements. The dashboard labels them separately.
- Missing model/dimension combinations are shown as unavailable; the app does not interpolate results.
