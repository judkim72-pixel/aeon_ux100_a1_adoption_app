# AEON UX100 · A-1 · AI Adoption App

## Run (local)
```bash
pip install -r requirements.txt
streamlit run apps/aeon_ux100_a1_adoption_app.py
```

## Repo structure (suggested)
```
/ (repo root)
├─ apps/
│  └─ aeon_ux100_a1_adoption_app.py
├─ data/  (Excel dataset: UX_Comparison_100_ALL_v2_withProxyIndicators.xlsx)
├─ docs/  (slides & notes)
├─ requirements.txt
└─ README.md
```

## Notes
- The app expects an Excel file (sheet name: **Data**) that contains **Company** and **AI Adoption Index (0–5)**.
- You can upload the file from the UI or type a server path.