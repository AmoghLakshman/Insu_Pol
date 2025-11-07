# Streamlit Insurance Policy Prediction Dashboard

This repository contains a Streamlit app (`app.py`) to explore an insurance dataset, train three tree-based classifiers (Decision Tree, Random Forest, Gradient Boosting) using stratified 5-fold CV, visualize metrics (confusion matrices, ROC, feature importances), and predict on new uploaded datasets.

Files in this package (no folders):
- `app.py` - main Streamlit application
- `sample_insurance.csv` - sample dataset (from your uploaded Insurance.csv)
- `requirements.txt` - list of packages (no versions) to install on Streamlit Cloud
- `README.md` - this file

## How to use
1. Push these files to a GitHub repository (root of the repo).
2. On Streamlit Cloud, create a new app and point it to `app.py` in the repo.
3. Streamlit Cloud will install dependencies listed in `requirements.txt`.
4. Open the app, use the sample dataset or upload your own CSV. Train models on the "Run Models" tab, then use "Upload & Predict" to run predictions on new CSVs and download.

Notes:
- The app uses simple preprocessing (median/mode imputation and ordinal encoding). For best results adapt the preprocessing to your data.
- Keep column names in uploaded CSVs compatible with the sample (or similar) for best results.