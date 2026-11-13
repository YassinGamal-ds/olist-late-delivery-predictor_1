# Olist Late-Delivery Predictor — Notebook Edition

The project is split into six separate notebooks:

1. `01_understanding.ipynb` — inspect original tables, structure, missing values, and relationships.
2. `02_clean.ipynb` — clean, aggregate, join tables, and create the late label.
3. `03_eda.ipynb` — explore target balance, distributions, distance, category and timing patterns.
4. `04_feature_engineering.ipynb` — create purchase-time features and a leakage-safe modeling table.
5. `05_model.ipynb` — baseline, three classifiers, tuning, evaluation, error analysis, and saved artifacts.
6. `06_website.ipynb` — create the Streamlit app source and explain how to launch it.

## Dataset
Download the Olist dataset from:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Put the extracted CSV files in `data/raw/`.

## Install
```bash
pip install -r requirements.txt
```

Open Jupyter from this folder:
```bash
pip install notebook
jupyter notebook
```

Run notebooks in order. `02_clean` creates `data/processed/orders_clean.csv`; `04_feature_engineering` creates `orders_features.csv`; `05_model` saves the model and seller summary; `06_website` writes `website/app.py`.

Launch the website:
```bash
pip install streamlit
streamlit run website/app.py
```

## Method notes
- Target: `late = 1` if actual customer delivery date is after estimated delivery date.
- Only delivered orders with both dates are included.
- The latest 20% of orders are held out chronologically for final testing.
- Imputation, scaling, and encoding are fitted inside a scikit-learn pipeline.
- The app reports an estimate, not a guarantee.
- This is historical Brazilian marketplace data, not actual Noon/Jumia data.
