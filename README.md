# Seoul Bike Sharing Demand Regression

A complete, beginner-friendly regression project that predicts the number of public bikes rented in Seoul each hour. It includes a real-world dataset, Excel EDA, a reproducible Python solution, model comparison, and feature-importance analysis.

## Why this dataset?

This is a good portfolio dataset because the target is intuitive, the data combines weather and time variables, and the problem is genuinely a regression task. It is less overused than house-price or salary datasets but remains easy to explain to a reviewer. The dataset contains 8,760 hourly records and has no missing values according to the official UCI documentation.

## Project question

> Given the hour, weather, season, holiday status, and system operating status, how many bikes will be rented?

## Repository structure

- `data/raw/SeoulBikeData.csv` — original UCI file, preserved for traceability.
- `data/processed/seoul_bike_clean.csv` — cleaned data with calendar and cyclic-hour features.
- `data/processed/test_predictions.csv` — held-out predictions from the best model selected by RMSE.
- `reports/seoul_bike_eda.xlsx` — Excel-first EDA workbook with formulas, summaries, charts, and model results.
- `reports/model_metrics.csv` — model comparison results.
- `reports/feature_importance.csv` — permutation importance on the held-out test set.
- `notebooks/seoul_bike_regression.ipynb` — notebook version of the workflow.
- `src/solution.py` — command-line reproducible solution.

## Excel EDA guide

Open `reports/seoul_bike_eda.xlsx` and review the sheets in this order: `README`, `Data_Dictionary`, `Summary`, `Numeric_Summary`, `Category_Summary`, `Hourly_Analysis`, `Monthly_Analysis`, `Model_Results`, and `Feature_Importance`. In Excel, press **Formulas → Calculate Now** if formula values do not refresh automatically.

Recommended EDA questions:

1. Which hours have the highest average demand?
2. Does demand differ by season, holiday, and functioning day?
3. Which weather variables have the strongest relationship with demand?
4. Are there unusual outliers during extreme rainfall or snowfall?

## Modeling choices

The solution uses an 80/20 chronological split. The final 20% is treated as future-like test data, which is more realistic than randomly shuffling hourly observations. It compares Linear Regression, Gradient Boosting, and Random Forest. The primary selection metric is RMSE; MAE and R² are reported alongside it.

## Run locally

```bash
pip install -r requirements.txt
python src/solution.py
```

To run the notebook:

```bash
jupyter notebook notebooks/seoul_bike_regression.ipynb
```

## Dataset attribution

Seoul Bike Sharing Demand. (2020). UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5F62R. Licensed under CC BY 4.0.

The raw file is included for educational and reproducibility purposes. Keep this attribution when publishing the repository.

## Deployment app

Run the Streamlit app from the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app provides an interactive prediction form, model comparison, dataset audit indicators, and simple EDA charts. It trains the selected regularized pipeline on all available cleaned data for prediction while displaying evaluation metrics from the chronological holdout.

## Full workflow

Run `python src/full_workflow.py` to regenerate the audit file, EDA CSV summaries, and the baseline/Ridge/Lasso/ElasticNet comparison.
