# SPEI-3 forecasting pipeline audit

Audit scope: `notebooks/02_silver_transformation.ipynb`,
`notebooks/03_gold_feature_engineering.ipynb`,
`notebooks/04_model_random_forest.ipynb`,
`notebooks/05_model_xgboost.ipynb`, `notebooks/06_model_lstm.ipynb`, and
the corresponding model/ensemble code under `src/drought_prediction/`.

## Current implementation

| Stage | What the repository currently does |
|---|---|
| Silver missing values | The Silver notebook clips to a spatial boundary and uses `dropna` for anomaly analysis, but there is no imputation/interpolation step. The Gold model table removes rows with missing feature/target values. |
| Silver outliers | It audits eight ERA5-Land variables with month/grid-specific Tukey IQR bounds (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`). It explicitly retains the source values rather than clipping or replacing them; physical checks reported zero invalid Kelvin observations. The code maps the outlier-detail table to a Silver flag, but its saved notebook output reports 22,037 IQR observations and zero published flagged observations, so the final flag should be checked. This is IQR flagging, not Huber treatment. |
| PET / water balance | PET is `abs(pev) * 1000 * days_in_month` from ERA5-Land potential evaporation; it is not Hargreaves-Samani or FAO-56 Penman-Monteith. The Gold balance is precipitation minus PET. |
| SPEI-3 distribution | The notebook defines a Gamma-based helper for SPI, but that helper is not used to produce SPEI. The actual `SPEI_3m` is a 3-month rolling water-balance sum standardized as `(x - mean) / (std + 1e-6)` for each grid cell, over the complete 2000-2025 series. This is a Z-score approximation, not the conventional SPEI log-logistic fit. The full-period mean/std also means the held-out years affect the target normalization. |
| Scaling | RF and XGBoost use unscaled feature values. LSTM uses `MinMaxScaler`, fit only on rows before 2019; validation/test rows are transformed using that fitted scaler. No `StandardScaler` is used. |
| Split / target | RF originally fits `spei_3m` at the same timestamp as its features and writes the same timestamp to the test file. It is not a one-month-ahead forecast and cannot be compared fairly to the other forecasts. XGBoost and LSTM shift target by one month, train on targets before 2019, validate on 2019-2021, and test on 2022-2025. Thus the code's actual split is 2000-2018 / 2019-2021 / 2022-2025, not a single 2000-2021 training set. |
| Random Forest | `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)`, 15 features, no scaling. |
| XGBoost | `XGBRegressor(objective="reg:squarederror", n_estimators=1200, max_depth=6, learning_rate=0.3, random_state=42, tree_method="hist")`, validation early stopping (`eval_metric="rmse"`, 50 rounds), 15 features, no scaling. |
| LSTM | 12 monthly inputs × 15 features; 2-layer unidirectional LSTM, hidden size 64, dropout 0.2; last timestep feeds `Linear(64,32) -> ReLU -> Dropout(0.2) -> Linear(32,1)`. It has no Bidirectional layer or Attention. Adam, MSE, gradient clipping, validation scheduler and early stopping. |
| Ensemble | `ensemble_pipeline.py` performs an outer join and arithmetic mean across LSTM, RF, XGBoost, skipping missing values. `future_forecaster.py` instead averages only RF and XGBoost. Neither is validation-weighted nor provides uncertainty intervals. |

## Leakage and validity findings

1. The target's 2000-2025 mean/std are calculated before the chronological split. This uses test-period climatology in the target definition; use a declared fixed climatological reference period or fit distribution parameters on training data and document the choice.
2. The original RF notebook predicts same-month SPEI using same-month meteorological features. Its existing test score is not a one-month-ahead forecast score. The upgraded evaluation retrains RF with a one-month lead before comparing models.
3. XGBoost's current-month SPEI feature is a valid autoregressive input only under a one-month-ahead setup where month `t` is observed before forecasting `t+1`. The prediction files label the forecast target month and record the input month.
4. Existing test artifacts cover 15,648 grid-month observations from 2022-01 through 2025-12. Existing `all_models_metrics.json` is retained unchanged; upgraded metrics are written separately.

## Hydrology-method interpretation

The current target should be described as a standardized 3-month water-balance anomaly derived from ERA5-Land `pev`, rather than a canonical distribution-fitted SPEI unless the SPEI distribution step is reimplemented. For a publication-grade SPEI-3, define PET consistently (prefer FAO-56 Penman-Monteith when required inputs are available), aggregate `P - PET` over rolling three-month windows, fit the chosen three-parameter log-logistic distribution per grid cell and calendar month over a declared reference climatology, then transform its CDF to standard normal scores. Preserve genuine extremes; use physical plausibility checks for sensor/reanalysis errors and separately flag, rather than automatically delete, statistical extremes.

## Reproducible upgrade

`src/drought_prediction/models/baseline_arima.py` provides a 12-month seasonal-persistence baseline. `src/drought_prediction/models/evaluate_models.py` generates an aligned, one-month-ahead RF, computes validation inverse-MSE weights, evaluates the five models by region, and writes 95% normal-approximation intervals. The interval is explicitly approximate; empirical coverage must be checked on validation residuals before representing it as calibrated.
