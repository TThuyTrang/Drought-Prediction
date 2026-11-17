# Methodology summary for Section 2.3

The one-month-ahead SPEI-3 forecasting experiment used a chronological split: training targets before January 2019, validation targets from January 2019 through December 2021, and a held-out test period from January 2022 through December 2025. All model forecasts were aligned by target month and grid cell. Random Forest was retrained using the next month's SPEI-3 as its target so that its forecast horizon matched XGBoost and LSTM. A 12-month seasonal-persistence forecast (the same grid cell's SPEI-3 from the same month of the preceding year) served as the classical time-series baseline.

The weighted ensemble combines Random Forest, XGBoost, and LSTM test forecasts using normalized inverse validation-MSE weights; no test outcomes are used to select the weights. We report MAE, MSE, RMSE, R², mean bias, and recall for severe drought observations (`actual SPEI-3 < -1.5`) for the whole Mekong Delta, seven coastal provinces, and six inland provinces. The ensemble's 95% interval is a normal approximation derived from validation MSEs, not an empirically calibrated conformal interval; its coverage must be verified before interpreting it as a calibrated probability interval.

The current project target is calculated by aggregating monthly precipitation minus ERA5-Land potential evaporation (`pev`) over three months and standardizing the resulting water-balance sum with each grid cell's full-period mean and standard deviation. It is therefore a standardized water-balance anomaly and not a distribution-fitted canonical SPEI. A publication-grade analysis should calculate PET with a declared method, fit the three-parameter log-logistic distribution to a declared reference climatology, and avoid using held-out test-period observations to estimate target-distribution parameters.

The canonical SPEI formulation follows Vicente-Serrano, Beguería, and López-Moreno (2010, *Journal of Climate*, 23, 1696–1718, doi:10.1175/2009JCLI2909.1). FAO-56 Penman–Monteith is the reference evapotranspiration method (Allen et al., 1998, FAO Irrigation and Drainage Paper 56); the present Silver variables do not contain all its required meteorological inputs, so `pev` is retained and described accurately rather than mislabeled as FAO-56 ET₀.

To regenerate forecasts and metrics from the repository root, run:

```bash
PYTHONPATH=src python -m drought_prediction.models.evaluate_models
```

Then `src/drought_prediction/data_processing/ensemble_pipeline.py` exports the weighted result and its interval while retaining the legacy arithmetic-mean column for compatibility.
