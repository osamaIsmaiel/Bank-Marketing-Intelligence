# Bank Marketing Intelligence — Predicting Customer Subscription

Dataset: UCI Bank Marketing, `bank-additional-full.csv` (41,188 rows, 21 columns).
Official source: https://archive.ics.uci.edu/dataset/222/bank+marketing

- Binary classification target: y (yes/no).
- 12 exact duplicate rows removed; unknown categories retained; no literal missing cells in raw file.
- Feature engineering: previous-contact indicator, cleaned previous-contact days, total campaign contacts, age group.
- `duration` excluded because it is only known after the call (leakage).
- Stratified 70/15/15 train/validation/test split.
- Three models: Logistic Regression (GridSearchCV tuned), Random Forest, Extra Trees.
- Embedded feature selection using ExtraTrees; threshold selected on validation data.
- Test results:
                      model  threshold  accuracy  precision   recall       f1  roc_auc  average_precision
              Random Forest   0.499748  0.878096   0.466744 0.574713 0.515132 0.805317           0.464475
Logistic Regression (tuned)   0.666935  0.873078   0.449309 0.560345 0.498721 0.803268           0.446997
                Extra Trees   0.469968  0.853003   0.386994 0.521552 0.444308 0.780737           0.413343

Best model: Random Forest; selected threshold: 0.4997.
Run: `pip install -r requirements.txt`; `python train.py`; `streamlit run app.py`.


## Final saved-pipeline test metrics
- precision: 0.4670
- recall: 0.5690
- f1: 0.5130
- accuracy: 0.8783
- roc_auc: 0.8090
- average_precision: 0.4735