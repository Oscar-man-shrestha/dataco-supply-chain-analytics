# HAND-OFF: Phase 3b — Late-delivery model (D)

## 1. Done

- Built a leakage-free logistic regression model for `Late_delivery_risk` using A's cleaned supply-chain dataset.
- Used an 80/20 stratified train/test split with `random_state = 42`.
- Used `Shipping Mode` instead of `Days for shipment (scheduled)`, since scheduled days are completely determined by Shipping Mode in this dataset.
- Scaled numeric features using training data only and evaluated the final model on the held-out test set.

## 2. Key numbers

Test set: 36,104 orders.

- Accuracy: 0.696765 (69.68%)
- Precision: 0.843745 (84.37%)
- Recall: 0.548545 (54.85%)
- F1: 0.664850 (66.49%)
- ROC-AUC: 0.740931

Confusion matrix:

- TN = 14,297
- FP = 2,011
- FN = 8,937
- TP = 10,859

Top odds ratios:

- Shipping Mode — First Class: OR = 34.2236
- Shipping Mode — Second Class: OR = 5.3869
- Type — TRANSFER: OR = 0.6408
- Shipping Mode — Same Day: OR = 1.3754

The model's accuracy and ROC-AUC are consistent with the expected performance range for the leakage-free late-delivery model.

## 3. Decisions

- `Shipping Mode` was retained and `Days for shipment (scheduled)` was excluded because the latter has exactly one value for each shipping mode: Same Day = 0, First Class = 1, Second Class = 2, Standard Class = 4.
- `Order Status` was excluded because it contains post-order information associated with the delivery outcome and can leak the target.
- `Delivery Status`, `Days for shipping (real)`, and `shipping date (DateOrders)` were excluded as delivery-time/post-delivery leakage and were already removed during Phase 1.
- `Benefit per order`, `Benefit_signed_log`, `loss`, and `late` were excluded because they are targets or transformations of targets rather than legitimate predictors.
- Numeric features were scaled after the train/test split, with the scaler fitted only on the training data.
- Categorical dummy groups use one reference level each to avoid redundant dummy variables in the logistic regression model.
- Results should be interpreted as predictive associations, not causal effects. Shipping Mode provided the strongest predictive signal.
- Odds ratios for standardized numeric features represent the change in odds for a one-standard-deviation increase; categorical odds ratios are interpreted relative to their dropped reference category.

## 4. Files

- `D_late_model_report/late_model.py`
- `D_late_model_report/outputs/late_model_results.csv`
- `D_late_model_report/outputs/late_predictions.csv`
- `D_late_model_report/outputs/late_odds_ratios.csv`
- `D_late_model_report/outputs/late_model_summary.txt`
- `D_late_model_report/outputs/plots/confusion_matrix.png`
- `D_late_model_report/outputs/plots/roc_curve.png`

## 5. Before you start

- Use the rows with `split = test` in `late_predictions.csv` for test-set model-result reporting.
- Use the saved accuracy, precision, recall, F1 and ROC-AUC values above in the report and Power BI.
- Do not describe Shipping Mode, market, segment or other predictors as causal "drivers"; describe them as predictive/associated features.
- A substantially higher accuracy, especially above roughly 90%, should trigger a leakage check.    