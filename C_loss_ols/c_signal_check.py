"""
Phase 3a supporting check: is the weak OLS / loss-model performance a limitation of
linear models, or is there simply no signal in order-time columns?

Run:  python C_loss_ols/c_signal_check.py
Reads : A_preprocessing_eda/outputs/cleaned_supply_chain.csv  (never modified)
Writes: C_loss_ols/outputs/signal_check.txt, signal_check.csv

Same split as c_ols_loss.py (80/20, stratified on loss, random_state = 42).
Gradient boosting is used only as a "ceiling" probe with every order-time column,
including Product Name. It is not a proposed model.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "A_preprocessing_eda" / "outputs" / "cleaned_supply_chain.csv"
OUT = ROOT / "C_loss_ols" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)
lines, rows = [], []


def say(s=""):
    print(s)
    lines.append(s)


# ---- 1. Ceiling probe: gradient boosting, all order-time columns incl. Product Name ----
NUM = ["Order Item Product Price", "Order Item Quantity", "Order Item Discount Rate",
       "Order Item Discount", "Sales", "Order Item Total", "order_month", "order_year",
       "order_weekday", "Days for shipment (scheduled)"]
CAT = ["Shipping Mode", "Market", "Customer Segment", "Type", "Department Name",
       "Category Name", "Order Region", "Order Country", "Product Name"]
X = df[NUM].astype(float).copy()
for c in CAT:
    X[c] = df[c].astype("category").cat.codes
cat_idx = [X.columns.get_loc(c) for c in CAT]

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED,
                                  stratify=df["loss"].values)
clf = HistGradientBoostingClassifier(random_state=SEED, categorical_features=cat_idx)
clf.fit(X.iloc[idx_tr], df["loss"].values[idx_tr])
auc = roc_auc_score(df["loss"].values[idx_te], clf.predict_proba(X.iloc[idx_te])[:, 1])
reg = HistGradientBoostingRegressor(random_state=SEED, categorical_features=cat_idx)
reg.fit(X.iloc[idx_tr], df["Benefit per order"].values[idx_tr])
r2 = r2_score(df["Benefit per order"].values[idx_te], reg.predict(X.iloc[idx_te]))
say("1) Ceiling probe (gradient boosting, all order-time columns incl. Product Name, test set)")
say(f"   loss ROC-AUC = {auc:.4f}   (0.5 = no skill; logistic regression got about 0.503)")
say(f"   Benefit per order R2 = {r2:.4f}   (OLS got 0.0200)")
rows += [{"check": "GB loss ROC_AUC (test)", "value": round(auc, 4)},
         {"check": "GB Benefit R2 (test)", "value": round(r2, 4)}]

# ---- 2. Does the product explain who makes a loss? ----
g = df.groupby("Product Name").agg(n=("loss", "size"), loss_rate=("loss", "mean"))
g = g[g.n >= 200]
overall = df["loss"].mean()
chance_sd = float(np.sqrt(overall * (1 - overall) / g.n).mean())
say("")
say("2) Loss rate across products with >= 200 order lines")
say(f"   products = {len(g)}, loss-rate range = {g.loss_rate.min():.3f} to {g.loss_rate.max():.3f}")
say(f"   sd of product loss rates = {g.loss_rate.std():.4f}; sd expected from pure chance = {chance_sd:.4f}")
rows += [{"check": "n products (>=200 lines)", "value": len(g)},
         {"check": "sd of product loss rates", "value": round(float(g.loss_rate.std()), 4)},
         {"check": "chance sd of product loss rates", "value": round(chance_sd, 4)}]

# ---- 3. Same product, same price, same quantity, low discount: how different are outcomes? ----
top = df["Product Name"].value_counts().index[0]
sub = df[(df["Product Name"] == top) & (df["Order Item Quantity"] == 1) &
         (df["Order Item Discount Rate"] <= 0.05)]
q = sub["Benefit per order"].quantile([0.05, 0.5, 0.95]).round(2).tolist()
say("")
say(f"3) One product held fixed: '{top}', quantity 1, discount rate <= 5%")
say(f"   order lines = {len(sub):,}; distinct unit prices = {sub['Order Item Product Price'].nunique()}")
say(f"   loss rate = {sub['loss'].mean():.3f}; Benefit per order 5th / 50th / 95th percentile = {q}")
say(f"   losses = {int(sub['loss'].sum()):,}, profits = {int((sub['loss'] == 0).sum()):,}")
rows += [{"check": "fixed-product order lines", "value": len(sub)},
         {"check": "fixed-product loss rate", "value": round(float(sub['loss'].mean()), 4)},
         {"check": "fixed-product Benefit p05", "value": q[0]},
         {"check": "fixed-product Benefit p50", "value": q[1]},
         {"check": "fixed-product Benefit p95", "value": q[2]}]

(OUT / "signal_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
pd.DataFrame(rows).to_csv(OUT / "signal_check.csv", index=False)
