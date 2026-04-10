
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
# Tuned: XGBoost (CV=0.80724)

data_dir        = Path("C:/Users/PC4/AppData/Local/Temp/ml_agent_2xoebdeg/data")
submission_path = Path("C:/Users/PC4/AppData/Local/Temp/ml_agent_2xoebdeg/submission_xgb.csv")

train = pd.read_csv(next(data_dir.rglob("train.csv")))
test  = pd.read_csv(next(data_dir.rglob("test.csv")))
sample_paths = list(data_dir.rglob("sample_submission.csv"))
if sample_paths:
    sample     = pd.read_csv(sample_paths[0])
    id_col     = sample.columns[0]
    target_col = sample.columns[1]
else:
    id_col     = train.columns[0]
    target_col = train.columns[-1]

print(f"id={id_col}  target={target_col}  train={train.shape}  test={test.shape}")
ids = test[id_col].reset_index(drop=True)
y   = train[target_col].copy()

def engineer(df):
    out = pd.DataFrame(index=df.index)
    cabin_deck = None
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", expand=True)
        cabin_deck = cabin[0].where(cabin[0] != "nan", other="__NA__")
        out["Cabin_Deck"] = cabin_deck
        raw_num = pd.to_numeric(cabin.get(1), errors="coerce")
        out["Cabin_Num"]    = raw_num
        out["CabinNum_bin"] = pd.cut(raw_num, bins=10, labels=False).astype("float")
        out["Cabin_Side"]   = cabin[2].where(cabin[2] != "nan", other="__NA__") if 2 in cabin.columns else "__NA__"
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", expand=True)
        out["Group"] = pd.to_numeric(pid[0], errors="coerce")
        grp_series = df["PassengerId"].astype(str).str.split("_").str[0]
        grp_size   = grp_series.map(grp_series.value_counts())
        out["GroupSize"] = grp_size.values
        out["IsSolo"]    = (grp_size == 1).astype(int).values
    spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
    existing   = [c for c in spend_cols if c in df.columns]
    if existing:
        spend = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0)
        total = spend.sum(axis=1)
        out["TotalSpend"]    = total
        out["AnySpend"]      = (total > 0).astype(int)
        out["LogTotalSpend"] = np.log1p(total)
        grp_size_safe = out.get("GroupSize", pd.Series(1, index=df.index)).replace(0, 1)
        out["SpendPerPerson"] = total / grp_size_safe
        for c in existing:
            out[c]           = spend[c]
            out[f"Log{c}"] = np.log1p(spend[c])
    for col in df.columns:
        uniq = set(str(v).lower() for v in df[col].dropna().unique())
        if uniq <= {"true", "false"}:
            out[f"{col}_int"] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(-1).astype(int)
    cryo_int = None
    if "CryoSleep" in df.columns:
        cryo_int = df["CryoSleep"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)
        if existing:
            out["CryoSpend_flag"] = ((cryo_int == 1) & (out.get("TotalSpend", pd.Series(0, index=df.index)) > 0)).astype(int)
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce")
        out["Age_bin"] = pd.cut(age, bins=[0,12,18,35,60,200], labels=False).astype("float")
        if cryo_int is not None:
            out["Age_x_Cryo"] = age.fillna(age.median()) * cryo_int
        if existing:
            total_spend = out.get("TotalSpend", pd.Series(0, index=df.index))
            out["Age_x_Spend"] = age.fillna(age.median()) * np.log1p(total_spend)
    if cabin_deck is not None and "Destination" in df.columns:
        out["Deck_x_Dest"] = cabin_deck.astype(str) + "_" + df["Destination"].astype(str).fillna("__NA__")
    if "HomePlanet" in df.columns and "Destination" in df.columns:
        out["Home_x_Dest"] = df["HomePlanet"].astype(str).fillna("__NA__") + "_" + df["Destination"].astype(str).fillna("__NA__")
    skip = {"Cabin", "PassengerId", "Name"}
    for c in df.columns:
        if c not in skip and c not in out.columns:
            out[c] = df[c]
    return out

_tr_feat = train.drop(columns=[c for c in [id_col, target_col] if c in train.columns])
_te_feat = test.drop(columns=[c for c in [id_col, target_col] if c in test.columns])
X_tr_raw = engineer(_tr_feat)
X_te_raw = engineer(_te_feat)

print(f"Features ({len(X_tr_raw.columns)}): {list(X_tr_raw.columns)}")
num_cols = X_tr_raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_tr_raw.columns if c not in num_cols]

X_tr = X_tr_raw.assign(**{c: X_tr_raw[c].astype(str) for c in cat_cols})
X_te = X_te_raw.assign(**{c: X_te_raw[c].astype(str) for c in cat_cols})

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="__NA__")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
transformers = []
if num_cols: transformers.append(("num", num_pipe, num_cols))
if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
pre = ColumnTransformer(transformers=transformers, remainder="drop")

clf = Pipeline([("pre", pre), ("model", XGBClassifier(n_estimators=503, max_depth=3, learning_rate=0.0704, subsample=0.97, colsample_bytree=0.84, min_child_weight=1, gamma=0.294, reg_alpha=0.488, reg_lambda=1.785, random_state=42, eval_metric='logloss', verbosity=0))])
print(f"Fitting XGBoost...")
clf.fit(X_tr, y)
preds = clf.predict(X_te)

sub = pd.DataFrame({id_col: ids, target_col: preds})
sub.to_csv(submission_path, index=False)
print(f"Done. {len(sub)} predictions saved to {submission_path}")
