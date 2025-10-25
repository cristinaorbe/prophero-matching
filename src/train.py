import argparse, os, yaml, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


NUM_COLS = [
"price_delta_norm","rooms","baths","total_price","yield_est","reno_cost",
"has_max_budget","budget_min","budget_max_num","mort_ok","rooms_exact","baths_exact",
"in_zone","reno_match","first_time_investor"
]
CAT_COLS = ["zone","reno_bucket"]


def train_model(pairs_path, models_dir, config):
os.makedirs(models_dir, exist_ok=True)
df = pd.read_csv(pairs_path)
X = df[NUM_COLS + CAT_COLS]
y = df['label'].astype(int)


pre = ColumnTransformer([
("num", "passthrough", NUM_COLS),
("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
])


model_name = config.get('model','random_forest')
if model_name == 'logistic_regression':
clf = LogisticRegression(max_iter=500)
else:
clf = RandomForestClassifier(n_estimators=300, random_state=0)


pipe = Pipeline([("pre", pre), ("clf", clf)])


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config['train_test_split'], random_state=7, stratify=y)
pipe.fit(Xtr, ytr)


proba = pipe.predict_proba(Xte)[:,1]
pred = (proba >= 0.5).astype(int)


metrics = dict(
model=model_name,
AUC=float(roc_auc_score(yte, proba)),
Accuracy=float(accuracy_score(yte, pred)),
F1=float(f1_score(yte, pred))
)


import pickle
main()
