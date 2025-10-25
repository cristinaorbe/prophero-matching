# src/model_start.py
import json, math, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ---- 1) Cargar datos
clients = pd.read_csv("src/data/clients_v3.csv")
props   = pd.read_csv("src/data/properties_v3.csv")

# ---- 2) Limpieza mínima / validaciones rápidas
# a) Tipos numéricos clave
num_cols_clients = ["budget_min"]
for c in num_cols_clients:
    clients[c] = pd.to_numeric(clients[c], errors="coerce")

num_cols_props = ["total_price","rooms","baths","yield_est","reno_cost","lat","lng"]
for c in num_cols_props:
    props[c] = pd.to_numeric(props[c], errors="coerce")

# b) Reglas PropHero
assert (clients["budget_min"] >= 50000).all(), "Hay budget_min < 50k"
assert (props["total_price"] >= 85000).all(), "Hay total_price < 85k"

# c) Normaliza booleans/comodines
def to_bool(x): 
    if isinstance(x, str): 
        return x.strip().lower() in ["true","1","sí","si","yes"]
    return bool(x)

for col in ["has_max_budget","first_time_investor","is_loft","mortgageable"]:
    if col in clients.columns: clients[col] = clients[col].apply(to_bool) if col in clients else clients.get(col)
    if col in props.columns:   props[col]   = props[col].apply(to_bool)   if col in props   else props.get(col)

# d) Parseo de listas (si vienen como texto/JSON)
def parse_list(s):
    if pd.isna(s) or str(s).strip()=="":
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list): return v
    except:
        pass
    return [x.strip() for x in str(s).split(",") if x.strip()]

# ---- 3) Construir pares cliente-propiedad válidos (filtros duros mínimos)
pairs = []
price_tol = 0.10

for _, c in clients.iterrows():
    # soporte budget_max vacío (sin tope)
    has_max = bool(c.get("has_max_budget", True))
    budget_max = c.get("budget_max", np.nan)
    try:
        budget_max = float(budget_max)
    except:
        budget_max = np.nan
    budget_max_num = budget_max if not np.isnan(budget_max) else c["budget_min"] + 20000

    rooms_pref = parse_list(c.get("rooms_pref", ""))
    baths_pref = parse_list(c.get("baths_pref", ""))
    zones_pref = set(parse_list(c.get("zones_pref","")))
    y_min = float(c.get("yield_min", 0) or 0)

    for _, p in props.iterrows():
        if str(p.get("status","")).lower() != "activa": 
            continue
        if str(c.get("mortgage_ok","")).lower() == "sí" and not bool(p.get("mortgageable", True)):
            continue

        total_price = float(p["total_price"])
        low_ok  = total_price >= c["budget_min"] * (1 - price_tol)
        high_ok = total_price <= budget_max_num * (1 + price_tol) if has_max else True
        if not (low_ok and high_ok): 
            continue

        # rooms: 0 = loft
        pr = int(p.get("rooms",0))
        is_loft = bool(p.get("is_loft", False))
        if rooms_pref:
            if (pr not in rooms_pref) and not (is_loft and 0 in rooms_pref):
                continue

        # baths
        if baths_pref and int(p.get("baths",1)) not in baths_pref:
            continue

        in_zone = (str(p.get("zone","")) in zones_pref) if zones_pref else True
        # etiqueta sintética simple (positivo si encaja precio + tipología + zona y yield >= umbral cliente)
        label = 1 if (in_zone and (p.get("yield_est",0) >= y_min)) else 0

        pairs.append({
            "client_id": c["client_id"],
            "property_id": p["property_id"],
            "total_price": total_price,
            "rooms": pr,
            "baths": int(p.get("baths",1)),
            "zone": str(p.get("zone","")),
            "yield_est": float(p.get("yield_est",0)),
            "reno_cost": float(p.get("reno_cost",0)),
            "has_max_budget": int(has_max),
            "budget_min": float(c["budget_min"]),
            "budget_max_num": float(budget_max_num),
            "mort_ok": int(str(c.get("mortgage_ok","")).lower() in ["indiferente","sí","si","yes"]),
            "rooms_exact": int((pr in rooms_pref) or (is_loft and 0 in rooms_pref)) if rooms_pref else 1,
            "baths_exact": int((not baths_pref) or (int(p.get("baths",1)) in baths_pref)),
            "in_zone": int(in_zone),
            "first_time_investor": int(bool(c.get("first_time_investor", True))),
            "reno_bucket": str(p.get("reno_bucket","")),
            "label": int(label)
        })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("src/data/pairs_to_score.csv", index=False)
print("Pairs:", pairs_df.shape)

# ---- 4) Train / Test + modelo
NUM_COLS = ["total_price","rooms","baths","yield_est","reno_cost",
            "has_max_budget","budget_min","budget_max_num","mort_ok",
            "rooms_exact","baths_exact","in_zone","first_time_investor"]
CAT_COLS = ["zone","reno_bucket"]

X = pairs_df[NUM_COLS + CAT_COLS]
y = pairs_df["label"].astype(int)

pre = ColumnTransformer([
    ("num","passthrough", NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
])

rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=0))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

rf.fit(Xtr, ytr)
proba = rf.predict_proba(Xte)[:,1]
pred  = (proba >= 0.5).astype(int)

print("AUC:", roc_auc_score(yte, proba))
print("Accuracy:", accuracy_score(yte, pred))
print("F1:", f1_score(yte, pred))

# ---- 5) Top-K por cliente (ejemplo K=5)
pairs_df["pred_proba"] = rf.predict_proba(X)[:,1]
topk = pairs_df.sort_values(["client_id","pred_proba"], ascending=[True, False]).groupby("client_id").head(5)
topk.to_csv("src/data/preds_top5.csv", index=False)
print("Top-5 por cliente guardado en data/preds_top5.csv")


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

print("Distribución de y_test:", np.bincount(yte))
cm = confusion_matrix(yte, pred)
print("Matriz de confusión:\n", cm)
print(classification_report(yte, pred, digits=3))


from sklearn.metrics import precision_recall_fscore_support

best = {"thr":0.5, "F1":-1, "P":0, "R":0}
for thr in np.linspace(0.1, 0.9, 17):
    pr = (proba >= thr).astype(int)
    P,R,F1,_ = precision_recall_fscore_support(yte, pr, average="binary", zero_division=0)
    if F1 > best["F1"]:
        best = {"thr":thr, "F1":F1, "P":P, "R":R}
print(f"Mejor F1={best['F1']:.3f} en umbral={best['thr']:.2f} (P={best['P']:.3f}, R={best['R']:.3f})")

# usa el mejor umbral para el informe de test
pred_best = (proba >= best["thr"]).astype(int)
print("Confusión con mejor umbral:\n", confusion_matrix(yte, pred_best))


# Obtener nombres de features tras el preprocesador
preproc = rf.named_steps["pre"]
clf     = rf.named_steps["clf"]

num_names = preproc.transformers_[0][2]
ohe       = preproc.transformers_[1][1]
cat_names = list(ohe.get_feature_names_out(["zone","reno_bucket"]))
feat_names = list(num_names) + cat_names

importances = clf.feature_importances_
imp = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
print(imp.head(15))
imp.to_csv("src/data/feature_importances.csv", index=False)


# Top-5 clientes por propiedad
topk_prop = pairs_df.sort_values(["property_id","pred_proba"], ascending=[True, False]).groupby("property_id").head(5)
topk_prop.to_csv("src/data/preds_top5_por_propiedad.csv", index=False)
print("Top-5 por propiedad guardado en data/preds_top5_por_propiedad.csv")
