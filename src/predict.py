import argparse, os, pandas as pd, pickle


def predict(pairs_path, model_path, out_path, topk):
df = pd.read_csv(pairs_path)
with open(model_path, 'rb') as f:
model = pickle.load(f)
proba = model.predict_proba(df.drop(columns=['label','score_heuristic','client_id','property_id']))[:,1]
df['pred_proba'] = proba


# Top-K por cliente
df_sorted = df.sort_values(['client_id','pred_proba'], ascending=[True, False])
topk_rows = df_sorted.groupby('client_id').head(topk)
topk_rows.to_csv(out_path, index=False)
print(f"✓ Top-{topk} por cliente guardado en {out_path}")




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--pairs', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--topk', type=int, default=5)
args = ap.parse_args()
predict(args.pairs, args.model, args.out, args.topk)


if __name__ == '__main__':
main()
