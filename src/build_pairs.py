import argparse, os, pandas as pd, numpy as np, yaml
client_id=c['client_id'], property_id=p['property_id'], score_heuristic=sc,
price_delta_norm=price_delta_norm, rooms=p['rooms'], baths=p['baths'], total_price=total_price,
yield_est=p['yield_est'], reno_cost=p['reno_cost'], has_max_budget=int(c['has_max_budget']),
budget_min=c['budget_min'], budget_max_num=float(c['budget_max_num']), mort_ok=mort_ok,
rooms_exact=rooms_exact, baths_exact=baths_exact, in_zone=in_zone, reno_match=reno_match,
first_time_investor=int(c['first_time_investor']), zone=p['zone'], reno_bucket=p['reno_bucket']
))


df = pd.DataFrame(rows)
# Etiqueta sintética para el primer entrenamiento
df['label'] = (df['score_heuristic'] >= config['score_threshold']).astype(int)
df.to_csv(out_path, index=False)
print(f"✓ Pairs with features saved to {out_path} ({len(df)} rows)")




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--clients', required=True)
ap.add_argument('--properties', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--config', default='config.yaml')
args = ap.parse_args()
with open(args.config, 'r') as f:
config = yaml.safe_load(f)
build_pairs(args.clients, args.properties, args.out, config)


if __name__ == '__main__':
main()
