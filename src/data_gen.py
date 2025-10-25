import argparse, os, random
feats = rng.sample(FEATURES, k=rng.randint(1,5))
reno_bucket = wchoice(["small","medium","high"],[0.5,0.3,0.2])
reno_cost = rng.randint(3000,5000) if reno_bucket=="small" else (rng.randint(10000,20000) if reno_bucket=="medium" else rng.randint(35000,40000))
total_price = price_purchase + reno_cost
if total_price < 85000:
price_purchase += (85000 - total_price)
total_price = price_purchase + reno_cost
yield_est = round(rng.uniform(0.05,0.07) + rng.uniform(-0.0025,0.0025), 4)
props.append(dict(
property_id=f"P{i:03d}", status=status, price_purchase=price_purchase, reno_cost=reno_cost,
total_price=total_price, rooms=rooms, is_loft=is_loft, baths=baths, zone=zone, lat=lat, lng=lng,
mortgageable=mortgageable, arras_end=arras_end, yield_est=yield_est, features=",".join(feats), reno_bucket=reno_bucket
))
return pd.DataFrame(props)




def gen_clients(n=100):
clients = []
for i in range(1, n+1):
budget_min = wchoice([50000,55000,60000,65000], [0.35,0.30,0.25,0.10])
r = rng.random()
if r < 0.85:
has_max, budget_max = True, rng.randint(max(60000, budget_min+5000), 80000)
elif r < 0.95:
has_max, budget_max = True, rng.randint(max(80001, budget_min+10000), 120000)
else:
has_max, budget_max = False, ""
rooms_pref = wchoice([[1],[2],[1,2],[0,1],[0,2],[0,1,2]], [0.28,0.34,0.22,0.06,0.06,0.04])
baths_pref = wchoice([[],[1],[1,2]],[0.55,0.35,0.10])
zones_pref = list(set(rng.choices([z[0] for z in ZONES_LIST], k=rng.randint(1,3))))
mortgage_ok = wchoice(["Sí","No","Indiferente"],[0.75,0.05,0.20])
buy_window_end = (datetime.today() + timedelta(days=90 + rng.randint(-15,15))).date().isoformat()
yield_min = wchoice([0.05,0.055,0.06,0.065,0.07],[0.3,0.25,0.25,0.12,0.08])
reno_pref = wchoice(["small","medium","high"],[0.5,0.35,0.15])
first_time = rng.random() < 0.8
clients.append(dict(
client_id=f"C{i:03d}", first_time_investor=first_time, budget_min=budget_min,
has_max_budget=has_max, budget_max=budget_max, rooms_pref=str(rooms_pref), baths_pref=str(baths_pref),
zones_pref=str(zones_pref), mortgage_ok=mortgage_ok, buy_window_end=buy_window_end, yield_min=yield_min,
reno_budget_pref=reno_pref, must_haves="", nice_to_haves=""
))
return pd.DataFrame(clients)




def main():
ap = argparse.ArgumentParser()
ap.add_argument('--n_clients', type=int, default=100)
ap.add_argument('--n_properties', type=int, default=50)
ap.add_argument('--out', type=str, default='data')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
gen_clients(args.n_clients).to_csv(os.path.join(args.out,'clients.csv'), index=False)
gen_properties(args.n_properties).to_csv(os.path.join(args.out,'properties.csv'), index=False)
print('✓ Synthetic datasets saved to', args.out)


if __name__ == '__main__':
main()
