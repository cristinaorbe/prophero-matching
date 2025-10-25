import math
rng = max(1.0, float(c["budget_max_num"]) - float(c["budget_min"]))
mid = (float(c["budget_max_num"]) + float(c["budget_min"])) / 2
else:
rng, mid = 20000.0, float(c["budget_min"]) + 10000.0
price_sc = max(0.0, 1 - abs(total_price - mid) / (rng / 2))


rooms_sc = 1.0 if (prop_rooms in rooms_pref or (is_loft and 0 in rooms_pref)) else 0.6


baths_pref = parse_list(c.get("baths_pref"))
b_sc = 1.0 if (not baths_pref or int(p["baths"]) in baths_pref) else 0.6


zones_pref = set(parse_list(c.get("zones_pref")))
in_zone = str(p["zone"]) in zones_pref
if zones_pref:
centroids = [ZONES[z] for z in zones_pref if z in ZONES]
lat0, lng0 = centroids[0] if centroids else (float(p["lat"]), float(p["lng"]))
else:
lat0, lng0 = float(p["lat"]), float(p["lng"])
dist = haversine_km(float(p["lat"]), float(p["lng"]), lat0, lng0)
dist_sc = max(0.0, 1 - dist / radius_km)
zone_sc = 1.0 if in_zone else dist_sc


mort_sc = 1.0 if (c.get("mortgage_ok") == "Indiferente" or (c.get("mortgage_ok") == "Sí" and bool(p.get("mortgageable"))) or (c.get("mortgage_ok") == "No" and not bool(p.get("mortgageable")))) else 0.0
time_sc = 1.0


musts = set(parse_list(c.get("must_haves")))
nices = set(parse_list(c.get("nice_to_haves")))
prop_feats = set(parse_list(p.get("features")))
must_ok = musts.issubset(prop_feats)
nice_sc = (len(nices & prop_feats) / max(1, len(nices))) if nices else 0.0
feat_sc = 0.5 * (1.0 if must_ok else 0.0) + 0.5 * nice_sc


y_min = float(c["yield_min"]) if c.get("yield_min") not in (None, "") else 0.0
y_sc = 1.0 if float(p["yield_est"]) >= y_min else max(0.0, 1 - (y_min - float(p["yield_est"])) / 0.02)


reno_match = 1.0 if str(c.get("reno_budget_pref")) == str(p.get("reno_bucket")) else 0.7


score = 100 * (
weights["price"]*price_sc + weights["rooms"]*rooms_sc + weights["baths"]*b_sc +
weights["zone"]*zone_sc + weights["mortgage"]*mort_sc + weights["timing"]*time_sc +
weights["features"]*feat_sc + weights["yld"]*y_sc + weights["reno"]*reno_match
)
return round(score, 2)
