from math import radians, sin, cos, atan2, sqrt
import json


def haversine_km(lat1, lon1, lat2, lon2):
R = 6371.0
dlat = radians(lat2 - lat1)
dlon = radians(lon2 - lon1)
a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1-a))
return R * c


def parse_list(s):
if s is None: return []
s = str(s).strip()
if not s: return []
try:
v = json.loads(s)
if isinstance(v, list):
return v
except Exception:
pass
return [x.strip() for x in s.split(',') if x.strip()]
