# PropHero Matching – Starter


## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate # en Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```


## 2) Generar datos sintéticos (mockup)
```bash
python -m src.data_gen --n_clients 100 --n_properties 50 --out data
```


## 3) Construir pares + features
```bash
python -m src.build_pairs --clients data/clients.csv --properties data/properties.csv --out data/pairs_to_score.csv
```


## 4) Entrenar
```bash
python -m src.train --pairs data/pairs_to_score.csv --models_dir models --config config.yaml
```


## 5) Predecir Top-K (ejemplo: Top-5 por cliente)
```bash
python -m src.predict --pairs data/pairs_to_score.csv --model models/model.pkl --out data/preds.csv --topk 5
```


## 6) Próximos pasos
- Sustituye `data/clients.csv` y `data/properties.csv` por export de Airtable.
- (Luego) añade un `airtable_io.py` para leer/escribir directamente.
