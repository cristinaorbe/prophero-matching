prophero-matching/
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ data/ # CSVs live here (ignored in git if you want)
│ ├─ clients.csv
│ ├─ properties.csv
│ └─ pairs_to_score.csv
├─ models/
│ └─ model.pkl
├─ src/
│ ├─ __init__.py
│ ├─ data_gen.py # genera datasets sintéticos coherentes con PropHero
│ ├─ build_pairs.py # crea pares cliente-prop y features
│ ├─ train.py # entrena y guarda modelo (AUC/Accuracy/F1)
│ ├─ predict.py # puntúa pares y saca Top-K por cliente/propiedad
│ ├─ features.py # funciones de ingeniería de variables
│ └─ utils.py # helpers (distancias, parser de listas, etc.)
└─ .gitignore
