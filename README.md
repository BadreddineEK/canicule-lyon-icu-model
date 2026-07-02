# 🌡️ Chaleur nocturne à Lyon — au grain fin

> *« Pendant la canicule, je me suis demandé pourquoi certains coins de Lyon restaient étouffants la nuit quand d'autres respiraient. J'ai pris les vraies données de la Métropole, îlot par îlot. Et j'ai découvert que mon plus beau R² était aussi le plus trompeur. »*

**▶️ Démo en ligne : [canicule-lyon-icu-model.streamlit.app](https://canicule-lyon-icu-model.streamlit.app/)**

Un dashboard Streamlit qui cartographie l'îlot de chaleur urbain de Lyon à partir de **vraies données ouvertes**, au grain de l'îlot (~29 657 points réels) — et qui montre au passage **le piège de l'agrégation** : en résumant chaque commune à un chiffre, on décroche un R² flatteur en jetant 99,8 % du signal.

## 🎯 L'idée

Question prise à l'endroit : **qu'est-ce qui explique la chaleur nocturne d'une commune ?**

On confronte trois ingrédients de la surface au sol au score d'exposition nocturne à la chaleur :

- 🌿 **Végétation** (% de surface : parcs, arbres, espaces verts)
- 🏙️ **Bâti compact** (% de surface : immeubles serrés, cœurs urbains denses)
- 🪨 **Sol minéral** (% de surface : béton, bitume, surfaces imperméables)

Méthode : régression linéaire sur variables standardisées, validation croisée (5 folds), puis analyse des écarts (résidus) entre prédiction et réalité.

## 🗂️ Les données (réelles)

Tout vient d'un seul jeu ouvert et officiel : la couche **« Îlots de chaleur urbains »** de [data.grandlyon.com](https://data.grandlyon.com), produite avec le logiciel scientifique **GEOCLIMATE** (méthode de l'Institut Paris Région).

- Chaque îlot est classé en **Local Climate Zone (LCZ)** — la typologie internationale de forme urbaine.
- Chaque îlot reçoit une **exposition nocturne à la chaleur** (de « effet rafraîchissant » à « fort »).
- Les **29 657 îlots** ont été agrégés par commune (pondéré par la surface) pour obtenir, d'un côté la composition du sol, de l'autre un **score d'exposition** moyen — la cible du modèle.
- Résultat : **67 communes / arrondissements** de la Métropole (celles comptant au moins 20 îlots).

Ce sont de **vraies mesures d'aménagement**, pas des relevés de thermomètre : l'exposition est un indicateur calculé, à lire comme tel.

## 🔬 Ce que montre le dashboard

- La **carte au grain fin** : ~29 657 îlots réels, colorés du bleu (rafraîchissant) au rouge (fort), lisibles rue par rue
- Le **piège de l'agrégation** : la même réalité moyennée en 67 communes, un R² de 87 %… et pourquoi il ment
- Un **explorateur** : zoom sur les îlots d'une commune pour voir son contraste interne (un parc frais à côté d'un cœur dense brûlant)
- La **tendance de fond** : bâti compact → chaleur, coefficients et classement des communes
- Un **simulateur** interactif de la relation encodée par la donnée

## 💡 La leçon data science

Le plus beau chiffre du projet, un **R² de 87 %**, est aussi le plus trompeur : il vient d'avoir agrégé 29 657 mesures en 67 points. La **validation croisée le ramène à ~64 %** (très instable), et le grain fin le démasque.

- **La bonne échelle** n'est pas la plus pratique, c'est celle où vit le phénomène — ici l'îlot, pas la commune.
- **Agréger gonfle mécaniquement les corrélations** : un chiffre flatteur mérite toujours qu'on cherche ce qu'il cache.
- **Honnêteté sur la donnée** : l'exposition est un indicateur calculé par GEOCLIMATE à partir de la forme urbaine. La relation « bâti compact → chaleur » est donc en partie encodée dans la donnée : on la **visualise**, on ne la « découvre » pas. C'est de la lecture de données, pas une preuve causale.

## 🛠️ Stack technique

- Python 3.11+
- Streamlit (dashboard interactif)
- Pandas, NumPy, Scikit-learn (modèle + validation croisée)
- Plotly (visualisations, carte)

## 🚀 Lancer l'app en local

```bash
git clone https://github.com/BadreddineEK/canicule-lyon-icu-model
cd canicule-lyon-icu-model
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre dans le navigateur (par défaut sur http://localhost:8501).

## 🔄 Reproduire le jeu de données

Les CSV sont déjà versionnés (`data/ilots_lyon.csv` et `data/communes_lyon.csv`), l'app fonctionne sans rien télécharger. Pour les régénérer depuis la source ouverte :

```bash
pip install requests
python data/build_dataset.py
```

Le script télécharge les 29 657 îlots via le service WFS de data.grandlyon.com, les classe par LCZ, écrit le grain fin géolocalisé puis l'agrégat par commune. `requests` n'est utilisé que par ce script (hors runtime de l'app).

## 📁 Structure du projet

```
├── app.py                      # App Streamlit principale
├── model/
│   └── naive_model.py          # Régression linéaire + validation croisée + résidus
├── data/
│   ├── ilots_lyon.csv          # Grain fin : ~29 657 îlots géolocalisés — versionné
│   ├── communes_lyon.csv       # Agrégat par commune (67 lignes) — versionné
│   ├── dataset.py              # Chargement des CSV + descriptions des variables
│   └── build_dataset.py        # Script de génération depuis l'open data (dev)
├── utils/
│   └── viz.py                  # Graphiques Plotly (charte commune)
├── .streamlit/config.toml      # Thème de l'app
├── requirements.txt
├── LICENSE
└── README.md
```

## 📄 Licence

MIT — voir [LICENSE](LICENSE). Données : Métropole de Lyon, couche « Îlots de chaleur urbains » calculée avec GEOCLIMATE, publiée sous **Licence Ouverte Etalab** sur [data.grandlyon.com](https://data.grandlyon.com).

---

*Badreddine EL KHAMLICHI · Ingénieur en mathématiques appliquées · Lyon · [badreddineek.com](https://badreddineek.com)*
