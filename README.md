# 🌡️ Îlot de Chaleur Urbain à Lyon — vraies données ouvertes

> *« Pendant la canicule, je me suis demandé pourquoi certaines communes de Lyon restent invivables la nuit quand d'autres respirent. J'ai creusé les vraies données de la Métropole. Voilà ce qu'elles disent — et pourquoi un beau R² peut mentir. »*

**▶️ Démo en ligne : [canicule-lyon-icu-model.streamlit.app](https://canicule-lyon-icu-model.streamlit.app/)**

Un dashboard Streamlit qui explique, à partir de **vraies données ouvertes de la Métropole de Lyon**, ce qui rend une commune chaude la nuit en été — et qui montre au passage pourquoi il faut se méfier d'un R² flatteur sur un petit échantillon.

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

- La **carte** de la chaleur nocturne sur toute la Métropole
- Le **poids de chaque variable** : le bâti compact est le moteur numéro un
- Un **simulateur** interactif : compose une commune et regarde le modèle recalculer l'exposition
- Le **réel vs prédit**, commune par commune, avec un explorateur pour fouiller chaque commune
- Les **résidus** : là où le modèle se trompe le plus
- La **leçon data** : R² en apprentissage vs validation croisée

## 💡 La leçon data science

Le modèle affiche un **R² d'environ 87 % en apprentissage**. Superbe — sauf que la **validation croisée le ramène autour de 64 %**, avec des écarts énormes d'un découpage à l'autre. Sur 67 communes seulement, le premier chiffre flatte, le second dit la vérité : la vraie capacité à généraliser est plus modeste et instable.

Le vrai travail n'est pas d'obtenir un beau chiffre, c'est de savoir à quel point lui faire confiance :

- **Petit échantillon** : 67 communes, un modèle mémorise vite et surestime sa performance.
- **Analyse agrégée** : la moyenne par commune lisse les contrastes internes (un parc et une dalle béton dans la même case).
- **Cible calculée** : l'exposition vient d'un modèle (GEOCLIMATE), pas d'un thermomètre.

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

Le CSV agrégé est déjà versionné (`data/communes_lyon.csv`), l'app fonctionne sans rien télécharger. Pour le régénérer depuis la source ouverte :

```bash
pip install requests
python data/build_dataset.py
```

Le script télécharge les 29 657 îlots via le service WFS de data.grandlyon.com, les classe par LCZ et les agrège par commune. `requests` n'est utilisé que par ce script (hors runtime de l'app).

## 📁 Structure du projet

```
├── app.py                      # App Streamlit principale
├── model/
│   └── naive_model.py          # Régression linéaire + validation croisée + résidus
├── data/
│   ├── communes_lyon.csv       # Jeu agrégé (67 communes) — versionné
│   ├── dataset.py              # Chargement du CSV + descriptions des variables
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
