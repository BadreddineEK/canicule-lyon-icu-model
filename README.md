# 🌡️ Îlot de Chaleur Urbain à Lyon — Modèle naïf vs réalité

> *« J'ai voulu prédire les quartiers les plus chauds de Lyon pendant la canicule avec une simple régression linéaire. Elle s'en sort bien... jusqu'à ce qu'on regarde de près. Voici pourquoi. »*

**▶️ Démo en ligne : [canicule-lyon-icu-model.streamlit.app](https://canicule-lyon-icu-model.streamlit.app/)**

Un dashboard Streamlit qui modélise l'îlot de chaleur urbain (ICU) à Lyon avec une simple régression linéaire, puis confronte les prédictions à la réalité pour montrer **pourquoi un modèle simple ne suffit pas** pour capturer la chaleur en ville.

## 🎯 L'idée

À partir de 3 variables faciles à obtenir, peut-on prédire l'écart de température entre un quartier et la campagne pendant une nuit de canicule ?

- 🌳 **Végétation** (indice NDVI estimé par quartier, entre 0 = béton et 1 = forêt)
- 🏘️ **Densité de population** (INSEE, proxy de l'artificialisation)
- 📍 **Distance au centre-ville** (Bellecour)

Méthode : régression linéaire sur variables standardisées, puis analyse des écarts (résidus) entre prédiction et réalité.

## 🔬 Ce que montre le dashboard

- La **formule** du modèle et le poids de chaque variable
- Le **réel vs prédit** quartier par quartier
- Les **résidus** : où le modèle se trompe le plus (l'écart entre réalité et prédiction)
- Une section **« Limites du modèle »** qui assume la fragilité des données et de la méthode

Le modèle explique une grande part de la variance sur cet échantillon, mais c'est justement là que se cache le piège : **14 quartiers, c'est trop peu pour conclure**, et les données sont des estimations, pas des mesures satellite. Le dashboard l'explique en toute transparence.

## 🛠️ Stack technique

- Python 3.11+
- Streamlit (dashboard interactif)
- Pandas, NumPy, Scikit-learn (modèle)
- Plotly (visualisations)

## 🚀 Lancer l'app en local

```bash
git clone https://github.com/BadreddineEK/canicule-lyon-icu-model
cd canicule-lyon-icu-model
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre ensuite dans le navigateur (par défaut sur http://localhost:8501).

## 📁 Structure du projet

```
├── app.py                      # App Streamlit principale
├── model/
│   └── naive_model.py          # Modèle naïf (régression linéaire) + détection des écarts
├── data/
│   └── quartiers_lyon.py       # Données des 14 quartiers (reproductibles)
├── utils/
│   └── viz.py                  # Graphiques Plotly (charte commune)
├── .streamlit/config.toml      # Thème de l'app
├── requirements.txt
├── LICENSE
└── README.md
```

## 💡 La leçon data science

Sur un si petit jeu de données, un R² élevé est plus une **alerte** qu'une victoire. Les phénomènes qui pilotent vraiment la chaleur urbaine — orientation des rues, albédo des matériaux, canyons urbains, présence d'eau, flux de chaleur nocturne — ne tiennent pas dans 3 colonnes de CSV.

C'est ça, la vraie complexité de la modélisation climatique urbaine : les modèles sérieux (MApUCE, TEB de Météo-France) demandent des années de calibration, pas une régression linéaire.

## ⚠️ Honnêteté sur les données

Projet **pédagogique**. Les valeurs NDVI sont des estimations par quartier (pas des mesures satellite pixel par pixel), et les écarts de température sont des ordres de grandeur tirés de publications Météo-France, pas des relevés station par station. Les chiffres illustrent un raisonnement, ils ne guident aucune décision d'aménagement.

## 📄 Licence

MIT — voir [LICENSE](LICENSE). Données ICU : publications Météo-France (Licence Ouverte Etalab). Population : INSEE 2021.

---

*Badreddine EL KHAMLICHI · Ingénieur en mathématiques appliquées · Lyon*

