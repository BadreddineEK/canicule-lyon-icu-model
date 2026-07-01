# 🌡️ Îlot de Chaleur Urbain à Lyon — Modèle Naïf vs Réalité

> **Projet LinkedIn post** — *"J'ai essayé de prédire les zones les plus chaudes de Lyon pendant la canicule. Mon modèle s'est trompé — et c'est ça qui est intéressant."*

## 🎯 Objectif

Modéliser naïvement la température ressentie dans différents quartiers de Lyon à partir de 3 variables simples, puis confronter ces prédictions à la réalité pour comprendre **pourquoi un modèle simple ne suffit pas** pour capturer l'îlot de chaleur urbain.

## 📊 Le modèle naïf

Variables utilisées :
- 🏙️ **Densité de végétation** (indice NDVI simplifié par quartier)
- 🏘️ **Densité de population** (INSEE)
- 📍 **Distance au centre-ville** (Bellecour)

Méthode : régression linéaire simple → prédiction de la température différentielle par rapport à une zone rurale de référence.

## 🔬 La confrontation

Comparaison des prédictions avec les données réelles de stations météo Météo-France et les cartographies ICU officielles.

**Spoiler** : ça rate sur plusieurs quartiers — et l'analyse du "pourquoi" est la vraie valeur pédagogique.

## 🛠️ Stack technique

- Python 3.11+
- Streamlit (dashboard interactif)
- Pandas, NumPy, Scikit-learn
- Plotly, Folium (visualisations)
- Données : Météo-France Open Data (Licence Ouverte Etalab)

## 🚀 Lancer l'app localement

```bash
git clone https://github.com/BadreddineEK/canicule-lyon-icu-model
cd canicule-lyon-icu-model
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure

```
├── app.py                  # App Streamlit principale
├── model/
│   ├── naive_model.py      # Modèle naïf (régression linéaire)
│   └── features.py         # Feature engineering
├── data/
│   ├── quartiers_lyon.py   # Données simulées des quartiers (reproductibles)
│   └── stations_ref.py     # Stations météo de référence
├── utils/
│   └── viz.py              # Helpers de visualisation
├── requirements.txt
└── README.md
```

## 💡 La leçon data science

Un modèle naïf capture ~60% de la variance thermique urbaine. Les 40% restants dépendent de variables que 3 colonnes CSV ne peuvent pas capturer : orientation des rues, albédo des matériaux, flux de chaleur nocturne, présence d'eau...

C'est ça, la vraie complexité de la modélisation climatique urbaine.

---

*Données Météo-France sous Licence Ouverte Etalab — Source : Météo-France*
