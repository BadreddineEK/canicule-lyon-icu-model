"""
App Streamlit — Îlot de chaleur urbain à Lyon (vraies données ouvertes).
Ce qui explique la chaleur nocturne d'une commune, et pourquoi un beau R²
peut cacher un modèle bien plus fragile qu'il n'en a l'air.
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.dataset import get_communes_data, get_feature_descriptions
from model.naive_model import train_naive_model, identify_outliers, get_model_formula, FEATURES
from utils.viz import (
    plot_top_communes,
    plot_residuals,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_scatter_reel_vs_predit,
    plot_heat_map,
    plot_gauge,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Chaleur urbaine à Lyon — vraies données",
    page_icon="🌡️",
    layout="wide",
)


def safe_plot(fig_func, *args, **kwargs):
    """Affiche un graphique en isolant les erreurs : un souci sur un graphe
    n'interrompt pas toute la page."""
    try:
        st.plotly_chart(fig_func(*args, **kwargs), use_container_width=True)
    except Exception:
        st.warning("Ce graphique n'a pas pu être généré. Le reste de l'analyse reste disponible.")


def get_active_theme() -> str:
    """Renvoie le thème configuré ('light' ou 'dark') pour adapter les graphiques."""
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"


@st.cache_data(show_spinner=False)
def load_and_train():
    df = get_communes_data()
    model, scaler, results = train_naive_model(df)
    df_out = identify_outliers(df, results.residuals)
    return df, model, scaler, results, df_out


try:
    df, model, scaler, results, df_out = load_and_train()
except Exception:
    st.error(
        "Impossible de charger les données ou d'entraîner le modèle. "
        "Vérifiez l'installation des dépendances avec `pip install -r requirements.txt`."
    )
    st.stop()

theme = get_active_theme()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🌡️ Îlot de chaleur urbain à Lyon")
st.subheader(
    "Pendant la canicule, pourquoi certaines communes restent invivables la nuit "
    "quand d'autres respirent ? J'ai creusé les vraies données de la Métropole."
)

st.caption(
    "Par [Badreddine EL KHAMLICHI](https://badreddineek.com) · ingénieur en mathématiques appliquées, Lyon · "
    "[Portfolio](https://portfolio.badreddineek.com) · "
    "[Code source](https://github.com/BadreddineEK/canicule-lyon-icu-model)"
)

st.divider()

# ─────────────────────────────────────────
# SECTION 0 : LE PHÉNOMÈNE, VU DU CIEL
# ─────────────────────────────────────────
st.markdown("## 🗺️ Le phénomène, vu du ciel")

st.markdown(
    "Chaque point est une commune ou un arrondissement. Plus il tire vers le rouge, plus il reste "
    "chaud la nuit en été. La Presqu'île et Part-Dieu cuisent ; les Monts d'Or et les bords de Saône "
    "gardent la fraîcheur. La logique se voit déjà : c'est une histoire de forme urbaine."
)

safe_plot(plot_heat_map, df, theme)

st.caption(
    "Données : « Îlots de chaleur urbains » de la Métropole de Lyon, calculées avec le logiciel "
    "GEOCLIMATE (Local Climate Zones + exposition à la chaleur). 29 657 îlots réels agrégés en "
    f"{len(df)} communes. Licence Ouverte Etalab."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 1 : LES DONNÉES
# ─────────────────────────────────────────
with st.expander(f"📂 Les données — {len(df)} communes de la Métropole de Lyon", expanded=False):
    feat_desc = get_feature_descriptions()
    st.markdown("**Variables :**")
    for feat, desc in feat_desc.items():
        st.markdown(f"- `{feat}` : {desc}")
    st.dataframe(
        df[["commune", "pct_vegetation", "pct_compact", "pct_mineral", "expo_nuit_score"]]
        .set_index("commune"),
        use_container_width=True,
    )
    st.markdown("""
**D'où viennent ces chiffres ?** Tout vient d'un seul jeu de données ouvert et officiel :
la couche **« Îlots de chaleur urbains »** de [data.grandlyon.com](https://data.grandlyon.com),
produite avec le logiciel scientifique **GEOCLIMATE** (méthode de l'Institut Paris Région).

- Chaque îlot est classé en **Local Climate Zone (LCZ)** — la typologie internationale de forme
  urbaine (bâti compact, bâti ouvert, végétation, eau, sol minéral…).
- Chaque îlot reçoit une **exposition nocturne à la chaleur** (de « effet rafraîchissant » à « fort »).
- J'ai agrégé les **29 657 îlots** par commune (pondéré par la surface) pour obtenir, d'un côté la
  composition du sol, de l'autre un **score d'exposition** moyen — la cible du modèle.

Ce sont donc de **vraies mesures d'aménagement**, pas des relevés de thermomètre : l'exposition est
un indicateur calculé, à lire comme tel.
""")

# ─────────────────────────────────────────
# SECTION 2 : CE QUI EXPLIQUE LA CHALEUR
# ─────────────────────────────────────────
st.markdown("## 🔍 Ce qui explique la chaleur")

st.markdown(
    "Question prise à l'endroit : qu'est-ce qui fait qu'une commune surchauffe la nuit ? "
    "On confronte trois ingrédients de sa surface — la végétation, le bâti compact et le sol "
    "minéral — au score d'exposition."
)

col_a, col_b = st.columns(2)
with col_a:
    safe_plot(plot_feature_importance, results.coefficients, theme)
with col_b:
    safe_plot(plot_correlation_matrix, df, theme)

st.markdown(
    "Le verdict est net et cohérent avec la physique : **le bâti compact est le moteur numéro un** "
    "de la chaleur nocturne (les immeubles serrés stockent la chaleur du jour et la relâchent la nuit). "
    "La végétation, elle, tire vers le frais. Rien de révolutionnaire — et c'est justement bon signe : "
    "le modèle retrouve ce que dit la science."
)

# ─────────────────────────────────────────
# SECTION 3 : LE MODÈLE (ET SON PIÈGE)
# ─────────────────────────────────────────
st.markdown("## 🧮 Le modèle, et son piège")

c1, c2, c3 = st.columns(3)
c1.metric("R² en apprentissage", f"{results.r2:.0%}",
          help="Sur les données qui ont servi à l'entraîner")
c2.metric("R² en validation croisée", f"{results.r2_cv:.0%}",
          delta=f"{results.r2_cv - results.r2:+.0%}",
          help="Sur des communes que le modèle n'a pas vues à l'entraînement")
c3.metric("Erreur moyenne (MAE)", f"{results.mae:.2f}", help="Erreur absolue moyenne, en points de score")

st.info(f"**Formule** : `{get_model_formula(results.coefficients, results.intercept)}`", icon="📐")

st.markdown(
    f"C'est là que ça devient intéressant. En apprentissage, le modèle affiche **{results.r2:.0%}** : "
    f"superbe. Mais testé sur des communes qu'il n'a jamais vues, il tombe à **{results.r2_cv:.0%}** en "
    f"moyenne — avec des écarts énormes d'un découpage à l'autre "
    f"(de {min(results.cv_scores):.0%} à {max(results.cv_scores):.0%}). "
    "Sur 67 communes seulement, le premier chiffre flatte, le second dit la vérité : la vraie capacité "
    "à généraliser est plus modeste et surtout instable."
)

# ─────────────────────────────────────────
# SECTION 3bis : SIMULATEUR (interactif)
# ─────────────────────────────────────────
st.markdown("### 🎮 À toi de jouer : compose une commune")
st.markdown(
    "Fais varier la composition du sol et regarde le modèle recalculer l'exposition nocturne en direct."
)

sim_left, sim_right = st.columns([1, 1])
with sim_left:
    sim_veg = st.slider("🌿 Végétation (% de surface)", 0, 60, 10, 1)
    sim_comp = st.slider("🏙️ Bâti compact (% de surface)", 0, 60, 20, 1)
    sim_min = st.slider("🪨 Sol minéral (% de surface)", 0, 30, 6, 1)

X_sim = pd.DataFrame([[sim_veg, sim_comp, sim_min]], columns=FEATURES)
pred_sim = float(model.predict(scaler.transform(X_sim))[0])

with sim_right:
    safe_plot(plot_gauge, pred_sim, theme)

plus_exposees = int((df["expo_nuit_score"] < pred_sim).sum())
st.markdown(
    f"Avec ces réglages, le modèle prédit un score de **{pred_sim:+.2f}**. Cette commune fictive serait "
    f"plus exposée à la chaleur nocturne que **{plus_exposees} des {len(df)}** communes réelles."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 4 : RÉEL VS PRÉDIT
# ─────────────────────────────────────────
st.markdown("## 🔬 Réel vs prédit, commune par commune")

st.markdown(
    "Chaque point est une commune : plus elle est proche de la diagonale, mieux elle est prédite. "
    "La tendance est bonne, mais quelques communes s'écartent nettement — les plus instructives."
)

col_c, col_d = st.columns(2)
with col_c:
    safe_plot(plot_scatter_reel_vs_predit, df, results.predictions, theme)
with col_d:
    safe_plot(plot_top_communes, df, 15, theme)

# Explorateur de commune
st.markdown("### 🔎 Explore une commune")
choix = st.selectbox("Commune", df_out["commune"].tolist(), label_visibility="collapsed")
row = df_out[df_out["commune"] == choix].iloc[0]
predit = row["expo_nuit_score"] - row["residual"]

m1, m2, m3 = st.columns(3)
m1.metric("Exposition réelle", f"{row['expo_nuit_score']:+.2f}")
m2.metric("Exposition prédite", f"{predit:+.2f}")
m3.metric("Écart du modèle", f"{row['residual']:+.2f}",
          help="Positif = le modèle sous-estime la chaleur ; négatif = il la surestime")

f1, f2, f3 = st.columns(3)
f1.metric("🌿 Végétation", f"{row['pct_vegetation']:.0f} %")
f2.metric("🏙️ Bâti compact", f"{row['pct_compact']:.0f} %")
f3.metric("🪨 Sol minéral", f"{row['pct_mineral']:.0f} %")

if row["is_outlier"]:
    st.warning(f"⚠️ {choix} fait partie des communes que le modèle prédit mal.")
else:
    st.success(f"✅ Sur {choix}, le modèle tombe assez juste.")

# ─────────────────────────────────────────
# SECTION 5 : OÙ LE MODÈLE SE TROMPE
# ─────────────────────────────────────────
st.markdown("## 🔴 Là où le modèle se trompe")

safe_plot(plot_residuals, df_out, 12, theme)

st.caption(
    "Ces écarts ne sont pas des bugs : ils rappellent que trois pourcentages de surface ne suffisent "
    "pas à tout expliquer. L'orientation des rues, les matériaux, la proximité de l'eau ou l'activité "
    "humaine jouent aussi — et n'entrent pas dans le modèle."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 6 : LA LEÇON
# ─────────────────────────────────────────
st.markdown("## 💡 Ce que je retiens")

st.markdown(f"""
Deux enseignements, un technique et un métier :

1. **Sur la chaleur urbaine** : à Lyon, la forme du bâti explique l'essentiel de l'exposition nocturne.
   Densifier sans végétaliser, c'est fabriquer des nuits chaudes. La donnée le montre noir sur blanc.

2. **Sur la data** : un R² de **{results.r2:.0%}** ne veut rien dire tout seul. La validation croisée
   le ramène à **{results.r2_cv:.0%}**, très instable sur 67 communes. Le vrai travail n'est pas
   d'obtenir un beau chiffre, c'est de savoir à quel point lui faire confiance.
""")

col_e, col_f, col_g = st.columns(3)
with col_e:
    st.markdown(
        "#### 📏 Petit échantillon\n"
        "67 communes, c'est peu. Sur si peu de points, un modèle mémorise vite ses données et "
        "surestime sa vraie performance."
    )
with col_f:
    st.markdown(
        "#### 🧩 Analyse agrégée\n"
        "Travailler par commune lisse les contrastes internes : un parc et une dalle béton peuvent "
        "se retrouver dans la même moyenne."
    )
with col_g:
    st.markdown(
        "#### 🛰️ Une cible calculée\n"
        "L'exposition vient d'un modèle (GEOCLIMATE), pas d'un thermomètre. On modélise donc un "
        "indicateur, à ne pas confondre avec une mesure de terrain."
    )

st.divider()

# ─────────────────────────────────────────
# APPEL À L'ACTION
# ─────────────────────────────────────────
cta_left, cta_right = st.columns([3, 2])
with cta_left:
    st.markdown("""
### 👋 On continue ailleurs ?
Je suis **Badreddine**, ingénieur en mathématiques appliquées à Lyon. Je construis des outils data,
des dashboards et des modèles, et je partage ce que j'apprends au passage.
""")
with cta_right:
    st.markdown(
        "🌐 **[badreddineek.com](https://badreddineek.com)**\n\n"
        "🧑‍💻 **[Mon portfolio](https://portfolio.badreddineek.com)**\n\n"
        "⭐ **[Le code sur GitHub](https://github.com/BadreddineEK/canicule-lyon-icu-model)**"
    )

st.divider()
st.caption(
    "🌡️ Projet pédagogique — Données : Métropole de Lyon, « Îlots de chaleur urbains » "
    "(GEOCLIMATE), Licence Ouverte Etalab · "
    "Code : [github.com/BadreddineEK/canicule-lyon-icu-model](https://github.com/BadreddineEK/canicule-lyon-icu-model)"
)
