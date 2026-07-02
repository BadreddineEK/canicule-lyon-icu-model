"""
App Streamlit — Îlot de Chaleur Urbain à Lyon
Modèle naïf vs réalité : pourquoi c'est compliqué de modéliser la chaleur en ville
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.quartiers_lyon import get_quartiers_data, get_feature_descriptions
from model.naive_model import train_naive_model, identify_outliers, get_model_formula, FEATURES
from utils.viz import (
    plot_reel_vs_predit,
    plot_residuals,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_scatter_reel_vs_predit,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ICU Lyon — Modèle naïf vs réalité",
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
    """Renvoie le thème actif ('light' ou 'dark') pour adapter les graphiques."""
    try:
        return getattr(st.context.theme, "type", None) or "dark"
    except Exception:
        return "dark"

# ─────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    df = get_quartiers_data()
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

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
st.title("🌡️ Îlot de chaleur urbain à Lyon")
st.subheader(
    f"Mon modèle simple explique {results.r2:.0%} des écarts de température entre quartiers. "
    "Et c'est justement pour ça qu'il faut s'en méfier."
)

st.markdown("""
> **Le pitch** : la nuit, en pleine canicule, le centre de Lyon peut rester plusieurs degrés
> plus chaud que la campagne autour — c'est l'îlot de chaleur urbain. J'ai voulu voir jusqu'où
> une régression linéaire toute simple, sur 3 variables, pouvait reproduire ce phénomène.
> Un score flatteur, quelques ratés révélateurs, et une bonne leçon sur ce qu'un « bon » chiffre cache.
""")

st.caption(
    "Par [Badreddine EL KHAMLICHI](https://badreddineek.com) · ingénieur en mathématiques appliquées, Lyon · "
    "[Portfolio](https://portfolio.badreddineek.com) · "
    "[Code source](https://github.com/BadreddineEK/canicule-lyon-icu-model)"
)

st.divider()

# ─────────────────────────────────────────
# SECTION 1 : DONNÉES
# ─────────────────────────────────────────
with st.expander("📂 Données utilisées — 14 quartiers de la métropole lyonnaise", expanded=False):
    feat_desc = get_feature_descriptions()
    st.markdown("**Variables du modèle naïf :**")
    for feat, desc in feat_desc.items():
        st.markdown(f"- `{feat}` : {desc}")
    st.dataframe(
        df[["quartier", "ndvi", "densite_pop", "dist_centre_km", "icu_reel"]]
        .rename(columns=feat_desc)
        .set_index("quartier"),
        use_container_width=True,
    )
    st.caption("Sources : Météo-France (ICU), INSEE 2021 (population), estimation NDVI par quartier")

# ─────────────────────────────────────────
# SECTION 2 : LE MODÈLE NAÏF
# ─────────────────────────────────────────
st.markdown("## 🧮 Le modèle naïf")

col1, col2, col3 = st.columns(3)
col1.metric("R² du modèle", f"{results.r2:.1%}", help="Part de variance expliquée")
col2.metric("Erreur moyenne (MAE)", f"{results.mae:.2f} °C", help="Erreur absolue moyenne sur les prédictions")
col3.metric("Quartiers mal prédits", f"{df_out['is_outlier'].sum()} / {len(df)}",
            help="Quartiers avec erreur > 0.8°C")

st.caption(
    "Le **R²** indique la part de la réalité que le modèle explique (100 % = prédiction parfaite). "
    "La **MAE** est l'erreur moyenne, en degrés. « Mal prédit » = écart supérieur à 0,8 °C."
)

st.info(
    f"**Formule** : `{get_model_formula(results.coefficients, results.intercept)}`",
    icon="📐"
)

st.markdown("""
**Hypothèse naïve** : si on connaît la végétation, la densité de population et la distance au centre,
on peut prédire l'écart de température. Simple, non ?
""")

st.caption(
    "Le **NDVI** est un indice de végétation entre 0 (béton nu) et 1 (forêt dense), estimé ici par quartier."
)

col_a, col_b = st.columns(2)
with col_a:
    safe_plot(plot_feature_importance, results.coefficients, theme)
with col_b:
    safe_plot(plot_correlation_matrix, df, theme)

# ─────────────────────────────────────────
# SECTION 3 : LA CONFRONTATION
# ─────────────────────────────────────────
st.markdown("## 🔬 La confrontation : réel vs prédit")

st.markdown("""
Le modèle suit bien la tendance générale : centre dense et chaud, périphérie végétalisée et fraîche.
Mais un chiffre global lisse toujours les détails. En regardant quartier par quartier, un cas sort
nettement du lot — et c'est le plus instructif.
""")

st.caption(
    "Un **résidu** est l'écart entre la température réelle et celle prédite : "
    "positif = le modèle a sous-estimé la chaleur, négatif = il l'a surestimée."
)

safe_plot(plot_reel_vs_predit, df, results.predictions, theme)

col_c, col_d = st.columns(2)
with col_c:
    safe_plot(plot_scatter_reel_vs_predit, df, results.predictions, theme)
with col_d:
    safe_plot(plot_residuals, df_out, theme)

# ─────────────────────────────────────────
# SECTION 4 : LES CAS PROBLÉMATIQUES
# ─────────────────────────────────────────
st.markdown("## 🔴 Là où le modèle se trompe le plus")

outliers = df_out[df_out["is_outlier"] == True]

if len(outliers) > 0:
    for _, row in outliers.iterrows():
        direction = "🔺 sous-estime" if row["residual"] > 0 else "🔻 sur-estime"
        st.warning(
            f"**{row['quartier']}** — {direction} de {abs(row['residual']):.2f}°C "
            f"(prédit : {row['icu_reel'] - row['residual']:.1f}°C / réel : {row['icu_reel']:.1f}°C)"
        )
else:
    st.success("Le modèle prédit tous les quartiers avec moins de 0.8°C d'erreur.")
st.caption(
    "Gerland est un ancien quartier industriel devenu pôle biotech, avec halles, laboratoires "
    "et grand stade. La chaleur y vient probablement aussi de l'activité et des matériaux, "
    "que la végétation, la densité et la distance au centre ne captent pas."
)
# ─────────────────────────────────────────# SECTION : LIMITES & HONNÊTETÉ SUR LES DONNÉES
# ─────────────────────────────────────
st.markdown("## ⚠️ Limites de ce modèle (à lire avant de commenter)")

st.markdown("""
Ce projet est **pédagogique** : il montre une démarche, pas un résultat de recherche. En toute transparence :

- **Les valeurs NDVI sont des estimations par quartier**, pas des mesures satellite précises. Un vrai NDVI se calcule pixel par pixel à partir d'images Sentinel ou Landsat.
- **Les écarts de température (ICU) sont des ordres de grandeur** tirés de cartographies et publications Météo-France, pas des relevés station par station horodatés.
- **14 quartiers, c'est un très petit échantillon.** Sur si peu de points, un R² élevé peut donner une fausse impression de fiabilité.
- **Le modèle est évalué sur les données qui ont servi à l'entraîner** (pas de validation croisée) : ses performances sur de nouveaux quartiers seraient plus faibles.
- **Une régression linéaire suppose des effets simples et additifs**, alors que la physique de la chaleur urbaine est fortement non linéaire.

Autrement dit : ces chiffres servent à illustrer un raisonnement, pas à guider une décision d'aménagement.
""")

st.divider()

# ─────────────────────────────────────# SECTION 5 : LA LEÇON
# ─────────────────────────────────────────
st.markdown("## 💡 Pourquoi le modèle simple ne suffit pas")

st.markdown(f"""
**{results.r2:.0%} de variance expliquée avec 3 variables, ça impressionne.** Mais sur seulement
14 quartiers, ce chiffre est plus une alerte qu'un trophée : quelques points bien alignés suffisent
à le gonfler. Et les {(1 - results.r2):.0%} qui manquent dépendent de phénomènes qu'aucune colonne
simple ne capture :
""")

col_e, col_f, col_g = st.columns(3)
with col_e:
    st.markdown(
        "#### 🌬️ Flux nocturnes\n"
        "La nuit, la chaleur stockée le jour dans le bitume et le béton se libère à un rythme "
        "qui dépend de l'**albédo** (la part de lumière qu'une surface renvoie). "
        "Deux rues aussi denses peuvent ne pas refroidir à la même vitesse."
    )
with col_f:
    st.markdown(
        "#### 🏙️ Morphologie urbaine\n"
        "L'**orientation des rues** et le rapport hauteur/largeur des bâtiments forment des "
        "« canyons urbains » qui piègent la chaleur, indépendamment de la végétation ou de la densité."
    )
with col_g:
    st.markdown(
        "#### 💧 Présence d'eau\n"
        "La proximité de la Saône ou du Rhône crée des couloirs de fraîcheur qu'un simple indice "
        "de végétation ignore. Dans les données, Confluence est d'ailleurs un peu moins chaude "
        "que ce que prédit le modèle."
    )

st.divider()

st.markdown("""### 🔜 Et maintenant ?
Un modèle plus réaliste nécessiterait :
- Des données à **haute résolution spatiale** (250m ou moins)
- L'**albédo** des surfaces (données satellitaires)
- La **morphologie urbaine** (ratio H/W, orientation)
- Les **flux de chaleur anthropique** (trafic, climatiseurs, industrie)

C'est ce que font Météo-France et les chercheurs avec des modèles comme **MApUCE** ou **TEB** — 
et ça demande des années de calibration, pas une régression linéaire sur 3 colonnes.
""")

# ─────────────────────────────────────────# APPEL À L'ACTION
# ─────────────────────────────────────
st.divider()

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

# ─────────────────────────────────────# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("""
🌡️ Projet pédagogique — Données ICU : publications Météo-France (Licence Ouverte Etalab) | 
Population : INSEE 2021 | NDVI : estimation par quartier | 
Code source : [github.com/BadreddineEK/canicule-lyon-icu-model](https://github.com/BadreddineEK/canicule-lyon-icu-model)
""")
