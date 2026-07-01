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

# ─────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = get_quartiers_data()
    model, scaler, results = train_naive_model(df)
    df_out = identify_outliers(df, results.residuals)
    return df, model, scaler, results, df_out

df, model, scaler, results, df_out = load_and_train()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🌡️ Îlot de Chaleur Urbain à Lyon")
st.subheader("J'ai essayé de prédire les zones les plus chaudes. Mon modèle s'est trompé — voici pourquoi.")

st.markdown("""
> **Contexte** : Pendant la canicule de juin 2026, l'écart de température entre la Presqu'île et Caluire pouvait 
> dépasser **4°C la nuit**. J'ai voulu reproduire ça avec un modèle simple. 
> Ce dashboard montre les résultats... et leurs limites.
""")

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

st.info(
    f"**Formule** : `{get_model_formula(results.coefficients, results.intercept)}`",
    icon="📐"
)

st.markdown("""
**Hypothèse naïve** : si on connaît la végétation, la densité de population et la distance au centre,
on peut prédire l'écart de température. Simple, non ?
""")

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_feature_importance(results.coefficients), use_container_width=True)
with col_b:
    st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

# ─────────────────────────────────────────
# SECTION 3 : LA CONFRONTATION
# ─────────────────────────────────────────
st.markdown("## 🔬 La confrontation : réel vs prédit")

st.markdown("""
Le modèle prédit plutôt bien les cas extrêmes (centre-ville chaud, périphérie rurale fraîche).
Mais il se plante sur plusieurs quartiers intermédiaires — et c'est là que ça devient intéressant.
""")

st.plotly_chart(plot_reel_vs_predit(df, results.predictions), use_container_width=True)

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(plot_scatter_reel_vs_predit(df, results.predictions), use_container_width=True)
with col_d:
    st.plotly_chart(plot_residuals(df_out), use_container_width=True)

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

# ─────────────────────────────────────────
# SECTION 5 : LA LEÇON
# ─────────────────────────────────────────
st.markdown("## 💡 Pourquoi le modèle simple ne suffit pas")

st.markdown(f"""
Le modèle naïf explique **{results.r2:.0%} de la variance thermique** entre quartiers. 
Pas mal pour 3 variables. Mais les {(1 - results.r2):.0%} restants dépendent de phénomènes 
qu'aucun fichier CSV classique ne capture :
""")

col_e, col_f, col_g = st.columns(3)
with col_e:
    st.markdown("""#### 🌬️ Flux nocturnes
    La nuit, la chaleur emmagasinée dans les matériaux (bitume, béton) 
    est relâchée différemment selon l'**albédo** des surfaces. 
    Deux rues avec la même densité peuvent avoir des températures nocturnes très différentes.
    """)
with col_f:
    st.markdown("""#### 🏙️ Morphologie urbaine
    L'**orientation des rues**, la hauteur des bâtiments et le ratio H/W (hauteur/largeur de rue) 
    créent des « canyons urbains » qui emprisonnent la chaleur indépendamment de la végétation ou de la densité.
    """)
with col_g:
    st.markdown("""#### 💧 Présence d'eau
    La proximité de la Saône ou du Rhône crée des corridors de fraîcheur 
    non captés par un simple indice de végétation. 
    Confluence devrait être plus chaude selon le modèle — en réalité, 
    l'eau atténue significativement l'ICU.
    """)

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

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("""
🌡️ Projet pédagogique — Données ICU : publications Météo-France (Licence Ouverte Etalab) | 
Population : INSEE 2021 | NDVI : estimation par quartier | 
Code source : [github.com/BadreddineEK/canicule-lyon-icu-model](https://github.com/BadreddineEK/canicule-lyon-icu-model)
""")
