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
    plot_heat_map,
    plot_gauge,
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
    """Renvoie le thème actif ('light' ou 'dark') pour adapter les graphiques.
    On se base sur le thème réellement configuré (config.toml) : c'est ce qui
    pilote l'apparence de la page, donc les graphiques restent toujours cohérents."""
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "light"

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
    f"Une simple régression linéaire explique {results.r2:.0%} des écarts de température entre quartiers. "
    "Et c'est justement pour ça qu'il faut s'en méfier."
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

st.markdown("""
Chaque point est un quartier de la métropole. Plus il est **gros et rouge**, plus il reste chaud
la nuit par rapport à la campagne autour. La logique se voit déjà à l'œil nu : ça chauffe au centre,
ça respire en périphérie. Reste à savoir si un modèle arrive vraiment à capturer ça, ou s'il fait
juste illusion.
""")

safe_plot(plot_heat_map, df, theme)

st.caption(
    "ΔT = écart de température nocturne par rapport à une zone rurale de référence, "
    "pendant un épisode de canicule (ordres de grandeur d'après les cartographies Météo-France)."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 1 : DONNÉES
# ─────────────────────────────────────────
with st.expander("📂 Données utilisées — 14 quartiers de la métropole lyonnaise", expanded=False):
    feat_desc = get_feature_descriptions()
    st.markdown("**Variables du modèle :**")
    for feat, desc in feat_desc.items():
        st.markdown(f"- `{feat}` : {desc}")
    st.dataframe(
        df[["quartier", "ndvi", "densite_pop", "dist_centre_km", "icu_reel"]]
        .rename(columns=feat_desc)
        .set_index("quartier"),
        use_container_width=True,
    )
    st.markdown("""
**D'où viennent ces chiffres ?** En toute transparence, ce sont des **valeurs reconstruites pour la
démonstration**, pas un jeu de données officiel téléchargé tel quel :

- **Densité de population** — ordres de grandeur d'après l'INSEE (recensement 2021).
- **Distance au centre** — mesurée depuis la place Bellecour.
- **NDVI (végétation)** — estimé par quartier d'après la couverture végétale connue, **pas** un calcul
  satellite pixel par pixel.
- **ΔT / ICU réel** — ordres de grandeur repris des cartographies publiques d'îlot de chaleur de
  Météo-France, **pas** des relevés de stations horodatés.

Autrement dit : les valeurs sont **plausibles et cohérentes entre elles**, mais elles servent à
illustrer une démarche, pas à publier une mesure.
""")

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
L'hypothèse : avec la végétation, la densité de population et la distance au centre, on devrait
pouvoir prédire l'écart de température d'un quartier.
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
# SECTION 2bis : SIMULATEUR (interactif)
# ─────────────────────────────────────────
st.markdown("## 🎮 À toi de jouer : invente un quartier")
st.markdown(
    "Bouge les curseurs pour composer un quartier imaginaire. Le modèle recalcule sa prédiction "
    "en direct : à toi de voir ce qui réchauffe et ce qui rafraîchit."
)

sim_left, sim_right = st.columns([1, 1])
with sim_left:
    sim_ndvi = st.slider("🌿 Végétation (NDVI)", 0.0, 0.8, 0.20, 0.01,
                         help="0 = béton nu, 0.8 = très végétalisé")
    sim_dens = st.slider("👥 Densité de population (hab/km²)", 0, 25000, 10000, 500)
    sim_dist = st.slider("📍 Distance au centre (km)", 0.0, 12.0, 3.0, 0.5)

X_sim = pd.DataFrame([[sim_ndvi, sim_dens, sim_dist]], columns=FEATURES)
pred_sim = float(model.predict(scaler.transform(X_sim))[0])
pred_affiche = max(pred_sim, 0.0)

with sim_right:
    safe_plot(plot_gauge, pred_affiche, theme, 7.0)

plus_chauds = int((df["icu_reel"] < pred_sim).sum())
ecart_moyen = pred_sim - df["icu_reel"].mean()
st.markdown(
    f"Avec ces réglages, le modèle prédit **{pred_affiche:.1f} °C** d'écart avec la campagne. "
    f"Ce quartier fictif serait plus chaud que **{plus_chauds} des {len(df)}** quartiers réels, "
    f"soit **{ecart_moyen:+.1f} °C** par rapport à la moyenne de l'échantillon."
)

# ─────────────────────────────────────────
# SECTION 3 : LA CONFRONTATION
# ─────────────────────────────────────────
st.markdown("## 🔬 La confrontation : réel vs prédit")

st.markdown("""
Le modèle suit bien la tendance générale : centre dense et chaud, périphérie végétalisée et fraîche.
Mais une moyenne cache les détails. Quartier par quartier, un cas sort du lot, et c'est le plus
intéressant à regarder.
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
# SECTION 3bis : EXPLORER UN QUARTIER (interactif)
# ─────────────────────────────────────────
st.markdown("### 🔎 Explore un quartier toi-même")
st.caption("Choisis un quartier pour voir ce que le modèle prédit, et de combien il se trompe.")

choix = st.selectbox(
    "Quartier",
    df_out["quartier"].tolist(),
    label_visibility="collapsed",
)
row = df_out[df_out["quartier"] == choix].iloc[0]
predit = row["icu_reel"] - row["residual"]

m1, m2, m3 = st.columns(3)
m1.metric("ΔT réel", f"{row['icu_reel']:.1f} °C")
m2.metric("ΔT prédit", f"{predit:.1f} °C")
m3.metric(
    "Écart du modèle", f"{row['residual']:+.1f} °C",
    help="Positif = le modèle sous-estime la chaleur ; négatif = il la surestime",
)

f1, f2, f3 = st.columns(3)
f1.metric("🌿 Végétation (NDVI)", f"{row['ndvi']:.2f}")
f2.metric("👥 Densité", f"{row['densite_pop']:,.0f} hab/km²".replace(",", " "))
f3.metric("📍 Distance au centre", f"{row['dist_centre_km']:.1f} km")

if row["is_outlier"]:
    st.warning(f"⚠️ {choix} fait partie des quartiers que le modèle prédit mal (écart > 0,8 °C).")
else:
    st.success(f"✅ Sur {choix}, le modèle tombe juste (écart < 0,8 °C).")

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
**{results.r2:.0%} de variance expliquée avec 3 variables, ça en jette.** Mais sur 14 quartiers
seulement, ce chiffre veut moins dire qu'on croit : quelques points bien alignés suffisent à le
gonfler. Et les {(1 - results.r2):.0%} qui manquent dépendent de phénomènes qu'aucune colonne
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

C'est ce que font Météo-France et les chercheurs avec des modèles comme **MApUCE** ou **TEB**. Ça
demande des années de calibration, pas une régression linéaire sur 3 colonnes.
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
