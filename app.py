"""
App Streamlit — Chaleur nocturne à Lyon, au grain fin.

Deux histoires en une :
  1. Le vrai visage de l'îlot de chaleur urbain, îlot par îlot (~29 657 points réels).
  2. Le piège de l'agrégation : en résumant chaque commune à un chiffre, on obtient
     un R² flatteur… en jetant 99,8 % du signal.
Données ouvertes : Métropole de Lyon, « Îlots de chaleur urbains » (GEOCLIMATE / LCZ).
"""

import streamlit as st
import pandas as pd
import numpy as np

from data.dataset import get_communes_data, get_ilots_data
from model.naive_model import train_naive_model, identify_outliers, get_model_formula, FEATURES
from utils.viz import (
    plot_ilots_map,
    plot_commune_ilots,
    plot_heat_map,
    plot_expo_by_lcz,
    plot_top_communes,
    plot_gauge,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Chaleur nocturne à Lyon — au grain fin",
    page_icon="🌡️",
    layout="wide",
)


def safe_plot(fig_func, *args, **kwargs):
    """Affiche un graphique en isolant les erreurs, pour ne jamais casser la page."""
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
def load_ilots():
    return get_ilots_data()


@st.cache_data(show_spinner=False)
def load_and_train():
    df = get_communes_data()
    model, scaler, results = train_naive_model(df)
    df_out = identify_outliers(df, results.residuals)
    return df, model, scaler, results, df_out


@st.cache_data(show_spinner=False)
def aggregation_stats(df_ilots: pd.DataFrame):
    """Chiffres qui illustrent la perte d'information due à l'agrégation."""
    grp = df_ilots.groupby("commune")["expo_score"]
    spans = grp.max() - grp.min()
    n_full = int(((grp.min() <= -1) & (grp.max() >= 2)).sum())
    return {
        "n_ilots": len(df_ilots),
        "span_median": float(spans.median()),
        "n_full": n_full,
        "n_communes": int(grp.ngroups),
    }


try:
    df_ilots = load_ilots()
    df, model, scaler, results, df_out = load_and_train()
    agg = aggregation_stats(df_ilots)
except Exception:
    st.error(
        "Impossible de charger les données. "
        "Vérifiez l'installation des dépendances avec `pip install -r requirements.txt`."
    )
    st.stop()

theme = get_active_theme()
n_ilots_fmt = f"{agg['n_ilots']:,}".replace(",", " ")

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🌡️ La nuit, Lyon ne refroidit pas partout pareil")
st.subheader(
    "Pendant la dernière canicule, je me suis demandé pourquoi certains coins de Lyon restaient "
    "étouffants la nuit quand d'autres respiraient. J'ai pris les vraies données de la Métropole, "
    "îlot par îlot. Voici ce qu'elles montrent."
)

st.caption(
    "Par [Badreddine EL KHAMLICHI](https://badreddineek.com) · ingénieur en mathématiques appliquées, Lyon · "
    "[Portfolio](https://portfolio.badreddineek.com) · "
    "[Code source](https://github.com/BadreddineEK/canicule-lyon-icu-model)"
)

st.divider()

# ─────────────────────────────────────────
# SECTION 1 : LA CARTE, AU GRAIN FIN (le visuel qui claque)
# ─────────────────────────────────────────
st.markdown("## 🗺️ Le vrai visage de l'îlot de chaleur")

st.markdown(
    f"Chaque point est un **îlot réel** — {n_ilots_fmt} au total, sans rien lisser. "
    "Rouge : la chaleur reste piégée la nuit. Bleu : ça se rafraîchit. La Presqu'île et la Part-Dieu "
    "brûlent, les Monts d'Or et les bords de Saône soufflent. Le phénomène n'est pas flou : il se lit "
    "rue par rue."
)

safe_plot(plot_ilots_map, df_ilots, theme)

st.caption(
    "Source : « Îlots de chaleur urbains », Métropole de Lyon (data.grandlyon.com), calculés avec le "
    "logiciel scientifique GEOCLIMATE — classification en Local Climate Zones et exposition à la "
    "chaleur. Licence Ouverte Etalab. Zoome et survole : la donnée tient à l'échelle du pâté de maisons."
)

st.markdown(
    "Ce que la carte dit tout de suite : **le bâti compact chauffe, la végétation et l'eau rafraîchissent**. "
    "Rien d'ésotérique — les immeubles serrés stockent la chaleur du jour et la relâchent la nuit. "
    "Mais regarde bien : même dans un arrondissement « chaud », il reste des poches bleues. "
    "Cette granularité, c'est toute l'histoire de la suite."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 2 : LE PIÈGE DE L'AGRÉGATION
# ─────────────────────────────────────────
st.markdown("## 🪤 Le piège que je me suis tendu tout seul")

st.markdown(
    "Première réaction de data scientist : *« résumons chaque commune par un chiffre et modélisons ça »*. "
    f"Je suis donc passé de **{n_ilots_fmt} îlots à {agg['n_communes']} communes**, "
    "une moyenne par commune. Voici la même réalité, une fois moyennée :"
)

col_map, col_model = st.columns([3, 2])
with col_map:
    safe_plot(plot_heat_map, df, theme)
with col_model:
    st.markdown("#### Une régression bien propre")
    m1, m2 = st.columns(2)
    m1.metric("R² en apprentissage", f"{results.r2:.0%}")
    m2.metric("R² validation croisée", f"{results.r2_cv:.0%}",
              delta=f"{results.r2_cv - results.r2:+.0%}")
    st.metric("Erreur moyenne (MAE)", f"{results.mae:.2f}")
    st.info(f"`{get_model_formula(results.coefficients, results.intercept)}`", icon="📐")
    st.markdown(
        f"**{results.r2:.0%}** de variance expliquée. Sur le papier, on dirait la maîtrise. "
        "Sauf que ce chiffre est un mirage."
    )

st.warning(
    f"**En moyennant, je résume {n_ilots_fmt} îlots à seulement {agg['n_communes']} points.** "
    "Un R² de 87 % sur 67 points, c'est facile à décrocher — et la validation croisée le confirme : "
    f"testé sur des communes jamais vues, le modèle tombe à **{results.r2_cv:.0%}**, très instable "
    f"(de {min(results.cv_scores):.0%} à {max(results.cv_scores):.0%} selon le découpage).",
    icon="⚠️",
)

st.markdown(
    f"Pire : la moyenne **efface le contraste interne**. La commune médiane s'étale sur **toute "
    f"l'échelle** (écart de {agg['span_median']:.0f} points, du plus frais au plus chaud), et "
    f"**{agg['n_full']} des {agg['n_communes']}** communes contiennent à la fois des îlots "
    "rafraîchissants et des îlots fortement exposés. Les résumer par un seul nombre, c'est mélanger "
    "un parc et une dalle de béton dans la même case. La preuve, en zoomant sur une seule commune :"
)

# ─────────────────────────────────────────
# SECTION 2bis : EXPLORATEUR (preuve du contraste interne)
# ─────────────────────────────────────────
communes_tries = df.sort_values("expo_nuit_score", ascending=False)["commune"].tolist()
default_idx = communes_tries.index("Lyon 7e Arrondissement") if "Lyon 7e Arrondissement" in communes_tries else 0
choix = st.selectbox("🔎 Choisis une commune et regarde ses îlots", communes_tries, index=default_idx)

d_com = df_ilots[df_ilots["commune"] == choix]
row = df[df["commune"] == choix].iloc[0]

col_cmap, col_cstat = st.columns([3, 2])
with col_cmap:
    safe_plot(plot_commune_ilots, df_ilots, choix, theme)
with col_cstat:
    st.markdown(f"#### {choix}")
    st.metric("Moyenne (le chiffre agrégé)", f"{row['expo_nuit_score']:+.2f}")
    s1, s2 = st.columns(2)
    s1.metric("Îlot le plus frais", f"{d_com['expo_score'].min():+.0f}")
    s2.metric("Îlot le plus chaud", f"{d_com['expo_score'].max():+.0f}")
    st.metric("Nombre d'îlots", f"{len(d_com)}")
    ecart = d_com["expo_score"].max() - d_com["expo_score"].min()
    if ecart >= 2:
        st.markdown(
            "👉 À l'intérieur de cette seule commune, l'écart va **du rafraîchissant au fortement "
            "exposé**. La moyenne ne le raconte pas."
        )
    else:
        st.markdown("👉 Ici, les îlots sont plus homogènes — la moyenne est moins trompeuse.")

st.divider()

# ─────────────────────────────────────────
# SECTION 3 : CE QUE LA MOYENNE LAISSE QUAND MÊME VOIR
# ─────────────────────────────────────────
st.markdown("## 🔍 La tendance de fond, au grain fin cette fois")

st.markdown(
    "L'agrégation gonfle la précision, pas le sens. En revenant aux îlots — sans rien moyenner — la "
    "relation reste nette : le bâti compact chauffe, la végétation et l'eau rafraîchissent. Mais "
    "mesurée honnêtement, îlot par îlot, la corrélation est **modérée (r ≈ 0,59)** — loin des 87 % "
    "que l'agrégation laissait croire. La vraie force du lien, la voici :"
)

col_lcz, col_top = st.columns(2)
with col_lcz:
    safe_plot(plot_expo_by_lcz, df_ilots, theme)
with col_top:
    safe_plot(plot_top_communes, df, 15, theme)

st.caption(
    "Note d'honnêteté : la cible (l'exposition) est elle-même calculée par GEOCLIMATE à partir de la "
    "forme urbaine. La relation « bâti compact → chaleur » est donc en partie encodée dans la donnée : "
    "je la visualise, je ne la « découvre » pas. C'est un travail de lecture de données, pas une preuve causale."
)

# ─────────────────────────────────────────
# SECTION 3bis : SIMULATEUR
# ─────────────────────────────────────────
st.markdown("### 🎮 La relation, en un curseur")
st.markdown(
    "Fais varier la composition d'une commune fictive et regarde le score que la donnée lui associerait."
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

plus_chaudes = int((df["expo_nuit_score"] < pred_sim).sum())
st.markdown(
    f"Score associé : **{pred_sim:+.2f}**. Une telle commune serait plus exposée la nuit que "
    f"**{plus_chaudes} des {len(df)}** communes réelles."
)

st.divider()

# ─────────────────────────────────────────
# SECTION 4 : LA LEÇON
# ─────────────────────────────────────────
st.markdown("## 💡 Ce que je retiens")

st.markdown(f"""
1. **Sur la ville** : à Lyon, la nuit de canicule ne se joue pas à l'échelle du quartier mais du
   pâté de maisons. Densifier sans végétaliser fabrique des poches invivables — et il en existe
   jusque dans les communes réputées « fraîches ».

2. **Sur la data** : mon plus beau chiffre, **{results.r2:.0%}**, était le plus trompeur. Il venait
   d'avoir agrégé {n_ilots_fmt} mesures en {agg['n_communes']} points. La validation croisée
   ({results.r2_cv:.0%}, instable) et le grain fin l'ont démasqué. Un bon résultat n'est pas un grand
   R², c'est un chiffre dont on connaît les limites.
""")

col_e, col_f, col_g = st.columns(3)
with col_e:
    st.markdown(
        "#### 📏 Le grain compte\n"
        "La bonne échelle d'analyse n'est pas la plus pratique, c'est celle où vit le phénomène. "
        "Ici, l'îlot, pas la commune."
    )
with col_f:
    st.markdown(
        "#### 🪤 Méfiance du beau R²\n"
        "Agréger gonfle mécaniquement les corrélations. Un chiffre flatteur mérite toujours qu'on "
        "cherche ce qu'il cache."
    )
with col_g:
    st.markdown(
        "#### 🛰️ Savoir d'où vient la donnée\n"
        "L'exposition est un indicateur calculé par GEOCLIMATE, pas un thermomètre. Je le lis pour "
        "ce qu'il est, sans surinterpréter."
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
des dashboards et des analyses, et je partage ma manière de raisonner au passage.
""")
with cta_right:
    st.markdown(
        "🌐 **[badreddineek.com](https://badreddineek.com)**\n\n"
        "🧑‍💻 **[Mon portfolio](https://portfolio.badreddineek.com)**\n\n"
        "⭐ **[Le code sur GitHub](https://github.com/BadreddineEK/canicule-lyon-icu-model)**"
    )

st.divider()
st.caption(
    "🌡️ Projet d'analyse — Données : Métropole de Lyon, « Îlots de chaleur urbains » (GEOCLIMATE / LCZ), "
    "Licence Ouverte Etalab · "
    "Code : [github.com/BadreddineEK/canicule-lyon-icu-model](https://github.com/BadreddineEK/canicule-lyon-icu-model)"
)
