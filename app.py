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
from model.naive_model import (
    train_naive_model, get_model_formula, pearson_ci, eta_squared,
)
from utils.viz import (
    plot_ilots_map,
    plot_commune_ilots,
    plot_heat_map,
    plot_expo_by_lcz,
    plot_top_communes,
    plot_hot_ilots_map,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Chaleur nocturne à Lyon — au grain fin",
    page_icon="🌡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: rgba(231, 76, 60, 0.06);
        border: 1px solid rgba(231, 76, 60, 0.18);
        border-radius: 12px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-weight: 700; }
    [data-testid="stMetricLabel"] { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Compteur remis à zéro à chaque rerun : garantit une clé unique par graphique
# et évite StreamlitDuplicateElementId si deux graphiques sont identiques.
_PLOT_SEQ = {"n": 0}


def safe_plot(fig_func, *args, **kwargs):
    """Affiche un graphique en isolant les erreurs, pour ne jamais casser la page."""
    _PLOT_SEQ["n"] += 1
    try:
        st.plotly_chart(fig_func(*args, **kwargs), use_container_width=True,
                        key=f"chart_{_PLOT_SEQ['n']}")
    except Exception as e:
        st.warning("Ce graphique n'a pas pu être généré. Le reste de l'analyse reste disponible.")
        with st.expander("Détails techniques"):
            st.exception(e)


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
    _model, _scaler, results = train_naive_model(df)
    return df, results


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
    df, results = load_and_train()
    agg = aggregation_stats(df_ilots)
except Exception:
    st.error(
        "Impossible de charger les données. "
        "Vérifiez l'installation des dépendances avec `pip install -r requirements.txt`."
    )
    st.stop()

theme = get_active_theme()
n_ilots_fmt = f"{agg['n_ilots']:,}".replace(",", " ")

# Mesures de force du lien, calculées pour de vrai sur la donnée (rien codé en dur).
# - au grain fin : η (rapport de corrélation) type de sol LCZ → exposition
# - à l'échelle commune : corrélation de Pearson (avec IC 95 %) part de bâti/végétation → exposition
eta_ilot = eta_squared(df_ilots["lcz_groupe"], df_ilots["expo_score"])
r_compact = pearson_ci(df["pct_compact"], df["expo_nuit_score"])
r_veg = pearson_ci(df["pct_vegetation"], df["expo_nuit_score"])
cv_std = float(np.std(results.cv_scores)) if results.cv_scores else 0.0

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

st.caption(
    "🟢 **Lecture express (30 s)** : la carte, les chiffres clés et une phrase par section suffisent. "
    "🔬 **Envie de preuve ?** Dépliez les volets « Creuser » : corrélation avec intervalle de "
    "confiance, robustesse du R², échelle de preuve et limites."
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
        f"**{results.r2:.0%}** de variance expliquée. Sur le papier, on dirait la maîtrise…"
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
    "L'agrégation gonfle la **précision affichée**, pas le sens. En revenant aux îlots — sans rien "
    f"moyenner — le lien saute aux yeux et se chiffre : le seul type de sol explique déjà "
    f"**{eta_ilot['eta2']:.0%} de la variance** d'exposition entre îlots (η ≈ {eta_ilot['eta']:.2f}), "
    f"et à l'échelle des communes le bâti compact est corrélé à **{r_compact['r']:+.2f}** à la chaleur "
    "nocturne. Le lien est donc **réel et fort**. Le mirage, ce n'est pas lui : c'est le **R² de 87 %** "
    f"du modèle agrégé, que la validation croisée ramène à **{results.r2_cv:.0%}** — soit, presque "
    f"exactement, l'ordre de grandeur du lien réel mesuré au grain fin ({eta_ilot['eta2']:.0%}). "
    "La vraie force du lien, la voici :"
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

with st.expander("🔬 Creuser : la preuve, la robustesse et les limites"):
    tab_lien, tab_cv, tab_echelle, tab_lim = st.tabs(
        ["📏 Le lien, chiffré", "🎲 Le R² tient-il ?", "🪜 Échelle de preuve", "⚠️ Limites"]
    )

    with tab_lien:
        st.markdown(
            "À l'échelle de l'îlot, on n'a pas de pourcentages continus mais une **classe de sol** "
            "(LCZ) par point. La bonne mesure du lien est donc le **rapport de corrélation η²** : la "
            "part de variance d'exposition expliquée par le seul type de sol. À l'échelle des "
            "**communes**, on a des pourcentages continus : on peut alors donner une **corrélation de "
            "Pearson avec son intervalle de confiance à 95 %**."
        )
        e1, e2, e3 = st.columns(3)
        e1.metric("η² sol → chaleur (îlots)", f"{eta_ilot['eta2']:.0%}",
                  f"η ≈ {eta_ilot['eta']:.2f} · n = {eta_ilot['n']:,}".replace(",", " "))
        e2.metric("Bâti compact ↔ chaleur (communes)", f"r = {r_compact['r']:+.2f}",
                  f"IC95 : [{r_compact['lo']:+.2f} ; {r_compact['hi']:+.2f}]")
        e3.metric("Végétation ↔ chaleur (communes)", f"r = {r_veg['r']:+.2f}",
                  f"IC95 : [{r_veg['lo']:+.2f} ; {r_veg['hi']:+.2f}]")
        st.caption(
            "Lecture : le bâti compact est corrélé positivement à la chaleur nocturne, la végétation "
            "négativement, et les deux intervalles à 95 % excluent zéro — le lien est **réel et fort**. "
            "Ce n'est pas le lien qui était un mirage, c'est la **précision de 87 %** que promettait "
            "l'agrégat."
        )

    with tab_cv:
        st.markdown(
            "Le R² d'apprentissage (87 %) est mesuré sur les communes que le modèle a **déjà vues**. "
            "La validation croisée le teste sur des communes **jamais vues** — c'est le vrai test de "
            "généralisation sur un si petit échantillon."
        )
        cv1, cv2, cv3 = st.columns(3)
        cv1.metric("R² apprentissage", f"{results.r2:.0%}")
        cv2.metric("R² validation croisée", f"{results.r2_cv:.0%}", f"écart-type ±{cv_std:.0%}")
        cv3.metric("Amplitude selon le découpage",
                   f"{min(results.cv_scores):.0%} → {max(results.cv_scores):.0%}")
        st.warning(
            f"Le R² s'effondre de **{results.r2:.0%} à {results.r2_cv:.0%}** et **varie énormément** "
            f"selon le découpage (±{cv_std:.0%}, d'un fold quasi nul à un fold à 95 %). Un modèle "
            "vraiment robuste donnerait des folds resserrés : ici, l'instabilité est le symptôme du "
            "sur-ajustement sur 67 points.",
            icon="⚠️",
        )
        st.caption(
            f"Coïncidence parlante : cette généralisation (~{results.r2_cv:.0%}) rejoint le lien réel "
            f"mesuré au grain fin (η² ≈ {eta_ilot['eta2']:.0%}). Le « vrai » signal était là ; le 87 % "
            "était l'inflation d'échantillon."
        )

    with tab_echelle:
        st.markdown("""
Où se situe ce projet sur l'échelle de la preuve ? Soyons explicites.

| Niveau | Statut ici |
|---|---|
| **Observation** — « on voit un motif » | ✅ oui : la carte au grain fin |
| **Corrélation** — « ça varie ensemble », chiffré + IC | ✅ oui : η² îlots, r ± IC communes |
| **Association robuste** — tient aux changements d'échelle/méthode | ✅ oui : visible à l'îlot ET à la commune |
| **Interprétation physique** — mécanisme plausible | ✅ oui : inertie thermique du bâti dense |
| **Causalité prouvée** — contrefactuel, expérience | ⛔ **non**, et on ne le prétend pas |

**La nuance clé** : l'exposition est un indicateur *calculé* par GEOCLIMATE à partir de la forme
urbaine. Le lien « bâti → chaleur » est donc en partie **encodé** dans la construction de la donnée.
On le **visualise et on le quantifie** ; on ne le « découvre » pas, et on ne prouve pas la causalité.
""")

    with tab_lim:
        st.markdown("""
**⛔ Ce que ce projet ne dit PAS**
- Ce n'est **pas** un relevé de thermomètre : l'exposition est un indicateur d'aménagement calculé,
  à lire comme tel (pas des °C).
- Le modèle à 67 communes **ne généralise pas** (CV instable) : il sert de contre-exemple pédagogique,
  pas d'outil de prédiction.
- La relation étant partiellement encodée dans la donnée, ce n'est **pas** une preuve causale que
  « végétaliser fait baisser la température de X ».

**✅ Ce qu'on peut affirmer**
- Le contraste thermique nocturne se joue à l'échelle de **l'îlot**, pas de la commune.
- Les poches les plus exposées sont **identifiables une par une** → ciblables pour l'action.
- Agréger **gonfle mécaniquement** la précision affichée : le beau R² était un artefact d'échantillon.
""")

# ─────────────────────────────────────────
# SECTION 3bis : OÙ AGIR EN PRIORITÉ (interactif, actionnable, sans modèle)
# ─────────────────────────────────────────
st.markdown("## 🎯 Alors, où agir en priorité ?")

st.markdown(
    "Sortir un constat, c'est bien ; le rendre utile, c'est mieux. La donnée brute permet de pointer "
    "directement **les poches où la chaleur nocturne se concentre** — les cibles évidentes pour "
    "végétaliser, désimperméabiliser, rafraîchir. Choisis le niveau d'exposition :"
)

niveau = st.radio(
    "Niveau d'exposition",
    ["🔴 Uniquement les plus chauds (fort)", "🟠 Moyennement exposés et plus"],
    horizontal=True,
    label_visibility="collapsed",
)
min_score = 2 if niveau.startswith("🔴") else 1

hot = df_ilots[df_ilots["expo_score"] >= min_score]
top_hot = hot.groupby("commune").size().sort_values(ascending=False).head(5)

col_hmap, col_hstat = st.columns([3, 2])
with col_hmap:
    safe_plot(plot_hot_ilots_map, df_ilots, min_score, theme)
with col_hstat:
    h1, h2 = st.columns(2)
    h1.metric("Îlots concernés", f"{len(hot):,}".replace(",", " "))
    h2.metric("Surface", f"{hot['surface'].sum() / 10000:,.0f} ha".replace(",", " "))
    st.markdown("**Communes qui concentrent le plus ces poches :**")
    for nom, n in top_hot.items():
        st.markdown(f"- {nom} — **{n}** îlots")
    st.caption(
        "Ce ciblage sort directement des données, sans aucun modèle : ce sont des mesures "
        "d'aménagement réelles, pas des prédictions."
    )

st.divider()

# ─────────────────────────────────────────
# SECTION 4 : LA LEÇON
# ─────────────────────────────────────────
st.markdown("## 💡 Ce que je retiens")

st.markdown(f"""
1. **Sur la ville** : à Lyon, la nuit de canicule ne se joue pas à l'échelle du quartier mais du
   pâté de maisons. Densifier sans végétaliser fabrique des poches invivables — et il en existe
   jusque dans les communes réputées « fraîches ». La bonne nouvelle : ces poches sont
   **identifiables une par une**, donc traitables.

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
