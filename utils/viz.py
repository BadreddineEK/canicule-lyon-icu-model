"""
Helpers de visualisation Plotly pour le dashboard.
Palette et mise en page centralisées pour garder tous les graphiques cohérents.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Seuil de résidu (en points de score) au-delà duquel on considère que le modèle
# se trompe nettement. Doit rester aligné avec model.naive_model.identify_outliers.
OUTLIER_THRESHOLD = 0.2

# Palette cohérente sur tous les graphiques
COLOR_REEL = "#E74C3C"      # rouge : valeur réelle
COLOR_PREDIT = "#3498DB"    # bleu : prédiction du modèle
COLOR_WARM = "#E74C3C"      # effet réchauffant (coefficient positif)
COLOR_COOL = "#3498DB"      # effet rafraîchissant (coefficient négatif)
COLOR_OUTLIER = "#F39C12"   # orange : commune mal prédite
COLOR_OK = "#2ECC71"        # vert : commune bien prédite


def _palette(theme: str) -> dict:
    """Couleurs de fond, texte, grille et lignes de repère selon le thème actif."""
    if theme == "light":
        return dict(
            bg="#ffffff", font="#1a1d24",
            grid="rgba(0,0,0,0.10)", ref="rgba(0,0,0,0.45)",
        )
    return dict(
        bg="#0e1117", font="#fafafa",
        grid="rgba(255,255,255,0.10)", ref="rgba(255,255,255,0.55)",
    )


def _apply_theme(fig: go.Figure, height: int, theme: str) -> go.Figure:
    """Applique la même charte (fond, police, marges, grille) à tous les graphiques."""
    p = _palette(theme)
    fig.update_layout(
        plot_bgcolor=p["bg"],
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=16),
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["grid"])
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["grid"])
    return fig


@st.cache_data(show_spinner=False)
def plot_top_communes(df: pd.DataFrame, n: int = 15, theme: str = "light") -> go.Figure:
    """Classement des communes les plus exposées à la chaleur nocturne."""
    d = df.sort_values("expo_nuit_score", ascending=True).tail(n)
    colors = [
        COLOR_REEL if v > 0.6 else (COLOR_PREDIT if v < 0 else "#E67E22")
        for v in d["expo_nuit_score"]
    ]
    fig = go.Figure(go.Bar(
        x=d["expo_nuit_score"],
        y=d["commune"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in d["expo_nuit_score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Les {n} communes les plus exposées à la chaleur la nuit",
        xaxis_title="Score d'exposition nocturne (−1 rafraîchissant → +2 fort)",
    )
    return _apply_theme(fig, height=480, theme=theme)


@st.cache_data(show_spinner=False)
def plot_residuals(df_out: pd.DataFrame, n: int = 12, theme: str = "light") -> go.Figure:
    """Communes où le modèle se trompe le plus (plus gros résidus)."""
    p = _palette(theme)
    d = df_out.sort_values("abs_residual", ascending=False).head(n)
    d = d.sort_values("residual")
    colors = [COLOR_OUTLIER if o else COLOR_OK for o in d["is_outlier"]]

    fig = go.Figure(go.Bar(
        x=d["residual"],
        y=d["commune"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in d["residual"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=p["ref"], opacity=0.6)
    fig.update_layout(
        title="Là où le modèle se trompe le plus (résidus)",
        xaxis_title="Erreur du modèle (réel − prédit)",
    )
    return _apply_theme(fig, height=440, theme=theme)


@st.cache_data(show_spinner=False)
def plot_correlation_matrix(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Heatmap de corrélation entre les variables et la cible."""
    cols = ["pct_vegetation", "pct_compact", "pct_mineral", "expo_nuit_score"]
    labels = ["Végétation", "Bâti compact", "Minéral", "Expo. nuit"]
    corr = df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Corrélations entre variables")
    return _apply_theme(fig, height=380, theme=theme)


@st.cache_data(show_spinner=False)
def plot_feature_importance(coefficients: dict, theme: str = "light") -> go.Figure:
    """Barres horizontales des coefficients standardisés du modèle."""
    p = _palette(theme)
    labels = {
        "pct_vegetation": "Végétation",
        "pct_compact": "Bâti compact",
        "pct_mineral": "Sol minéral",
    }
    feats = list(coefficients.keys())
    coefs = list(coefficients.values())
    # Bleu = fait baisser l'exposition (coef négatif), rouge = fait monter (positif)
    colors = [COLOR_COOL if c < 0 else COLOR_WARM for c in coefs]

    fig = go.Figure(go.Bar(
        x=coefs,
        y=[labels[f] for f in feats],
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.3f}" for c in coefs],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color=p["ref"], opacity=0.6)
    fig.update_layout(
        title="Ce qui pèse sur la chaleur nocturne (coefficients standardisés)",
        xaxis_title="Coefficient",
    )
    return _apply_theme(fig, height=280, theme=theme)


@st.cache_data(show_spinner=False)
def plot_scatter_reel_vs_predit(df: pd.DataFrame, predictions: np.ndarray, theme: str = "light") -> go.Figure:
    """Nuage réel vs prédit sur les 67 communes, avec la droite parfaite."""
    p = _palette(theme)
    y_reel = df["expo_nuit_score"].values
    resid = y_reel - predictions
    min_v = min(y_reel.min(), predictions.min())
    max_v = max(y_reel.max(), predictions.max())

    # On n'étiquette que les communes mal prédites, pour rester lisible.
    labels = [c if abs(r) > OUTLIER_THRESHOLD else "" for c, r in zip(df["commune"], resid)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines",
        line=dict(color=p["ref"], dash="dot", width=1),
        name="Prédiction parfaite",
    ))
    fig.add_trace(go.Scatter(
        x=y_reel,
        y=predictions,
        mode="markers+text",
        marker=dict(
            size=9,
            color=[COLOR_OUTLIER if abs(r) > OUTLIER_THRESHOLD else COLOR_OK for r in resid],
        ),
        text=labels,
        textposition="top center",
        textfont=dict(size=9),
        name="Communes",
        hovertext=df["commune"],
        hovertemplate="<b>%{hovertext}</b><br>réel : %{x:.2f}<br>prédit : %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Score réel vs prédit (une commune = un point)",
        xaxis_title="Exposition nocturne réelle",
        yaxis_title="Exposition nocturne prédite",
        showlegend=False,
    )
    return _apply_theme(fig, height=460, theme=theme)


@st.cache_data(show_spinner=False)
def plot_heat_map(df: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Carte des communes colorées selon leur exposition nocturne à la chaleur."""
    p = _palette(theme)
    map_style = "carto-positron" if theme == "light" else "carto-darkmatter"

    fig = go.Figure(go.Scattermapbox(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=dict(
            size=(df["expo_nuit_score"] - df["expo_nuit_score"].min()) * 12 + 9,
            color=df["expo_nuit_score"],
            colorscale="RdYlBu_r",
            cmid=0,
            colorbar=dict(title="Expo.<br>nuit"),
            opacity=0.85,
        ),
        text=df["commune"],
        customdata=df["expo_nuit_score"],
        hovertemplate="<b>%{text}</b><br>Exposition nocturne : %{customdata:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Carte de la chaleur nocturne — Métropole de Lyon",
        mapbox=dict(style=map_style, center=dict(lat=45.75, lon=4.87), zoom=9.6),
        height=540,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=13),
    )
    return fig


def _ilot_sizes(surface: pd.Series, lo: float, hi: float) -> np.ndarray:
    """Taille des marqueurs proportionnelle à la racine de la surface (bornée)."""
    return np.clip(np.sqrt(surface.clip(lower=1)) / 18.0, lo, hi)


@st.cache_data(show_spinner=False)
def plot_ilots_map(df_ilots: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Carte au grain fin : un point par îlot (~29 000), coloré par la chaleur nocturne.
    C'est le vrai visage du phénomène, avant toute agrégation."""
    p = _palette(theme)
    map_style = "carto-positron" if theme == "light" else "carto-darkmatter"

    fig = go.Figure(go.Scattermapbox(
        lat=df_ilots["lat"],
        lon=df_ilots["lon"],
        mode="markers",
        marker=dict(
            size=_ilot_sizes(df_ilots["surface"], 3, 12),
            color=df_ilots["expo_score"],
            colorscale="RdYlBu_r",
            cmin=-1, cmax=2, cmid=0,
            colorbar=dict(
                title="Chaleur<br>nocturne",
                tickvals=[-1, 0, 1, 2],
                ticktext=["Rafraîch.", "Faible", "Moyen", "Fort"],
            ),
            opacity=0.72,
        ),
        text=df_ilots["commune"],
        customdata=df_ilots[["lcz_groupe", "expo_score"]],
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Chaleur nocturne : %{customdata[1]:+.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Chaleur nocturne à Lyon — {len(df_ilots):,} îlots réels".replace(",", " "),
        mapbox=dict(style=map_style, center=dict(lat=45.752, lon=4.85), zoom=10.3),
        height=620,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=17),
        hoverlabel=dict(font_size=13),
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_commune_ilots(df_ilots: pd.DataFrame, commune: str, theme: str = "light") -> go.Figure:
    """Zoom sur les îlots d'une seule commune : on voit le contraste interne
    (un parc frais à côté d'un cœur dense brûlant) que la moyenne efface."""
    p = _palette(theme)
    map_style = "carto-positron" if theme == "light" else "carto-darkmatter"
    d = df_ilots[df_ilots["commune"] == commune]

    fig = go.Figure(go.Scattermapbox(
        lat=d["lat"],
        lon=d["lon"],
        mode="markers",
        marker=dict(
            size=_ilot_sizes(d["surface"], 6, 22),
            color=d["expo_score"],
            colorscale="RdYlBu_r",
            cmin=-1, cmax=2, cmid=0,
            colorbar=dict(
                title="Chaleur<br>nocturne",
                tickvals=[-1, 0, 1, 2],
                ticktext=["Rafraîch.", "Faible", "Moyen", "Fort"],
            ),
            opacity=0.85,
        ),
        text=d["lcz_groupe"],
        customdata=d["expo_score"],
        hovertemplate="<b>%{text}</b><br>Chaleur nocturne : %{customdata:+.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{commune} — {len(d)} îlots, du plus frais au plus chaud",
        mapbox=dict(
            style=map_style,
            center=dict(lat=float(d["lat"].mean()), lon=float(d["lon"].mean())),
            zoom=12.3,
        ),
        height=520,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=13),
    )
    return fig


@st.cache_data(show_spinner=False)
def plot_expo_by_lcz(df_ilots: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Exposition nocturne moyenne par type de sol, calculée directement sur les
    îlots (aucune agrégation communale). La relation, à l'échelle honnête."""
    p = _palette(theme)
    order = ["Bâti compact", "Bâti diffus", "Minéral", "Végétation", "Eau"]
    g = df_ilots[df_ilots["lcz_groupe"].isin(order)].groupby("lcz_groupe")["expo_score"]
    stats = g.agg(["mean", "count"]).reindex(order).dropna()
    stats = stats.sort_values("mean")

    fig = go.Figure(go.Bar(
        x=stats["mean"],
        y=stats.index,
        orientation="h",
        marker=dict(
            color=stats["mean"], colorscale="RdYlBu_r", cmin=-1, cmax=2, cmid=0,
        ),
        text=[f"{v:+.2f}  ({int(n):,} îlots)".replace(",", " ")
              for v, n in zip(stats["mean"], stats["count"])],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=p["ref"], opacity=0.6)
    fig.update_layout(
        title="Chaleur nocturne moyenne par type de sol (au grain fin)",
        xaxis_title="Score d'exposition nocturne (−1 rafraîchissant → +2 fort)",
        xaxis=dict(range=[-1.2, 2.3]),
    )
    return _apply_theme(fig, height=340, theme=theme)


@st.cache_data(show_spinner=False)
def plot_hot_ilots_map(df_ilots: pd.DataFrame, min_score: float, theme: str = "light") -> go.Figure:
    """Ne montre que les îlots les plus exposés (score ≥ seuil) : les poches où
    la chaleur nocturne se concentre, donc les priorités d'action."""
    p = _palette(theme)
    map_style = "carto-positron" if theme == "light" else "carto-darkmatter"
    d = df_ilots[df_ilots["expo_score"] >= min_score]

    fig = go.Figure(go.Scattermapbox(
        lat=d["lat"],
        lon=d["lon"],
        mode="markers",
        marker=dict(
            size=_ilot_sizes(d["surface"], 4, 14),
            color=d["expo_score"],
            colorscale="YlOrRd",
            cmin=0, cmax=2,
            colorbar=dict(title="Chaleur<br>nocturne", tickvals=[1, 2], ticktext=["Moyen", "Fort"]),
            opacity=0.8,
        ),
        text=d["commune"],
        customdata=d[["lcz_groupe", "expo_score"]],
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Chaleur nocturne : %{customdata[1]:+.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{len(d):,} îlots à cibler (chaleur nocturne ≥ {min_score:+.0f})".replace(",", " "),
        mapbox=dict(style=map_style, center=dict(lat=45.752, lon=4.85), zoom=10.3),
        height=560,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=13),
    )
    return fig


def plot_gauge(value: float, theme: str = "light", vmin: float = -1.0, vmax: float = 2.0) -> go.Figure:
    """Jauge affichant un score d'exposition nocturne prédit."""
    p = _palette(theme)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 42, "color": p["font"]}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickcolor": p["font"], "tickfont": {"color": p["font"]}},
            "bar": {"color": COLOR_REEL},
            "bgcolor": p["bg"],
            "borderwidth": 0,
            "steps": [
                {"range": [vmin, 0], "color": "#AED6F1"},
                {"range": [0, 1], "color": "#FAD7A0"},
                {"range": [1, vmax], "color": "#F5B7B1"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"]),
    )
    return fig
