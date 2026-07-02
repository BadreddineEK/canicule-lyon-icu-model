"""
Helpers de visualisation Plotly pour le dashboard.
Palette et mise en page centralisées pour garder tous les graphiques cohérents.
Le seuil d'écart utilisé pour repérer un quartier mal prédit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Seuil (en °C) au-delà duquel une prédiction est considérée comme "ratée".
# Doit rester aligné avec model.naive_model.identify_outliers.
OUTLIER_THRESHOLD = 0.8

# Palette cohérente sur tous les graphiques
COLOR_REEL = "#E74C3C"      # rouge : température réelle
COLOR_PREDIT = "#3498DB"    # bleu : prédiction du modèle
COLOR_WARM = "#E74C3C"      # effet réchauffant (coefficient positif)
COLOR_COOL = "#3498DB"      # effet rafraîchissant (coefficient négatif)
COLOR_OUTLIER = "#F39C12"   # orange : quartier mal prédit
COLOR_OK = "#2ECC71"        # vert : quartier bien prédit

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
def plot_reel_vs_predit(df: pd.DataFrame, predictions: np.ndarray, theme: str = "dark") -> go.Figure:
    """Graphique en barres groupées : ICU réel vs prédit par quartier."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="ICU réel (référence Météo-France)",
        x=df["quartier"],
        y=df["icu_reel"],
        marker_color=COLOR_REEL,
        opacity=0.85,
    ))

    fig.add_trace(go.Bar(
        name="ICU prédit (modèle naïf)",
        x=df["quartier"],
        y=predictions,
        marker_color=COLOR_PREDIT,
        opacity=0.85,
    ))

    fig.update_layout(
        barmode="group",
        title="Écart de température ICU : réalité vs modèle naïf",
        xaxis_title="Quartier",
        yaxis_title="ΔT vs zone rurale de référence (°C)",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", y=1.15),
    )
    return _apply_theme(fig, height=450, theme=theme)


@st.cache_data(show_spinner=False)
def plot_residuals(df_out: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Graphique des résidus — met en évidence les quartiers problématiques."""
    p = _palette(theme)
    colors = [COLOR_OUTLIER if o else COLOR_OK for o in df_out["is_outlier"]]

    fig = go.Figure(go.Bar(
        x=df_out["quartier"],
        y=df_out["residual"],
        marker_color=colors,
        text=[f"{v:+.2f}°C" for v in df_out["residual"]],
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=p["ref"], opacity=0.6)
    fig.add_hline(y=OUTLIER_THRESHOLD, line_dash="dash", line_color=COLOR_OUTLIER, opacity=0.5,
                  annotation_text=f"seuil d'alerte (+{OUTLIER_THRESHOLD}°C)", annotation_position="top right")
    fig.add_hline(y=-OUTLIER_THRESHOLD, line_dash="dash", line_color=COLOR_OUTLIER, opacity=0.5)

    fig.update_layout(
        title="Résidus : écart entre réalité et prédiction (°C)",
        xaxis_title="Quartier",
        yaxis_title="Erreur du modèle (°C)",
        xaxis_tickangle=-35,
    )
    return _apply_theme(fig, height=420, theme=theme)


@st.cache_data(show_spinner=False)
def plot_correlation_matrix(df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Heatmap de corrélation entre les variables et la cible."""
    cols = ["ndvi", "densite_pop", "dist_centre_km", "icu_reel"]
    labels = ["Végétation (NDVI)", "Densité pop.", "Dist. centre (km)", "ICU réel (°C)"]
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
def plot_feature_importance(coefficients: dict, theme: str = "dark") -> go.Figure:
    """Barres horizontales des coefficients standardisés du modèle."""
    p = _palette(theme)
    labels = {
        "ndvi": "Végétation (NDVI)",
        "densite_pop": "Densité population",
        "dist_centre_km": "Distance au centre",
    }
    feats = list(coefficients.keys())
    coefs = list(coefficients.values())
    # Bleu = fait baisser la température (coef négatif), rouge = fait monter (coef positif)
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
        title="Importance des variables (coefficients standardisés)",
        xaxis_title="Coefficient",
    )
    return _apply_theme(fig, height=280, theme=theme)


@st.cache_data(show_spinner=False)
def plot_scatter_reel_vs_predit(df: pd.DataFrame, predictions: np.ndarray, theme: str = "dark") -> go.Figure:
    """Nuage de points réel vs prédit avec la droite de prédiction parfaite."""
    p = _palette(theme)
    y_reel = df["icu_reel"].values
    min_v, max_v = min(y_reel.min(), predictions.min()), max(y_reel.max(), predictions.max())

    fig = go.Figure()

    # Ligne parfaite y=x
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines",
        line=dict(color=p["ref"], dash="dot", width=1),
        name="Prédiction parfaite",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=y_reel,
        y=predictions,
        mode="markers+text",
        marker=dict(
            size=10,
            color=[COLOR_OUTLIER if abs(r - p) > OUTLIER_THRESHOLD else COLOR_OK
                   for r, p in zip(y_reel, predictions)],
        ),
        text=df["quartier"],
        textposition="top center",
        textfont=dict(size=9),
        name="Quartiers",
    ))

    fig.update_layout(
        title="Valeurs réelles vs prédites",
        xaxis_title="ICU réel (°C)",
        yaxis_title="ICU prédit (°C)",
    )
    return _apply_theme(fig, height=420, theme=theme)


@st.cache_data(show_spinner=False)
def plot_heat_map(df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """Carte de Lyon : chaque quartier coloré et dimensionné selon l'intensité
    de l'îlot de chaleur (écart de température réel vs zone rurale)."""
    p = _palette(theme)
    map_style = "carto-positron" if theme == "light" else "carto-darkmatter"

    fig = go.Figure(go.Scattermapbox(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=dict(
            size=df["icu_reel"] * 5 + 12,
            color=df["icu_reel"],
            colorscale="YlOrRd",
            cmin=0,
            cmax=float(df["icu_reel"].max()),
            colorbar=dict(title="ΔT (°C)"),
            opacity=0.9,
        ),
        text=df["quartier"],
        customdata=df["icu_reel"],
        hovertemplate="<b>%{text}</b><br>ΔT vs campagne : %{customdata:.1f} °C<extra></extra>",
    ))

    fig.update_layout(
        title="Où se concentre la chaleur ? Les îlots de chaleur de Lyon",
        mapbox=dict(
            style=map_style,
            center=dict(lat=45.762, lon=4.86),
            zoom=10.1,
        ),
        height=520,
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor=p["bg"],
        font=dict(color=p["font"], size=13),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=13),
    )
    return fig
