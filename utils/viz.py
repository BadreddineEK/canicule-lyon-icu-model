"""
Helpers de visualisation Plotly pour le dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Palette couleurs cohérente
COLOR_REEL = "#E74C3C"
COLOR_PREDIT = "#3498DB"
COLOR_OUTLIER = "#F39C12"
COLOR_OK = "#2ECC71"


def plot_reel_vs_predit(df: pd.DataFrame, predictions: np.ndarray) -> go.Figure:
    """Graphique en barres groupées : ICU réel vs prédit par quartier."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="ICU réel mesuré",
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
        yaxis_title="ΔT par rapport à la campagne (°C)",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", y=1.12),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=450,
    )
    return fig


def plot_residuals(df_out: pd.DataFrame) -> go.Figure:
    """Graphique des résidus — met en évidence les quartiers problématiques."""
    colors = [COLOR_OUTLIER if o else COLOR_OK for o in df_out["is_outlier"]]

    fig = go.Figure(go.Bar(
        x=df_out["quartier"],
        y=df_out["residual"],
        marker_color=colors,
        text=[f"{v:+.2f}°C" for v in df_out["residual"]],
        textposition="outside",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.4)
    fig.add_hline(y=0.8, line_dash="dash", line_color=COLOR_OUTLIER, opacity=0.5,
                  annotation_text="seuil d'alerte (+0.8°C)", annotation_position="top right")
    fig.add_hline(y=-0.8, line_dash="dash", line_color=COLOR_OUTLIER, opacity=0.5)

    fig.update_layout(
        title="Résidus : là où le modèle se trompe le plus",
        xaxis_title="Quartier",
        yaxis_title="Erreur du modèle (°C)",
        xaxis_tickangle=-35,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=420,
    )
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Heatmap de corrélation entre les features et la cible."""
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

    fig.update_layout(
        title="Corrélations entre variables",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=380,
    )
    return fig


def plot_feature_importance(coefficients: dict) -> go.Figure:
    """Barres horizontales des coefficients standardisés du modèle."""
    labels = {
        "ndvi": "Végétation (NDVI)",
        "densite_pop": "Densité population",
        "dist_centre_km": "Distance au centre",
    }
    feats = list(coefficients.keys())
    coefs = list(coefficients.values())
    colors = [COLOR_REEL if c < 0 else COLOR_PREDIT for c in coefs]

    fig = go.Figure(go.Bar(
        x=coefs,
        y=[labels[f] for f in feats],
        orientation="h",
        marker_color=colors,
        text=[f"{c:+.3f}" for c in coefs],
        textposition="outside",
    ))

    fig.add_vline(x=0, line_color="white", opacity=0.4)
    fig.update_layout(
        title="Importance des variables (coefficients standardisés)",
        xaxis_title="Coefficient",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=280,
    )
    return fig


def plot_scatter_reel_vs_predit(df: pd.DataFrame, predictions: np.ndarray) -> go.Figure:
    """Scatter plot réel vs prédit avec droite de régression parfaite."""
    y_reel = df["icu_reel"].values
    min_v, max_v = min(y_reel.min(), predictions.min()), max(y_reel.max(), predictions.max())

    fig = go.Figure()

    # Ligne parfaite y=x
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode="lines",
        line=dict(color="white", dash="dot", width=1),
        name="Prédiction parfaite",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=y_reel,
        y=predictions,
        mode="markers+text",
        marker=dict(
            size=10,
            color=[COLOR_OUTLIER if abs(r - p) > 0.8 else COLOR_OK
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
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=420,
    )
    return fig
