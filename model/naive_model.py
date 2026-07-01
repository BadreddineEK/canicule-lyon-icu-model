"""
Modèle naïf de prédiction de l'îlot de chaleur urbain.
Régression linéaire sur 3 variables simples.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelResults:
    r2: float
    mae: float
    predictions: np.ndarray
    residuals: np.ndarray
    coefficients: dict
    intercept: float


FEATURES = ["ndvi", "densite_pop", "dist_centre_km"]
TARGET = "icu_reel"


def train_naive_model(df: pd.DataFrame) -> Tuple[LinearRegression, StandardScaler, ModelResults]:
    """
    Entraîne le modèle naïf sur l'ensemble des données.
    Retourne le modèle, le scaler, et les résultats.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    residuals = y - preds

    coefficients = {
        feat: round(coef, 3)
        for feat, coef in zip(FEATURES, model.coef_)
    }

    results = ModelResults(
        r2=round(r2_score(y, preds), 3),
        mae=round(mean_absolute_error(y, preds), 3),
        predictions=preds,
        residuals=residuals,
        coefficients=coefficients,
        intercept=round(model.intercept_, 3),
    )

    return model, scaler, results


def get_model_formula(coefficients: dict, intercept: float) -> str:
    """Retourne la formule du modèle sous forme lisible."""
    terms = []
    labels = {
        "ndvi": "Végétation",
        "densite_pop": "Densité pop.",
        "dist_centre_km": "Distance centre",
    }
    for feat, coef in coefficients.items():
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.2f} × {labels[feat]}")
    return f"ΔT_ICU = {intercept:.2f} " + " ".join(terms) + " (variables standardisées)"


def identify_outliers(df: pd.DataFrame, residuals: np.ndarray, threshold: float = 0.8) -> pd.DataFrame:
    """
    Identifie les quartiers où le modèle se trompe le plus.
    Ce sont ces cas qui servent à expliquer les limites du modèle naïf.
    """
    df_out = df.copy()
    df_out["residual"] = residuals
    df_out["abs_residual"] = np.abs(residuals)
    df_out["is_outlier"] = df_out["abs_residual"] > threshold
    df_out["error_direction"] = np.where(
        residuals > 0, "Modèle sous-estime", "Modèle sur-estime"
    )
    return df_out.sort_values("abs_residual", ascending=False)
