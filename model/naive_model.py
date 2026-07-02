"""
Modèle de l'exposition nocturne à la chaleur des communes de la Métropole de Lyon.

Régression linéaire sur la composition réelle du sol (végétation, bâti compact,
minéral), issue des Local Climate Zones. On compare volontairement le R² en
apprentissage au R² en validation croisée : c'est tout l'objet du projet.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelResults:
    r2: float                 # R² en apprentissage (sur les données vues)
    r2_cv: float              # R² moyen en validation croisée (généralisation)
    mae: float
    predictions: np.ndarray
    residuals: np.ndarray
    coefficients: dict
    intercept: float
    cv_scores: list = field(default_factory=list)


FEATURES = ["pct_vegetation", "pct_compact", "pct_mineral"]
TARGET = "expo_nuit_score"


def train_naive_model(df: pd.DataFrame) -> Tuple[LinearRegression, StandardScaler, ModelResults]:
    """Entraîne la régression et mesure sa performance en apprentissage ET en
    validation croisée. Retourne le modèle, le scaler et les résultats."""
    X = df[FEATURES].values
    y = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    residuals = y - preds

    # Validation croisée : le vrai test d'honnêteté sur un petit échantillon.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")

    coefficients = {feat: round(coef, 3) for feat, coef in zip(FEATURES, model.coef_)}

    results = ModelResults(
        r2=round(r2_score(y, preds), 3),
        r2_cv=round(float(cv_scores.mean()), 3),
        mae=round(mean_absolute_error(y, preds), 3),
        predictions=preds,
        residuals=residuals,
        coefficients=coefficients,
        intercept=round(model.intercept_, 3),
        cv_scores=[round(float(s), 3) for s in cv_scores],
    )

    return model, scaler, results


def get_model_formula(coefficients: dict, intercept: float) -> str:
    """Retourne la formule du modèle sous forme lisible."""
    labels = {
        "pct_vegetation": "Végétation",
        "pct_compact": "Bâti compact",
        "pct_mineral": "Minéral",
    }
    terms = []
    for feat, coef in coefficients.items():
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.2f} × {labels[feat]}")
    return f"Expo_nuit = {intercept:.2f} " + " ".join(terms) + " (variables standardisées)"


def identify_outliers(df: pd.DataFrame, residuals: np.ndarray, threshold: float = 0.2) -> pd.DataFrame:
    """Identifie les communes où le modèle se trompe le plus (résidus élevés)."""
    df_out = df.copy()
    df_out["residual"] = residuals
    df_out["abs_residual"] = np.abs(residuals)
    df_out["is_outlier"] = df_out["abs_residual"] > threshold
    df_out["error_direction"] = np.where(
        residuals > 0, "Modèle sous-estime", "Modèle sur-estime"
    )
    return df_out.sort_values("abs_residual", ascending=False)
