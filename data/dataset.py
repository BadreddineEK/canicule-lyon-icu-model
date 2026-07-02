"""
Chargement du jeu de données réel de la Métropole de Lyon.

Données : « Îlots de chaleur urbains » (data.grandlyon.com), produites avec le
logiciel GEOCLIMATE — classification en Local Climate Zones (LCZ) et exposition
à la chaleur, méthode de l'Institut Paris Région. Licence Ouverte Etalab.

29 657 îlots réels ont été agrégés par commune / arrondissement
(voir data/build_dataset.py). Pour chaque commune on dispose de la composition
du sol et d'un score d'exposition nocturne à la chaleur.
"""

from pathlib import Path
import pandas as pd

_CSV = Path(__file__).with_name("communes_lyon.csv")
_CSV_ILOTS = Path(__file__).with_name("ilots_lyon.csv")


def get_communes_data() -> pd.DataFrame:
    """Retourne le DataFrame des communes de la Métropole de Lyon."""
    return pd.read_csv(_CSV, encoding="utf-8")


def get_ilots_data() -> pd.DataFrame:
    """Retourne les ~29 657 îlots géolocalisés (grain fin).

    Colonnes : commune, lcz_groupe, expo_score, surface, lat, lon.
    """
    return pd.read_csv(_CSV_ILOTS, encoding="utf-8")


def get_feature_descriptions() -> dict:
    """Description lisible de chaque variable, pour l'interface."""
    return {
        "pct_vegetation": "Part de surface végétalisée (arbres, herbe) — LCZ A/B/D",
        "pct_compact": "Part de bâti compact (immeubles serrés) — LCZ 1/2/3",
        "pct_mineral": "Part de sol minéral (pavés, macadam, roche nue) — LCZ E",
        "expo_nuit_score": "Exposition nocturne à la chaleur (−1 rafraîchissant → +2 fort)",
    }
