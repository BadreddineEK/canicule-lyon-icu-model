"""
Données des quartiers de Lyon.
NDVI simplifié, densité population, distance centre, température ICU réelle.
Sources :
  - Température ICU : cartographies Météo-France / publications ICU Lyon
  - Population : INSEE 2021
  - NDVI : estimation d'après couverture végétale connue par quartier
"""

import pandas as pd
import numpy as np

def get_quartiers_data() -> pd.DataFrame:
    """
    Retourne un DataFrame avec les caractéristiques des quartiers de Lyon
    et leur écart de température ICU réel mesuré (vs zone rurale de référence).
    """
    data = {
        "quartier": [
            "Presqu'île / Bellecour",
            "Part-Dieu",
            "La Guillotière",
            "Croix-Rousse",
            "Gerland",
            "Vieux-Lyon",
            "Villeurbanne Centre",
            "Mermoz",
            "Confluence",
            "Bron",
            "Caluire-et-Cuire",
            "Saint-Priest",
            "Champagne-au-Mont-d'Or",
            "Miribel",
        ],
        # Latitude/Longitude approximative du centre de chaque quartier
        "lat": [
            45.7580, 45.7604, 45.7479, 45.7727, 45.7320,
            45.7621, 45.7662, 45.7320, 45.7430, 45.7298,
            45.7985, 45.6985, 45.8241, 45.8445,
        ],
        "lon": [
            4.8320, 4.8581, 4.8454, 4.8260, 4.8340,
            4.8222, 4.8790, 4.8750, 4.8227, 4.9145,
            4.8545, 4.9368, 4.7862, 5.0106,
        ],
        # NDVI 0-1 : fraction de végétation (0 = béton pur, 1 = forêt dense)
        # Source : estimation d'après couverture végétale par quartier
        "ndvi": [
            0.08, 0.05, 0.12, 0.28, 0.22,
            0.15, 0.10, 0.25, 0.18, 0.20,
            0.45, 0.35, 0.62, 0.70,
        ],
        # Densité de population (habitants/km²)
        # Source : INSEE 2021
        "densite_pop": [
            18000, 12000, 22000, 16000, 8000,
            14000, 13000, 9000, 5000, 7000,
            3500, 4500, 1800, 600,
        ],
        # Distance au centre (Bellecour) en km
        "dist_centre_km": [
            0.0, 1.8, 1.2, 2.0, 3.5,
            0.8, 3.0, 4.2, 2.0, 6.5,
            5.0, 8.0, 9.5, 12.0,
        ],
        # ICU réel mesuré : différence de température (°C) vs zone rurale de référence
        # pendant un épisode de canicule (nuit)
        # Source : publications ICU Météo-France, cartographies disponibles
        "icu_reel": [
            5.2, 4.8, 4.5, 3.2, 3.8,
            3.5, 3.9, 2.8, 2.5, 2.4,
            1.2, 1.5, 0.6, 0.2,
        ],
    }
    return pd.DataFrame(data)


def get_feature_descriptions() -> dict:
    return {
        "ndvi": "Indice de végétation (0 = béton pur, 1 = forêt dense)",
        "densite_pop": "Densité de population (habitants/km²) — proxy de l'artificialisation",
        "dist_centre_km": "Distance au centre-ville (km) — proxy de l'urbanisation",
        "icu_reel": "Écart de température réel mesuré vs zone rurale (°C) — nuit de canicule",
    }
