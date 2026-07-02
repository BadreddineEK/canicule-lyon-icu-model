"""
Construit un vrai jeu de données à partir des données ouvertes de la Métropole de Lyon.

Source : « Îlots de chaleur urbains » (data.grandlyon.com), calculé avec le logiciel
GEOCLIMATE (classification Local Climate Zone + exposition à la chaleur), Licence Ouverte Etalab.

Chaque îlot porte une Local Climate Zone (LCZ) et une exposition nocturne à la chaleur.
On agrège par commune / arrondissement (pondéré par la surface) pour obtenir :
  - la composition du sol (végétation, bâti compact, minéral)
  - un score continu d'exposition nocturne à la chaleur (la cible à expliquer)
Sortie : data/communes_lyon.csv
"""

import requests
import pandas as pd

WFS = ("https://download.data.grandlyon.com/wfs/grandlyon"
       "?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
       "&typename=metropole-de-lyon:ilots-de-chaleur-urbains"
       "&outputFormat=application/json;%20subtype=geojson&SRSNAME=EPSG:4326")

# Exposition nocturne -> score ordinal (plus c'est haut, plus il fait chaud la nuit)
EXPO_SCORE = {
    "effet rafraîchissant": -1.0,
    "faible": 0.0,
    "moyen": 1.0,
    "fort": 2.0,
}

# Regroupement des Local Climate Zones
VEGETATION = {"LCZ A", "LCZ B", "LCZ C", "LCZ D"}   # arbres denses/épars, buissons, herbe
EAU = {"LCZ G"}                                       # eau
MINERAL = {"LCZ E"}                                   # roche nue, pavés, macadam
COMPACT = {"LCZ 1", "LCZ 2", "LCZ 3"}                 # bâti compact (haut/moyen/bas)
BATI = {f"LCZ {i}" for i in range(1, 11)}             # tout le bâti (LCZ 1 à 10)


def fetch_all():
    feats, start, page = [], 0, 5000
    while True:
        url = f"{WFS}&count={page}&startIndex={start}"
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        batch = r.json().get("features", [])
        feats.extend(batch)
        if len(batch) < page:
            break
        start += page
    return feats


def centroid(geom):
    """Centroïde approximatif d'un polygone (moyenne des sommets de l'anneau externe)."""
    if geom is None:
        return None, None
    coords = geom["coordinates"]
    t = geom["type"]
    if t == "Polygon":
        ring = coords[0]
    elif t == "MultiPolygon":
        ring = coords[0][0]
    else:
        return None, None
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def build():
    feats = fetch_all()
    rows = []
    for f in feats:
        p = f["properties"]
        expo = p.get("expo_nuit")
        if expo not in EXPO_SCORE:
            continue
        lcz = p.get("lcz_code")
        surf = p.get("surface") or 0.0
        if surf <= 0:
            continue
        lon, lat = centroid(f.get("geometry"))
        rows.append({
            "commune": p.get("commune"),
            "insee": p.get("insee"),
            "surface": surf,
            "lcz": lcz,
            "expo_score": EXPO_SCORE[expo],
            "is_vegetation": lcz in VEGETATION,
            "is_eau": lcz in EAU,
            "is_mineral": lcz in MINERAL,
            "is_compact": lcz in COMPACT,
            "is_bati": lcz in BATI,
            "lon": lon,
            "lat": lat,
        })

    df = pd.DataFrame(rows)
    print("îlots exploités :", len(df))

    def agg(g):
        s = g["surface"].sum()
        return pd.Series({
            "insee": g["insee"].iloc[0],
            "n_ilots": len(g),
            "surface_ha": s / 10000.0,
            "pct_vegetation": 100 * (g.loc[g["is_vegetation"], "surface"].sum()) / s,
            "pct_mineral": 100 * (g.loc[g["is_mineral"], "surface"].sum()) / s,
            "pct_compact": 100 * (g.loc[g["is_compact"], "surface"].sum()) / s,
            "pct_bati": 100 * (g.loc[g["is_bati"], "surface"].sum()) / s,
            # cible : score d'exposition nocturne moyen, pondéré par la surface
            "expo_nuit_score": (g["expo_score"] * g["surface"]).sum() / s,
            # part de surface fortement exposée la nuit
            "pct_forte_nuit": 100 * (g.loc[g["expo_score"] >= 2, "surface"].sum()) / s,
            "lat": (g["lat"] * g["surface"]).sum() / s,
            "lon": (g["lon"] * g["surface"]).sum() / s,
        })

    communes = df.groupby("commune").apply(agg, include_groups=False).reset_index()
    # On garde les communes avec assez d'îlots pour des pourcentages stables
    communes = communes[communes["n_ilots"] >= 20].copy()
    communes = communes.sort_values("expo_nuit_score", ascending=False).reset_index(drop=True)

    for c in ["surface_ha", "pct_vegetation", "pct_mineral", "pct_compact",
              "pct_bati", "expo_nuit_score", "pct_forte_nuit", "lat", "lon"]:
        communes[c] = communes[c].round(3)

    out = "data/communes_lyon.csv"
    communes.to_csv(out, index=False, encoding="utf-8")
    print("communes retenues :", len(communes))
    print(communes[["commune", "pct_vegetation", "pct_compact",
                    "pct_mineral", "expo_nuit_score"]].head(12).to_string(index=False))
    print("...")
    print(communes[["commune", "expo_nuit_score"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    build()
