"""Profile complexity and tolerance features from the cross-section polygon.

Extrusion cost drivers and their proxies:
- metal used: area → weight per metre (density of 6xxx aluminium, 2.70 kg/dm³);
- press size: circumscribing circle diameter (CCD) of the section;
- die complexity and press speed: perimeter per area (thin, long walls extrude slowly),
  number of hollows (bridge dies), number of vertices, solidity (deep channels and fins
  have a low area / convex-hull area), CCD / minimum wall (large thin sections deform);
- surface treatments are priced per m², so outer surface per kg drives finish cost.
"""
import numpy as np

DENSITY_KG_PER_MM2_M = 2.70e-3  # kg per metre of length per mm² of section


def polygon_area(poly) -> float:
    """Absolute area by the shoelace formula."""
    x, y = np.asarray(poly, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(poly) -> float:
    p = np.asarray(poly, dtype=float)
    return float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(poly) -> np.ndarray:
    """Convex hull by Andrew's monotone chain."""
    pts = sorted(map(tuple, np.asarray(poly, dtype=float)))

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and _cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def profile_features(outer, holes, min_wall_mm: float) -> dict:
    area = polygon_area(outer) - sum(polygon_area(h) for h in holes)
    if area <= 0:
        raise ValueError("cross-section area must be positive (holes larger than outer contour)")
    outer_perimeter = polygon_perimeter(outer)
    perimeter = outer_perimeter + sum(polygon_perimeter(h) for h in holes)
    pts = np.asarray(outer, dtype=float)
    ccd = 2 * float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    weight = area * DENSITY_KG_PER_MM2_M
    return {
        "area_mm2": area,
        "weight_kg_per_m": weight,
        "ccd_mm": ccd,
        "perimeter_per_area": perimeter / area,
        "n_hollows": len(holes),
        "n_vertices": len(outer) + sum(len(h) for h in holes),
        "solidity": area / polygon_area(convex_hull(outer)),
        "ccd_per_wall": ccd / min_wall_mm,
        "min_wall_mm": min_wall_mm,
        "surface_m2_per_kg": outer_perimeter * 1e-3 / weight,
    }


# Standard (untoleranced) values; a simplified rule where the linear tolerance grows with
# section size, as in extrusion tolerance standards.
def standard_linear_tol(ccd_mm: float) -> float:
    return 0.15 + 0.0025 * ccd_mm


STANDARD_GDT = {"flatness_mm": 0.3, "straightness_mm_per_m": 1.0, "twist_deg_per_m": 1.0}


def tolerance_features(linear_tol_mm, gdt: dict, ccd_mm: float) -> dict:
    """Tightness = log2(standard / specified): 0 for standard, +1 for each halving.

    An unspecified tolerance means the standard applies on a drawing, so None maps to 0
    rather than being imputed from other quotes.
    """
    def tightness(value, standard):
        return 0.0 if value is None else float(np.log2(standard / value))

    out = {"tight_linear": tightness(linear_tol_mm, standard_linear_tol(ccd_mm))}
    for name, standard in STANDARD_GDT.items():
        out[f"tight_{name.split('_')[0]}"] = tightness(gdt.get(name), standard)
    out["n_tight_gdt"] = sum(out[f"tight_{n.split('_')[0]}"] > 0 for n in STANDARD_GDT)
    return out
