"""Feature table from validated (current-schema) records."""
import numpy as np
import pandas as pd

from .geometry import profile_features, tolerance_features
from .market import lme_features

CATEGORICAL = ["alloy", "finish", "segment", "profile_type"]
HIGH_CARDINALITY = ["customer_id"]
NUMERIC = [
    "area_mm2", "weight_kg_per_m", "ccd_mm", "perimeter_per_area", "n_hollows", "n_vertices", "solidity",
    "ccd_per_wall", "min_wall_mm", "surface_m2_per_kg", "length_m", "quantity_pcs", "order_kg", "log_order_kg",
    "setup_kg_inv", "new_die", "lead_time_days", "tight_linear", "tight_flatness", "tight_straightness",
    "tight_twist", "n_tight_gdt", "lme_lag1", "lme_ma1m", "lme_ma3m", "lme_ret1m", "lme_vol1m",
]
FEATURES = HIGH_CARDINALITY + CATEGORICAL + NUMERIC


def record_row(r) -> dict:
    geo, man = r.geometry, r.manufacturing
    row = {
        "quote_id": getattr(r, "quote_id", None),
        "quote_date": pd.Timestamp(r.quote_date),
        "customer_id": r.customer.customer_id,
        "segment": r.customer.segment.value if r.customer.segment else None,
        "alloy": r.material.alloy.value,
        "finish": r.material.finish.value if r.material.finish else None,
        "profile_type": geo.profile_type,
        "length_m": geo.length_mm / 1000,
        "quantity_pcs": man.quantity_pcs,
        "new_die": float(man.new_die),
        "lead_time_days": r.commercial.lead_time_days,
        "quoted_price_per_kg": getattr(r, "quoted_price_per_kg", None),
    }
    row.update(profile_features(geo.cross_section.outer, geo.cross_section.holes, geo.min_wall_mm))
    row.update(tolerance_features(man.linear_tol_mm, man.gdt.model_dump(), row["ccd_mm"]))
    row["order_kg"] = row["weight_kg_per_m"] * row["length_m"] * row["quantity_pcs"]
    row["log_order_kg"] = np.log(row["order_kg"])
    # fixed costs are spread over the order weight: price per kg is linear in 1 / kg
    row["setup_kg_inv"] = (1 + 2 * row["new_die"]) / row["order_kg"]
    return row


def build_features(records, lme: pd.Series) -> pd.DataFrame:
    """One row per record: geometry, tolerance, order, customer and market features."""
    df = pd.DataFrame([record_row(r) for r in records])
    df = pd.concat([df, lme_features(lme, df["quote_date"])], axis=1)
    for c in CATEGORICAL + HIGH_CARDINALITY:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    df["lead_time_days"] = df["lead_time_days"].astype(float)
    return df
