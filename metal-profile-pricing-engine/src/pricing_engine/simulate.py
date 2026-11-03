"""Simulated market prices and historical quotes for two tenants with a known pricing mechanism.

Price per kg = metal + margin · customer · rush · (conversion · tolerance + finish + setup)
  metal       = 1-month mean market price / 1000 + regional premium + alloy premium
  conversion  = base · complexity(perimeter/area, hollows, thin walls, size, solidity)
  tolerance   = 1 + cost per halving of linear and GD&T tolerances below standard
  finish      = treatment price per m² · outer surface per kg
  setup       = (setup + new die cost) / order weight
with log-normal noise, a few outliers (rush jobs, entry errors), missing optional fields
and a handful of invalid records. Tenant A sends schema v1, tenant B schema v2 with
GD&T; they differ in cost base, margin, customers and history. A later feedback batch
for tenant A has a higher conversion cost (energy surcharge) — concept drift.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import STANDARD_GDT, profile_features, standard_linear_tol, tolerance_features
from .market import lme_features

PROFILE_TYPES = {"bar": .12, "rect_tube": .15, "round_tube": .12, "angle": .12, "channel": .14,
                 "tee": .08, "multi_chamber": .14, "heat_sink": .13}
ALLOYS = {"6060": .45, "6063": .25, "6005A": .15, "6082": .15}
FINISHES = {"mill": .4, "brushed": .1, "anodised": .3, "powder_coated": .2}
SEGMENTS = {"oem": .4, "distributor": .35, "construction": .25}

TRUTH = {
    "regional_premium": 0.35,
    "alloy_premium": {"6060": 0.0, "6063": 0.05, "6005A": 0.12, "6082": 0.25},
    "complexity": {"perimeter_per_area_ref": 0.3, "perimeter_per_area_exp": 0.3, "per_hollow": 0.12,
                   "thin_wall_below_mm": 2.5, "thin_wall_per_mm": 0.25, "large_ccd_above_mm": 120,
                   "large_ccd_per_mm": 0.003, "per_unit_concavity": 0.2},
    "tolerance": {"linear_per_halving": 0.12, "gdt_per_halving": 0.06},
    "finish_usd_per_m2": {"mill": 0.0, "brushed": 2.5, "anodised": 6.0, "powder_coated": 8.0},
    "setup_usd": 300, "new_die_usd": 700, "new_die_per_hollow_usd": 500,
    "rush": {"below_days": 21, "per_16_days": 0.12},
    "segment_factor": {"oem": 1.0, "distributor": 0.94, "construction": 0.97},
    "customer_discount_log_sd": 0.07,
    "noise_log_sd": 0.035,
    "outlier_share": 0.015,
    "tenants": {
        "tenant_a": {"schema_version": 1, "n_quotes": 900, "n_customers": 120, "start": "2023-01-02",
                     "end": "2025-06-30", "conversion_base": 1.5, "margin": 1.10},
        "tenant_b": {"schema_version": 2, "n_quotes": 600, "n_customers": 70, "start": "2023-07-03",
                     "end": "2025-06-30", "conversion_base": 1.35, "margin": 1.22},
    },
    "feedback": {"tenant": "tenant_a", "n_quotes": 200, "start": "2025-07-01", "end": "2025-10-31",
                 "conversion_base_factor": 1.25},
}


def simulate_lme(rng, start="2022-01-03", end="2025-10-31") -> pd.Series:
    """Log mean-reverting daily price on trading days."""
    days = pd.bdate_range(start, end)
    x = np.empty(len(days))
    x[0], mu = np.log(2250), np.log(2400)
    for t in range(1, len(days)):
        x[t] = x[t - 1] + 0.01 * (mu - x[t - 1]) + 0.013 * rng.standard_normal()
    return pd.Series(np.round(np.exp(x), 1), index=days, name="price_usd_t")


def _rect(x0, y0, w, h):
    return [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]


def make_profile(rng, kind):
    """Cross-section (outer, holes) and minimum wall of a parametric profile, mm."""
    u = rng.uniform
    if kind == "bar":
        w, h = u(10, 80), u(5, 40)
        return _rect(0, 0, w, h), [], min(w, h)
    if kind == "rect_tube":
        w, h, t = u(20, 100), u(15, 80), u(1.4, 5)
        return _rect(0, 0, w, h), [_rect(t, t, w - 2 * t, h - 2 * t)], t
    if kind == "round_tube":
        d = u(15, 100)
        t = u(1.4, min(5, d / 6))
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        circle = lambda r: list(zip(r * np.cos(a), r * np.sin(a)))
        return circle(d / 2), [circle(d / 2 - t)], t
    if kind == "angle":
        a, b, t = u(15, 80), u(15, 80), u(1.5, 6)
        return [(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)], [], t
    if kind == "channel":
        w, h, t = u(20, 100), u(15, 60), u(1.5, 6)
        return [(0, 0), (w, 0), (w, h), (w - t, h), (w - t, t), (t, t), (t, h), (0, h)], [], t
    if kind == "tee":
        w, h, t = u(20, 80), u(20, 60), u(1.5, 6)
        c = w / 2
        return [(c - t / 2, 0), (c + t / 2, 0), (c + t / 2, h - t), (w, h - t), (w, h), (0, h), (0, h - t),
                (c - t / 2, h - t)], [], t
    if kind == "multi_chamber":
        w, h, t, k = u(40, 160), u(20, 60), u(1.4, 4), int(rng.integers(2, 5))
        cw = (w - (k + 1) * t) / k
        return _rect(0, 0, w, h), [_rect(t + i * (cw + t), t, cw, h - 2 * t) for i in range(k)], t
    if kind == "heat_sink":
        w, base, fin_t, fin_h = u(40, 160), u(3, 8), u(1.2, 3), u(10, 40)
        n = int(rng.integers(4, max(5, min(15, int(w / (fin_t + 3))))))
        pitch = (w - fin_t) / (n - 1)
        pts = [(0, 0), (w, 0)]
        for i in reversed(range(n)):
            x = i * pitch
            pts += [(x + fin_t, base), (x + fin_t, base + fin_h), (x, base + fin_h), (x, base)]
        return pts, [], min(fin_t, base)
    raise ValueError(kind)


def _clean_polygon(poly):
    """Round to 0.1 mm and drop consecutive duplicate points."""
    out = []
    for x, y in poly:
        p = (round(float(x), 1), round(float(y), 1))
        if not out or p != out[-1]:
            out.append(p)
    if out[0] == out[-1]:
        out.pop()
    return out


def _pick(rng, probs: dict, size=None):
    return rng.choice(list(probs), p=list(probs.values()), size=size)


def _quote_dates(rng, start, end, n):
    days = pd.bdate_range(start, end)
    return sorted(rng.choice(days, size=n))


def expected_price(spec: dict, finish_true: str, lead_true: int, lme_ma1m: float, customer_factor: float,
                   conversion_base: float, margin: float) -> float:
    """Price per kg without noise, from the simulated mechanism."""
    T = TRUTH
    geo, man = spec["geometry"], spec["manufacturing"]
    f = profile_features(geo["cross_section"]["outer"], geo["cross_section"]["holes"], geo["min_wall_mm"])
    lin = man.get("linear_tol_mm", man.get("tolerance_mm"))
    tol = tolerance_features(lin, man.get("gdt", {}), f["ccd_mm"])
    c = T["complexity"]
    complexity = ((f["perimeter_per_area"] / c["perimeter_per_area_ref"]) ** c["perimeter_per_area_exp"]
                  * (1 + c["per_hollow"] * f["n_hollows"])
                  * (1 + c["thin_wall_per_mm"] * max(0, c["thin_wall_below_mm"] - f["min_wall_mm"]))
                  * (1 + c["large_ccd_per_mm"] * max(0, f["ccd_mm"] - c["large_ccd_above_mm"]))
                  * (1 + c["per_unit_concavity"] * (1 - f["solidity"])))
    tolerance = (1 + T["tolerance"]["linear_per_halving"] * max(0, tol["tight_linear"])
                 + T["tolerance"]["gdt_per_halving"] * sum(max(0, tol[f"tight_{n.split('_')[0]}"])
                                                           for n in STANDARD_GDT))
    finish = T["finish_usd_per_m2"][finish_true] * f["surface_m2_per_kg"]
    order_kg = f["weight_kg_per_m"] * geo["length_mm"] / 1000 * man["quantity_pcs"]
    setup = (T["setup_usd"] + man["new_die"] * (T["new_die_usd"] + T["new_die_per_hollow_usd"] * f["n_hollows"])
             ) / order_kg
    rush = 1 + T["rush"]["per_16_days"] * max(0, (T["rush"]["below_days"] - lead_true) / 16)
    metal = lme_ma1m / 1000 + T["regional_premium"] + T["alloy_premium"][spec["material"]["alloy"]]
    return metal + margin * customer_factor * rush * (conversion_base * complexity * tolerance + finish + setup)


def simulate_quotes(rng, lme, tenant, cfg, n=None, start=None, end=None, base_factor=1.0, customers=None,
                    id_prefix="Q"):
    """Quotes of one tenant as raw JSON records and their truth rows."""
    n = n or cfg["n_quotes"]
    v = cfg["schema_version"]
    if customers is None:
        k = cfg["n_customers"]
        weights = 1 / np.arange(1, k + 1) ** 1.1
        customers = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in rng.permutation(np.arange(1000, 10000))[:k]],
            "segment": _pick(rng, SEGMENTS, k),
            "discount": np.exp(rng.normal(0, TRUTH["customer_discount_log_sd"], k)),
            "weight": weights / weights.sum(),
        })
    dates = _quote_dates(rng, start or cfg["start"], end or cfg["end"], n)
    lme_ma = lme_features(lme, dates)["lme_ma1m"].to_numpy()
    records, truth = [], []
    for i, d in enumerate(dates):
        cust = customers.iloc[rng.choice(len(customers), p=customers["weight"])]
        kind = _pick(rng, PROFILE_TYPES)
        outer, holes, wall = make_profile(rng, kind)
        outer, holes = _clean_polygon(outer), [_clean_polygon(h) for h in holes]
        ccd = profile_features(outer, holes, wall)["ccd_mm"]
        lin = None if rng.random() < .55 else round(standard_linear_tol(ccd) * rng.uniform(.3, 1), 2)
        man = {"quantity_pcs": int(np.exp(rng.uniform(np.log(20), np.log(5000)))),
               "new_die": bool(rng.random() < .35)}
        if v == 1:
            man["tolerance_mm"] = lin
        else:
            man["linear_tol_mm"] = lin
            man["gdt"] = {name: (round(std * rng.uniform(.25, 1), 3) if rng.random() < .25 else None)
                          for name, std in STANDARD_GDT.items()}
        finish = str(_pick(rng, FINISHES))
        lead = int(rng.integers(5, 61))
        spec = {
            "schema_version": v,
            "quote_id": f"{id_prefix}{i + 1:05d}",
            "quote_date": pd.Timestamp(d).date().isoformat(),
            "customer": {"customer_id": cust["customer_id"],
                         "segment": None if rng.random() < .08 else cust["segment"]},
            "material": {"alloy": str(_pick(rng, ALLOYS)), "finish": None if rng.random() < .03 else finish},
            "geometry": {"profile_type": kind, "cross_section": {"outer": outer, "holes": holes},
                         "min_wall_mm": round(wall, 2),
                         "length_mm": float(rng.choice([3000, 4000, 5000, 6000, 6500]))},
            "manufacturing": man,
            "commercial": {"lead_time_days": None if rng.random() < .05 else lead},
        }
        factor = TRUTH["segment_factor"][cust["segment"]] * cust["discount"]
        mean = expected_price(spec, finish, lead, lme_ma[i], factor, cfg["conversion_base"] * base_factor,
                              cfg["margin"])
        price = mean * np.exp(rng.normal(0, TRUTH["noise_log_sd"]))
        outlier = rng.random() < TRUTH["outlier_share"]
        if outlier:
            price *= rng.uniform(1.5, 2.2) if rng.random() < .6 else rng.uniform(.55, .75)
        spec["quoted_price_per_kg"] = round(float(price), 2)
        records.append(spec)
        truth.append({"tenant": tenant, "quote_id": spec["quote_id"], "expected_price_per_kg": round(mean, 4),
                      "is_outlier": outlier, "profile_type": kind})
    return records, pd.DataFrame(truth), customers


def add_invalid(records, rng):
    """A few records that must be rejected by schema validation."""
    bad = [json.loads(json.dumps(records[i])) for i in rng.choice(len(records), 4, replace=False)]
    bad[0]["manufacturing"]["quantity_pcs"] = 0
    bad[1]["material"]["alloy"] = "7075"
    del bad[2]["quoted_price_per_kg"]
    bad[3]["geometry"]["min_wall_mm"] = -1.0
    for j, b in enumerate(bad):
        b["quote_id"] = f"BAD{j + 1}"
    return records + bad


def write_jsonl(path, records):
    with open(path, "w") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data", help="output directory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lme = simulate_lme(rng)
    lme.rename_axis("date").reset_index().to_csv(out / "lme.csv", index=False)
    truths, customers_of = [], {}
    for tenant, cfg in TRUTH["tenants"].items():
        records, truth, customers_of[tenant] = simulate_quotes(rng, lme, tenant, cfg, id_prefix=tenant[-1].upper())
        write_jsonl(out / f"quotes_{tenant}.jsonl", add_invalid(records, rng))
        truths.append(truth)
    fb = TRUTH["feedback"]
    records, truth, _ = simulate_quotes(rng, lme, fb["tenant"], TRUTH["tenants"][fb["tenant"]], n=fb["n_quotes"],
                                        start=fb["start"], end=fb["end"], base_factor=fb["conversion_base_factor"],
                                        customers=customers_of[fb["tenant"]], id_prefix="F")
    write_jsonl(out / f"feedback_{fb['tenant']}.jsonl", records)
    truths.append(truth)
    pd.concat(truths).to_csv(out / "truth_prices.csv", index=False)
    (out / "truth.json").write_text(json.dumps({"seed": args.seed, **TRUTH}, indent=2))
    print(f"wrote {out}/: lme.csv, quotes_*.jsonl, feedback_{fb['tenant']}.jsonl, truth.json, truth_prices.csv")


if __name__ == "__main__":
    main()
