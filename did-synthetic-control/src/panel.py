"""Panel data helpers: long table -> unit x week matrix of log outcomes, launch weeks and attributes."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    Y: np.ndarray            # units x weeks, log outcome
    G: np.ndarray            # launch week per unit, inf if never launched
    weeks: np.ndarray        # week labels
    units: np.ndarray        # unit labels
    attrs: pd.DataFrame      # one row per unit, same order as Y
    effect: np.ndarray | None = None   # true effect (simulation only), same shape as Y

    @property
    def treated(self) -> np.ndarray:
        return np.isfinite(self.G)

    @property
    def D(self) -> np.ndarray:
        """Treatment indicator per unit and week."""
        return (self.weeks[None, :] >= self.G[:, None]).astype(float)

    def subset(self, rows) -> "Panel":
        return Panel(self.Y[rows], self.G[rows], self.weeks, self.units[rows], self.attrs.iloc[rows].reset_index(drop=True),
                     None if self.effect is None else self.effect[rows])


def to_panel(df: pd.DataFrame, unit: str, time: str = "week", outcome: str = "sales",
             launch: str = "launch_week", log: bool = True) -> Panel:
    """Balanced long panel -> Panel. Raises if the panel is unbalanced or has missing outcomes."""
    wide = df.pivot(index=unit, columns=time, values=outcome).sort_index()
    if wide.isna().any().any():
        raise ValueError("panel must be balanced: every unit observed in every period")
    Y = wide.to_numpy(float)
    if log:
        if (Y <= 0).any():
            raise ValueError("log outcome needs positive values")
        Y = np.log(Y)
    per_unit = df.drop_duplicates(unit).set_index(unit).loc[wide.index]
    G = per_unit[launch].to_numpy(float)
    G = np.where(np.isnan(G), np.inf, G)
    attrs = per_unit.drop(columns=[c for c in [time, outcome, launch, "true_effect"] if c in per_unit]).reset_index()
    effect = None
    if "true_effect" in df:
        effect = df.pivot(index=unit, columns=time, values="true_effect").sort_index().to_numpy(float)
    return Panel(Y, G, wide.columns.to_numpy(), wide.index.to_numpy(), attrs, effect)
