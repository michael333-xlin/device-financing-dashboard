"""Decision rules and business-value calculations. Pure functions, no Streamlit.

Profit framework (from the preparation document):
  expected profit per funded plan = (1 - p) * G - p * L
  break-even default probability    = G / (G + L)
  incremental value                 = proposed total profit - status quo total profit
Declined or walked-away applications contribute zero profit (explicit assumption).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as cfg


# ----------------------------------------------------------------------------- basics
def expected_profit(p, gain, loss):
    return (1 - p) * gain - p * loss


def break_even(gain: float, loss: float) -> float:
    return gain / (gain + loss)


def payoff_lookup(payoffs: pd.DataFrame, regime: str) -> dict:
    sub = payoffs[payoffs["REGIME"] == regime].set_index("DEVICE_TIER")
    return {t: (float(r["GAIN"]), float(r["LOSS"])) for t, r in sub.iterrows()}


# ----------------------------------------------------------------------------- application review
LABELS = {"CREDIT_SCORE": "Credit score", "INTERNAL_RISK_SCORE": "Internal risk score",
          "AGE": "Age", "INCOME": "Income"}


def current_lane(internal_score: float, policy: pd.DataFrame, regime: str) -> str:
    row = policy[policy["REGIME"] == regime].iloc[0]
    if internal_score <= row["APPROVE_MAX"]:
        return "Approve"
    if internal_score <= row["CONDITIONAL_MAX"]:
        return "Conditional"
    return "Deny"


def validate_application(rec: dict, bounds: pd.DataFrame, regime: str) -> tuple[list, list]:
    """Return (errors, warnings). Errors route to Review; warnings are shown to the rep."""
    errors, warnings = [], []
    for feat, (lo, hi) in cfg.INPUT_LIMITS.items():
        v = rec.get(feat)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{LABELS[feat]} is missing.")
        elif not lo <= v <= hi:
            errors.append(f"{LABELS[feat]} {v:,} is outside the valid range {lo:,}–{hi:,}.")
    if rec.get("DEVICE_REQUESTED") not in cfg.TIERS:
        errors.append("Requested device tier must be Low, Mid or High.")
    if rec.get("EXISTING_CUSTOMER") not in (0, 1):
        errors.append("Customer status must be new or existing.")
    if errors:
        return errors, warnings

    b = bounds[bounds["REGIME"] == regime].set_index("FEATURE")
    for feat in ("CREDIT_SCORE", "INTERNAL_RISK_SCORE"):
        lo, hi = b.loc[feat, "FUNDED_MIN"], b.loc[feat, "FUNDED_MAX"]
        if not lo <= rec[feat] <= hi:
            warnings.append(
                f"{LABELS[feat]} {rec[feat]:,.3g} is outside the range ever funded in "
                f"{regime} ({lo:,.3g} to {hi:,.3g}). The model would be extrapolating."
            )
    return errors, warnings


@dataclass
class Recommendation:
    action: str          # Approve / Negotiate / Decline / Review
    terms: str           # what the rep offers
    basis: str           # current rule, model + policy, or validation
    reasons: list = field(default_factory=list)


def recommend(pd_hgb: float, lane: str, errors: list, warnings: list,
              ceiling: float, be: float) -> Recommendation:
    """Proposed policy: keep the current Approve and Deny lanes; use the model only inside
    the Conditional lane, where outcomes are observed and walk-aways cost profit."""
    if errors:
        return Recommendation("Review", "Hold — supervisor review", "Input validation", errors)
    if warnings:
        return Recommendation("Review", "Hold — supervisor review", "Outside funded range", warnings)
    if pd_hgb >= be:
        return Recommendation("Decline", "No financing offer", "Model + break-even",
                              [f"Predicted default {pd_hgb:.1%} is at or above break-even {be:.1%}."])
    if lane == "Approve":
        return Recommendation("Approve", "Requested device, standard down payment", "Current policy rule",
                              ["Internal score is in the current approve band; the proposal leaves this lane unchanged."])
    if lane == "Deny":
        return Recommendation("Decline", "No financing offer", "Current policy rule",
                              ["Internal score is in the current deny band; the proposal leaves this lane unchanged."])
    if pd_hgb < ceiling:
        return Recommendation(
            "Approve", "Requested device, standard down payment", "Model fast-track (proposed)",
            [f"Conditional band, but predicted default {pd_hgb:.1%} is below the fast-track ceiling "
             f"{ceiling:.0%}. Skipping negotiation avoids the historical walk-away rate."],
        )
    return Recommendation(
        "Negotiate", "Revised terms — compare offers", "Current negotiation process",
        [f"Conditional band and predicted default {pd_hgb:.1%} is at or above the fast-track ceiling "
         f"{ceiling:.0%}. Use Offer Comparison to choose revised terms."],
    )


# ----------------------------------------------------------------------------- offers
def offer_table(offers: list[dict], payoffs: dict) -> pd.DataFrame:
    rows = []
    for o in offers:
        gain, loss = payoffs[o["tier"]]
        down = o["price"] * o["down_pct"]
        ep = expected_profit(o["pd"], gain, loss)
        rows.append(dict(
            OFFER=o["name"], TIER=o["tier"], PRICE=o["price"], DOWN_PCT=o["down_pct"],
            DOWN_AMOUNT=down, FINANCED=o["price"] - down, PD=o["pd"], PD_SOURCE=o["pd_source"],
            GAIN=gain, LOSS=loss, BREAK_EVEN=break_even(gain, loss),
            EP_IF_ACCEPTED=ep, ACCEPTANCE=o["accept"], EXPECTED_VALUE=o["accept"] * ep,
            PERMITTED=o.get("permitted", True),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 1M scenario
@dataclass
class ScenarioParams:
    ceiling: float = 0.16          # fast-track ceiling on predicted default (pilot setting)
    accept: float = 0.50           # share of former walk-aways who accept original terms
    risk_mult: float = 1.5         # true default = multiplier x predicted (selection-bias haircut)
    include_downgrade: bool = True  # revised plans keep the requested device under fast-track
    pooled_dd_pp: float = 0.0      # extra pooled default reduction on the existing book (pp)
    var_cost: float = 0.10         # USD per application (inference, monitoring, ops)
    fixed_cost: float = 500_000.0  # build, validation, pilot, training
    applications: int = cfg.SCENARIO_APPLICATIONS


def status_quo(apps: pd.DataFrame) -> dict:
    """apps: final-regime applications with FUNDED, DEFAULT_FLAG, PROFIT."""
    n = len(apps)
    funded = int(apps["FUNDED"].sum())
    defaults = float(apps["DEFAULT_FLAG"].fillna(0).sum())
    return dict(n=n, funded=funded, defaults=defaults, profit=float(apps["PROFIT"].sum()))


def run_scenario(sq: dict, cond: pd.DataFrame, pay: dict, p: ScenarioParams) -> dict:
    """cond: final-regime Conditional applications with PD_HGB, DEVICE_REQUESTED,
    DEVICE_FINAL, FINAL_STATUS. pay: {tier: (G, L)} incl. 'ALL'."""
    scale = p.applications / sq["n"]
    g_req = cond["DEVICE_REQUESTED"].map(lambda t: pay[t][0])
    l_req = cond["DEVICE_REQUESTED"].map(lambda t: pay[t][1])
    g_fin = cond["DEVICE_FINAL"].map(lambda t: pay[t][0])
    l_fin = cond["DEVICE_FINAL"].map(lambda t: pay[t][1])
    q = np.clip(p.risk_mult * cond["PD_HGB"], 0, 1)
    eligible = cond["PD_HGB"] < p.ceiling

    # A. Walk-away recovery: Conditional, not funded, now offered original terms.
    walk = eligible & (cond["FINAL_STATUS"] == "Not_Funded")
    ev_walk = p.accept * ((1 - q) * g_req - q * l_req)
    lever_a = float(ev_walk[walk].sum()) * scale
    added_funded = p.accept * float(walk.sum()) * scale
    added_defaults = p.accept * float(q[walk].sum()) * scale

    # C. Downgrade avoidance: revised plans that were downgraded now keep the requested tier.
    dg = eligible & (cond["FINAL_STATUS"] == "Funded_Revised") & (cond["DEVICE_REQUESTED"] != cond["DEVICE_FINAL"])
    inc = (1 - q) * (g_req - g_fin) - q * (l_req - l_fin)
    lever_c = float(inc[dg].sum()) * scale if p.include_downgrade else 0.0

    # B. Pooled default reduction on the existing funded book (unmeasured; default 0).
    g_all, l_all = pay["ALL"]
    sq_funded = sq["funded"] * scale
    lever_b = sq_funded * (p.pooled_dd_pp / 100) * (g_all + l_all)

    costs = p.var_cost * p.applications + p.fixed_cost
    sq_profit = sq["profit"] * scale
    net = lever_a + lever_b + lever_c - costs

    prop_funded = sq_funded + added_funded
    prop_defaults = sq["defaults"] * scale + added_defaults - sq_funded * p.pooled_dd_pp / 100
    return dict(
        scale=scale,
        eligible_share=float(eligible.mean()),
        walk_count=float(walk.sum()) * scale,
        downgrade_count=float(dg.sum()) * scale if p.include_downgrade else 0.0,
        lever_a=lever_a, lever_b=lever_b, lever_c=lever_c, costs=costs, net=net,
        table=pd.DataFrame(
            {
                "Status quo (observed)": [
                    p.applications, sq_funded, sq_funded / p.applications,
                    sq["defaults"] * scale / sq_funded, sq_profit, 0.0, sq_profit,
                ],
                "Proposed (scenario)": [
                    p.applications, prop_funded, prop_funded / p.applications,
                    prop_defaults / prop_funded, sq_profit + lever_a + lever_b + lever_c,
                    costs, sq_profit + net,
                ],
            },
            index=["Applications", "Funded plans", "Funding rate", "Default rate (funded)",
                   "Contribution profit", "Incremental costs", "Net profit"],
        ),
    )


def breakeven_multiplier(sq, cond, pay, p: ScenarioParams, hi: float = 10.0) -> float | None:
    """Risk multiplier at which the proposal's net incremental value reaches zero."""
    def net(m):
        return run_scenario(sq, cond, pay, ScenarioParams(**{**p.__dict__, "risk_mult": m}))["net"]

    lo = 0.0
    if net(lo) <= 0:
        return 0.0
    if net(hi) > 0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def sensitivity(sq, cond, pay, p: ScenarioParams,
                mults=(1.0, 1.5, 2.0, 2.5), accepts=(0.25, 0.50, 0.75)) -> pd.DataFrame:
    grid = {}
    for a in accepts:
        grid[f"Acceptance {a:.0%}"] = [
            run_scenario(sq, cond, pay, ScenarioParams(**{**p.__dict__, "risk_mult": m, "accept": a}))["net"]
            for m in mults
        ]
    return pd.DataFrame(grid, index=[f"Risk ×{m:g}" for m in mults])


def pooled_illustration(funded: float, dd: float, gain: float, loss: float) -> float:
    """Preparation-document check: 800,000 x 0.005 x 468.06 = 1,872,240."""
    return funded * dd * (gain + loss)