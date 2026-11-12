"""Device-financing decision support — Streamlit in Snowflake prototype.

Pages
  Application review  – score one applicant, proposed action, reasons
  Offer comparison    – compare financing offers with explicit risk/acceptance assumptions
  Portfolio overview  – historical KPIs, model evidence, one-million-application value
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
import finance_rules as fin
import ui
from data_layer import T, base_cte, get_backend, lookup_tables, query, sql_in
from scoring import FEATURES, get_scorer, score

st.set_page_config(page_title="Device financing decisions", layout="wide")
ui.inject_css()

PAGES = ["Application review", "Offer comparison", "Portfolio overview"]

EXAMPLES = {
    "Conditional band, moderate risk": dict(CREDIT_SCORE=640, INTERNAL_RISK_SCORE=1.05, AGE=34,
                                            INCOME=72_000, EXISTING_CUSTOMER=0, DEVICE_REQUESTED="High"),
    "Conditional band, higher risk": dict(CREDIT_SCORE=520, INTERNAL_RISK_SCORE=1.40, AGE=22,
                                          INCOME=28_000, EXISTING_CUSTOMER=1, DEVICE_REQUESTED="High"),
    "Approve band, typical": dict(CREDIT_SCORE=700, INTERNAL_RISK_SCORE=-0.40, AGE=45,
                                  INCOME=85_000, EXISTING_CUSTOMER=1, DEVICE_REQUESTED="Mid"),
    "Outside the funded range": dict(CREDIT_SCORE=560, INTERNAL_RISK_SCORE=2.10, AGE=29,
                                     INCOME=41_000, EXISTING_CUSTOMER=0, DEVICE_REQUESTED="Mid"),
}
SERIES_NAMES = {"ACTUAL": "Actual default rate", "PRED_INTERNAL": "Internal risk score only",
                "PRED_HGB": "Gradient boosting (proposed)"}
FEATURE_LABELS = {
    "CREDIT_SCORE": "Credit score", "INTERNAL_RISK_SCORE": "Internal risk score", "AGE": "Age",
    "INCOME": "Income", "EXISTING_CUSTOMER": "Customer status", "DEVICE_REQUESTED": "Requested device",
}


# ============================================================================ sidebar
def sidebar():
    with st.sidebar:
        st.markdown("### Device financing\nDecision-support prototype")
        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.divider()
        regime = st.selectbox(
            "Policy regime assumption", cfg.REGIMES, index=2, format_func=cfg.REGIME_LABELS.get,
            help="The model uses the policy regime as an input. New applications are scored as the "
                 "current regime; the model cannot anticipate a future scoring or economic regime.",
        )
        ceiling = st.slider(
            "Fast-track ceiling (predicted default)", 10.0, 25.0, 16.0, 0.5, format="%.1f%%",
            help="Conditional-band applicants below this predicted default are approved on their "
                 "requested terms instead of being negotiated. Team-agreed demonstration rule.",
        ) / 100
        st.divider()
        scorer = get_scorer()
        st.caption(f"**Data** · {get_backend().name}")
        st.caption(f"**Model** · {scorer.source}")
        st.caption(scorer.detail)
    return page, regime, ceiling, scorer


# ============================================================================ page 1
def page_application(L, scorer, regime, ceiling):
    ui.page_header(
        "Application review",
        "Score one application before the financing conversation. The company's approve and deny "
        "bands stay as they are; the model decides only who in the conditional band can skip negotiation.",
    )
    pay = fin.payoff_lookup(L["payoffs"], regime)
    g_all, l_all = pay["ALL"]
    be = fin.break_even(g_all, l_all)

    left, right = st.columns([0.85, 1.45], gap="large")
    with left:
        preset = st.selectbox("Start from an example", list(EXAMPLES))
        ex = EXAMPLES[preset]
        with st.form("application"):
            k = preset  # keys tied to the example so switching examples refills the form
            credit = st.number_input("Credit score (FICO)", 300, 850, ex["CREDIT_SCORE"], 1, key=f"c{k}")
            internal = st.number_input("The company's internal risk score", -3.74, 6.87,
                                       float(ex["INTERNAL_RISK_SCORE"]), 0.01, format="%.2f", key=f"i{k}")
            age = st.number_input("Age", 18, 79, ex["AGE"], 1, key=f"a{k}")
            income = st.number_input("Annual income (USD)", 15_000, 200_000, ex["INCOME"], 1_000, key=f"n{k}")
            customer = st.radio("Customer status", ["New customer", "Existing customer"],
                                index=ex["EXISTING_CUSTOMER"], horizontal=True, key=f"e{k}")
            device = st.radio("Requested device tier", cfg.TIERS, index=cfg.TIERS.index(ex["DEVICE_REQUESTED"]),
                              horizontal=True, key=f"d{k}")
            submitted = st.form_submit_button("Score application", type="primary", **ui.FULL)

    if submitted:
        rec = dict(CREDIT_SCORE=int(credit), INTERNAL_RISK_SCORE=float(internal), AGE=int(age),
                   INCOME=float(income), EXISTING_CUSTOMER=1 if customer.startswith("Existing") else 0,
                   DEVICE_REQUESTED=device, REGIME=regime)
        st.session_state["review"] = evaluate_application(rec, L, scorer, regime, ceiling, be)

    rev = st.session_state.get("review")
    with right:
        if rev is None:
            st.info("Choose an example or enter an application, then select **Score application**.")
            return
        if rev["regime"] != regime or abs(rev["ceiling"] - ceiling) > 1e-9:
            st.warning("Policy settings changed since this result. Select **Score application** to refresh.")
        r = rev["rec"]
        ui.decision_card(rev["action"].action, rev["action"].terms, rev["action"].basis)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted default", ui.pct(rev["pd"]["PD_HGB"]),
                  help="Gradient-boosting model, application-stage inputs only.")
        m2.metric("Expected profit if funded", ui.usd(rev["ep"]),
                  help=ui.md(f"(1 − p) × {ui.usd(rev['g'])} − p × {ui.usd(rev['l'])}, "
                             f"{r['DEVICE_REQUESTED']} tier, {regime} payoffs."))
        m3.metric("Break-even default", ui.pct(be), help="Pooled G / (G + L) for the regime.")
        for reason in rev["action"].reasons:
            st.markdown(f"- {reason}")
        st.caption(f"Current company's rule places this application in the **{rev['lane']}** band "
                   f"(internal score {r['INTERNAL_RISK_SCORE']:.2f}; inferred cutoffs "
                   f"{rev['cutoffs'][0]:.3f} / {rev['cutoffs'][1]:.3f}).")
        for w in rev["warnings"]:
            st.warning(w)

        st.subheader("Same applicant, three models")
        cmp = pd.DataFrame({
            "Model": [cfg.MODEL_LABELS[k] for k in ("HGB", "INTERNAL", "CREDIT")],
            "Predicted default": [rev["pd"][f"PD_{k}"] for k in ("HGB", "INTERNAL", "CREDIT")],
        })
        st.dataframe(cmp.style.format({"Predicted default": "{:.1%}"}), hide_index=True,
                     **ui.FULL)

        st.subheader("What moves this estimate")
        st.caption("Change in predicted default if each input were at the typical funded value for "
                   "the final regime, holding the others fixed. A sensitivity check, not a causal effect.")
        drivers = rev["drivers"]
        bars = alt.Chart(drivers).mark_bar(size=18).encode(
            x=alt.X("DELTA_PP:Q", title="Percentage points added to predicted default"),
            y=alt.Y("LABEL:N", sort="-x", title=None),
            color=alt.condition("datum.DELTA_PP > 0", alt.value(ui.BRAND), alt.value("#4A5866")),
            tooltip=[alt.Tooltip("LABEL:N", title="Input"), alt.Tooltip("DETAIL:N", title="Compared with"),
                     alt.Tooltip("DELTA_PP:Q", title="pp", format="+.2f")],
        )
        text = bars.mark_text(align="left", dx=4, color=ui.INK).encode(
            text=alt.Text("DELTA_PP:Q", format="+.2f"))
        ui.show_chart((bars + text).properties(height=230))
        with st.expander("Model input and source"):
            st.json({**r, "model_source": scorer.source, "model_detail": scorer.detail})


def evaluate_application(rec, L, scorer, regime, ceiling, be):
    errors, warnings = fin.validate_application(rec, L["bounds"], regime)
    ref = L["reference"].set_index("FEATURE")["REFERENCE_VALUE"].to_dict()
    rows = [rec]
    labels = []
    for feat in ["CREDIT_SCORE", "INTERNAL_RISK_SCORE", "AGE", "INCOME", "EXISTING_CUSTOMER", "DEVICE_REQUESTED"]:
        alt_val = "Mid" if feat == "DEVICE_REQUESTED" else ref[feat]
        if feat == "EXISTING_CUSTOMER":
            alt_val = 1 - rec[feat]
        if feat in ("CREDIT_SCORE", "AGE", "EXISTING_CUSTOMER"):
            alt_val = int(alt_val)
        rows.append({**rec, feat: alt_val})
        shown = {0: "new customer", 1: "existing customer"}.get(alt_val) if feat == "EXISTING_CUSTOMER" else alt_val
        labels.append((FEATURE_LABELS[feat], f"{shown:,.2f}" if isinstance(shown, float) else f"{shown}"))

    preds = score(pd.DataFrame(rows))
    main = preds.iloc[0]
    drivers = pd.DataFrame({
        "LABEL": [l for l, _ in labels], "DETAIL": [d for _, d in labels],
        "DELTA_PP": (main["PD_HGB"] - preds["PD_HGB"].iloc[1:].to_numpy()) * 100,
    })
    lane = fin.current_lane(rec["INTERNAL_RISK_SCORE"], L["policy"], regime)
    action = fin.recommend(main["PD_HGB"], lane, errors, warnings, ceiling, be)
    pay = fin.payoff_lookup(L["payoffs"], regime)
    g, l = pay[rec["DEVICE_REQUESTED"]]
    pol = L["policy"][L["policy"]["REGIME"] == regime].iloc[0]
    return dict(rec=rec, pd=main.to_dict(), lane=lane, action=action, warnings=warnings if not errors else [],
                drivers=drivers, ep=fin.expected_profit(main["PD_HGB"], g, l), g=g, l=l,
                regime=regime, ceiling=ceiling, cutoffs=(pol["APPROVE_MAX"], pol["CONDITIONAL_MAX"]))


# ============================================================================ page 2
def page_offers(L, regime):
    ui.page_header(
        "Offer comparison",
        "Compare financing options for the reviewed applicant. Down-payment amounts are arithmetic; "
        "any change in default risk from revised terms is an entered assumption, not a model output.",
    )
    rev = st.session_state.get("review")
    pay = fin.payoff_lookup(L["payoffs"], regime)
    tiers = L["tiers"].set_index("DEVICE_TIER")
    accept_hist = revised_acceptance(regime)

    if rev is None:
        st.info("No reviewed application yet. Score one on **Application review**, or set a reference below.")
        c1, c2 = st.columns(2)
        ref_pd = c1.number_input("Reference predicted default (%)", 1.0, 60.0, 15.0, 0.1) / 100
        req = c2.radio("Requested device tier", cfg.TIERS, index=2, horizontal=True)
        action = "Negotiate"
    else:
        ref_pd, req, action = rev["pd"]["PD_HGB"], rev["rec"]["DEVICE_REQUESTED"], rev["action"].action
        ui.tags(f"Applicant: {req} tier", f"Predicted default {ui.pct(ref_pd)}",
                f"Proposed action: {action}", regime)

    lower = {"High": "Mid", "Mid": "Low", "Low": "Low"}[req]
    permitted_all = action == "Approve"
    defaults = [
        dict(name="A · Requested device, standard terms", tier=req, down_add=0.0, accept=1.0,
             permitted=permitted_all, note="Original offer"),
        dict(name="B · Requested device, higher down payment", tier=req, down_add=0.10, accept=accept_hist,
             permitted=action in ("Approve", "Negotiate"), note="Revised terms"),
        dict(name="C · One tier lower", tier=lower, down_add=0.0, accept=accept_hist,
             permitted=action in ("Approve", "Negotiate"), note="Downgrade"),
    ]
    st.caption(f"Default acceptance for revised offers is the historical share of conditional applicants "
               f"who accepted revised terms in {regime}: {accept_hist:.0%}. Original-offer acceptance is "
               "assumed to be 100%.")

    offers = []
    cols = st.columns(3, gap="medium")
    for col, d in zip(cols, defaults):
        with col:
            st.markdown(f"**{d['name']}**")
            tier = st.selectbox("Device tier", cfg.TIERS, index=cfg.TIERS.index(d["tier"]), key=f"t{d['name']}")
            price = st.number_input("Device price (USD)", 300, 1500, int(tiers.loc[tier, "MEDIAN_PRICE"]), 10,
                                    key=f"p{d['name']}{tier}")
            base = float(tiers.loc[tier, "MEAN_BASE_DOWN"])
            down = st.slider("Down payment", 0.0, 50.0, round((base + d["down_add"]) * 100, 1), 0.5,
                             format="%.1f%%", key=f"dp{d['name']}{tier}") / 100
            pd_in = st.number_input("Default risk for this offer (%)", 0.5, 80.0, round(ref_pd * 100, 2), 0.1,
                                    key=f"r{d['name']}", help="Defaults to the application-stage prediction. "
                                    "Lowering it here is a scenario assumption the model does not support.") / 100
            acc = st.slider("Acceptance", 0, 100, int(round(d["accept"] * 100)), 5, format="%d%%",
                            key=f"ac{d['name']}") / 100
            source = "Application-stage model" if abs(pd_in - ref_pd) < 5e-5 else "Entered scenario risk"
            offers.append(dict(name=d["name"], tier=tier, price=price, down_pct=down, pd=pd_in,
                               pd_source=source, accept=acc, permitted=d["permitted"]))

    tbl = fin.offer_table(offers, pay)
    if (tbl["PD_SOURCE"] == "Entered scenario risk").any():
        ui.scenario_note("At least one offer uses an entered default risk. The current model has no "
                         "down-payment input and no validated effect of revised terms on default.")

    tbl["STATUS"] = np.where(tbl["PERMITTED"], "Permitted", "Not permitted under proposed action")
    chart_df = tbl.assign(LABEL=tbl["EXPECTED_VALUE"].map(ui.usd))
    bars = alt.Chart(chart_df).mark_bar(size=34).encode(
        y=alt.Y("OFFER:N", title=None, sort=None),
        x=alt.X("EXPECTED_VALUE:Q", title="Expected value per application (USD) = acceptance × expected profit"),
        color=alt.Color("STATUS:N", title=None,
                        scale=alt.Scale(domain=["Permitted", "Not permitted under proposed action"],
                                        range=["#1E7B4F", "#B9C0C8"])),
        tooltip=["OFFER", alt.Tooltip("EXPECTED_VALUE:Q", format="$,.0f")],
    )
    labels = bars.mark_text(align="left", dx=5, color=ui.INK).encode(text="LABEL:N")
    ui.show_chart((bars + labels).properties(height=190))

    show = tbl[["OFFER", "TIER", "PRICE", "DOWN_AMOUNT", "FINANCED", "PD", "PD_SOURCE", "GAIN", "LOSS",
                "EP_IF_ACCEPTED", "ACCEPTANCE", "EXPECTED_VALUE", "STATUS"]].rename(columns={
        "OFFER": "Offer", "TIER": "Tier", "PRICE": "Price", "DOWN_AMOUNT": "Down payment",
        "FINANCED": "Financed", "PD": "Default risk", "PD_SOURCE": "Risk source", "GAIN": "Gain (G)",
        "LOSS": "Loss (L)", "EP_IF_ACCEPTED": "Expected profit if accepted", "ACCEPTANCE": "Acceptance",
        "EXPECTED_VALUE": "Expected value", "STATUS": "Policy"})
    money = {c: "${:,.0f}" for c in ["Price", "Down payment", "Financed", "Gain (G)", "Loss (L)",
                                     "Expected profit if accepted", "Expected value"]}
    st.dataframe(show.style.format({**money, "Default risk": "{:.1%}", "Acceptance": "{:.0%}"}),
                 hide_index=True, **ui.FULL)

    allowed = tbl[tbl["PERMITTED"]]
    if allowed.empty:
        st.markdown(f"**No offer is permitted** because the proposed action is *{action}*. "
                    "The table is shown for discussion only.")
    else:
        best = allowed.sort_values("EXPECTED_VALUE", ascending=False).iloc[0]
        st.markdown(ui.md(f"**Highest expected value among permitted offers:** {best['OFFER']} "
                          f"({ui.usd(best['EXPECTED_VALUE'])} per application)."))
    st.caption(
        f"Why downgrades rarely pay: in {regime} the break-even default is about "
        f"{tbl['BREAK_EVEN'].min():.1%}–{tbl['BREAK_EVEN'].max():.1%} for every tier, because gain and loss "
        "scale together with device value. A lower tier reduces dollars at risk and dollars earned in "
        "proportion; historically, downgraded revised plans defaulted at about the same rate as "
        "non-downgraded ones."
    )


def revised_acceptance(regime: str) -> float:
    df = query(f"""{base_cte()}
        SELECT AVG(CASE WHEN FINAL_STATUS = 'Funded_Revised' THEN 1.0 ELSE 0.0 END) AS ACC
        FROM A WHERE INITIAL_DECISION = 'Conditional' AND REGIME = ?""", (regime,))
    return float(df["ACC"].iloc[0])


# ============================================================================ page 3
def page_portfolio(L, regime, ceiling):
    ui.page_header(
        "Portfolio overview",
        "Historical performance, model evidence on fixed samples, and the value of the proposed policy "
        "on a hypothetical base of one million applications.",
    )
    t1, t2, t3 = st.tabs(["Historical performance", "Model evidence", "Business value · 1M applications"])
    with t1:
        tab_history()
    with t2:
        tab_evidence(L)
    with t3:
        tab_value(L, ceiling)


def tab_history():
    c1, c2, c3 = st.columns(3)
    regimes = c1.multiselect("Policy regime", cfg.REGIMES, default=cfg.REGIMES)
    cust = c2.multiselect("Customer status", ["New", "Existing"], default=["New", "Existing"])
    dev = c3.multiselect("Requested device", cfg.TIERS, default=cfg.TIERS)
    cust_vals = ", ".join(str(int(c == "Existing")) for c in cust) or "NULL"
    where = (f"REGIME IN {sql_in(regimes, cfg.REGIMES)} AND EXISTING_CUSTOMER IN ({cust_vals}) "
             f"AND DEVICE_REQUESTED IN {sql_in(dev, cfg.TIERS)}")

    k = query(f"""{base_cte()}
        SELECT COUNT(*) AS APPS,
               AVG(CASE WHEN INITIAL_DECISION = 'Approve' THEN 1.0 ELSE 0.0 END) AS APPROVE_RATE,
               AVG(FUNDED * 1.0) AS FUND_RATE,
               SUM(DEFAULT_FLAG) * 1.0 / NULLIF(COUNT(DEFAULT_FLAG), 0) AS DEFAULT_RATE,
               AVG(PROFIT) AS PROFIT_PER_APP, SUM(PROFIT) AS TOTAL_PROFIT,
               SUM(CASE WHEN INITIAL_DECISION = 'Conditional' AND FUNDED = 0 THEN 1 ELSE 0 END) * 1.0
                 / NULLIF(SUM(CASE WHEN INITIAL_DECISION = 'Conditional' THEN 1 ELSE 0 END), 0) AS WALK_RATE
        FROM A WHERE {where}""")
    if k.empty or not k["APPS"].iloc[0]:
        st.info("No applications match these filters. Widen the selection.")
        return
    k = k.iloc[0]
    st.caption(f"Active filters: {', '.join(regimes) or 'none'} · {', '.join(cust) or 'none'} · "
               f"{', '.join(dev) or 'none'} — {int(k['APPS']):,} applications")
    m = st.columns(6)
    m[0].metric("Applications", f"{int(k['APPS']):,}")
    m[1].metric("Initial approval rate", ui.pct(k["APPROVE_RATE"]), help="Share with initial decision Approve.")
    m[2].metric("Funding rate", ui.pct(k["FUND_RATE"]), help="Funded plans ÷ applications.")
    m[3].metric("Default rate (funded)", ui.pct(k["DEFAULT_RATE"]),
                help="Defaults ÷ funded plans with known outcomes. Missing outcomes are never counted as non-defaults.")
    m[4].metric("Profit per application", ui.usd(k["PROFIT_PER_APP"], 2))
    m[5].metric("Conditional walk-away", ui.pct(k["WALK_RATE"]),
                help="Share of conditional applicants who were not funded after negotiation.")

    trend = query(f"""{base_cte()}
        SELECT DATE_TRUNC('month', APPLICATION_DATE) AS MONTH,
               AVG(CASE WHEN INITIAL_DECISION = 'Approve' THEN 1.0 ELSE 0.0 END) AS APPROVE_RATE,
               AVG(FUNDED * 1.0) AS FUND_RATE,
               SUM(DEFAULT_FLAG) * 1.0 / NULLIF(COUNT(DEFAULT_FLAG), 0) AS DEFAULT_RATE
        FROM A WHERE {where} GROUP BY 1 ORDER BY 1""")
    trend["MONTH"] = pd.to_datetime(trend["MONTH"])
    trend = trend.rename(columns={"APPROVE_RATE": "Initial approval rate", "FUND_RATE": "Funding rate",
                                  "DEFAULT_RATE": "Default rate (funded)"})
    long = trend.melt("MONTH", var_name="METRIC", value_name="VALUE")
    metric_order = ["Initial approval rate", "Funding rate", "Default rate (funded)"]
    base = alt.Chart(long).encode(
        x=alt.X("MONTH:T", title=None),
        y=alt.Y("VALUE:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("METRIC:N", title=None, scale=alt.Scale(domain=metric_order,
                                                               range=["#4A5866", ui.INK, ui.BRAND])),
        strokeDash=alt.StrokeDash("METRIC:N", title=None, scale=alt.Scale(domain=metric_order,
                                                                       range=[[4, 3], [1, 0], [1, 0]])),
        tooltip=[alt.Tooltip("MONTH:T", format="%b %Y"), "METRIC:N", alt.Tooltip("VALUE:Q", format=".1%")],
    )
    lines = base.mark_line(strokeWidth=2.2) + base.mark_point(filled=True, size=28)
    st.subheader("Monthly rates with policy breaks")
    ui.show_chart((lines + regime_rules()).properties(height=300))

    by_regime = query(f"""{base_cte()}
        SELECT REGIME, COUNT(*) AS APPLICATIONS, AVG(FUNDED * 1.0) AS FUNDING_RATE,
               SUM(DEFAULT_FLAG) * 1.0 / NULLIF(COUNT(DEFAULT_FLAG), 0) AS DEFAULT_RATE,
               AVG(PROFIT) AS PROFIT_PER_APP,
               SUM(CASE WHEN INITIAL_DECISION = 'Conditional' AND FUNDED = 0 THEN 1 ELSE 0 END) * 1.0
                 / NULLIF(SUM(CASE WHEN INITIAL_DECISION = 'Conditional' THEN 1 ELSE 0 END), 0) AS WALK_RATE
        FROM A WHERE {where} GROUP BY 1 ORDER BY 1""")
    by_regime.columns = ["Regime", "Applications", "Funding rate", "Default rate (funded)",
                         "Profit per application", "Conditional walk-away"]
    st.dataframe(by_regime.style.format({"Applications": "{:,.0f}", "Funding rate": "{:.1%}",
                                         "Default rate (funded)": "{:.2%}", "Profit per application": "${:,.2f}",
                                         "Conditional walk-away": "{:.1%}"}),
                 hide_index=True, **ui.FULL)
    st.caption("Monthly patterns pooled across years are not evidence of seasonality: the default-rate rise "
               "coincides with the 1 May 2024 policy break.")


def regime_rules():
    rules = pd.DataFrame({"DATE": pd.to_datetime(list(cfg.REGIME_BREAKS)),
                          "LABEL": ["Policy change 1 Aug 2023", "Policy change 1 May 2024"]})
    r = alt.Chart(rules).mark_rule(color="#8B939C", strokeDash=[2, 2]).encode(x="DATE:T")
    t = alt.Chart(rules).mark_text(align="left", dx=4, dy=-6, color=ui.MUTED, fontSize=11).encode(
        x="DATE:T", y=alt.value(8), text="LABEL:N")
    return r + t


def tab_evidence(L):
    metrics = L["metrics"]
    samples = list(dict.fromkeys(metrics["EVAL_SAMPLE"]))
    st.subheader("Predictive accuracy on a common holdout")
    sample = st.radio("Evaluation sample", samples, horizontal=True)
    mm = metrics[metrics["EVAL_SAMPLE"] == sample].copy()
    mm["LABEL"] = mm["LOG_LOSS"].map(lambda v: f"{v:.6f}")
    order = [cfg.MODEL_LABELS[k] for k in ("CREDIT", "INTERNAL", "HGB")]
    lo = mm["LOG_LOSS"].min()
    bars = alt.Chart(mm).mark_bar(size=26, clip=True).encode(
        y=alt.Y("MODEL_LABEL:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("LOG_LOSS:Q", title="Test log loss (lower is better)",
                scale=alt.Scale(domain=[lo * 0.97, mm["LOG_LOSS"].max() * 1.005])),
        color=alt.Color("MODEL_LABEL:N", scale=ui.model_color_scale(order), legend=None),
    )
    ui.show_chart((bars + bars.mark_text(align="left", dx=4).encode(text="LABEL:N", color=alt.value(ui.INK)))
                  .properties(height=150))
    h = mm.set_index("MODEL")["LOG_LOSS"]
    n = int(mm["N"].iloc[0])
    st.markdown(
        f"Gradient boosting lowers log loss by **{(h['INTERNAL'] - h['HGB']) / h['INTERNAL']:.2%}** versus the "
        f"internal-score model and **{(h['CREDIT'] - h['HGB']) / h['CREDIT']:.2%}** versus credit score, on the "
        f"same {n:,} funded plans. The x-axis is truncated to make a modest predictive gain visible; "
        "it is not by itself a measured profit gain."
    )
    st.dataframe(mm[["MODEL_LABEL", "N", "LOG_LOSS", "AUC", "MEAN_ACTUAL", "MEAN_PREDICTED"]]
                 .rename(columns={"MODEL_LABEL": "Model", "LOG_LOSS": "Log loss", "MEAN_ACTUAL": "Actual default",
                                  "MEAN_PREDICTED": "Mean predicted"})
                 .style.format({"N": "{:,.0f}", "Log loss": "{:.6f}", "AUC": "{:.3f}",
                                "Actual default": "{:.2%}", "Mean predicted": "{:.2%}"}),
                 hide_index=True, **ui.FULL)

    st.subheader("Who the internal score misjudges")
    cal = query(f"""{base_cte()}
        SELECT CASE WHEN EXISTING_CUSTOMER = 1 THEN 'Existing customers' ELSE 'New customers' END AS GROUP_NAME,
               COUNT(*) AS N, AVG(DEFAULT_FLAG * 1.0) AS ACTUAL,
               AVG(PD_INTERNAL) AS PRED_INTERNAL, AVG(PD_HGB) AS PRED_HGB
        FROM A WHERE SPLIT = 'test' AND REGIME = 'Regime 3' GROUP BY 1 ORDER BY 1""")
    cal = cal.rename(columns=SERIES_NAMES)
    long = cal.melt(["GROUP_NAME", "N"], var_name="SERIES", value_name="RATE")
    long["LABEL"] = long["RATE"].map(lambda v: f"{v:.1%}")
    series = ["Actual default rate", "Internal risk score only", "Gradient boosting (proposed)"]
    ui.show_chart(layered_group_bars(long, series))
    st.caption(f"Final-regime holdout, funded plans ({int(cal['N'].sum()):,}). The internal-score model "
               "underpredicts existing customers' default; the proposed model, which uses customer status, "
               "closes most of that gap.")

    st.subheader("Calibration by predicted-risk decile")
    c1, c2 = st.columns([1, 1])
    model_key = c1.radio("Model", ["HGB", "INTERNAL"], format_func=cfg.MODEL_LABELS.get, horizontal=True)
    reg = c2.radio("Regime", ["All"] + cfg.REGIMES, horizontal=True, index=3)
    reg_filter = "" if reg == "All" else f"AND REGIME = '{reg}'" if reg in cfg.REGIMES else ""
    col = f"PD_{model_key}"
    dec = query(f"""{base_cte()},
        D AS (SELECT {col} AS P, DEFAULT_FLAG, EXISTING_CUSTOMER,
                     NTILE(10) OVER (ORDER BY {col}) AS DECILE
              FROM A WHERE SPLIT = 'test' {reg_filter})
        SELECT DECILE, CASE WHEN EXISTING_CUSTOMER = 1 THEN 'Existing customers' ELSE 'New customers' END AS GROUP_NAME,
               COUNT(*) AS N, AVG(P) AS PREDICTED, AVG(DEFAULT_FLAG * 1.0) AS ACTUAL
        FROM D GROUP BY 1, 2 ORDER BY 1, 2""")
    mx = float(max(dec["PREDICTED"].max(), dec["ACTUAL"].max())) * 1.08
    diag = alt.Chart(pd.DataFrame({"x": [0, mx], "y": [0, mx]})).mark_line(
        color="#9AA2AB", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(dec).mark_point(filled=True, size=80).encode(
        x=alt.X("PREDICTED:Q", title="Mean predicted default", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, mx])),
        y=alt.Y("ACTUAL:Q", title="Observed default", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, mx])),
        color=alt.Color("GROUP_NAME:N", title=None, scale=alt.Scale(range=["#4A5866", ui.BRAND])),
        shape=alt.Shape("GROUP_NAME:N", title=None, scale=alt.Scale(range=["circle", "triangle-up"])),
        tooltip=["GROUP_NAME", "DECILE", "N", alt.Tooltip("PREDICTED:Q", format=".2%"),
                 alt.Tooltip("ACTUAL:Q", format=".2%")],
    )
    ui.show_chart((diag + pts).properties(height=320))
    st.caption("Points on the dashed line are well calibrated. Holdout funded plans; deciles of the selected "
               "model's prediction; counts in the tooltip.")

    st.subheader("Monitoring view: actual vs predicted by month")
    mon = query(f"""{base_cte()}
        SELECT DATE_TRUNC('month', APPLICATION_DATE) AS MONTH,
               AVG(DEFAULT_FLAG * 1.0) AS ACTUAL, AVG(PD_HGB) AS PRED_HGB, AVG(PD_INTERNAL) AS PRED_INTERNAL
        FROM A WHERE SPLIT = 'test' GROUP BY 1 ORDER BY 1""")
    mon = mon.rename(columns=SERIES_NAMES)
    mon["MONTH"] = pd.to_datetime(mon["MONTH"])
    ml = mon.melt("MONTH", var_name="SERIES", value_name="RATE")
    series = ["Actual default rate", "Gradient boosting (proposed)", "Internal risk score only"]
    line = alt.Chart(ml).mark_line(strokeWidth=2.2, point=True).encode(
        x=alt.X("MONTH:T", title=None), y=alt.Y("RATE:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("SERIES:N", title=None, scale=ui.model_color_scale(series)),
        strokeDash=alt.StrokeDash("SERIES:N", title=None, scale=alt.Scale(domain=series,
                                                                       range=[[1, 0], [1, 0], [5, 3]])),
        tooltip=[alt.Tooltip("MONTH:T", format="%b %Y"), "SERIES", alt.Tooltip("RATE:Q", format=".2%")],
    )
    ui.show_chart((line + regime_rules()).properties(height=280))
    st.caption("In a pilot this panel would run on new funded plans as outcomes mature, with an alert when "
               "the monthly gap exceeds an agreed tolerance.")

    st.subheader("Subgroup audit")
    dims = {
        "Customer status": "CASE WHEN EXISTING_CUSTOMER = 1 THEN 'Existing' ELSE 'New' END",
        "Age group": "CASE WHEN AGE < 25 THEN '18-24' WHEN AGE < 40 THEN '25-39' WHEN AGE < 65 THEN '40-64' ELSE '65+' END",
        "Requested device": "DEVICE_REQUESTED", "Gender": "GENDER", "Race": "RACE", "Hispanic ethnicity": "HISP",
        "Policy regime": "REGIME",
    }
    dim = st.selectbox("Group by", list(dims))
    sub = query(f"""{base_cte()}
        SELECT {dims[dim]} AS GRP, COUNT(*) AS N,
               AVG(DEFAULT_FLAG * 1.0) AS ACTUAL, AVG(PD_HGB) AS PRED_HGB, AVG(PD_INTERNAL) AS PRED_INTERNAL,
               -AVG(DEFAULT_FLAG * LN(PD_HGB) + (1 - DEFAULT_FLAG) * LN(1 - PD_HGB)) AS LL_HGB,
               -AVG(DEFAULT_FLAG * LN(PD_INTERNAL) + (1 - DEFAULT_FLAG) * LN(1 - PD_INTERNAL)) AS LL_INTERNAL
        FROM A WHERE SPLIT = 'test' GROUP BY 1 ORDER BY 1""")
    sub["LL_GAIN"] = (sub["LL_INTERNAL"] - sub["LL_HGB"]) / sub["LL_INTERNAL"]
    sub = sub.rename(columns={"GRP": "Group", "ACTUAL": "Actual default", "PRED_HGB": "Predicted (proposed)",
                              "PRED_INTERNAL": "Predicted (internal)", "LL_HGB": "Log loss (proposed)",
                              "LL_INTERNAL": "Log loss (internal)", "LL_GAIN": "Log-loss improvement"})
    st.dataframe(sub.style.format({"N": "{:,.0f}", "Actual default": "{:.2%}", "Predicted (proposed)": "{:.2%}",
                                   "Predicted (internal)": "{:.2%}", "Log loss (proposed)": "{:.4f}",
                                   "Log loss (internal)": "{:.4f}", "Log-loss improvement": "{:+.2%}"}),
                 hide_index=True, **ui.FULL)
    st.caption("Demographic columns are used for auditing only and are not model inputs. Age is a model input "
               "and needs policy review; the age-25 step is not used as an operational cutoff.")

    st.subheader("Where the model cannot be trusted")
    sb = query(f"""{base_cte()}
        SELECT INITIAL_DECISION, COUNT(*) AS NOT_FUNDED,
               SUM(CASE WHEN IN_SUPPORT THEN 0 ELSE 1 END) AS OUTSIDE_RANGE
        FROM A WHERE REGIME = 'Regime 3' AND FUNDED = 0 GROUP BY 1 ORDER BY 1""")
    tot, out = int(sb["NOT_FUNDED"].sum()), int(sb["OUTSIDE_RANGE"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Not funded, final regime", f"{tot:,}")
    c2.metric("Above every funded internal score", f"{out / tot:.2%}", help=f"{out:,} applicants")
    c3.metric("Inside the funded range", f"{tot - out:,}", help="All of these were conditional walk-aways.")
    st.caption("Outcomes exist only for funded plans. Every denied applicant sits above the highest funded "
               "internal score, so the proposal does not expand approvals there. The in-range group is the "
               "conditional walk-aways — the only not-funded group the model can speak to, with a risk haircut.")


def layered_group_bars(long: pd.DataFrame, series: list[str]):
    """Grouped bars with direct value labels (no colour-only encoding)."""
    base = alt.Chart(long).encode(
        x=alt.X("SERIES:N", sort=series, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("RATE:Q", title="Default rate", axis=alt.Axis(format="%")),
    )
    bars = base.mark_bar(size=46).encode(color=alt.Color("SERIES:N", sort=series, title=None,
                                                         scale=ui.model_color_scale(series)))
    text = base.mark_text(dy=-8, color=ui.INK, fontSize=12).encode(text="LABEL:N")
    return alt.layer(bars, text).properties(width=230, height=240).facet(
        column=alt.Column("GROUP_NAME:N", title=None, header=alt.Header(labelFontSize=13, labelColor=ui.INK)))


def tab_value(L, ceiling):
    pay = fin.payoff_lookup(L["payoffs"], "Regime 3")
    sqdf = query(f"""{base_cte()}
        SELECT COUNT(*) AS N, SUM(FUNDED) AS FUNDED, SUM(COALESCE(DEFAULT_FLAG, 0)) AS DEFAULTS,
               SUM(PROFIT) AS PROFIT FROM A WHERE REGIME = 'Regime 3'""").iloc[0]
    sq = dict(n=int(sqdf["N"]), funded=int(sqdf["FUNDED"]), defaults=float(sqdf["DEFAULTS"]),
              profit=float(sqdf["PROFIT"]))
    cond = query(f"""{base_cte()}
        SELECT PD_HGB, DEVICE_REQUESTED, DEVICE_FINAL, FINAL_STATUS
        FROM A WHERE REGIME = 'Regime 3' AND INITIAL_DECISION = 'Conditional'""")

    st.markdown(
        "**Proposal.** Keep the company's approve and deny bands. Inside the conditional band, approve applicants "
        "whose predicted default is below the fast-track ceiling on their requested terms. Today about four in "
        "ten conditional applicants walk away during negotiation, and negotiated downgrades cut profit "
        "without lowering observed default."
    )
    c = st.columns(4)
    accept = c[0].slider("Walk-aways who accept original terms", 0, 100, 50, 5, format="%d%%") / 100
    mult = c[1].slider("Risk multiplier on predicted default", 1.0, 3.0, 1.5, 0.1, format="×%.1f",
                       help="Selection-bias haircut: walk-aways may be riskier than funded look-alikes.")
    var_cost = c[2].number_input("Variable cost per application (USD)", 0.0, 5.0, 0.10, 0.05)
    fixed = c[3].number_input("Fixed cost: build, validation, pilot (USD)", 0, 5_000_000, 500_000, 50_000)
    c2 = st.columns([1, 1, 2])
    dg = c2[0].checkbox("Fast-tracked plans keep requested device", True,
                        help="Removes negotiated downgrades for fast-tracked applicants. Assumes default risk is "
                             "unchanged by the larger device — validate in the pilot.")
    dd = c2[1].number_input("Extra pooled default reduction (pp)", 0.0, 1.0, 0.0, 0.05,
                            help="Unmeasured. The historical replay found no rejection-driven gain; keep at 0 "
                                 "unless validated.")
    c2[2].caption(ui.md(f"Fast-track ceiling is set in the sidebar: **{ceiling:.1%}**. Scenario uses final-regime "
                  f"application mix scaled to {cfg.SCENARIO_APPLICATIONS:,} applications and final-regime "
                  f"payoffs (pooled G {ui.usd(pay['ALL'][0], 2)}, L {ui.usd(pay['ALL'][1], 2)})."))

    p = fin.ScenarioParams(ceiling=ceiling, accept=accept, risk_mult=mult, include_downgrade=dg,
                           pooled_dd_pp=dd, var_cost=var_cost, fixed_cost=float(fixed))
    r = fin.run_scenario(sq, cond, pay, p)
    bem = fin.breakeven_multiplier(sq, cond, pay, p)

    m = st.columns(4)
    m[0].metric("Net incremental profit per 1M", ui.usd_m(r["net"]))
    m[1].metric("Break-even risk multiplier", "above ×10" if bem is None else f"×{bem:.2f}",
                help="How many times worse than predicted the fast-tracked applicants' default would have to be "
                     "before the proposal stops paying.")
    m[2].metric("Plans recovered from walk-aways", f"{r['walk_count'] * accept:,.0f}")
    t = r["table"]
    m[3].metric("Default rate among funded", ui.pct(t.loc["Default rate (funded)", "Proposed (scenario)"], 2),
                f"{(t.loc['Default rate (funded)', 'Proposed (scenario)'] - t.loc['Default rate (funded)', 'Status quo (observed)']) * 100:+.2f} pp",
                delta_color="inverse")
    ui.scenario_note("Acceptance, risk multiplier, device effect and costs are inputs, not measured outcomes. "
                     "Status-quo figures are observed final-regime results scaled to one million applications.")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Value bridge")
        steps = [("Walk-away recovery", r["lever_a"]), ("Downgrades avoided", r["lever_c"]),
                 ("Pooled default reduction", r["lever_b"]), ("Costs", -r["costs"])]
        steps = [s for s in steps if abs(s[1]) > 0.5]
        rows, run = [], 0.0
        for name, v in steps:
            rows.append(dict(STEP=name, START=run, END=run + v, KIND="Increase" if v >= 0 else "Decrease",
                             LABEL=("+" if v >= 0 else "") + ui.usd_m(v)))
            run += v
        rows.append(dict(STEP="Net incremental", START=0.0, END=run, KIND="Net", LABEL=ui.usd_m(run)))
        wf = pd.DataFrame(rows)
        wf["START_M"], wf["END_M"] = wf["START"] / 1e6, wf["END"] / 1e6
        order = list(wf["STEP"])
        bars = alt.Chart(wf).mark_bar(size=46).encode(
            x=alt.X("STEP:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("START_M:Q", title="USD millions per 1M applications"), y2="END_M:Q",
            color=alt.Color("KIND:N", title=None, scale=alt.Scale(domain=["Increase", "Decrease", "Net"],
                                                                  range=["#1E7B4F", "#8E1B1B", ui.INK])),
        )
        labels = alt.Chart(wf).mark_text(dy=-8, color=ui.INK, fontSize=12).encode(
            x=alt.X("STEP:N", sort=order), y=alt.Y("TOP:Q"), text="LABEL:N"
        ).transform_calculate(TOP="max(datum.START_M, datum.END_M)")
        ui.show_chart((bars + labels).properties(height=300))
    with right:
        st.subheader("Status quo vs proposal")
        tf = t.copy().astype(object)
        for idx in tf.index:
            for col in tf.columns:
                v = t.loc[idx, col]
                tf.loc[idx, col] = (f"{v:,.0f}" if idx in ("Applications", "Funded plans") else
                                    f"{v:.2%}" if "rate" in idx.lower() else ui.usd_m(v))
        st.dataframe(tf, **ui.FULL)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Choosing the fast-track ceiling")
        grid = []
        for cval in np.arange(0.12, 0.2201, 0.005):
            rr = fin.run_scenario(sq, cond, pay, fin.ScenarioParams(**{**p.__dict__, "ceiling": float(cval)}))
            grid.append(dict(CEILING=cval, NET_M=rr["net"] / 1e6, SHARE=rr["eligible_share"]))
        g = pd.DataFrame(grid)
        line = alt.Chart(g).mark_line(color=ui.BRAND, strokeWidth=2.4, point=True).encode(
            x=alt.X("CEILING:Q", title="Fast-track ceiling (predicted default)", axis=alt.Axis(format="%")),
            y=alt.Y("NET_M:Q", title="Net incremental, USD millions"),
            tooltip=[alt.Tooltip("CEILING:Q", format=".1%"), alt.Tooltip("SHARE:Q", title="Conditional lane fast-tracked", format=".0%"),
                     alt.Tooltip("NET_M:Q", format=",.2f")],
        )
        now = alt.Chart(pd.DataFrame({"CEILING": [ceiling]})).mark_rule(color=ui.MUTED, strokeDash=[3, 3]).encode(x="CEILING:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9AA2AB").encode(y="y:Q")
        ui.show_chart((zero + line + now).properties(height=260))
        st.caption(f"At {ceiling:.1%}, {r['eligible_share']:.0%} of the conditional lane is fast-tracked. "
                   "A pilot can start low and raise the ceiling as observed defaults confirm the predictions.")
    with right:
        st.subheader("Sensitivity of net value")
        sens = fin.sensitivity(sq, cond, pay, p)
        styler = sens.style.format(ui.usd_m)
        cell_style = getattr(styler, "map", None) or styler.applymap  # pandas < 2.1 in some runtimes
        st.dataframe(cell_style(lambda v: f"color: {'#8E1B1B' if v < 0 else ui.INK}; "
                                          f"font-weight: {'700' if v < 0 else '400'}"),
                     **ui.FULL)
        st.caption("Rows: true default as a multiple of predicted. Columns: share of walk-aways who accept the "
                   "original offer. Negative cells are shown in bold.")

    with st.expander("Formula and assumptions"):
        st.markdown(ui.md(
            "For each policy *s*: expected profit per application = Σₖ wₖ · Σₐ qₖₐ · [(1 − pₖₐ)·Gₖₐ − pₖₐ·Lₖₐ] − cₛ; "
            "total = 1,000,000 × that − Fₛ; incremental value = proposed − status quo.\n\n"
            "| Input | Value | Type |\n|---|---|---|\n"
            f"| Application mix, funding, defaults, profit | Final regime, scaled ×{r['scale']:.2f} | Observed |\n"
            f"| G, L by device tier | High {ui.usd(pay['High'][0])}/{ui.usd(pay['High'][1])}, Mid {ui.usd(pay['Mid'][0])}/{ui.usd(pay['Mid'][1])}, Low {ui.usd(pay['Low'][0])}/{ui.usd(pay['Low'][1])} | Observed (training split) |\n"
            f"| Default of fast-tracked walk-aways | model prediction × {mult:.1f} | Assumption |\n"
            f"| Acceptance of original offer by walk-aways | {accept:.0%} | Assumption |\n"
            f"| Device effect on default for kept upgrades | none | Assumption, unvalidated |\n"
            f"| Declined / walked-away profit | $0 | Assumption |\n"
            f"| Costs | {ui.usd(var_cost, 2)} per application + {ui.usd(fixed)} fixed | Assumption |\n"
        ))


# ============================================================================ main
def main():
    page, regime, ceiling, scorer = sidebar()
    try:
        L = lookup_tables()
    except Exception as exc:
        st.error(f"Could not read the evidence tables. Run model/train_register.py first. Details: {exc}")
        st.stop()
    if not scorer.ready and page != "Portfolio overview":
        st.error(f"No scoring model is available. {scorer.detail}")
        st.stop()
    if page == "Application review":
        page_application(L, scorer, regime, ceiling)
    elif page == "Offer comparison":
        page_offers(L, regime)
    else:
        page_portfolio(L, regime, ceiling)


main()
