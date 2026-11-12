"""Visual system: palette, formatting, small UI components."""
from __future__ import annotations

import altair as alt
import streamlit as st

INK = "#1F2328"
MUTED = "#5B6470"
RULE = "#D8DCE1"
BRAND = "#2F5D8C"

MODEL_COLORS = {
    "Gradient boosting (proposed)": BRAND,
    "Internal risk score only": "#4A5866",
    "Credit score only": "#A3ABB5",
    "Actual default rate": INK,
}
ACTIONS = {
    #            colour     symbol   plain-language label
    "Approve":   ("#1E7B4F", "✓", "Approve"),
    "Negotiate": ("#9A5800", "↔", "Negotiate"),
    "Decline":   ("#8E1B1B", "✕", "Decline"),
    "Review":    ("#4A5866", "!", "Review"),
}

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
  h2, h3 {{ font-weight: 650; }}
  [data-testid="stMetricValue"] {{ font-size: 1.65rem; font-variant-numeric: tabular-nums; }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
  .app-kicker {{ color: {MUTED}; font-size: 0.95rem; margin: -0.6rem 0 1.2rem 0; max-width: 72ch; }}
  .app-decision {{ border-left: 6px solid var(--c); padding: 0.9rem 1.1rem; background: #F6F7F9;
                  border-radius: 0 6px 6px 0; margin-bottom: 0.8rem; }}
  .app-decision .big {{ font-size: 1.9rem; font-weight: 750; color: var(--c); line-height: 1.1; }}
  .app-decision .terms {{ font-size: 1.02rem; color: {INK}; margin-top: 0.2rem; }}
  .app-decision .basis {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.35rem; }}
  .app-tag {{ display: inline-block; font-size: 0.78rem; padding: 0.05rem 0.5rem; border-radius: 999px;
             border: 1px solid {RULE}; color: {MUTED}; margin-right: 0.3rem; }}
  .app-scenario {{ border: 1px dashed #9A5800; padding: 0.6rem 0.9rem; border-radius: 6px;
                  color: #6B3D00; background: #FFF8EE; font-size: 0.9rem; margin: 0.4rem 0 0.8rem 0; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, kicker: str) -> None:
    st.title(title)
    st.markdown(f'<div class="app-kicker">{kicker}</div>', unsafe_allow_html=True)


def decision_card(action: str, terms: str, basis: str) -> None:
    colour, symbol, label = ACTIONS[action]
    st.markdown(
        f'<div class="app-decision" style="--c:{colour}">'
        f'<div class="big">{symbol} {label}</div>'
        f'<div class="terms">{terms}</div>'
        f'<div class="basis">Basis: {basis}</div></div>',
        unsafe_allow_html=True,
    )


def scenario_note(text: str) -> None:
    st.markdown(f'<div class="app-scenario">Scenario assumption · {text}</div>', unsafe_allow_html=True)


def tags(*labels: str) -> None:
    st.markdown("".join(f'<span class="app-tag">{l}</span>' for l in labels), unsafe_allow_html=True)


def _st_version() -> tuple:
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


# Streamlit in Snowflake may run an older Streamlit; newer releases replace
# use_container_width with width="stretch". FULL works on both.
FULL = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}


def show_chart(chart) -> None:
    chart = chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ECEEF1",
                                 labelFontSize=12, titleFontSize=12) \
                 .configure_view(stroke=None) \
                 .configure_legend(labelFontSize=12, titleFontSize=12, orient="bottom")
    try:
        st.altair_chart(chart, **FULL)
    except TypeError:
        # Some Streamlit versions accept width="stretch" for tables but not for charts.
        st.altair_chart(chart, use_container_width=True)


def md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not render them as LaTeX."""
    return text.replace("$", "\\$")


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}%}"


def usd(x: float, d: int = 0) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x):,.{d}f}"


def usd_m(x: float) -> str:
    sign = "−" if x < 0 else ""
    return f"{sign}${abs(x) / 1e6:,.2f}M"


def model_color_scale(domain=None):
    domain = domain or list(MODEL_COLORS)
    return alt.Scale(domain=domain, range=[MODEL_COLORS[d] for d in domain])