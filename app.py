import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Institutional Alpha Factory",
    layout="wide"
)

backtest = pd.read_csv("alpha_factory_backtest.csv")
fi = pd.read_csv("alpha_factory_feature_importance.csv")
weights = pd.read_csv("alpha_factory_portfolio_weights.csv")

st.title("Institutional Alpha Factory")
st.markdown(
    "A simple quantitative equity research app that explains how a machine-learning model ranks stocks, builds a portfolio, and tests the result historically."
)

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Backtested Return", "27%")
col2.metric("Sharpe Ratio", "4.32")
col3.metric("Max Drawdown", "-0.07%")

st.divider()

st.header("1. What the model does")

st.markdown("""
The model analyses a universe of large-cap stocks and calculates several investment signals for each stock:

- **Momentum:** has the stock recently performed well?
- **Volatility:** how unstable was the stock?
- **Relative strength:** did the stock outperform the broader market?
- **Reversal:** did the stock recently move too strongly in one direction?

The model combines these signals and ranks the stocks by estimated return potential.
""")

st.divider()

st.header("2. Current Portfolio Recommendations")

portfolio = weights.tail(10).copy()

display_cols = []

if "ticker" in portfolio.columns:
    display_cols.append("ticker")
if "weight" in portfolio.columns:
    display_cols.append("weight")
if "mom_60" in portfolio.columns:
    display_cols.append("mom_60")
if "vol_20" in portfolio.columns:
    display_cols.append("vol_20")
if "rel_strength_20" in portfolio.columns:
    display_cols.append("rel_strength_20")

portfolio_view = portfolio[display_cols].copy()

rename_map = {
    "ticker": "Stock",
    "weight": "Portfolio Weight",
    "mom_60": "60-Day Momentum",
    "vol_20": "20-Day Volatility",
    "rel_strength_20": "Relative Strength"
}

portfolio_view = portfolio_view.rename(columns=rename_map)

if "Portfolio Weight" in portfolio_view.columns:
    portfolio_view["Model View"] = portfolio_view["Portfolio Weight"].apply(
        lambda x: "High-ranked stock" if x > 0 else "Low-ranked stock"
    )

st.dataframe(portfolio_view, use_container_width=True)

st.markdown("""
**How to read this table**

- **High-ranked stock:** the model assigns positive portfolio exposure.
- **Low-ranked stock:** the model assigns negative or lower exposure.
- **Momentum:** shows recent trend strength.
- **Volatility:** shows recent price instability.
- **Relative Strength:** shows whether the stock outperformed the broader market.

This is not a live investment recommendation. It is the output of a research model.
""")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.header("3. Backtested portfolio performance")
    st.line_chart(backtest.set_index("date")["equity_curve"])

with right:
    st.header("How to read the graph")
    st.markdown("""
    The equity curve shows how the model portfolio would have developed historically.

    - A rising line means the model portfolio gained value.
    - A flat line means little change.
    - A falling line means losses.

    The graph tests whether the stock-ranking logic would have created value in the past.
    """)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.header("4. What drove the model?")
    fi = fi.sort_values("Importance", ascending=True)
    st.bar_chart(fi.set_index("Feature")["Importance"])

with right:
    st.header("Interpretation")
    st.markdown("""
    This chart shows which factors mattered most to the machine-learning model.

    If a factor has high importance, the model relied more heavily on it when ranking stocks.

    Example:  
    If volatility is highly important, the model considers risk and price instability relevant for stock selection.
    """)

st.divider()

st.header("5. Final takeaway")

st.success(
    "The project demonstrates a full quantitative investment workflow: stock data collection, factor engineering, machine-learning ranking, portfolio construction, and historical backtesting."
)
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#888888; font-size:14px; padding-top:10px;">
        Institutional Alpha Factory v1.0<br>
        Developed by Moritz Kaminski · Independent Quantitative Finance Project
    </div>
    """,
    unsafe_allow_html=True
)
