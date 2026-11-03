import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtester.backtest_engine import BacktestEngine
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover

# Set page layout
st.set_page_config(page_title="Crypto Backtester", layout="wide")
st.title("📈 Algorithmic Strategy Backtester")

# Sidebar for User Inputs
st.sidebar.header("Data")
data_file = st.sidebar.text_input("OHLCV CSV", value="data/historical/BTC-USD_1h.csv")

st.sidebar.header("Strategy Parameters")
fast_ma = st.sidebar.slider("Fast SMA Window", min_value=5, max_value=50, value=10)
slow_ma = st.sidebar.slider("Slow SMA Window", min_value=20, max_value=200, value=50)

st.sidebar.header("Risk Management")
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
sl_pct = st.sidebar.number_input("Stop Loss (%)", min_value=0.5, max_value=20.0, value=5.0, step=0.5) / 100
tp_pct = st.sidebar.number_input("Take Profit (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0) / 100
fee = st.sidebar.number_input("Fee per side (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.05) / 100
slippage = st.sidebar.number_input("Slippage per fill (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.01) / 100

if st.sidebar.button("Run Backtest"):
    if fast_ma >= slow_ma:
        st.error("The fast window must be shorter than the slow window.")
        st.stop()
    try:
        # 1. Initialize Components
        risk_manager = RiskManager(risk_per_trade_pct=risk_pct, sl_pct=sl_pct, tp_pct=tp_pct, fee_rate=fee)
        strategy = SMACrossover(name=f"SMA_{fast_ma}_{slow_ma}", risk_manager=risk_manager,
                                fast_window=fast_ma, slow_window=slow_ma)

        # 2. Run Engine
        engine = BacktestEngine(data_file, strategy, initial_capital=10000.0, fee_rate=fee, slippage=slippage)
        m = engine.run(verbose=False)

        # 3. Display Metrics
        st.subheader("Performance Metrics")
        cols = st.columns(6)
        cols[0].metric("Final Capital", f"${m['final_capital']:,.2f}", f"{m['return_pct']:.2f}%")
        cols[1].metric("Buy & Hold", f"{m['buy_and_hold_pct']:.2f}%")
        cols[2].metric("Total Trades", m['trades'])
        cols[3].metric("Win Rate", f"{m['win_rate_pct']:.1f}%")
        cols[4].metric("Max Drawdown", f"{m['max_drawdown_pct']:.2f}%")
        cols[5].metric("Sharpe Ratio", f"{m['sharpe']:.2f}")

        # 4. Plot Chart
        st.subheader("Price & Moving Averages")
        df = engine.df

        # Calculate MAs for the chart
        df['Fast_MA'] = df['close'].rolling(window=fast_ma).mean()
        df['Slow_MA'] = df['close'].rolling(window=slow_ma).mean()

        fig = go.Figure()
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name='Price'))
        # Fast MA
        fig.add_trace(go.Scatter(x=df.index, y=df['Fast_MA'], line={'color': 'blue', 'width': 1.5},
                                 name=f'Fast MA ({fast_ma})'))
        # Slow MA
        fig.add_trace(go.Scatter(x=df.index, y=df['Slow_MA'], line={'color': 'orange', 'width': 1.5},
                                 name=f'Slow MA ({slow_ma})'))

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # 5. Equity curve and drawdown
        st.subheader("Equity & Drawdown")
        eq = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        eq.add_trace(go.Scatter(x=engine.equity.index, y=engine.equity, name='Equity'), row=1, col=1)
        eq.add_trace(go.Scatter(x=engine.equity.index, y=engine.drawdown(), name='Drawdown (%)', fill='tozeroy',
                                line={'color': 'red'}), row=2, col=1)
        eq.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(eq, use_container_width=True)

        # 6. Display Trade History Table
        st.subheader("Trade History")
        if engine.trade_history:
            st.dataframe(pd.DataFrame(engine.trade_history))

    except FileNotFoundError:
        st.error(f"Could not find data file at {data_file}. Please run data_downloader.py first.")
