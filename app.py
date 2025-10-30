import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Cross-Asset Market Regime Monitor",
    layout="wide"
)

# ==================== DATA CACHING ====================
@st.cache_data(ttl=86400, show_spinner=False)
def download_market_data(tickers, start_date, end_date):
    """Download market data with caching for 24 hours"""
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )["Close"]
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def download_treasury_data():
    """Download US Treasury yield curve data"""
    try:
        tickers = {
            "^IRX": "3M",
            "^FVX": "5Y",
            "^TNX": "10Y",
            "^TYX": "30Y"
        }
        
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        data = yf.download(
            list(tickers.keys()),
            start=start_date,
            end=end_date,
            progress=False
        )["Close"]
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        data = data.rename(columns=tickers)
        return data.ffill().dropna()
    except Exception as e:
        st.warning(f"Impossible de charger les données Treasury: {str(e)}")
        return pd.DataFrame()

# ==================== MARKET REGIME CALCULATION ====================
def calculate_market_regimes(prices):
    """Calculate market regime indicators: Growth, Inflation, Volatility"""
    regimes = {}
    
    if len(prices) < 63:
        return {"Growth": 0, "Inflation": 0, "Volatility": 0}
    
    # Growth indicator
    if "Equities" in prices.columns:
        equity_returns = prices["Equities"].pct_change(63).iloc[-1]
        regimes["Growth"] = equity_returns * 100
    else:
        regimes["Growth"] = 0
    
    # Inflation indicator
    inflation_assets = []
    if "Gold" in prices.columns:
        inflation_assets.append(prices["Gold"].pct_change(63).iloc[-1])
    if "Crude Oil" in prices.columns:
        inflation_assets.append(prices["Crude Oil"].pct_change(63).iloc[-1])
    
    regimes["Inflation"] = np.mean(inflation_assets) * 100 if inflation_assets else 0
    
    # Volatility indicator
    if "Equities" in prices.columns:
        returns = prices["Equities"].pct_change()
        rolling_vol = returns.rolling(window=21).std()
        current_vol = rolling_vol.iloc[-1] * np.sqrt(252) * 100
        avg_vol = rolling_vol.mean() * np.sqrt(252) * 100
        regimes["Volatility"] = ((current_vol / avg_vol) - 1) * 100
    else:
        regimes["Volatility"] = 0
    
    return regimes

# ==================== VISUALIZATION FUNCTIONS ====================
def create_line_chart(data):
    """Line chart for tracking time series performance"""
    fig = go.Figure()
    
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[col],
            mode='lines',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Performance",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_yield_curve(treasury_data):
    """Visualize US Treasury yield curve"""
    if treasury_data.empty:
        return None
    
    latest = treasury_data.iloc[-1]
    
    maturities = ["3M", "5Y", "10Y", "30Y"]
    maturity_years = [0.25, 5, 10, 30]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=maturity_years,
        y=[latest[m] for m in maturities],
        mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="US Treasury Yield Curve",
        xaxis_title="Maturity (years)",
        yaxis_title="Yield (%)",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_term_structure(futures_data):
    """Term structure graphs for futures"""
    if futures_data.empty or len(futures_data) < 30:
        return None
    
    base_prices = futures_data.iloc[-30]
    current_prices = futures_data.iloc[-1]
    relative_perf = ((current_prices / base_prices) - 1) * 100
    
    fig = go.Figure()
    
    for col in futures_data.columns:
        fig.add_trace(go.Bar(
            x=[col],
            y=[relative_perf[col]],
            name=col
        ))
    
    fig.update_layout(
        title="Futures Term Structure (30-day performance)",
        xaxis_title="Contract",
        yaxis_title="Performance (%)",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_heatmap(data):
    """Heatmap for comparing performance across asset classes"""
    if data.empty or len(data) < 63:
        return None
    
    # Performance over different periods
    periods = [21, 63, 252]
    period_names = ["1M", "3M", "1Y"]
    
    perf_data = []
    for period in periods:
        if len(data) > period:
            perf = ((data.iloc[-1] / data.iloc[-period]) - 1) * 100
            perf_data.append(perf.values)
    
    if not perf_data:
        return None
    
    perf_matrix = np.array(perf_data)
    
    fig = go.Figure(data=go.Heatmap(
        z=perf_matrix,
        x=data.columns,
        y=period_names[:len(perf_data)],
        colorscale='RdYlGn',
        zmid=0,
        text=perf_matrix.round(2),
        texttemplate='%{text}%',
        textfont={"size": 11}
    ))
    
    fig.update_layout(
        title="Performance Heatmap Across Asset Classes and Time Periods",
        xaxis_title="Asset Class",
        yaxis_title="Period",
        template='plotly_white',
        height=400
    )
    
    return fig

# ==================== MAIN APP ====================
def main():
    st.title("Cross-Asset Market Regime Monitor")
    st.caption("MSc in Financial Markets and Investments - SKEMA Business School")
    
    # ==================== SIDEBAR FILTERS ====================
    st.sidebar.title("Filters")
    
    # Asset selection
    ASSET_MAP = {
        "ES=F": "Equities",
        "ZN=F": "Nominal Bonds",
        "GC=F": "Gold",
        "CL=F": "Crude Oil",
        "ZW=F": "Wheat",
        "DX=F": "Dollar"
    }
    
    selected_tickers = st.sidebar.multiselect(
        "Asset Classes",
        list(ASSET_MAP.keys()),
        default=list(ASSET_MAP.keys()),
        format_func=lambda x: ASSET_MAP[x]
    )
    
    # Time period filter
    end_date = st.sidebar.date_input(
        "End Date",
        value=date.today(),
        max_value=date.today()
    )
    
    time_periods = {
        "3 months": 90,
        "6 months": 180,
        "1 year": 365,
        "2 years": 730,
        "3 years": 1095
    }
    
    period_choice = st.sidebar.selectbox(
        "Time Period",
        list(time_periods.keys()),
        index=2
    )
    
    start_date = end_date - timedelta(days=time_periods[period_choice])
    
    # Market regime filter
    show_regimes = st.sidebar.checkbox("Show Market Regimes", value=True)
    
    # ==================== DATA LOADING ====================
    if not selected_tickers:
        st.warning("Please select at least one asset class.")
        st.stop()
    
    with st.spinner("Loading market data..."):
        prices = download_market_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("Unable to load data. Please try again later.")
            st.stop()
        
        # Rename columns
        prices = prices.rename(columns=ASSET_MAP)
        
        # Normalize prices
        returns = prices.pct_change().fillna(0.0)
        normalized_prices = (1 + returns).cumprod() * 100
    
    # ==================== MARKET REGIME INDICATORS ====================
    if show_regimes:
        st.subheader("Current Market Regime Indicators")
        
        regimes = calculate_market_regimes(prices)
        
        cols = st.columns(3)
        
        cols[0].metric("Growth", f"{regimes['Growth']:.2f}%")
        cols[1].metric("Inflation", f"{regimes['Inflation']:.2f}%")
        cols[2].metric("Volatility", f"{regimes['Volatility']:.2f}%")
        
        st.markdown("---")
    
    # ==================== LINE CHARTS ====================
    st.subheader("Asset Performance (Time Series)")
    
    fig_line = create_line_chart(normalized_prices)
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== TERM STRUCTURE & YIELD CURVE ====================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Term Structure of Futures")
        
        futures_tickers = ["ES=F", "GC=F", "CL=F", "ZW=F"]
        futures_data = download_market_data(
            futures_tickers,
            end_date - timedelta(days=180),
            end_date
        )
        
        if not futures_data.empty:
            futures_data = futures_data.rename(columns={
                "ES=F": "Equity",
                "GC=F": "Gold",
                "CL=F": "Crude Oil",
                "ZW=F": "Wheat"
            })
            
            fig_term = create_term_structure(futures_data)
            if fig_term:
                st.plotly_chart(fig_term, use_container_width=True)
        else:
            st.info("Futures term structure data not available")
    
    with col2:
        st.subheader("US Treasury Yield Curve")
        
        treasury_data = download_treasury_data()
        
        if not treasury_data.empty:
            fig_yield = create_yield_curve(treasury_data)
            if fig_yield:
                st.plotly_chart(fig_yield, use_container_width=True)
        else:
            st.info("Treasury yield curve data not available")
    
    st.markdown("---")
    
    # ==================== HEATMAP ====================
    st.subheader("Performance Heatmap Across Asset Classes and Market Regimes")
    
    fig_heatmap = create_heatmap(prices)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Insufficient data for heatmap")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
