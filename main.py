import os
import pandas as pd
import pandas_ta as ta
import streamlit as st
from datetime import date, timedelta
from typing import TypedDict, List, Dict
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Financial Analysis Imports ---
from nsepy import get_history
from nsetools import Nse

# --- AI and LangGraph Imports ---
from openai import AzureOpenAI
from langgraph.graph import StateGraph, END

# --- Initialize NSE Tools and App Configuration ---
try:
    nse = Nse()
except Exception as e:
    st.error(f"Could not connect to NSE. Some features may be unavailable. Error: {e}")
    nse = None

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent Stock Screener", page_icon="📈", layout="wide")
st.title("📈 Intelligent Stock Screener V2")
st.markdown("An advanced tool for the Indian Market with backtesting, AI analysis, and interactive charts.")


# ==============================================================================
# 1. ENHANCED DATA FETCHING & ANALYSIS
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_top_gainers_losers():
    """Fetches top gainers and losers with robust error handling."""
    if not nse:
        return pd.DataFrame(), pd.DataFrame()
    try:
        gainers = pd.DataFrame(nse.get_top_gainers())
        losers = pd.DataFrame(nse.get_top_losers())
        return gainers.head(10), losers.head(10)
    except Exception as e:
        st.error(f"Error fetching top movers: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache once a day
def get_breakout_stocks(num_stocks=100):
    """Scans for stocks near their 52-week high."""
    if not nse:
        return pd.DataFrame()
    try:
        # Using a more reliable way to get a list of symbols if available
        symbols = nse.get_stock_codes().keys()
        
        breakouts = []
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        progress_bar = st.progress(0, text="Initializing breakout scan...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(get_history, symbol=symbol, start=start_date, end=end_date): symbol for symbol in list(symbols)[:num_stocks]}
            
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    stock_data = future.result()
                    if not stock_data.empty:
                        last_price = stock_data['Close'].iloc[-1]
                        high_52_week = stock_data['High'].max()

                        if last_price >= high_52_week * 0.98:  # Within 2% of 52w high
                            breakouts.append({
                                'Symbol': symbol,
                                'Last Price': last_price,
                                '52-Week High': high_52_week,
                                'Proximity': f"{((last_price / high_52_week) * 100):.2f}%"
                            })
                except Exception:
                    continue # Ignore symbols that fail to fetch
                
                progress_bar.progress((i + 1) / num_stocks, text=f"Scanning {symbol}...")
        
        progress_bar.empty()
        return pd.DataFrame(breakouts)
    except Exception as e:
        st.error(f"Error fetching breakout stocks: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a comprehensive set of technical indicators using pandas-ta."""
    if df.empty:
        return df
    
    # Add all indicators in one go
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.supertrend(length=7, multiplier=3, append=True) # Common SuperTrend settings
    df.ta.vwap(append=True)
    
    return df.dropna()

def run_backtest(df: pd.DataFrame, capital=100000, risk_percent=1.5) -> Dict:
    """Runs a simple vector-based backtest on the provided data with the strategy."""
    if df.empty or len(df) < 50:
        return {"error": "Not enough data for a meaningful backtest."}

    # --- Strategy Logic ---
    # Entry Signal: SuperTrend turns bullish (SUPERTd_7_3.0 == 1) AND ADX > 20 (indicates a trend)
    df['signal'] = (df['SUPERTd_7_3.0'] == 1) & (df['ADX_14'] > 20)
    df['entry'] = df['signal'] & ~df['signal'].shift(1).fillna(False)

    trades = []
    position_open = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    for i, row in df.iterrows():
        if not position_open and row['entry']:
            # --- ENTER TRADE ---
            entry_price = row['Close']
            atr_val = row['ATRr_14']
            stop_loss = entry_price - (2 * atr_val)  # Stop-loss at 2x ATR
            take_profit = entry_price + (3 * atr_val) # Target at 3x ATR (RRR of 1.5)
            position_open = True
            
            trades.append({'entry_date': i, 'entry_price': entry_price, 'status': 'open'})
            continue

        if position_open:
            # --- EXIT TRADE ---
            if row['High'] >= take_profit:
                trades[-1].update({'exit_date': i, 'exit_price': take_profit, 'status': 'win'})
                position_open = False
            elif row['Low'] <= stop_loss:
                trades[-1].update({'exit_date': i, 'exit_price': stop_loss, 'status': 'loss'})
                position_open = False
            elif row['SUPERTd_7_3.0'] == -1: # Exit if trend reverses
                trades[-1].update({'exit_date': i, 'exit_price': row['Close'], 'status': 'trend_reversal'})
                position_open = False

    if not trades:
        return {"message": "No trade signals found in the last year."}

    # Calculate metrics
    trade_df = pd.DataFrame(trades)
    trade_df['pnl'] = trade_df['exit_price'] - trade_df['entry_price']
    
    wins = trade_df[trade_df['pnl'] > 0]
    losses = trade_df[trade_df['pnl'] <= 0]
    
    win_rate = len(wins) / len(trade_df) if not trade_df.empty else 0
    avg_win = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * abs(avg_loss))

    return {
        "Total Trades": len(trade_df),
        "Win Rate (%)": f"{win_rate * 100:.2f}",
        "Expectancy (Points)": f"{expectancy:.2f}",
        "Average Win (Points)": f"{avg_win:.2f}",
        "Average Loss (Points)": f"{abs(avg_loss):.2f}"
    }

def analyze_stock(symbol: str) -> Dict:
    """Main function to fetch, analyze, and backtest a single stock."""
    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * 2) # 2 years for more backtest data
        df = get_history(symbol=symbol, start=start_date, end=end_date)
        
        if df.empty:
            return {"error": "No historical data found."}

        df_indicators = calculate_technical_indicators(df.copy())
        if df_indicators.empty:
            return {"error": "Not enough data to compute indicators."}

        latest = df_indicators.iloc[-1]
        backtest_results = run_backtest(df_indicators.tail(252).copy()) # Backtest on last year
        
        # --- Current Signal Generation ---
        is_uptrend = latest['SUPERTd_7_3.0'] == 1
        is_strong_trend = latest['ADX_14'] > 25
        is_pullback_zone = latest['Close'] < latest['EMA_20']
        
        verdict = "Hold / Neutral"
        if is_uptrend and is_strong_trend:
            verdict = "Bullish Trend - Monitor for Entry"
            if is_pullback_zone:
                verdict = "Potential Buy on Dip"
        elif not is_uptrend:
            verdict = "Bearish Trend - Avoid"
        
        # Position Sizing for a 1 Lakh capital
        risk_per_trade = 1500  # 1.5% of 1 lakh
        stop_loss_points = 2 * latest['ATRr_14']
        position_size = int(risk_per_trade / stop_loss_points) if stop_loss_points > 0 else 0
        
        return {
            "symbol": symbol,
            "data": df_indicators,
            "latest_metrics": {
                "Verdict": verdict,
                "Price": latest['Close'],
                "ADX": latest['ADX_14'],
                "RSI": latest['RSI_14'],
                "ATR": latest['ATRr_14'],
                "SuperTrend": "Up" if latest['SUPERTd_7_3.0'] == 1 else "Down",
                "VWAP": latest['VWAP_D']
            },
            "position_sizing": {
                "Capital": 100000,
                "Risk per Trade (INR)": risk_per_trade,
                "ATR-Based Stop Loss (Points)": stop_loss_points,
                "Suggested Position Size (Shares)": position_size,
                "Stop Loss Price": latest['Close'] - stop_loss_points,
                "Take Profit Price": latest['Close'] + (3 * latest['ATRr_14']), # 1.5 RR
                "Risk/Reward Ratio": "1:1.5"
            },
            "backtest": backtest_results
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

def create_interactive_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Generates an interactive Plotly chart."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'))
    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='20-EMA', line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='50-EMA', line=dict(color='orange', width=1.5)))
    
    # SuperTrend
    st_up = df[df['SUPERTd_7_3.0'] == 1]['SUPERT_7_3.0']
    st_down = df[df['SUPERTd_7_3.0'] == -1]['SUPERT_7_3.0']
    fig.add_trace(go.Scatter(x=st_up.index, y=st_up, mode='lines', name='SuperTrend Up', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=st_down.index, y=st_down, mode='lines', name='SuperTrend Down', line=dict(color='red', width=2)))

    fig.update_layout(
        title=f'{symbol} Technical Chart',
        yaxis_title='Price (INR)',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ==============================================================================
# 2. AI AGENT LOGIC (LANGGRAPH)
# ==============================================================================

class AgentState(TypedDict):
    analysis_results: Dict
    final_report: str

def generate_report_node(state: AgentState):
    """Uses a powerful LLM prompt to generate a professional report."""
    try:
        client = AzureOpenAI(
            api_key=st.secrets["azure"]["api_key"],
            azure_endpoint=st.secrets["azure"]["endpoint"],
            api_version=st.secrets["azure"]["api_version"]
        )
    except Exception as e:
        st.error(f"Azure credentials not found or invalid. Please check your secrets.toml. Error: {e}")
        state['final_report'] = "Could not generate AI report due to configuration error."
        return state

    prompt_template = """
    You are an expert financial analyst providing a risk-first swing trading report for the Indian market.
    Synthesize the provided technical data, backtest results, and position sizing calculations into a clear, actionable markdown report.

    For each stock provided:

    ### **Stock: {symbol}**

    1.  **Executive Summary & Verdict**:
        * Start with a clear, one-word verdict: **BUY**, **HOLD**, or **AVOID**.
        * Provide a 2-3 sentence strategic rationale. Mention the trend direction (SuperTrend, EMAs), trend strength (ADX), and momentum (RSI).

    2.  **Detailed Technical Outlook**:
        * **Trend**: Is the SuperTrend bullish or bearish? Is the ADX above 25, indicating a strong trend?
        * **Momentum**: Is the RSI in a healthy range (30-70)? Is it overbought or oversold?
        * **Volatility**: What does the current ATR value suggest about the stock's daily price movement?

    3.  **Actionable Trading Plan (For BUY Verdicts Only)**:
        * **Portfolio Allocation**: Based on a ₹1,00,000 portfolio.
        * **Entry Strategy**: Suggest an ideal price range for entry (e.g., "Enter on a pullback near the 20-EMA at approx. ₹{price}").
        * **Position Size**: State the calculated number of shares to buy based on the 1.5% risk rule.
        * **Stop-Loss**: Provide the firm ATR-based stop-loss price. Emphasize its importance.
        * **Profit Target**: State the initial profit target based on a 1:1.5 Risk-to-Reward ratio.
        * **Capital at Risk**: Calculate and state the total potential loss in INR if the stop-loss is triggered.

    4.  **Backtesting Insights**:
        * Briefly comment on the historical performance based on the backtest results (e.g., "The backtest shows a promising win rate of {win_rate}% over {num_trades} trades, suggesting the strategy has been effective for this stock.").

    ---

    **Data Provided:**
    {analysis_data}
    """
    
    formatted_analysis = ""
    for symbol, data in state['analysis_results'].items():
        if "error" in data:
            formatted_analysis += f"\nCould not process {symbol}: {data['error']}\n"
        else:
            formatted_analysis += f"\n**Symbol: {symbol}**\n"
            formatted_analysis += f"**Latest Metrics:** {data['latest_metrics']}\n"
            formatted_analysis += f"**Position Sizing (for 1 Lakh Capital):** {data['position_sizing']}\n"
            formatted_analysis += f"**Backtest Results (1-Year):** {data['backtest']}\n"

    messages = [
        {"role": "system", "content": "You are an expert financial analyst creating a trading report."},
        {"role": "user", "content": prompt_template.format(symbol="[EACH STOCK]", analysis_data=formatted_analysis)}
    ]
    response = client.chat.completions.create(
        model=st.secrets["azure"]["model"], messages=messages, temperature=0.4
    )
    state['final_report'] = response.choices[0].message.content
    return state

# --- Build and Compile the LangGraph Agent ---
workflow = StateGraph(AgentState)
workflow.add_node("generateReport", generate_report_node)
workflow.set_entry_point("generateReport")
workflow.add_edge("generateReport", END)
agent_app = workflow.compile()

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================

tab1, tab2, tab3 = st.tabs(["🔥 Market Breakouts", "🚀 Top Movers", "🔍 Custom Analysis & AI Report"])

with tab1:
    st.header("🔥 Market Breakout Scanner")
    st.info("Scans for stocks trading near their 52-week high, indicating strong momentum. Scan is performed on the top 100 most active stocks.", icon="💡")
    if st.button("Scan for Breakouts", key='scan_breakouts'):
        with st.spinner("Scanning... This may take a few minutes."):
            breakout_stocks = get_breakout_stocks()
        if not breakout_stocks.empty:
            st.subheader("Potential Breakout Stocks Found")
            st.dataframe(breakout_stocks, use_container_width=True)
        else:
            st.warning("No breakout stocks found or an error occurred during the scan.")

with tab2:
    st.header("🚀 Top Market Movers (Daily)")
    st.info("The top 10 gainers and losers from the most recent trading day.", icon="💡")
    gainers, losers = get_top_gainers_losers()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Gainers")
        st.dataframe(gainers, use_container_width=True)
    with col2:
        st.subheader("Top 10 Losers")
        st.dataframe(losers, use_container_width=True)

with tab3:
    st.header("🔍 In-Depth Custom Stock Analysis")
    st.info("Enter NSE symbols for a detailed technical analysis, backtest, and an AI-powered trading plan.", icon="💡")
    symbols_str = st.text_input("Enter NSE stock symbols (comma-separated)", "SBIN,RELIANCE,TCS,TATAMOTORS")
    
    if st.button("Analyze & Generate AI Report", key='analyze_custom'):
        if symbols_str:
            symbols_list = [s.strip().upper() for s in symbols_str.split(',')]
            all_analyses = {}
            
            with st.spinner(f"Analyzing {len(symbols_list)} stocks in parallel..."):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(analyze_stock, symbol) for symbol in symbols_list]
                    for future in as_completed(futures):
                        result = future.result()
                        all_analyses[result['symbol']] = result

            st.subheader("Analysis Results")
            for symbol in symbols_list: # Display in the order user entered
                result = all_analyses.get(symbol)
                with st.expander(f"**{symbol}** - Verdict: {result.get('latest_metrics', {}).get('Verdict', 'N/A')}", expanded=False):
                    if "error" in result:
                        st.error(f"Could not analyze {symbol}: {result['error']}")
                        continue

                    # Display Chart
                    st.plotly_chart(create_interactive_chart(result['data'].tail(252), symbol), use_container_width=True)
                    
                    # Display Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Key Technicals")
                        st.table(pd.DataFrame.from_dict(result['latest_metrics'], orient='index', columns=['Value']))
                        st.subheader("Position Sizing (₹1L Capital)")
                        st.table(pd.DataFrame.from_dict(result['position_sizing'], orient='index', columns=['Value']))
                    with col2:
                        st.subheader("Strategy Backtest (1-Year)")
                        if "error" in result['backtest']:
                             st.warning(result['backtest']['error'])
                        elif "message" in result['backtest']:
                             st.info(result['backtest']['message'])
                        else:
                             st.table(pd.DataFrame.from_dict(result['backtest'], orient='index', columns=['Result']))
            
            # AI Report Generation
            st.subheader("🤖 AI-Generated Financial Report")
            with st.spinner("The AI Analyst is drafting the report..."):
                initial_state = {"analysis_results": all_analyses}
                final_state = agent_app.invoke(initial_state)
                report = final_state['final_report']
                st.markdown(report)
                
                # Download Report Button
                st.download_button(
                    label="Download Report as Markdown",
                    data=report,
                    file_name=f"analysis_report_{date.today()}.md",
                    mime="text/markdown",
                )
        else:
            st.warning("Please enter at least one stock symbol.")
