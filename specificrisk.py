import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Quant Risk Lab", layout="wide")

# --- LEGAL DISCLAIMER ---
with st.expander("⚖️ LEGAL DISCLAIMER / ZASTRZEŻENIE PRAWNE"):
    st.caption("""
    **ENG:** This application is for educational purposes only and presents quantitative financial models 
    based on historical data. It does not constitute investment advice or a recommendation to buy 
    or sell any financial instruments within the meaning of the Regulation of the Minister of Finance 
    or any other applicable law.
    
    **PL:** Program ma charakter wyłącznie edukacyjny i nie stanowi rekomendacji inwestycyjnej 
    w rozumieniu przepisów polskiego prawa. Inwestowanie wiąże się z ryzykiem utraty kapitału.
    """)

st.divider()
st.title("🛡️ Quant Risk Lab: Stock portfolio Risk & Factor Analysis")

# --- COMPREHENSIVE METHODOLOGY INTRODUCTION ---
st.markdown(r"""
### 🏗️ Sharpe's Single-Index Model (SIM) & Risk Decomposition
**Author: Eryk Syldatk CAI**

This tool is designed to audit the efficiency of your portfolio diversification. We utilize **Sharpe's Single-Index Model (SIM)** to decompose the total risk of your portfolio into two fundamental components. According to the model, the portfolio return ($R_p$) is a linear function of the market return ($R_m$):

$$ R_p = \alpha_p + \beta_p R_m + e_p $$
            
#### 📋 Breakdown of the Model Variables:
* **$R_p$ (Portfolio Return):** The total realized return of your portfolio.
* **$\alpha_p$ (Alpha):** The intercept. It represents the "excess return" or "added value" provided by the manager. If $\alpha > 0$, you are beating the market on a risk-adjusted basis.
* **$\beta_p$ (Beta):** The slope coefficient. It measures the sensitivity of your portfolio to market movements. (e.g., $\beta = 1.2$ means your portfolio is 20% more volatile than the market).
* **$R_m$ (Market Return):** The return of your chosen benchmark (e.g., S&P 500).
* **$e_p$ (Residual/Error Term):** The "noise." It represents the part of the return that **cannot be explained by the market**. This is where your **Specific Risk** lives.

#### 🔬 How do we measure your risk?
The total volatility of your investments (Total Variance $\sigma_p^2$) consists of:

1.  **Systematic Risk (Market Risk):** Driven by your exposure to the broad market index. This risk is inherent to the entire market and **cannot be eliminated** through diversification. 
    $$ \text{Market Risk} = \beta_p^2 \cdot \sigma_m^2 $$
2.  **Specific Risk (Idiosyncratic Risk):** Driven by news and events unique to your specific companies. **In a well-constructed portfolio, this risk should be diversified away.**
    $$ \text{Specific Risk} = \sigma^2(e_p) $$



#### 🎯 Key Metric: $R^2$ (The Diversification Index)
The coefficient of determination **$R^2$** indicates the percentage of your total risk that is attributed to market movements:
$$ R^2 = \frac{\text{Systematic Risk}}{\text{Total Risk}} $$

* **$R^2 > 85\%$:** **Professional Diversification.** Your performance is primarily driven by global market trends. You have successfully minimized idiosyncratic exposure.
* **$R^2 < 60\%$:** **Concentration Risk.** Your portfolio lacks proper diversification. Your returns are essentially a "bet" on specific corporate events rather than market growth. You are highly exposed to the risk of a single company's failure!
""")

st.divider()

# ==============================================================================
# 2. SIDEBAR - CONFIGURATION
# ==============================================================================
st.sidebar.header("🧪 Risk Analytics Settings")

# PORTFOLIO SELECTION
st.sidebar.subheader("1. Stock portfolio Components")
tickers_input = st.sidebar.text_input("Enter Yahoo Tickers (comma separated)", value="UNH, MSTR, AMZN, MELI")
# Zmienione na 'tickers', żeby pasowało do reszty kodu
tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]

# Wagi
weights_input = st.sidebar.text_input("Weights (comma separated, e.g. 0.2, 0.2...)", value="")
if weights_input:
    weights = [float(w.strip()) for w in weights_input.split(",")]
else:
    weights = [1.0/len(tickers)] * len(tickers) if tickers else []

# Sprawdzenie wag
if len(weights) != len(tickers):
    st.sidebar.error("❌ Number of weights must match number of tickers!")
    st.stop()
    
# BENCHMARK SELECTION
st.sidebar.subheader("2. Benchmark & Risk-Free")
benchmark_ticker = st.sidebar.text_input("Market Proxy (Benchmark)", value="SPY")
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=4.0) / 100

# TIME PARAMETERS
st.sidebar.subheader("3. Time Horizon")
start_date = st.sidebar.date_input("Analysis Start Date", value=datetime.date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date.today())

st.sidebar.divider()
# Zmienione na 'run_analysis', żeby blok niżej go widział
run_analysis = st.sidebar.button("🛡️ ANALYZE PORTFOLIO RISK")

# ==============================================================================
# 3. CORE CALCULATION ENGINE
# ==============================================================================

@st.cache_data(ttl=3600)
def fetch_portfolio_data(tickers_list, benchmark, start, end):
    try:
        all_symbols = list(set(tickers_list + [benchmark]))
        # Pobieramy dane
        raw_data = yf.download(all_symbols, start=start, end=end, interval="1d", progress=False)
        
        if raw_data.empty:
            st.error("❌ No data found for the given tickers or date range.")
            return None
            
        # --- ROZWIĄZANIE BŁĘDU KEYERROR (MultiIndex Flattening) ---
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Używamy .xs, aby precyzyjnie wyciąć poziom Adj Close lub Close
            if 'Adj Close' in raw_data.columns.levels[0]:
                df = raw_data.xs('Adj Close', axis=1, level=0)
            else:
                df = raw_data.xs('Close', axis=1, level=0)
        else:
            # Jeśli jest tylko jeden ticker, yfinance nie tworzy MultiIndex
            df = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]
        
        # Kluczowy krok: Czyścimy nazwy kolumn, żeby były czystymi stringami
        df.columns = [str(c).strip().upper() for c in df.columns] # Wymuszamy DUŻE LITERY I ZERO SPACJI
        
        # Log zwroty (Total Return)
        returns = np.log(df / df.shift(1)).dropna()
        
        # Debugging (opcjonalne, możesz odkomentować w razie problemów):
        # st.write("Dostępne kolumny w returns:", list(returns.columns))
        
        return returns
        
    except Exception as e:
        st.error(f"🚨 Critical Error during data fetching: {e}")
        return None

# Placeholder dla danych w sesji
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if run_analysis:
    with st.spinner("📥 Synchronizing market data..."):
        # 1. Czyścimy listę wejściową od razu
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        benchmark_ticker = benchmark_ticker.strip().upper()
        
        returns_df = fetch_portfolio_data(tickers, benchmark_ticker, start_date, end_date)
        
        if returns_df is not None:
            # --- DEBUG (Tylko jeśli chcesz sprawdzić co jest w środku) ---
            # st.write("Kolumny w tabeli:", list(returns_df.columns))
            # st.write("Szukane tickery:", tickers)

            # 2. Bezpieczne pobieranie serii danych
            # Sprawdzamy, który ticker z listy faktycznie jest w tabeli
            available_tickers = [t for t in tickers if t in returns_df.columns]
            
            if not available_tickers:
                st.error("❌ Some of the tickers were not found.")
                st.stop()

            # Teraz bierzemy pierwszy DOSTĘPNY ticker do checku długości
            stock_series = returns_df[available_tickers[0]].dropna()
            bench_series = returns_df[benchmark_ticker].dropna()
            
            bench_len = len(bench_series)
            stock_len = len(stock_series)

            if bench_len <= 1:
                st.error(f"🚨 **Critical Data Error: Benchmark is empty!**")
                st.markdown(f"""
                Yahoo Finance returned only **{bench_len}** days for `{benchmark_ticker}`.
                """)
                st.stop() # Zatrzymujemy kod tutaj - nie idziemy dalej w błędy

            if bench_len < (stock_len * 0.7):
                st.warning(f"⚠️ **Data Mismatch:** The benchmark has significantly less history ({bench_len} days) than your stocks ({stock_len} days). Beta and R² might be unreliable.")
            
            # --- KONIEC CHECKU ---

            # Kontynuacja separacji danych
            stock_returns = returns_df[tickers]
            bench_returns = returns_df[benchmark_ticker]
            
            # Zwrot portfela (ważony)
            portfolio_returns = stock_returns.dot(np.array(weights))
            
            # Zapis do sesji
            st.session_state.weights = weights
            st.session_state.tickers = tickers
            st.session_state.benchmark = benchmark_ticker
            st.session_state.stock_returns = stock_returns
            st.session_state.bench_returns = bench_returns
            st.session_state.portfolio_returns = portfolio_returns
            st.session_state.analysis_done = True
            st.success(f"✅ Market data synced ({bench_len} days of benchmark found). Single-Index Model ready.")
# Strażnik – nie idź dalej, jeśli nie ma danych
if not st.session_state.analysis_done:
    st.info("👈 Configure your portfolio and click 'Analyze' to start.")
    st.stop()
# ==============================================================================
# 4. PERFORMANCE & RISK AUDIT (ANNUALIZED METRICS)
# ==============================================================================
stock_returns = st.session_state.stock_returns
bench_returns = st.session_state.bench_returns
portfolio_returns = st.session_state.portfolio_returns
weights = st.session_state.weights
tickers = st.session_state.tickers

st.header(f"🔍 Portfolio Audit: {', '.join(tickers)}")

def get_metrics(returns, benchmark, rf):
    # 1. CAGR (Compound Annual Growth Rate) - The geometric mean return
    total_return = np.exp(returns.sum())
    n_years = len(returns) / 252
    cagr = (total_return ** (1 / n_years)) - 1
    
    # 2. Volatility (Annualized Standard Deviation) - Total Risk
    vol = returns.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio - Reward-to-Risk efficiency
    sharpe = (cagr - rf) / vol if vol != 0 else 0
    
    # 4. OLS Regression (Single Index Model)
    X = sm.add_constant(benchmark)
    model = sm.OLS(returns, X).fit()
    beta = model.params[1]
    r2 = model.rsquared # This is our Diversification Index
    
    return cagr, vol, sharpe, beta, r2

analysis_data = []

# Calculation for individual stocks
for ticker in tickers:
    cagr, vol, sharpe, beta, r2 = get_metrics(stock_returns[ticker], bench_returns, rf_rate)
    analysis_data.append({
        "Ticker": ticker,
        "CAGR (Ann. Return)": f"{cagr:.2%}",
        "Volatility (Total Risk)": f"{vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Beta (Market Sensitivity)": round(beta, 2),
        "Market Risk (R²)": f"{r2:.1%}",
        "Specific Risk (1-R²)": f"{1-r2:.1%}"
    })

# Calculation for the entire portfolio
p_cagr, p_vol, p_sharpe, p_beta, p_r2 = get_metrics(portfolio_returns, bench_returns, rf_rate)
analysis_data.append({
    "Ticker": "⭐ PORTFOLIO",
    "CAGR (Ann. Return)": f"{p_cagr:.2%}",
    "Volatility (Total Risk)": f"{p_vol:.2%}",
    "Sharpe Ratio": round(p_sharpe, 2),
    "Beta (Market Sensitivity)": round(p_beta, 2),
    "Market Risk (R²)": f"{p_r2:.1%}",
    "Specific Risk (1-R²)": f"{1-p_r2:.1%}"
})

# Display the Audit Table
st.subheader("📋 Audit Results: Selection vs. Diversification")
df_audit = pd.DataFrame(analysis_data)
st.dataframe(df_audit.set_index("Ticker"), use_container_width=True)

# --- METRIC DICTIONARY & INTERPRETATION ---
with st.expander("📖 Audit Metric Dictionary - How to interpret the results?"):
    st.markdown("""
    | Metric | Definition | How to interpret? |
    | :--- | :--- | :--- |
    | **CAGR** | Compound Annual Growth Rate. | Your average annual gain. Compare this to the Benchmark. |
    | **Volatility** | Annualized standard deviation of returns. | Represents the 'total bumpiness' of the asset. |
    | **Sharpe Ratio** | Excess return per unit of volatility. | Higher is better. > 1.0 is considered excellent. |
    | **Beta (β)** | Sensitivity to the Market (Benchmark). | 1.0 = same as market. > 1.0 = more aggressive. < 1.0 = defensive. |
    | **Market Risk (R²)** | The percentage of risk driven by the market. | **This is your Diversification Score.** High R² means you have eliminated most specific risks. |
    | **Specific Risk** | The percentage of risk unique to the company. | Risk caused by earnings, scandals, or product failures. A diversified portfolio should minimize this. |
    """)

st.divider()

# ==============================================================================
# 5. DIVERSIFICATION DEEP-DIVE: CORRELATION & COINTEGRATION
# ==============================================================================
st.header("🔗 Diversification Deep-Dive")

st.markdown(r"""
#### 🧠 Correlation vs. Cointegration: The Inefficiency Proof
While **Correlation** measures short-term co-movement, **Cointegration** is a much deeper statistical link. 
If two stocks are cointegrated, they are tethered by a long-term "equilibrium rope." 

**The EMH Conflict:** According to the **Weak-form Efficient Market Hypothesis (EMH)**, stock prices should follow a "Random Walk," making them unpredictable. 
* **The Smoking Gun:** If two stocks are cointegrated, their price spread is **mean-reverting**. 
* **Predictability:** This implies that if the gap between them gets too wide, we can predict they will come back together. This predictability is a direct contradiction of pure market efficiency.
* **Risk Warning:** Cointegrated stocks offer **zero long-term diversification**. You are essentially holding the same asset under two different names.
""")

c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("🕵️‍♂️ Twin Stock Detector (Correlation)")
    st.caption("Short-term linear relationship (-1 to +1)")
    
    # Calculate Correlation Matrix
    corr_matrix = stock_returns.corr()
    
    # Plotting Heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    im = ax_corr.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    fig_corr.colorbar(im)
    
    # Labels
    ax_corr.set_xticks(np.arange(len(tickers)))
    ax_corr.set_yticks(np.arange(len(tickers)))
    ax_corr.set_xticklabels(tickers)
    ax_corr.set_yticklabels(tickers)
    
    # Adding text values
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")
    st.pyplot(fig_corr)

with c2:
    st.subheader("⚖️ Inefficiency Test (Cointegration)")
    st.caption("Long-term 'Leash' detection (Engle-Granger Test on Log-Prices)")
    
    from statsmodels.tsa.stattools import coint
    
    # 1. PRACA NA LOGARYTMACH: log_prices to bezpośrednio suma zwrotów
    log_prices_df = stock_returns.cumsum()
    
    coint_results = []
    
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            
            # Test na logarytmach (analitycznie poprawny)
            _, p_value, _ = coint(log_prices_df[t1], log_prices_df[t2])
            
            # Twoje oryginalne progi i opisy
            if p_value < 0.05:
                status = "🚨 Cointegrated"
                verdict = "Market Inefficiency detected. These stocks move as ONE in the long run."
            else:
                status = "✅ Independent"
                verdict = "True diversification. No long-term tethering found."
                
            coint_results.append({
                "Pair": f"{t1} / {t2}",
                "P-Value": round(p_value, 4),
                "Status": status,
                "Audit Verdict": verdict,
                "is_significant": p_value < 0.05 # Flaga techniczna
            })
            
    # Zapisujemy do sesji, żeby sekcja 7 mogła pobrać dane
    st.session_state.coint_results_data = coint_results
            
    if coint_results:
        st.dataframe(pd.DataFrame(coint_results).drop(columns="is_significant").set_index("Pair"), use_container_width=True)
    else:
        st.info("Add more tickers to perform pairwise Cointegration tests.")
st.divider()

# ==============================================================================
# 6. STRATEGIC RISK VERDICT (GATEWAY LOGIC)
# ==============================================================================
st.header("🏁 Strategic Risk Verdict")

# 1. Definiujemy próg dywersyfikacji (standardowo 70%)
DIVERSIFICATION_THRESHOLD = 0.70
is_diversified = p_r2 >= DIVERSIFICATION_THRESHOLD

# 2. Status Card - Wyświetlamy to ZAWSZE na starcie sekcji
st.subheader("📡 Diversification Status Check")

if is_diversified:
    st.success(f"✅ **PORTFOLIO DIVERSIFIED** (R² = {p_r2:.1%})")
    st.write("Your idiosyncratic risk is sufficiently neutralized. Advanced metrics like Treynor Ratio are now valid and unlocked.")
else:
    st.error(f"🚨 **PORTFOLIO CONCENTRATED** (R² = {p_r2:.1%})")
    st.write("Specific (idiosyncratic) risk dominates this portfolio. Diversification is insufficient to rely on Beta-based performance metrics.")

st.divider()

# 3. Warunkowe odpalenie podsumowania
if is_diversified:
    # --- FULL REPORT FOR DIVERSIFIED PORTFOLIO ---
    st.subheader("⚔️ Institutional Grade Performance Audit")

    def calculate_full_metrics(returns, rf, b_beta):
        total_ret = np.exp(returns.sum())
        n_years = len(returns) / 252
        cagr = (total_ret ** (1 / n_years)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (cagr - rf) / vol if vol != 0 else 0
        
        # Treynor - valid here because portfolio is diversified
        treynor = (cagr - rf) / b_beta if b_beta != 0 else 0
        
        # Risk stats
        cumulative = np.exp(returns.cumsum())
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
        var_95 = returns.mean() - 1.645 * returns.std()
        
        return [f"{cagr:.2%}", f"{vol:.2%}", f"{sharpe:.2f}", f"{treynor:.2f}", f"{max_dd:.2%}", f"{var_95:.2%}"]

    comp_data = {
        "Metric": ["Annual Return (CAGR)", "Annual Volatility", "Sharpe Ratio", "Treynor Ratio", "Max Drawdown", "Daily VaR (95%)"],
        "Portfolio": calculate_full_metrics(portfolio_returns, rf_rate, p_beta),
        "Benchmark": calculate_full_metrics(bench_returns, rf_rate, 1.0)
    }
    
    st.table(pd.DataFrame(comp_data).set_index("Metric"))
    
    st.info("💡 **Insight:** Since your R² is high, the Treynor Ratio provides a reliable measure of how well you are compensated for systemic (market) risk.")

else:
    # --- LIMITED REPORT FOR NON-DIVERSIFIED PORTFOLIO ---
    st.subheader("⚠️ Limited Performance Audit")
    st.warning("Treynor Ratio is disabled (requires diversification). VaR and Sharpe remain valid as they measure Total Risk.")

    def calculate_limited_metrics(returns, rf):
        total_ret = np.exp(returns.sum())
        n_years = len(returns) / 252
        cagr = (total_ret ** (1 / n_years)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = (cagr - rf) / vol if vol != 0 else 0
        
        cumulative = np.exp(returns.cumsum())
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
        
        # VaR - odblokowany, bo bazuje na całkowitym odchyleniu (vol)
        var_95 = returns.mean() - 1.645 * returns.std()
        
        return [f"{cagr:.2%}", f"{vol:.2%}", f"{sharpe:.2f}", "N/A (Not Diversified)", f"{max_dd:.2%}", f"{var_95:.2%}"]

    comp_data = {
        "Metric": ["Annual Return (CAGR)", "Annual Volatility", "Sharpe Ratio", "Treynor Ratio", "Max Drawdown", "Daily VaR (95%)"],
        "Portfolio": calculate_limited_metrics(portfolio_returns, rf_rate),
        "Benchmark": calculate_limited_metrics(bench_returns, rf_rate)
    }
    
    st.table(pd.DataFrame(comp_data).set_index("Metric"))
    
    st.error("""
    **⚠️ Critical Note on VaR in Concentrated Portfolios:**
    Since your diversification is low, this VaR calculation (based on Normal Distribution) might 
    **underestimate** your true risk. Individual stocks often have 'Fat Tails', meaning 
    extreme losses occur more often than the math suggests.
    """)

# ==============================================================================
# 8. ACTIVE STRATEGY: KELLY CRITERION REBALANCING
# ==============================================================================
st.divider()
st.header("🎯 Active Strategy: Kelly Criterion Optimizer")
st.markdown(r"""
Use this module to calculate optimal portfolio weights based on **your forward-looking views**. 
Unlike passive models, the Kelly Criterion maximizes the long-term growth of capital by exploiting the asymmetry between reward and risk.

The formula used is the **Kelly Fraction**:
$$f^* = \frac{bp - q}{b}$$
where $b$ is the odds ($Upside/Downside$), $p$ is the probability of success, and $q$ is the probability of failure ($1-p$).
""")

# 1. Retrieve data from session
tickers = st.session_state.tickers
stock_returns = st.session_state.stock_returns
current_weights = st.session_state.weights
prices_df = np.exp(stock_returns.cumsum())
last_prices = prices_df.iloc[-1]

# 2. User Input Interface
kelly_inputs = []

with st.expander("🛠️ Configure Market Views (Expected Payoff & Probability)", expanded=True):
    st.info("Pro Tip: Reference the 'Volatility' column in the Audit Results above to set realistic Stop Loss levels.")
    cols = st.columns(len(tickers))
    
    for i, ticker in enumerate(tickers):
        with cols[i]:
            st.subheader(f"📍 {ticker}")
            # Individual views
            u_upside = st.number_input(f"Expected Upside (%)", key=f"k_up_{ticker}", value=20.0) / 100
            u_downside = st.number_input(f"Stop Loss (%)", key=f"k_down_{ticker}", value=10.0) / 100
            u_prob = st.slider(f"Conviction (p)", 0.0, 1.0, 0.5, key=f"k_p_{ticker}")
            
            # Kelly Mathematics
            # b = Win Amount / Loss Amount
            b = u_upside / u_downside if u_downside != 0 else 0
            q = 1 - u_prob
            
            # f* calculation
            f_star = (b * u_prob - q) / b if b > 0 else 0
            
            kelly_inputs.append({
                "Ticker": ticker,
                "Price": round(last_prices[ticker], 2),
                "Potential (b)": b,
                "Conviction (p)": u_prob,
                "Raw Kelly": f_star
            })

# 3. Weight Processing (Half-Kelly for safety)
df_k = pd.DataFrame(kelly_inputs)
# Applying Half-Kelly to reduce drawdown volatility
df_k["Half-Kelly Weight"] = df_k["Raw Kelly"].apply(lambda x: max(0, x / 2))

# Normalizing weights to sum to 100%
total_k = df_k["Half-Kelly Weight"].sum()
if total_k > 0:
    df_k["Kelly Allocation (%)"] = (df_k["Half-Kelly Weight"] / total_k)
else:
    df_k["Kelly Allocation (%)"] = 0

# 4. Comparison Table & Chart
st.subheader("⚖️ Allocation Comparison: Current vs. Kelly Suggestion")

comparison_df = pd.DataFrame({
    "Ticker": tickers,
    "Current Weight (%)": [w * 100 for w in current_weights],
    "Kelly Suggestion (%)": df_k["Kelly Allocation (%)"].values * 100
}).set_index("Ticker")

c1, c2 = st.columns([1, 1.5])

with c1:
    st.table(comparison_df.style.format("{:.2f}%"))

with c2:
    st.bar_chart(comparison_df)

# 5. "What-If" Impact Analysis
st.subheader("🧪 Portfolio 'What-If' Simulation")
new_weights = df_k["Kelly Allocation (%)"].values
new_portfolio_returns = (stock_returns * new_weights).sum(axis=1)

def quick_audit(returns):
    total_ret = np.exp(returns.sum())
    n_years = len(returns) / 252
    cagr = (total_ret ** (1 / n_years)) - 1
    vol = returns.std() * np.sqrt(252)
    return cagr, vol

old_cagr, old_vol = quick_audit(st.session_state.portfolio_returns)
new_cagr, new_vol = quick_audit(new_portfolio_returns)

# Metric visualization
m1, m2 = st.columns(2)
m1.metric("CAGR (Expected Return)", f"{new_cagr:.2%}", f"{new_cagr - old_cagr:+.2%}")
m2.metric("Volatility (Total Risk)", f"{new_vol:.2%}", f"{new_vol - old_vol:+.2%}", delta_color="inverse")



st.info("""
💡 **Analytical Insight:** If the Kelly Criterion suggests an allocation significantly different from your current one, your portfolio is not mathematically optimized for your market convictions. 
Note the change in **Volatility**—optimizing for growth via Kelly often leads to higher portfolio variance.
""")
