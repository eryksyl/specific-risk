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
