import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from financial_analyzer import FinancialAnalyzer
from market_data import MarketDataService
from document_parser import DocumentParser
from investment_strategy import InvestmentStrategy

# Initialize services
@st.cache_resource
def init_services():
    analyzer = FinancialAnalyzer()
    market_service = MarketDataService()
    doc_parser = DocumentParser()
    investment_strategy = InvestmentStrategy()
    return analyzer, market_service, doc_parser, investment_strategy

def main():
    st.set_page_config(
        page_title="FinDocAI - Financial Document Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 FinDocAI - AI-Powered Financial Document Analysis")
    st.markdown("Upload financial documents and get AI-powered insights, forecasts, and investment recommendations")
    
    # Initialize services
    analyzer, market_service, doc_parser, investment_strategy = init_services()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab_selection = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Document Upload & Q&A", "Market Analysis", "Investment Strategy", "Portfolio Dashboard"]
    )
    
    if tab_selection == "Document Upload & Q&A":
        document_qa_tab(analyzer, doc_parser)
    elif tab_selection == "Market Analysis":
        market_analysis_tab(market_service, analyzer)
    elif tab_selection == "Investment Strategy":
        investment_strategy_tab(investment_strategy, market_service)
    else:
        portfolio_dashboard_tab(market_service, investment_strategy)

def document_qa_tab(analyzer, doc_parser):
    st.header("📄 Document Analysis & Q&A")
    
    # Document upload
    uploaded_file = st.file_uploader(
        "Upload Financial Document",
        type=['pdf', 'txt', 'docx'],
        help="Upload earnings reports, financial filings, or press releases (Max 50MB)"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📎 Uploaded: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
        
        # Parse document
        with st.spinner("Parsing document..."):
            try:
                document_text = doc_parser.parse_document(uploaded_file)
                st.session_state['document_text'] = document_text
                st.success("✅ Document parsed successfully!")
                
                # Show document preview
                with st.expander("📖 Document Preview"):
                    st.text_area("Document Content (First 1000 characters)", 
                                document_text[:1000] + "..." if len(document_text) > 1000 else document_text,
                                height=200, disabled=True)
                
            except Exception as e:
                st.error(f"❌ Error parsing document: {str(e)}")
                return
        
        # Document Analysis
        st.subheader("🔍 Quick Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Analyze Sentiment", key="sentiment_btn"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment_result = analyzer.analyze_sentiment(document_text)
                    
                    sentiment_color = {
                        'positive': 'green',
                        'negative': 'red', 
                        'neutral': 'gray'
                    }
                    
                    st.metric(
                        "Sentiment",
                        sentiment_result['sentiment'].title(),
                        delta=f"Confidence: {sentiment_result['confidence']:.1%}"
                    )
                    
                    # Sentiment visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment_result['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Sentiment Score"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'bar': {'color': sentiment_color[sentiment_result['sentiment']]},
                            'steps': [
                                {'range': [-1, -0.3], 'color': "lightcoral"},
                                {'range': [-0.3, 0.3], 'color': "lightgray"},
                                {'range': [0.3, 1], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("⚠️ Detect Anomalies", key="anomaly_btn"):
                with st.spinner("Detecting anomalies..."):
                    anomalies = analyzer.detect_anomalies(document_text)
                    
                    if anomalies['anomalies_found']:
                        st.warning(f"⚠️ {len(anomalies['anomalies'])} anomalies detected")
                        for i, anomaly in enumerate(anomalies['anomalies'], 1):
                            st.write(f"{i}. **{anomaly['type']}**: {anomaly['description']}")
                            st.write(f"   *Severity: {anomaly['severity']}/10*")
                    else:
                        st.success("✅ No significant anomalies detected")
        
        # Q&A Section
        st.subheader("💬 Ask Questions About Your Document")
        
        # Predefined questions
        st.write("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 What's the revenue?"):
                st.session_state['user_question'] = "What is the company's revenue for this period?"
        
        with col2:
            if st.button("⚠️ What are the risks?"):
                st.session_state['user_question'] = "What are the main risks mentioned in this document?"
                
        with col3:
            if st.button("📊 Key metrics?"):
                st.session_state['user_question'] = "What are the key financial metrics mentioned?"
        
        # Custom question input
        user_question = st.text_input(
            "Or ask your own question:",
            value=st.session_state.get('user_question', ''),
            placeholder="e.g., What is the company's outlook for next quarter?"
        )
        
        if st.button("🔍 Get Answer") and user_question and 'document_text' in st.session_state:
            with st.spinner("Analyzing document to answer your question..."):
                try:
                    answer = analyzer.answer_question(st.session_state['document_text'], user_question)
                    st.success("✅ Analysis Complete")
                    st.write("**Answer:**")
                    st.write(answer['answer'])
                    
                    if answer.get('confidence'):
                        st.write(f"*Confidence: {answer['confidence']:.1%}*")
                        
                    if answer.get('key_excerpts'):
                        with st.expander("📋 Supporting Evidence"):
                            for excerpt in answer['key_excerpts']:
                                st.write(f"• {excerpt}")
                                
                except Exception as e:
                    st.error(f"❌ Error answering question: {str(e)}")

def market_analysis_tab(market_service, analyzer):
    st.header("📈 Market Analysis & Forecasting")
    
    # Stock symbol input
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, GOOGL, MSFT")
    with col2:
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"])
    
    if st.button("📊 Analyze Stock") and symbol:
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                # Get market data
                stock_data = market_service.get_stock_data(symbol, period)
                company_info = market_service.get_company_info(symbol)
                
                if stock_data.empty:
                    st.error(f"❌ No data found for symbol: {symbol}")
                    return
                
                # Display company info
                st.subheader(f"📊 {company_info.get('longName', symbol)} Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
                
                with col2:
                    st.metric("Market Cap", company_info.get('marketCap', 'N/A'))
                
                with col3:
                    st.metric("P/E Ratio", f"{company_info.get('forwardPE', 'N/A'):.2f}" if company_info.get('forwardPE') else 'N/A')
                
                with col4:
                    st.metric("52W High", f"${company_info.get('fiftyTwoWeekHigh', 'N/A')}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title=f"{symbol} Stock Price - {period}",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    name='Volume',
                    marker_color='orange'
                ))
                
                fig_volume.update_layout(
                    title=f"{symbol} Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume"
                )
                
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Technical analysis
                st.subheader("🔍 Technical Analysis")
                with st.spinner("Performing technical analysis..."):
                    tech_analysis = analyzer.technical_analysis(stock_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Moving Averages:**")
                        for ma, value in tech_analysis['moving_averages'].items():
                            st.write(f"• {ma}: ${value:.2f}")
                    
                    with col2:
                        st.write("**Technical Indicators:**")
                        st.write(f"• RSI: {tech_analysis['rsi']:.2f}")
                        st.write(f"• Volatility: {tech_analysis['volatility']:.2%}")
                        st.write(f"• Trend: {tech_analysis['trend']}")
                
                # Forecasting
                st.subheader("🔮 Price Forecast")
                if st.button("Generate 30-Day Forecast"):
                    with st.spinner("Generating forecast..."):
                        forecast = analyzer.generate_forecast(stock_data, symbol)
                        
                        # Create forecast chart
                        fig_forecast = go.Figure()
                        
                        # Historical data
                        fig_forecast.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast data
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast['dates'],
                            y=forecast['prices'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_forecast.update_layout(
                            title=f"{symbol} Price Forecast (30 Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)"
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast summary
                        st.write("**Forecast Summary:**")
                        st.write(f"• Predicted price in 30 days: ${forecast['target_price']:.2f}")
                        st.write(f"• Expected return: {forecast['expected_return']:+.1%}")
                        st.write(f"• Confidence: {forecast['confidence']:.1%}")
                        st.write(f"• Risk level: {forecast['risk_level']}")
                
        except Exception as e:
            st.error(f"❌ Error analyzing stock: {str(e)}")

def investment_strategy_tab(investment_strategy, market_service):
    st.header("💰 Investment Strategy & Recommendations")
    
    # Portfolio input
    st.subheader("📝 Portfolio Configuration")
    
    # Add stocks to portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        new_symbol = st.text_input("Add Stock Symbol", placeholder="e.g., AAPL")
    with col2:
        shares = st.number_input("Shares", min_value=1, value=100)
    with col3:
        if st.button("➕ Add to Portfolio"):
            if new_symbol:
                st.session_state.portfolio.append({
                    'symbol': new_symbol.upper(),
                    'shares': shares,
                    'added_date': datetime.now()
                })
                st.success(f"Added {new_symbol.upper()} to portfolio")
                st.rerun()
    
    # Display current portfolio
    if st.session_state.portfolio:
        st.subheader("📊 Current Portfolio")
        
        portfolio_data = []
        total_value = 0
        
        for item in st.session_state.portfolio:
            try:
                stock_data = market_service.get_stock_data(item['symbol'], '1d')
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    value = current_price * item['shares']
                    total_value += value
                    
                    portfolio_data.append({
                        'Symbol': item['symbol'],
                        'Shares': item['shares'],
                        'Current Price': f"${current_price:.2f}",
                        'Value': f"${value:.2f}"
                    })
            except:
                portfolio_data.append({
                    'Symbol': item['symbol'],
                    'Shares': item['shares'],
                    'Current Price': 'Error',
                    'Value': 'Error'
                })
        
        df_portfolio = pd.DataFrame(portfolio_data)
        st.dataframe(df_portfolio, use_container_width=True)
        
        st.metric("Total Portfolio Value", f"${total_value:.2f}")
        
        # Clear portfolio button
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = []
            st.rerun()
        
        # Generate recommendations
        if st.button("🎯 Generate Investment Recommendations"):
            with st.spinner("Analyzing portfolio and generating recommendations..."):
                try:
                    recommendations = investment_strategy.generate_recommendations(st.session_state.portfolio, market_service)
                    
                    st.subheader("🎯 Investment Recommendations")
                    
                    for rec in recommendations['recommendations']:
                        # Create colored boxes for different recommendation types
                        color_map = {
                            'BUY': 'green',
                            'SELL': 'red',
                            'HOLD': 'gray'
                        }
                        
                        st.markdown(f"""
                        <div style="padding: 10px; margin: 5px 0; border-left: 5px solid {color_map.get(rec['action'], 'gray')}; background-color: rgba(128,128,128,0.1);">
                        <h4>{rec['symbol']} - {rec['action']}</h4>
                        <p><strong>Confidence:</strong> {rec['confidence']:.1%}</p>
                        <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
                        <p><strong>Target Price:</strong> ${rec['target_price']:.2f}</p>
                        <p><strong>Risk Level:</strong> {rec['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Portfolio analysis
                    st.subheader("📈 Portfolio Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Score", f"{recommendations['portfolio_score']:.1f}/10")
                    
                    with col2:
                        st.metric("Risk Level", recommendations['risk_assessment'])
                    
                    with col3:
                        st.metric("Diversification", recommendations['diversification_score'])
                    
                    # Recommendations summary
                    st.write("**Portfolio Insights:**")
                    for insight in recommendations['insights']:
                        st.write(f"• {insight}")
                
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")

def portfolio_dashboard_tab(market_service, investment_strategy):
    st.header("📊 Portfolio Dashboard")
    
    st.info("💡 This dashboard provides an overview of market trends and investment opportunities.")
    
    # Market overview
    st.subheader("🌍 Market Overview")
    
    # Major indices
    indices = ['SPY', 'QQQ', 'IWM', 'VTI']
    
    cols = st.columns(len(indices))
    
    for i, index in enumerate(indices):
        try:
            with cols[i]:
                data = market_service.get_stock_data(index, '5d')
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[0]
                    change_pct = ((current - prev) / prev) * 100
                    
                    st.metric(
                        index,
                        f"${current:.2f}",
                        f"{change_pct:+.1f}%"
                    )
        except:
            with cols[i]:
                st.metric(index, "Error", "N/A")
    
    # Sector performance
    st.subheader("🏭 Sector Performance")
    
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV', 
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Consumer': 'XLY',
        'Industrials': 'XLI'
    }
    
    sector_data = []
    for sector, etf in sector_etfs.items():
        try:
            data = market_service.get_stock_data(etf, '1mo')
            if not data.empty:
                current = data['Close'].iloc[-1]
                start = data['Close'].iloc[0]
                performance = ((current - start) / start) * 100
                
                sector_data.append({
                    'Sector': sector,
                    'ETF': etf,
                    'Performance (1M)': f"{performance:+.1f}%",
                    'Current Price': f"${current:.2f}"
                })
        except:
            sector_data.append({
                'Sector': sector,
                'ETF': etf,
                'Performance (1M)': 'Error',
                'Current Price': 'Error'
            })
    
    if sector_data:
        df_sectors = pd.DataFrame(sector_data)
        st.dataframe(df_sectors, use_container_width=True)
    
    # Market sentiment
    st.subheader("📊 Market Sentiment Analysis")
    
    if st.button("📈 Analyze Current Market Sentiment"):
        with st.spinner("Analyzing market sentiment..."):
            try:
                # This would typically analyze news, social media, etc.
                # For demo purposes, we'll show a simulated analysis
                sentiment_data = {
                    'overall_sentiment': 'Cautiously Optimistic',
                    'sentiment_score': 0.65,
                    'key_factors': [
                        'Positive earnings reports from major tech companies',
                        'Federal Reserve policy uncertainty',
                        'Geopolitical tensions affecting energy markets',
                        'Strong consumer spending data'
                    ],
                    'recommendation': 'Mixed signals suggest a balanced approach to portfolio allocation'
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Market Sentiment", sentiment_data['overall_sentiment'])
                    
                    # Sentiment gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment_data['sentiment_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Sentiment Score"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "lightblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightcoral"},
                                {'range': [0.3, 0.7], 'color': "lightyellow"},
                                {'range': [0.7, 1], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Key Market Factors:**")
                    for factor in sentiment_data['key_factors']:
                        st.write(f"• {factor}")
                    
                    st.write("**Recommendation:**")
                    st.info(sentiment_data['recommendation'])
            
            except Exception as e:
                st.error(f"❌ Error analyzing market sentiment: {str(e)}")

if __name__ == "__main__":
    main()
