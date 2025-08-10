import json
import os
from datetime import datetime, timedelta
import numpy as np
from google import genai
from google.genai import types

class InvestmentStrategy:
    def __init__(self):
        # Using Google Gemini API instead of OpenAI
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
    def generate_recommendations(self, portfolio, market_service):
        """Generate investment recommendations based on portfolio and market data"""
        try:
            recommendations = []
            portfolio_analysis = self._analyze_portfolio(portfolio, market_service)
            
            for stock in portfolio:
                symbol = stock['symbol']
                shares = stock['shares']
                
                # Get current market data
                stock_data = market_service.get_stock_data(symbol, '3mo')
                company_info = market_service.get_company_info(symbol)
                
                if stock_data.empty:
                    continue
                
                # Analyze individual stock
                stock_analysis = self._analyze_individual_stock(
                    symbol, stock_data, company_info, market_service
                )
                
                # Generate AI-powered recommendation
                recommendation = self._generate_ai_recommendation(
                    symbol, stock_analysis, portfolio_analysis
                )
                
                recommendations.append(recommendation)
            
            # Overall portfolio assessment
            overall_assessment = self._assess_portfolio_health(
                portfolio, recommendations, market_service
            )
            
            return {
                'recommendations': recommendations,
                'portfolio_score': overall_assessment['score'],
                'risk_assessment': overall_assessment['risk_level'],
                'diversification_score': overall_assessment['diversification'],
                'insights': overall_assessment['insights']
            }
            
        except Exception as e:
            return {
                'recommendations': [],
                'portfolio_score': 5.0,
                'risk_assessment': 'unknown',
                'diversification_score': 'unknown',
                'insights': [f'Error generating recommendations: {str(e)}']
            }
    
    def _analyze_portfolio(self, portfolio, market_service):
        """Analyze overall portfolio characteristics"""
        try:
            sectors = {}
            total_value = 0
            portfolio_data = []
            
            for stock in portfolio:
                symbol = stock['symbol']
                shares = stock['shares']
                
                # Get company info for sector analysis
                company_info = market_service.get_company_info(symbol)
                stock_data = market_service.get_stock_data(symbol, '1d')
                
                if not stock_data.empty and 'currentPrice' in company_info:
                    current_price = company_info.get('currentPrice', stock_data['Close'].iloc[-1])
                    value = current_price * shares
                    total_value += value
                    
                    sector = company_info.get('sector', 'Unknown')
                    if sector not in sectors:
                        sectors[sector] = 0
                    sectors[sector] += value
                    
                    portfolio_data.append({
                        'symbol': symbol,
                        'shares': shares,
                        'current_price': current_price,
                        'value': value,
                        'sector': sector,
                        'company_info': company_info
                    })
            
            # Calculate sector allocation percentages
            sector_allocation = {}
            for sector, value in sectors.items():
                sector_allocation[sector] = (value / total_value) * 100 if total_value > 0 else 0
            
            return {
                'total_value': total_value,
                'stock_count': len(portfolio),
                'sector_allocation': sector_allocation,
                'portfolio_data': portfolio_data,
                'diversification_score': self._calculate_diversification_score(sector_allocation)
            }
            
        except Exception as e:
            return {
                'total_value': 0,
                'stock_count': len(portfolio),
                'sector_allocation': {},
                'portfolio_data': [],
                'diversification_score': 0,
                'error': str(e)
            }
    
    def _analyze_individual_stock(self, symbol, stock_data, company_info, market_service):
        """Analyze individual stock metrics"""
        try:
            current_price = stock_data['Close'].iloc[-1]
            
            # Price performance
            price_30d_ago = stock_data['Close'].iloc[-30] if len(stock_data) >= 30 else stock_data['Close'].iloc[0]
            price_performance_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
            
            # Volatility
            returns = stock_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Volume analysis
            avg_volume = stock_data['Volume'].mean()
            recent_volume = stock_data['Volume'].tail(5).mean()
            volume_trend = ((recent_volume - avg_volume) / avg_volume) * 100
            
            # Technical indicators
            ma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = stock_data['Close'].rolling(50).mean().iloc[-1] if len(stock_data) >= 50 else None
            
            # RSI calculation
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_performance_30d': price_performance_30d,
                'volatility': volatility,
                'volume_trend': volume_trend,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'rsi': rsi,
                'pe_ratio': company_info.get('forwardPE'),
                'market_cap': company_info.get('marketCap'),
                'sector': company_info.get('sector'),
                'beta': company_info.get('beta'),
                'dividend_yield': company_info.get('dividendYield')
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def _generate_ai_recommendation(self, symbol, stock_analysis, portfolio_analysis):
        """Generate AI-powered investment recommendation"""
        try:
            # Prepare analysis summary for AI
            analysis_summary = f"""
            Stock: {symbol}
            Current Price: ${stock_analysis.get('current_price', 0):.2f}
            30-day Performance: {stock_analysis.get('price_performance_30d', 0):.1f}%
            Volatility: {stock_analysis.get('volatility', 0):.2f}
            RSI: {stock_analysis.get('rsi', 50):.1f}
            P/E Ratio: {stock_analysis.get('pe_ratio', 'N/A')}
            Sector: {stock_analysis.get('sector', 'Unknown')}
            Beta: {stock_analysis.get('beta', 'N/A')}
            
            Portfolio Context:
            Total Portfolio Value: ${portfolio_analysis.get('total_value', 0):,.2f}
            Portfolio Diversification Score: {portfolio_analysis.get('diversification_score', 0):.1f}/10
            """
            
            system_prompt = """You are a professional investment advisor. Based on the stock analysis and portfolio context, provide investment recommendations.
            
            Consider:
            - Technical indicators (RSI, moving averages, volatility)
            - Fundamental metrics (P/E ratio, market cap, sector)
            - Portfolio diversification needs
            - Risk management principles
            
            Provide recommendations as BUY, SELL, or HOLD with clear reasoning.
            
            Respond in JSON format: {
                "action": "BUY|SELL|HOLD",
                "confidence": float (0 to 1),
                "target_price": float,
                "reasoning": "detailed explanation",
                "risk_level": "low|medium|high",
                "time_horizon": "short|medium|long",
                "key_factors": ["factor1", "factor2"]
            }"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=f"Analyze and recommend action for:\n{analysis_summary}")])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json"
                ),
            )
            
            result = json.loads(response.text)
            result['symbol'] = symbol
            
            return result
            
        except Exception as e:
            # Fallback recommendation
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.5,
                'target_price': stock_analysis.get('current_price', 0),
                'reasoning': f'Unable to generate detailed analysis: {str(e)}',
                'risk_level': 'medium',
                'time_horizon': 'medium',
                'key_factors': ['Analysis error']
            }
    
    def _assess_portfolio_health(self, portfolio, recommendations, market_service):
        """Assess overall portfolio health and provide insights"""
        try:
            # Calculate portfolio metrics
            buy_count = sum(1 for rec in recommendations if rec['action'] == 'BUY')
            sell_count = sum(1 for rec in recommendations if rec['action'] == 'SELL')
            hold_count = sum(1 for rec in recommendations if rec['action'] == 'HOLD')
            
            total_stocks = len(recommendations)
            
            # Calculate average confidence
            avg_confidence = np.mean([rec['confidence'] for rec in recommendations]) if recommendations else 0.5
            
            # Risk assessment
            high_risk_count = sum(1 for rec in recommendations if rec['risk_level'] == 'high')
            risk_percentage = (high_risk_count / total_stocks * 100) if total_stocks > 0 else 0
            
            if risk_percentage > 60:
                risk_level = 'High'
            elif risk_percentage > 30:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Portfolio score (1-10)
            score_factors = []
            
            # Diversification factor
            sectors = set()
            for stock in portfolio:
                try:
                    company_info = market_service.get_company_info(stock['symbol'])
                    sector = company_info.get('sector', 'Unknown')
                    sectors.add(sector)
                except:
                    pass
            
            diversification_score = min(10, len(sectors) * 2)  # Max 10 points for 5+ sectors
            score_factors.append(diversification_score)
            
            # Recommendation balance factor
            balance_score = 10 - abs((buy_count - sell_count) / total_stocks * 10) if total_stocks > 0 else 5
            score_factors.append(balance_score)
            
            # Confidence factor
            confidence_score = avg_confidence * 10
            score_factors.append(confidence_score)
            
            overall_score = np.mean(score_factors)
            
            # Generate insights
            insights = []
            
            if diversification_score < 6:
                insights.append("Consider diversifying across more sectors to reduce risk")
            
            if sell_count > buy_count:
                insights.append("Portfolio shows more sell signals - consider rebalancing")
            elif buy_count > sell_count:
                insights.append("Portfolio shows growth opportunities with multiple buy signals")
            
            if risk_percentage > 50:
                insights.append("High concentration of risky assets - consider risk management")
            
            if avg_confidence < 0.6:
                insights.append("Market conditions show uncertainty - consider cautious approach")
            
            if not insights:
                insights.append("Portfolio appears well-balanced with reasonable risk distribution")
            
            return {
                'score': overall_score,
                'risk_level': risk_level,
                'diversification': f"{diversification_score:.1f}/10",
                'insights': insights,
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'hold_signals': hold_count,
                'average_confidence': avg_confidence
            }
            
        except Exception as e:
            return {
                'score': 5.0,
                'risk_level': 'Medium',
                'diversification': 'Unknown',
                'insights': [f'Error assessing portfolio: {str(e)}'],
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'average_confidence': 0.5
            }
    
    def _calculate_diversification_score(self, sector_allocation):
        """Calculate diversification score based on sector allocation"""
        try:
            if not sector_allocation:
                return 0
            
            # Calculate Herfindahl-Hirschman Index (lower is more diversified)
            hhi = sum((percentage/100) ** 2 for percentage in sector_allocation.values())
            
            # Convert to score (0-10, where 10 is perfectly diversified)
            # Perfect diversification (equal weights) would have HHI of 1/n
            # We'll use inverse relationship: lower HHI = higher score
            diversification_score = max(0, min(10, (1 - hhi) * 10))
            
            return diversification_score
            
        except Exception as e:
            return 0
    
    def generate_market_outlook(self, market_service):
        """Generate overall market outlook and recommendations"""
        try:
            # Get market indices data
            indices_data = market_service.get_market_indices()
            
            # Get sector performance
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer Discretionary': 'XLY',
                'Industrials': 'XLI'
            }
            
            sector_performance = market_service.get_sector_performance(sector_etfs)
            
            # Prepare market summary for AI analysis
            market_summary = f"""
            Market Indices Performance (5-day):
            """
            
            for name, data in indices_data.items():
                market_summary += f"{name}: {data['change_percent']:+.2f}%\n"
            
            market_summary += "\nSector Performance (1-month):\n"
            for sector, data in sector_performance.items():
                market_summary += f"{sector}: {data['performance']:+.2f}%\n"
            
            system_prompt = """You are a market analyst providing outlook based on recent market performance.
            
            Analyze market conditions and provide:
            - Overall market sentiment
            - Key trends and patterns
            - Investment themes to consider
            - Risk factors to watch
            
            Respond in JSON format: {
                "market_sentiment": "bullish|bearish|neutral",
                "confidence": float (0 to 1),
                "key_trends": ["trend1", "trend2"],
                "investment_themes": ["theme1", "theme2"],
                "risk_factors": ["risk1", "risk2"],
                "outlook": "detailed market outlook",
                "time_horizon": "short|medium|long"
            }"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=f"Analyze current market conditions:\n{market_summary}")])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json"
                ),
            )
            
            result = json.loads(response.text)
            return result
            
        except Exception as e:
            return {
                'market_sentiment': 'neutral',
                'confidence': 0.5,
                'key_trends': ['Analysis unavailable'],
                'investment_themes': ['Diversified approach recommended'],
                'risk_factors': ['Market uncertainty'],
                'outlook': f'Unable to generate detailed outlook: {str(e)}',
                'time_horizon': 'medium'
            }
