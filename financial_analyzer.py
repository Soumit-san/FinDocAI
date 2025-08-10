import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class FinancialAnalyzer:
    def __init__(self):
        # Using Google Gemini API instead of OpenAI
        if genai is None:
            raise ImportError("google-generativeai package not properly installed")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def analyze_sentiment(self, document_text):
        """Analyze sentiment of financial documents"""
        try:
            system_prompt = """You are a financial sentiment analysis expert. Analyze the sentiment of financial documents and classify as positive, negative, or neutral. 
            Consider factors like:
            - Revenue growth/decline
            - Profit margins
            - Market outlook statements
            - Risk factors mentioned
            - Management confidence
            
            Provide response in JSON format: {
                "sentiment": "positive|negative|neutral",
                "confidence": float between 0 and 1,
                "score": float between -1 and 1,
                "key_phrases": ["phrase1", "phrase2"],
                "reasoning": "explanation of analysis"
            }"""

            prompt = f"{system_prompt}\n\nAnalyze the sentiment of this financial document:\n\n{document_text[:4000]}"
            response = self.model.generate_content(prompt)
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        response_text = response_text[json_start:json_end].strip()
                
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, analyze the text response
                text_lower = response_text.lower()
                if "positive" in text_lower:
                    sentiment = "positive"
                    score = 0.5
                elif "negative" in text_lower:
                    sentiment = "negative"
                    score = -0.5
                else:
                    sentiment = "neutral"
                    score = 0.0
                
                return {
                    "sentiment": sentiment,
                    "confidence": 0.6,
                    "score": score,
                    "key_phrases": [],
                    "reasoning": response_text
                }
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "key_phrases": [],
                    "reasoning": "AI service at capacity. Please try again later or check your API quota."
                }
            else:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "score": 0.0,
                    "key_phrases": [],
                    "reasoning": f"Error in sentiment analysis: {str(e)}"
                }
    
    def detect_anomalies(self, document_text):
        """Detect anomalies in financial documents"""
        try:
            system_prompt = """You are a financial anomaly detection expert. Analyze financial documents for unusual patterns, inconsistencies, or red flags.
            
            Look for:
            - Unusual changes in financial metrics
            - Inconsistent data or calculations
            - Concerning risk factors
            - Irregular accounting practices
            - Significant one-time charges
            - Management changes or governance issues
            
            Provide response in JSON format: {
                "anomalies_found": boolean,
                "anomalies": [
                    {
                        "type": "category of anomaly",
                        "description": "detailed description",
                        "severity": integer 1-10,
                        "recommendation": "what to investigate further"
                    }
                ],
                "overall_risk": "low|medium|high"
            }"""

            prompt = f"{system_prompt}\n\nAnalyze this financial document for anomalies:\n\n{document_text[:4000]}"
            response = self.model.generate_content(prompt)
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        response_text = response_text[json_start:json_end].strip()
                
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create basic response
                return {
                    "anomalies_found": False,
                    "anomalies": [],
                    "overall_risk": "unknown",
                    "analysis": response_text
                }
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return {
                    "anomalies_found": False,
                    "anomalies": [],
                    "overall_risk": "unknown",
                    "error": "AI service at capacity. Please try again later or check your API quota."
                }
            else:
                return {
                    "anomalies_found": False,
                    "anomalies": [],
                    "overall_risk": "unknown",
                    "error": str(e)
                }
    
    def answer_question(self, document_text, question):
        """Answer questions about financial documents using Q&A"""
        try:
            system_prompt = """You are a financial document analysis expert. Answer questions about financial documents with precision and cite relevant information.
            
            When answering:
            - Be specific and quantitative when possible
            - Cite relevant numbers, percentages, or metrics
            - Explain the context and significance
            - If information is not available, state that clearly
            - Provide confidence level in your answer
            
            Provide response in JSON format: {
                "answer": "detailed answer to the question",
                "confidence": float between 0 and 1,
                "key_excerpts": ["relevant text excerpts from document"],
                "metrics_mentioned": ["specific numbers or percentages found"],
                "data_availability": "complete|partial|insufficient"
            }"""

            prompt = f"{system_prompt}\n\nDocument:\n{document_text[:4000]}\n\nQuestion: {question}"
            response = self.model.generate_content(prompt)
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON block in response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end != -1:
                        response_text = response_text[json_start:json_end].strip()
                
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                return {
                    "answer": response_text,
                    "confidence": 0.7,
                    "key_excerpts": [],
                    "metrics_mentioned": [],
                    "data_availability": "partial"
                }
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                return {
                    "answer": "The AI service is currently at capacity. Please try again later or check your API quota.",
                    "confidence": 0.0,
                    "key_excerpts": [],
                    "metrics_mentioned": [],
                    "data_availability": "service_unavailable"
                }
            else:
                return {
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": 0.0,
                    "key_excerpts": [],
                    "metrics_mentioned": [],
                    "data_availability": "error"
                }
    
    def technical_analysis(self, stock_data):
        """Perform technical analysis on stock data"""
        try:
            # Calculate moving averages
            ma_20 = stock_data['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = stock_data['Close'].rolling(window=50).mean().iloc[-1] if len(stock_data) >= 50 else None
            ma_200 = stock_data['Close'].rolling(window=200).mean().iloc[-1] if len(stock_data) >= 200 else None
            
            # Calculate RSI
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Calculate volatility
            returns = stock_data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Determine trend
            current_price = stock_data['Close'].iloc[-1]
            if ma_20 and current_price > ma_20:
                if ma_50 and ma_20 > ma_50:
                    trend = "Strong Uptrend"
                else:
                    trend = "Uptrend"
            elif ma_20 and current_price < ma_20:
                if ma_50 and ma_20 < ma_50:
                    trend = "Strong Downtrend"
                else:
                    trend = "Downtrend"
            else:
                trend = "Sideways"
            
            moving_averages = {"20-Day MA": ma_20}
            if ma_50:
                moving_averages["50-Day MA"] = ma_50
            if ma_200:
                moving_averages["200-Day MA"] = ma_200
            
            return {
                "moving_averages": moving_averages,
                "rsi": rsi,
                "volatility": volatility,
                "trend": trend
            }
            
        except Exception as e:
            return {
                "moving_averages": {},
                "rsi": 50,
                "volatility": 0,
                "trend": "Unknown",
                "error": str(e)
            }
    
    def generate_forecast(self, stock_data, symbol):
        """Generate AI-powered price forecast"""
        try:
            # Prepare recent data for analysis
            recent_data = stock_data.tail(30)  # Last 30 days
            current_price = recent_data['Close'].iloc[-1]
            
            # Calculate basic statistics
            avg_return = recent_data['Close'].pct_change().mean()
            volatility = recent_data['Close'].pct_change().std()
            volume_trend = recent_data['Volume'].pct_change().mean()
            
            # Create technical summary for AI analysis
            tech_summary = f"""
            Current Price: ${current_price:.2f}
            Average Daily Return: {avg_return:.4f}
            Volatility: {volatility:.4f}
            Volume Trend: {volume_trend:.4f}
            Recent High: ${recent_data['High'].max():.2f}
            Recent Low: ${recent_data['Low'].min():.2f}
            """
            
            system_prompt = """You are a financial forecasting expert. Based on recent stock performance data, generate a 30-day price forecast.
            
            Consider:
            - Recent price trends and momentum
            - Volatility patterns
            - Trading volume changes
            - Market conditions
            
            Provide a realistic forecast with JSON format: {
                "target_price": float (price in 30 days),
                "expected_return": float (percentage),
                "confidence": float (0 to 1),
                "risk_level": "low|medium|high",
                "key_factors": ["factor1", "factor2"],
                "price_range": {"low": float, "high": float}
            }"""

            prompt = f"{system_prompt}\n\nGenerate 30-day forecast for {symbol} based on:\n{tech_summary}"
            response = self.model.generate_content(prompt)
            
            result = json.loads(response.text)
            
            # Generate forecast dates and prices
            forecast_dates = pd.date_range(
                start=recent_data.index[-1] + timedelta(days=1),
                periods=30,
                freq='D'
            )
            
            # Simple linear interpolation for visualization
            start_price = current_price
            end_price = result['target_price']
            forecast_prices = np.linspace(start_price, end_price, 30)
            
            # Add some randomness based on volatility
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, volatility * start_price, 30)
            forecast_prices = forecast_prices + noise
            
            result.update({
                "dates": forecast_dates.tolist(),
                "prices": forecast_prices.tolist()
            })
            
            return result
            
        except Exception as e:
            # Fallback forecast - define current_price if not available
            try:
                current_price = stock_data['Close'].iloc[-1]
            except:
                current_price = 100.0  # Fallback price
                
            return {
                "target_price": current_price * 1.05,  # 5% increase
                "expected_return": 0.05,
                "confidence": 0.5,
                "risk_level": "medium",
                "key_factors": ["Error in detailed analysis"],
                "price_range": {"low": current_price * 0.95, "high": current_price * 1.15},
                "dates": pd.date_range(start=datetime.now() + timedelta(days=1), periods=30, freq='D').tolist(),
                "prices": [current_price * (1 + 0.05 * i/30) for i in range(30)],
                "error": str(e)
            }
