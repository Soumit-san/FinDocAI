import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class MarketDataService:
    def __init__(self):
        pass
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol):
        """Get company information and key metrics"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extract key information
            company_data = {
                'symbol': symbol,
                'longName': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': self._format_large_number(info.get('marketCap')),
                'forwardPE': info.get('forwardPE'),
                'trailingPE': info.get('trailingPE'),
                'priceToBook': info.get('priceToBook'),
                'debtToEquity': info.get('debtToEquity'),
                'returnOnEquity': info.get('returnOnEquity'),
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
                'currentPrice': info.get('currentPrice'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'averageVolume': info.get('averageVolume'),
                'businessSummary': info.get('businessSummary', 'No summary available')
            }
            
            return company_data
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_stocks(self, symbols, period="1mo"):
        """Fetch data for multiple stocks"""
        stock_data = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period)
                if not data.empty:
                    stock_data[symbol] = data
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
                continue
        
        return stock_data
    
    def get_sector_performance(self, sector_etfs):
        """Get performance data for sector ETFs"""
        sector_data = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = self.get_stock_data(etf, "1mo")
                if not data.empty:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    performance = ((end_price - start_price) / start_price) * 100
                    
                    sector_data[sector] = {
                        'etf': etf,
                        'performance': performance,
                        'current_price': end_price,
                        'data': data
                    }
            except Exception as e:
                print(f"Error fetching sector data for {sector}: {str(e)}")
                continue
        
        return sector_data
    
    def get_market_indices(self):
        """Get data for major market indices"""
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        index_data = {}
        
        for name, symbol in indices.items():
            try:
                data = self.get_stock_data(symbol, "5d")
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[0]
                    change = ((current - previous) / previous) * 100
                    
                    index_data[name] = {
                        'symbol': symbol,
                        'current_value': current,
                        'change_percent': change,
                        'data': data
                    }
            except Exception as e:
                print(f"Error fetching index data for {name}: {str(e)}")
                continue
        
        return index_data
    
    def calculate_correlation(self, symbols, period="1y"):
        """Calculate correlation matrix between stocks"""
        try:
            stock_data = {}
            
            for symbol in symbols:
                data = self.get_stock_data(symbol, period)
                if not data.empty:
                    stock_data[symbol] = data['Close']
            
            if len(stock_data) < 2:
                return pd.DataFrame()
            
            # Create DataFrame with closing prices
            df = pd.DataFrame(stock_data)
            
            # Calculate daily returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            print(f"Error calculating correlation: {str(e)}")
            return pd.DataFrame()
    
    def get_earnings_calendar(self, symbol):
        """Get earnings calendar for a stock"""
        try:
            stock = yf.Ticker(symbol)
            calendar = stock.calendar
            
            if calendar is not None and isinstance(calendar, pd.DataFrame) and not calendar.empty:
                return calendar
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching earnings calendar for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _format_large_number(self, number):
        """Format large numbers (market cap, etc.) for display"""
        if number is None:
            return "N/A"
        
        try:
            number = float(number)
            if number >= 1e12:
                return f"${number/1e12:.2f}T"
            elif number >= 1e9:
                return f"${number/1e9:.2f}B"
            elif number >= 1e6:
                return f"${number/1e6:.2f}M"
            else:
                return f"${number:,.2f}"
        except:
            return "N/A"
    
    def get_financial_ratios(self, symbol):
        """Get key financial ratios for analysis"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            ratios = {
                'P/E Ratio': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'P/B Ratio': info.get('priceToBook'),
                'P/S Ratio': info.get('priceToSalesTrailing12Months'),
                'Debt/Equity': info.get('debtToEquity'),
                'ROE': info.get('returnOnEquity'),
                'ROA': info.get('returnOnAssets'),
                'Current Ratio': info.get('currentRatio'),
                'Quick Ratio': info.get('quickRatio'),
                'Profit Margin': info.get('profitMargins'),
                'Operating Margin': info.get('operatingMargins'),
                'Gross Margin': info.get('grossMargins')
            }
            
            # Filter out None values
            filtered_ratios = {k: v for k, v in ratios.items() if v is not None}
            
            return filtered_ratios
            
        except Exception as e:
            print(f"Error fetching financial ratios for {symbol}: {str(e)}")
            return {}
