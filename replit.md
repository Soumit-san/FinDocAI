# FinDocAI - AI-Powered Financial Document Analysis

## Overview

FinDocAI is a comprehensive financial analysis platform that combines AI-powered document processing with real-time market data analysis to provide investment insights and recommendations. The application processes financial documents (PDFs, Word docs, text files), performs sentiment analysis, integrates live market data from Yahoo Finance, and generates AI-driven investment strategies. Built with Streamlit for the frontend and leveraging OpenAI's GPT-4o model for advanced financial analysis, the platform serves as an end-to-end solution for financial professionals and investors seeking data-driven investment decisions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-tab navigation
- **Layout**: Wide layout configuration with expandable sidebar for navigation
- **Components**: Four main sections - Document Upload & Q&A, Market Analysis, Investment Strategy, and Portfolio Dashboard
- **Caching**: Uses Streamlit's `@st.cache_resource` decorator for service initialization to optimize performance

### Backend Architecture
- **Modular Design**: Service-oriented architecture with separate modules for distinct functionalities
- **Core Services**:
  - `FinancialAnalyzer`: Handles AI-powered financial analysis and sentiment processing
  - `MarketDataService`: Manages real-time market data retrieval and company information
  - `DocumentParser`: Processes various document formats (PDF, DOCX, TXT)
  - `InvestmentStrategy`: Generates AI-driven investment recommendations

### AI and Machine Learning Integration
- **Primary AI Model**: OpenAI GPT-4o for financial analysis, sentiment analysis, and investment recommendations
- **Analysis Capabilities**: 
  - Document sentiment analysis with confidence scoring
  - Individual stock analysis and portfolio assessment
  - AI-powered investment recommendations with risk assessment

### Document Processing Pipeline
- **Multi-format Support**: Handles PDF (PyPDF2), Word documents (docx), and text files
- **Text Extraction**: Robust parsing with error handling and content validation
- **Integration**: Seamless connection between document content and AI analysis services

### Data Sources and Market Integration
- **Market Data Provider**: Yahoo Finance API (yfinance) for stock prices, company information, and financial metrics
- **Data Processing**: Pandas for data manipulation and NumPy for numerical computations
- **Visualization**: Plotly Express and Plotly Graph Objects for interactive financial charts and dashboards

### Investment Strategy Engine
- **Portfolio Analysis**: Comprehensive portfolio health assessment with scoring system
- **Risk Assessment**: Multi-level risk evaluation (low, medium, high)
- **Recommendation System**: AI-driven buy/sell recommendations with detailed reasoning
- **Performance Tracking**: Historical analysis and trend prediction capabilities

## External Dependencies

### AI and Language Models
- **OpenAI API**: GPT-4o model for financial analysis, sentiment analysis, and investment strategy generation
- **API Key Management**: Environment variable-based authentication for secure API access

### Market Data Services
- **Yahoo Finance (yfinance)**: Primary source for real-time stock data, company information, financial metrics, and historical market data
- **Data Coverage**: Stock prices, market cap, financial ratios, earnings data, sector information, and trading volumes

### Document Processing Libraries
- **PyPDF2**: PDF document parsing and text extraction
- **python-docx**: Microsoft Word document processing
- **Built-in I/O**: Text file processing and BytesIO for file handling

### Web Framework and Visualization
- **Streamlit**: Complete web application framework with built-in components, caching, and state management
- **Plotly**: Interactive charting library (Express and Graph Objects) for financial data visualization
- **Pandas**: Data manipulation and analysis framework
- **NumPy**: Numerical computing library for financial calculations

### Development and Utility Libraries
- **JSON**: Data serialization for AI model responses and configuration management
- **DateTime**: Time-based analysis and historical data processing
- **OS**: Environment variable management for API keys and configuration