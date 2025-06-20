"""
Init file for the volatility analysis package.
This makes the package importable in Python.
"""

from .volatility_pipeline import *

__all__ = [
    # Data preparation and sentiment analysis
    'initialize_sentiment_model',
    'infer_sentiment',
    'get_titles',
    'calculate_sentiment',
    'enhanced_sentiment_calculation',
    'split_data',
    'prepare_lstm_data',
    'create_sequences',
    'create_validation_split',
    
    # New sentiment caching and batching
    'SentimentCache',
    'calculate_batch_sentiments',
    
    # Feature engineering
    'improve_sentiment_features',
    
    # Model and training
    'ImprovedLSTMVolatility',
    'train_with_early_stopping',
    'evaluate_lstm_model',
    
    # Visualization
    'plot_volatility_news_count',
    'plot_volatility_sentiment',
    'plot_prediction_results',
    
    
    # Main pipeline
    'run_volatility_pipeline'
]
