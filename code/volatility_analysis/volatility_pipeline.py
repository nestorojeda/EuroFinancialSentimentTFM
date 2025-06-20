"""
Volatility Analysis Library for Financial Market Data

This module provides functions and classes for analyzing financial market volatility
in relation to news sentiment. It includes tools for sentiment analysis of financial news,
LSTM-based volatility prediction, and visualization.

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from volatility_plotter import VolatilityPlotter
from sklearn.feature_selection import SelectKBest, f_regression


class SentimentCache:
    """
    Manages caching of sentiment predictions to avoid redundant computations.
    """
    
    def __init__(self, cache_dir: str = "../cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, news_df: pd.DataFrame, date_range: Tuple[str, str], model_name: str) -> str:
        """Generate a unique cache key based on news data and parameters."""
        # Create a hash of the news titles and dates for this date range
        relevant_news = news_df[
            (news_df['date'].dt.date >= pd.to_datetime(date_range[0]).date()) &
            (news_df['date'].dt.date <= pd.to_datetime(date_range[1]).date())
        ]
        
        # Combine titles and dates into a string for hashing
        content = relevant_news['title'].str.cat(
            relevant_news['date'].dt.strftime('%Y-%m-%d'), sep='|'
        ).sum()
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Include model name in the key
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        
        return f"sentiment_cache_{model_hash}_{content_hash}.pkl"
    
    def get_cached_sentiments(self, news_df: pd.DataFrame, date_range: Tuple[str, str], 
                            model_name: str) -> Optional[Dict[str, float]]:
        """Retrieve cached sentiment predictions if available."""
        cache_key = self._get_cache_key(news_df, date_range, model_name)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load sentiment cache: {e}")
                return None
        return None
    
    def save_sentiments(self, sentiments: Dict[str, float], news_df: pd.DataFrame, 
                       date_range: Tuple[str, str], model_name: str):
        """Save sentiment predictions to cache."""
        cache_key = self._get_cache_key(news_df, date_range, model_name)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(sentiments, f)
        except Exception as e:
            print(f"Warning: Failed to save sentiment cache: {e}")


def calculate_batch_sentiments(news_df: pd.DataFrame, tokenizer: Any, model: Any, 
                             cache: SentimentCache, model_name: str,
                             batch_size: int = 32, verbose: bool = True) -> Dict[str, float]:
    """
    Calculate sentiment scores for all news in batches with caching support.
    
    Args:
        news_df: DataFrame containing news data
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        cache: SentimentCache instance
        model_name: Name of the model for cache key generation
        batch_size: Size of batches for processing
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping date strings to sentiment scores
    """
    # Set reproducible seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Determine device
    device = next(model.parameters()).device
    
    # Get date range for caching
    min_date = news_df['date'].min().strftime('%Y-%m-%d')
    max_date = news_df['date'].max().strftime('%Y-%m-%d')
    date_range = (min_date, max_date)
    
    # Try to load from cache first
    cached_sentiments = cache.get_cached_sentiments(news_df, date_range, model_name)
    if cached_sentiments is not None:
        if verbose:
            print(f"Loaded {len(cached_sentiments)} cached sentiment predictions")
        return cached_sentiments
    
    if verbose:
        print("Computing sentiment predictions from scratch...")
    
    # Group news by date
    news_by_date = news_df.groupby(news_df['date'].dt.date)['title'].apply(list).to_dict()
    
    # Prepare all titles for batch processing
    all_titles = []
    title_to_date = {}
    
    for date, titles in news_by_date.items():
        for title in titles:
            all_titles.append(title)
            title_to_date[title] = date.strftime('%Y-%m-%d')
    
    if verbose:
        print(f"Processing {len(all_titles)} news titles in batches of {batch_size}")
    
    # Process in batches
    all_sentiments = []
    all_confidences = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_titles), batch_size):
            batch_titles = all_titles[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_titles, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Get predictions
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Get confidence (max probability) and sentiment for each item in batch
            confidences = torch.max(probs, dim=1).values.cpu().numpy()
            sentiments = probs.argmax(dim=1).cpu().numpy()
            
            # Map sentiment values (0, 1, 2) to (1, 0, -1) for positive, neutral, negative
            mapped_sentiments = np.where(sentiments == 0, 1, np.where(sentiments == 1, 0, -1))
            
            all_sentiments.extend(mapped_sentiments)
            all_confidences.extend(confidences)
    
    # Group by date and calculate weighted averages
    date_sentiments = {}
    
    for title, sentiment, confidence in zip(all_titles, all_sentiments, all_confidences):
        date_str = title_to_date[title]
        
        if date_str not in date_sentiments:
            date_sentiments[date_str] = {'sentiments': [], 'confidences': []}
        
        date_sentiments[date_str]['sentiments'].append(sentiment)
        date_sentiments[date_str]['confidences'].append(confidence)
    
    # Calculate weighted sentiment for each date
    final_sentiments = {}
    for date_str, data in date_sentiments.items():
        if data['confidences']:
            weighted_sentiment = np.average(data['sentiments'], weights=data['confidences'])
            final_sentiments[date_str] = float(weighted_sentiment)
        else:
            final_sentiments[date_str] = 0.0
    
    # Cache the results
    cache.save_sentiments(final_sentiments, news_df, date_range, model_name)
    
    if verbose:
        print(f"Computed and cached sentiment predictions for {len(final_sentiments)} dates")
    
    return final_sentiments

# Run the entire volatility analysis pipeline
def run_volatility_pipeline(news_df: pd.DataFrame, 
                           stock_data: pd.DataFrame,
                           market_name: str,
                           cut_date: str,
                           output_dir: str = "../news",
                           seq_len: int = 10,
                           epochs: int = 50,
                           learning_rate: float = 0.001,
                           verbose: bool = True,
                           patience: int = 15
) -> Dict:
    """
    Run the entire volatility analysis pipeline.
    
    Args:
        news_df: DataFrame containing news data.
        stock_data: DataFrame containing stock market data.
        market_name: Name of the market (e.g., 'FTSE 100', 'IBEX 35').
        cut_date: Date string to split training/testing data (format: 'YYYY-MM-DD').
        output_dir: Directory to save output plots.
        seq_len: Sequence length for LSTM.
        epochs: Number of epochs for LSTM training.
        learning_rate: Learning rate for optimizer.
        use_sentiment: Whether to use sentiment inference for predictions. If False, only uses previous volatility.
        verbose: Whether to print progress information.
    
    Returns:
        Dictionary containing model metrics and other results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the right date format
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_daily = news_df.resample('D', on='date').count().reset_index()
    
    # Ensure stock data is properly formatted
    if 'Date' in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        # Make timezone-naive if needed
        if hasattr(stock_data['Date'].dtype, 'tz') and stock_data['Date'].dt.tz is not None:
            stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
    
    # Merge data
    merged = pd.merge(news_daily, stock_data, left_on='date', right_on='Date', how='inner')
    merged['Volatility_Smooth'] = merged['Volatility'].rolling(window=5, min_periods=1).mean()
    merged.rename(columns={'title': 'count',}, inplace=True)
    
    # Clean up the merged dataframe
    if 'Date' in merged.columns:
        merged.drop(columns=['Date'], inplace=True)    # Add technical indicators to stock data first

    # Re-merge with enhanced stock data to get technical indicators
    merged = pd.merge(news_daily, stock_data, left_on='date', right_on='Date', how='inner')
    merged['Volatility_Smooth'] = merged['Volatility'].rolling(window=5, min_periods=1).mean()
    merged.rename(columns={'title': 'count'}, inplace=True)
    if 'Date' in merged.columns:
        merged.drop(columns=['Date'], inplace=True)
    
    # Split data FIRST to prevent data leakage
    if verbose:
        print(f"Splitting data at {cut_date}...")
    train_raw, test_raw, val_raw = split_data_temporal(merged, cut_date, val_ratio=0.3)
    
    # Now process features separately for train and test to prevent leakage
    # Start with a base set of features that we know will be available
    base_features = ['Volatility_Smooth']
    tech_features = []
    sentiment_features = []

    # Initialize sentiment model and cache
    tokenizer, model = initialize_sentiment_model()
    sentiment_cache = SentimentCache()
    
    if verbose:
        print("Calculating enhanced sentiment scores...")
    
    # Calculate sentiments for all news data once using batched processing
    model_name = "nojedag/xlm-roberta-finetuned-financial-news-sentiment-analysis-european"
    sentiment_predictions = calculate_batch_sentiments(
        news_df, tokenizer, model, sentiment_cache, model_name, verbose=verbose
    )
    
    # Apply sentiment predictions to each split
    def apply_sentiment_to_split(df: pd.DataFrame) -> pd.DataFrame:
        """Apply cached sentiment predictions to a data split."""
        df = df.copy()
        df['sentiment'] = df['date'].apply(
            lambda date: sentiment_predictions.get(date.strftime('%Y-%m-%d'), None)
        )
        return df
    
    train_processed = apply_sentiment_to_split(train_raw)
    test_processed = apply_sentiment_to_split(test_raw)  
    val_processed = apply_sentiment_to_split(val_raw)
    
    # Create sentiment features on training data only and capture stats
    train_processed = improve_sentiment_features(train_processed, train_stats=None)
    
    # Get training statistics for consistent application to test/val
    train_stats = {
        'sentiment_mean': train_processed['sentiment'].mean(),
        'sentiment_std': train_processed['sentiment'].std()
    }
    
    # Apply the same feature engineering to test data using training statistics
    test_processed = apply_sentiment_features_test(test_processed, train_processed)
    val_processed = apply_sentiment_features_test(val_processed, train_processed)
    
    # Feature selection based ONLY on training data
    sentiment_cols = [
        'sentiment_vol',
        'news_sentiment_interaction',
        'sentiment_mean_3d',
        'sentiment_std_3d',
        'sentiment_mean_5d',
        'sentiment_std_5d',
        'sentiment_mean_7d',
        'sentiment_std_7d',
        'sentiment_lag_1',
        'sentiment_lag_2',
        'sentiment_lag_3',
        'sentiment_per_news',
        'sentiment_zscore',
        'sentiment_momentum',
        'sentiment_vol_interaction',
        'sentiment_extreme'
    ]

    selector = SelectKBest(score_func=f_regression, k=min(5, len(sentiment_cols)))        
    X_train_sentiment = train_processed[sentiment_cols].fillna(0)
    y_train_target = train_processed['Volatility_Smooth']

    # Fit selector on training data
    selector.fit(X_train_sentiment, y_train_target)
    selected_features = [sentiment_cols[i] for i in selector.get_support(indices=True)]
    sentiment_features = selected_features
    
    # Reconstruct merged dataframe for plotting (using training data statistics)
    merged_for_plotting = pd.concat([train_processed, test_processed, val_processed], ignore_index=True).sort_values('date')
    
    # Plot sentiment distribution
    if verbose:
        print("Plotting sentiment distribution...")
        plot_sentiment_distribution(merged_for_plotting, market_name, output_dir, show_plot=verbose)
        
    # Combine all selected features
    feature_cols = base_features + tech_features
    
    feature_cols += sentiment_features
        
    # Ensure all columns exist in the dataframes
    for df_name, df in [('train', train_processed), ('test', test_processed), ('val', val_processed)]:
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            if verbose:
                print(f"Warning: Missing columns in {df_name} data: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                df[col] = 0
            
    if verbose:
        print(f"Final feature columns for model: {feature_cols}")
    
    # Plot volatility and news count
    news_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_news_count_plot.png")
    plot_volatility_news_count(merged_for_plotting, market_name, news_plot_path, show_plot=verbose)

    sentiment_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_sentiment_plot.png")
    plot_volatility_sentiment(merged_for_plotting, market_name, sentiment_plot_path, show_plot=verbose)
    
    # Use the processed data splits
    train, test, val = train_processed, test_processed, val_processed
    
    # Prepare LSTM data with validation set
    if verbose:
        print("Preparing data for LSTM model...")
    (X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq, 
     scaler_x, scaler_y, scaler_y_single) = prepare_lstm_data(train, test, val, feature_cols=feature_cols, seq_len=seq_len)
      # Prepare validation data using the training scalers to maintain consistency
    # Train LSTM model with early stopping
    if verbose:
        print(f"Training enhanced LSTM model with early stopping...")
    input_size = X_train_seq.shape[2]
    output_size = y_train_seq.shape[1]
    model = train_with_early_stopping(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                     input_size, output_size, epochs=epochs, 
                                     learning_rate=learning_rate, verbose=verbose, patience=patience)
    
    # Evaluate model
    if verbose:
        print("Evaluating enhanced LSTM model...")
    y_pred_inv, y_test_inv, metrics = evaluate_lstm_model(model, X_test_seq, y_test_seq, scaler_y)
    
    # Plot prediction results
    test_dates = test['date'].iloc[seq_len:seq_len+len(y_test_inv)].values
    pred_plot_path = os.path.join(output_dir, f"{market_name.lower().replace(' ', '_')}_prediction_plot.png")
    plot_prediction_results(test_dates, y_test_inv, y_pred_inv, market_name, pred_plot_path, show_plot=verbose)
    
    return {
        'model': model,
        'metrics': metrics,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test),
        'feature_cols': feature_cols,
        'y_pred': y_pred_inv,
        'y_actual': y_test_inv,
        'test_dates': test_dates
    }


# Initialize sentiment analysis model and tokenizer
def initialize_sentiment_model(model_name: str = "nojedag/xlm-roberta-finetuned-financial-news-sentiment-analysis-european") -> Tuple:
    """
    Initialize the sentiment analysis model and tokenizer.
    
    Args:
        model_name: The name or path of the pre-trained model to use.
    
    Returns:
        Tuple containing the tokenizer and model objects.
    """
    # Set reproducible seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Determine device (GPU if available, CPU otherwise)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    model = model.to(device)
    
    print(f"Sentiment model loaded on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return tokenizer, model

def split_data_temporal(merged_df: pd.DataFrame, cut_date: str, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on a cut date WITHOUT data leakage.
    
    Args:
        merged_df: DataFrame containing merged news and volatility data.
        cut_date: Date string to split the data on (format: 'YYYY-MM-DD').
        val_ratio: Ratio of training data to use for validation.
    
    Returns:
        Tuple of (training_data, testing_data, validation_data)
    """
    # Sort by date to ensure temporal order
    merged_df_sorted = merged_df.sort_values('date').copy()
    
    # Split without any preprocessing to avoid data leakage
    train = merged_df_sorted[merged_df_sorted['date'] < cut_date].copy()
    test = merged_df_sorted[merged_df_sorted['date'] >= cut_date].copy()

    val_size = int(len(train) * val_ratio)
    val = train[-val_size:].copy()  # Take last val_size rows from original train set
    train = train[:-val_size].copy()  # Remove those rows from train set

    return train, test, val

# Legacy function for backwards compatibility
def split_data(merged_df: pd.DataFrame, cut_date: str, val_ratio: float = 0.2 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    LEGACY: Split data into training and testing sets based on a cut date.
    
    WARNING: This function has data leakage issues. Use split_data_temporal instead.
    """
    
    # Fill NaN values in sentiment features with 0 (neutral sentiment)
    # and technical indicators with their median values
    for col in merged_df.columns:
        if col.startswith('sentiment'):
            merged_df[col] = merged_df[col].fillna(0)
        elif col != 'date' and merged_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # For numeric columns other than date, fill with median
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    train = merged_df[merged_df['date'] < cut_date]
    test = merged_df[merged_df['date'] >= cut_date]

    val_size = int(len(train) * val_ratio)
    val = train[-val_size:]  # Take last val_size rows from original train set
    train = train[:-val_size]  # Remove those rows from train set

    return train, test, val


# Create sequences for LSTM
def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding output values for LSTM training.
    
    Args:
        X: Input features array.
        y: Target values array.
        seq_length: Length of each sequence.
    
    Returns:
        Tuple of (input_sequences, output_values)
    """
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

# Evaluate LSTM model
def evaluate_lstm_model(model: nn.Module, X_test_seq: torch.Tensor, y_test_seq: torch.Tensor, 
                       scaler_y: MinMaxScaler) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate an LSTM model and calculate performance metrics.
    
    Args:
        model: Trained LSTM model.
        X_test_seq: Input sequences for testing.
        y_test_seq: Target values for testing.
        scaler_y: Scaler used to normalize target values.
    
    Returns:
        Tuple of (predictions, actual_values, metrics_dict)    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_seq).numpy()
        # Use the original scaler for both predictions and test data
        y_pred_inv = scaler_y.inverse_transform(y_pred)[:, 0:1]  # Extract the volatility column
        y_test_inv = scaler_y.inverse_transform(y_test_seq.numpy())[:, 0:1]  # Extract the volatility column

    # Calculate error metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    print(f"Model Performance Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred_inv, y_test_inv, metrics

# Get titles for a specific date
def get_titles(date: datetime, news_df: pd.DataFrame) -> List[str]:
    """
    Get news titles for a specific date from a news DataFrame.
    
    Args:
        date: The date to get titles for.
        news_df: DataFrame containing news data with 'date' and 'title' columns.
    
    Returns:
        List of news article titles for the given date.
    """
    return news_df[news_df['date'].dt.date == date.date()]['title'].tolist()

def prepare_lstm_data(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame,
                     feature_cols: List[str] = ['Volatility_Smooth', 'sentiment'], 
                     target_col: str = 'Volatility_Smooth', 
                     seq_len: int = 10) -> Tuple:
    """
    Prepare data for LSTM model training and testing WITHOUT data leakage.
    
    Args:
        train: Training data DataFrame.
        test: Testing data DataFrame.
        val: Validation data DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column.
        seq_len: Sequence length for LSTM.
    
    Returns:
        Tuple containing prepared data and scalers.
    """    
    # Verify all required columns exist in all dataframes
    missing_train = [col for col in feature_cols if col not in train.columns]
    missing_test = [col for col in feature_cols if col not in test.columns]
    missing_val = [col for col in feature_cols if col not in val.columns]
    
    if missing_train or missing_test or missing_val:
        print(f"Warning: Missing columns - train: {missing_train}, test: {missing_test}, val: {missing_val}")
        # Add missing columns with zeros
        for col in missing_train:
            train[col] = 0
        for col in missing_test:
            test[col] = 0
        for col in missing_val:
            val[col] = 0
    
    # Handle NaN values WITHOUT data leakage
    train_clean = train.copy()
    test_clean = test.copy()
    val_clean = val.copy()
    
    # Fill NaN values in sentiment features with 0 (neutral sentiment)
    # and technical indicators with their TRAINING SET median values
    for col in feature_cols + [target_col]:
        if col.startswith('sentiment'):
            # Sentiment features: fill with 0 (neutral)
            train_clean[col] = train_clean[col].fillna(0)
            test_clean[col] = test_clean[col].fillna(0)
            val_clean[col] = val_clean[col].fillna(0)
        elif col != 'date' and train_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # For numeric columns: use TRAINING data median to fill ALL datasets
            train_median = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(train_median)
            test_clean[col] = test_clean[col].fillna(train_median)
            val_clean[col] = val_clean[col].fillna(train_median)
    
    # Select features and target
    train_data = train_clean[feature_cols + [target_col]].copy()
    test_data = test_clean[feature_cols + [target_col]].copy()
    val_data = val_clean[feature_cols + [target_col]].copy() if not val_clean.empty else pd.DataFrame(columns=feature_cols + [target_col])
    
    # Normalize features and target - fit scalers ONLY on training data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_y_single = MinMaxScaler()  # For single column output

    # Fit scalers on training data only
    X_train = scaler_x.fit_transform(train_data[feature_cols])
    y_train = scaler_y.fit_transform(train_data[[target_col]])
    scaler_y_single.fit(train_data[[target_col]].values.reshape(-1, 1))

    # Transform test and validation data using training scalers
    X_test = scaler_x.transform(test_data[feature_cols])
    y_test = scaler_y.transform(test_data[[target_col]])

    X_val = scaler_x.transform(val_data[feature_cols])
    y_val = scaler_y.transform(val_data[[target_col]])

    # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    
    # Convert to torch tensors
    X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32)
    X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_seq = torch.tensor(y_val_seq, dtype=torch.float32)
    
    # Input data doesn't need gradients, but ensure they're properly formatted
    X_train_seq.requires_grad_(False)
    y_train_seq.requires_grad_(False)
    X_test_seq.requires_grad_(False)
    y_test_seq.requires_grad_(False)
    X_val_seq.requires_grad_(False)
    y_val_seq.requires_grad_(False)

    return (X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_val_seq, y_val_seq,
            scaler_x, scaler_y, scaler_y_single)

def improve_sentiment_features(merged_df: pd.DataFrame, train_stats: Dict = None) -> pd.DataFrame:
    """Add improved sentiment features with multiple rolling windows, lags, and normalization."""
    merged_df = merged_df.copy()
    
    # Rolling means and stds for multiple windows
    for window in [3, 5, 7]:
        merged_df[f'sentiment_mean_{window}d'] = merged_df['sentiment'].rolling(window=window, min_periods=1).mean()
        merged_df[f'sentiment_std_{window}d'] = merged_df['sentiment'].rolling(window=window, min_periods=1).std()

    # Lagged sentiment features (previous 1, 2, 3 days)
    for lag in [1, 2, 3]:
        merged_df[f'sentiment_lag_{lag}'] = merged_df['sentiment'].shift(lag)

    # Sentiment volatility (5-day rolling std, legacy)
    merged_df['sentiment_vol'] = merged_df['sentiment'].rolling(window=5, min_periods=1).std()

    # News volume impact
    merged_df['news_sentiment_interaction'] = merged_df['count'] * merged_df['sentiment']

    # Sentiment normalized by news count (avoid division by zero)
    merged_df['sentiment_per_news'] = merged_df['sentiment'] / merged_df['count'].replace(0, np.nan)
    merged_df['sentiment_per_news'] = merged_df['sentiment_per_news'].fillna(0)

    # Sentiment momentum (change from previous day)
    merged_df['sentiment_momentum'] = merged_df['sentiment'].diff()
    
    # Sentiment-volatility interaction (current sentiment with previous volatility)
    merged_df['sentiment_vol_interaction'] = merged_df['sentiment'] * merged_df['Volatility_Smooth'].shift(1)
    
    # Use provided stats or calculate from current data
    if train_stats is None:
        sentiment_mean = merged_df['sentiment'].mean()
        sentiment_std = merged_df['sentiment'].std()
    else:
        sentiment_mean = train_stats['sentiment_mean']
        sentiment_std = train_stats['sentiment_std']
    
    # Extreme sentiment indicator (beyond 1 std)
    merged_df['sentiment_extreme'] = ((merged_df['sentiment'] - sentiment_mean).abs() > sentiment_std).astype(int)
    
    # Z-score normalization for sentiment
    merged_df['sentiment_zscore'] = (merged_df['sentiment'] - sentiment_mean) / (sentiment_std + 1e-8)

    # Fill NaNs in all new features with 0
    sentiment_cols = [col for col in merged_df.columns if col.startswith('sentiment') or 'news_sentiment_interaction' in col]
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

    return merged_df

def apply_sentiment_features_test(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply sentiment features to test data using training data statistics to prevent data leakage.
    """
    test_df = test_df.copy()
    
    # Rolling means and stds for multiple windows (calculate on test data itself)
    for window in [3, 5, 7]:
        test_df[f'sentiment_mean_{window}d'] = test_df['sentiment'].rolling(window=window, min_periods=1).mean()
        test_df[f'sentiment_std_{window}d'] = test_df['sentiment'].rolling(window=window, min_periods=1).std()

    # Lagged sentiment features (previous 1, 2, 3 days)
    for lag in [1, 2, 3]:
        test_df[f'sentiment_lag_{lag}'] = test_df['sentiment'].shift(lag)

    # Sentiment volatility (5-day rolling std, legacy)
    test_df['sentiment_vol'] = test_df['sentiment'].rolling(window=5, min_periods=1).std()

    # News volume impact
    test_df['news_sentiment_interaction'] = test_df['count'] * test_df['sentiment']

    # Sentiment normalized by news count (avoid division by zero)
    test_df['sentiment_per_news'] = test_df['sentiment'] / test_df['count'].replace(0, np.nan)
    test_df['sentiment_per_news'] = test_df['sentiment_per_news'].fillna(0)

    # Sentiment momentum (change from previous day)
    test_df['sentiment_momentum'] = test_df['sentiment'].diff()
    
    # Sentiment-volatility interaction
    test_df['sentiment_vol_interaction'] = test_df['sentiment'] * test_df['Volatility_Smooth'].shift(1)
    
    # Use TRAINING data statistics for extreme sentiment and z-score
    if 'sentiment' in train_df.columns:
        sentiment_mean = train_df['sentiment'].mean()
        sentiment_std = train_df['sentiment'].std()
        test_df['sentiment_extreme'] = ((test_df['sentiment'] - sentiment_mean).abs() > sentiment_std).astype(int)
        test_df['sentiment_zscore'] = (test_df['sentiment'] - sentiment_mean) / (sentiment_std + 1e-8)
    else:
        test_df['sentiment_extreme'] = 0
        test_df['sentiment_zscore'] = 0

    # Fill NaNs in all new features with 0
    sentiment_cols = [col for col in test_df.columns if col.startswith('sentiment') or 'news_sentiment_interaction' in col]
    test_df[sentiment_cols] = test_df[sentiment_cols].fillna(0)

    return test_df

def sentiment_calculation(date: datetime, news_df: pd.DataFrame, tokenizer: Any, model: Any) -> Optional[float]:
    """
    Calculate sentiment for a given date using a pre-trained model.
    
    DEPRECATED: This function is kept for backward compatibility.
    Use calculate_batch_sentiments() for better performance and caching.
    
    Args:
        date: The date to calculate sentiment for.
        news_df: DataFrame containing news data with 'date' and 'title' columns.
        tokenizer: Tokenizer for the sentiment model.
        model: Pre-trained sentiment analysis model.
    
    Returns:
        Sentiment score for the given date, or None if no titles are found.
    """
    titles = get_titles(date, news_df)
    if not titles:
        return None
    
    # Determine device (same as model)
    device = next(model.parameters()).device
    
    inputs = tokenizer(titles, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        
        # Average sentiment across all titles
        sentiment = probs.mean(dim=0).argmax().item()
        
        # Map sentiment values (0, 1, 2) to (1, 0, -1) for positive, neutral, negative
        if sentiment == 0:
            sentiment = 1
        elif sentiment == 2:
            sentiment = -1
            
    return sentiment

def enhanced_sentiment_calculation(date: datetime, news_df: pd.DataFrame, tokenizer: Any, model: Any) -> Optional[float]:
    """
    Enhanced sentiment with confidence weighting.
    
    DEPRECATED: This function is kept for backward compatibility.
    Use calculate_batch_sentiments() for better performance and caching.
    """
    titles = get_titles(date, news_df)
    if not titles:
        return None
    
    # Determine device (same as model)
    device = next(model.parameters()).device
    
    sentiments = []
    confidences = []
    
    for title in titles:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # Get confidence (max probability)
            confidence = torch.max(probs).item()
            sentiment = probs.argmax(dim=1).item()
            
            # Map sentiment values (0, 1, 2) to (1, 0, -1) for positive, neutral, negative
            if sentiment == 0:
                sentiment = 1
            elif sentiment == 2:
                sentiment = -1
                
            sentiments.append(sentiment)
            confidences.append(confidence)
    
    # Weighted average by confidence
    if confidences:
        weighted_sentiment = np.average(sentiments, weights=confidences)
        return weighted_sentiment
    
    return np.mean(sentiments)


def train_with_early_stopping(X_train_seq: torch.Tensor, y_train_seq: torch.Tensor, 
                              X_val_seq: torch.Tensor, y_val_seq: torch.Tensor,
                              input_size: int, output_size: int, 
                              epochs: int = 100, patience: int = 15, 
                              learning_rate: float = 0.001, verbose: bool = True) -> nn.Module:
    """
    Train model with early stopping and learning rate scheduling.
    
    Args:
        X_train_seq: Input sequences for training.
        y_train_seq: Target values for training.
        X_val_seq: Input sequences for validation.
        y_val_seq: Target values for validation.
        input_size: Number of input features.
        output_size: Number of output values.
        epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before stopping.
        learning_rate: Learning rate for optimizer.
        verbose: Whether to print training progress.
        model_type: Type of model to use ('simple' or 'improved').
      Returns:
        Trained LSTM model.    """
    
    torch.set_grad_enabled(True)
    
    # Clear any existing computation graphs to prevent issues between runs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Ensure input tensors are properly formatted (clone to avoid sharing references)
    # Note: We don't detach here as we need gradients to flow through the computation graph    X_train_seq = X_train_seq.clone()
    y_train_seq = y_train_seq.clone()
    X_val_seq = X_val_seq.clone()
    y_val_seq = y_val_seq.clone()
    
    from models.SimpleLSTM import LSTMVolatility
    model = LSTMVolatility(input_size, output_size=output_size, hidden_size=32, num_layers=1, seed=42)

    # Explicitly ensure model parameters require gradients and model is in training mode
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    
    # Verify gradient setup
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Model initialized with {param_count} trainable parameters")
    
    # Reset PyTorch autograd state to prevent issues between runs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Increased from 1e-5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for i in range(0, len(X_train_seq), 32):  # Batch processing
            batch_x = X_train_seq[i:i+32]
            batch_y = y_train_seq[i:i+32]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
          # Validation
        model.eval()
        with torch.no_grad():
            if len(X_val_seq) > 0:
                val_output = model(X_val_seq)
                val_loss = criterion(val_output, y_val_seq).item()
            else:
                # If no validation data, use training loss as validation loss
                val_loss = train_loss / max(1, len(X_train_seq) // 32)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if verbose and ((epoch+1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(X_train_seq)*32:.5f}, Val Loss: {val_loss:.5f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def plot_sentiment_distribution(merged_df: pd.DataFrame, market_name: str, 
                               save_path: Optional[str] = None, show_plot: bool = True) -> None:
    """
    Plot sentiment distribution with gradient colors.
    
    Args:
        merged_df: DataFrame containing sentiment data.
        market_name: Name of the market for plot title.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    plotter = VolatilityPlotter()
    plotter.plot_sentiment_distribution(merged_df, market_name, save_path, show_plot)

# Plot volatility and news count
def plot_volatility_news_count(merged_df: pd.DataFrame, 
                              market_name: str = 'Market',
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Plot volatility and news article counts.
    
    Args:
        merged_df: DataFrame with merged volatility and news data.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    plotter = VolatilityPlotter()
    plotter.plot_volatility_news_count(merged_df, market_name, save_path, show_plot)


# Plot volatility and sentiment
def plot_volatility_sentiment(merged_df: pd.DataFrame, 
                             market_name: str = 'Market',
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> None:
    """
    Plot volatility and news sentiment.
    
    Args:
        merged_df: DataFrame with merged volatility and sentiment data.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    plotter = VolatilityPlotter()
    plotter.plot_volatility_sentiment(merged_df, market_name, save_path, show_plot)



def plot_prediction_results(test_dates: np.ndarray, y_test_inv: np.ndarray, y_pred_inv: np.ndarray,
                           market_name: str = 'Market',
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    Plot predicted vs actual volatility.
    
    Args:
        test_dates: Array of dates for the x-axis.
        y_test_inv: Array of actual volatility values.
        y_pred_inv: Array of predicted volatility values.
        market_name: Name of the market for plot titles and labels.
        save_path: Path to save the plot image, or None to skip saving.
        show_plot: Whether to display the plot.
    """
    plotter = VolatilityPlotter()
    plotter.plot_prediction_results(test_dates, y_test_inv, y_pred_inv, market_name, save_path, show_plot)