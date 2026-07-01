# stock-price-prediction-deep-learning
Deep learning and big data framework for stock price forecasting using LSTM, GRU, BiLSTM and BiGRU
# Stock Price Prediction Using Deep Learning & Big Data (LSTM, GRU, BiLSTM, BiGRU)

## Overview

This project proposes an end-to-end framework for stock price forecasting that integrates 
distributed big data preprocessing with deep learning-based predictive modeling. Using Apple 
Inc. (AAPL) daily stock data from Yahoo Finance (2021–2024), four recurrent neural network 
architectures are compared to identify the most accurate model for financial time series 
prediction. This work was written up as an IEEE-format term paper.

## Objectives

- Build a scalable preprocessing pipeline using Hadoop (HDFS) and Apache Spark (PySpark)
- Implement and compare four RNN architectures: LSTM, GRU, BiLSTM, and BiGRU
- Evaluate model performance using RMSE, MAE, and R² metrics
- Identify the best-performing architecture for financial time series forecasting
- Demonstrate the benefit of bidirectional models over unidirectional counterparts

## Tools & Technologies

- Python (Pandas, NumPy, PyTorch)
- Hadoop (HDFS) and Apache Spark (PySpark) for distributed preprocessing
- yfinance for data acquisition from Yahoo Finance
- Scikit-learn (MinMaxScaler, evaluation metrics)
- Matplotlib, Seaborn for visualisation

## Methodology

- Data stored in HDFS and preprocessed using PySpark (cleaning, EDA, stationarity testing via ADF test)
- Sliding window sequence generation with 30-day time step
- 80/10/20 train/validation/test split with MinMaxScaler normalisation
- Hyperparameter tuning across hidden size, dropout rate, and number of layers
- Models trained for 30 epochs with Adam optimizer and MSE loss function

## Key Findings
- BiGRU achieved the best overall performance (Test RMSE = 5.55, R² = 0.914)
- Bidirectional architectures (BiLSTM, BiGRU) consistently outperformed 
  unidirectional models (LSTM, GRU) across all metrics
- All models showed good generalisation with stable convergence after 6–8 epochs
