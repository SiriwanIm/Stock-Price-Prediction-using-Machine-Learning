# Stock Price Prediction using Machine Learning
This project applies machine learning techniques to predict the price movements of three Thai stocks (ERW, SPRC, and TISCO) in 15-minute timeframes. The model classifies whether stock prices are likely to rise (buy) or fall (sell) based on historical data and technical indicators.

## Solution Architecture
The project develops three separate classification models to predict binary outcomes:
- **Buy signal (1)**: Price is predicted to rise
- **Sell signal (-1)**: Price is predicted to fall

### Classification Algorithms Evaluated
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression

### Features/Predictors
All models use the same feature set, selected based on domain expertise in stock trading:

#### Historical Stock Data (15-minute timeframe)
- Open Price
- Close Price
- High Price
- Low Price
- Volume

#### Technical Indicators
**Trend Following Indicators:**
- Simple Moving Average (SMA)
- Moving Average Convergence Divergence (MACD)
- Parabolic SAR (Stop and Reverse)

**Momentum Indicators:**
- Relative Strength Index (RSI)
- Stochastic Oscillator (%K and %D)

**Volatility Indicators:**
- Standard deviation of returns

**Market Index Data:**
- SET50 Index (for TISCO)
- SET100 Index (for ERW and SPRC)

# Methodology

### Data Collection
- Historical stock price data retrieved using the `yfinance` library
- Primary data included Open, Close, Low, High prices and Volume

### Feature Engineering
1. **SMA**: Calculated using 10-day window with Pandas rolling mean
2. **RSI**: Implemented using `talib.RSI` with 14-period lookback
3. **Stochastic Oscillator**:
   - Used 14 periods to generate the fast signal (%K)
   - Implemented with `DataFrame.rolling()` to process previous 14 values
   - Applied `max()` and `min()` functions to determine price ranges for each period
   - Calculated %K as a percentage of where current price is relative to the high-low range
   - Computed %D as a 3-period simple moving average of %K.
   - After we got %D and %K, they were added as the features.
   - Incorporated the difference between %K and %D as a feature.
 4. **Volatility**: Computed as standard deviation of 5-period returns
5. **MACD**: Used 12-period and 26-period EMAs with 9-period signal line
6. **Parabolic SAR**: Implemented using `talib.SAR`
7. **Stochastic Correlation**: Custom feature showing correlation between close price and stochastic oscillator

### Data Preparation
- Handled infinity values in Stochastic Correlation (capped at -1 and 1)
- Imputed missing values using appropriate strategies (mean imputation for normally distributed features)
- Addressed class imbalance through weighted classes during model training

### Model Training
- Split dataset (80% training, 20% testing) with stratified sampling
- Standardized features using `StandardScaler`
- Performed hyperparameter tuning through grid search with 10-fold cross-validation
- Key hyperparameters tuned:
  - Solver algorithms
  - Regularization (penalty)
  - C parameter (inverse regularization strength)

### Model Evaluation
- Used ROC-AUC score as the primary performance metric
- Compared performance across all algorithms

## Results
After evaluation, Logistic Regression was selected as the best algorithm for all three stocks:

| Stock | Model Algorithm | ROC-AUC Score |
|-------|----------------|--------------|
| ERW   | Logistic Regression | 0.757 |
| SPRC  | Logistic Regression | 0.747 |
| TISCO | Logistic Regression | 0.836 |



  
