# Time Series Analysis for Supply Chain Forecasting

## Table of Contents
1. [Introduction to Time Series Data](#introduction-to-time-series-data)
2. [Components of Time Series](#components-of-time-series)
3. [Time Series in Supply Chain Management](#time-series-in-supply-chain-management)
4. [Forecasting Techniques for Supply Chain](#forecasting-techniques-for-supply-chain)
5. [Deseasonalization Methods](#deseasonalization-methods)
6. [Practical Implementation](#practical-implementation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Advanced Topics](#advanced-topics)
9. [Resources for Further Learning](#resources-for-further-learning)

## Introduction to Time Series Data

### What is Time Series Data?
Time series data is a sequence of data points collected or recorded at specific time intervals. Unlike cross-sectional data, time series data maintains the temporal ordering of observations, making it essential for analyzing trends over time.

### Key Characteristics of Time Series Data
- **Temporal Dependency**: Observations close in time tend to be related
- **Seasonality**: Regular patterns that repeat at fixed intervals
- **Trend**: Long-term movement in the data
- **Cyclical Patterns**: Oscillations that do not have fixed periods
- **Irregular Fluctuations**: Random variations that cannot be predicted

### Common Examples in Supply Chain
- Daily sales volumes
- Monthly inventory levels
- Weekly production outputs
- Quarterly demand forecasts
- Shipping and logistics timestamps

### Importance of Time Series Analysis
Time series analysis provides insights into how variables change over time, enabling businesses to:
- Understand historical patterns
- Identify anomalies
- Make informed predictions about future values
- Optimize operations based on temporal patterns

## Components of Time Series

### Trend Component
The trend represents the long-term progression of the series. It indicates whether the time series tends to increase, decrease, or remain stable over extended periods.

**Types of Trends:**
- Linear trends
- Polynomial trends
- Exponential trends
- Logarithmic trends

### Seasonal Component
Seasonality refers to fluctuations that occur at regular intervals due to seasonal factors such as:
- Time of year (quarterly, monthly, etc.)
- Day of week
- Hour of day

These patterns repeat with predictable frequency and amplitude.

### Cyclical Component
Cycles are fluctuations that do not occur at fixed intervals. They often relate to business or economic cycles and typically last longer than seasonal patterns, often spanning multiple years.

### Irregular Component (Residuals)
After accounting for trend, seasonality, and cyclical patterns, the remaining variations are considered random or irregular components. These unpredictable fluctuations can result from:
- Random events
- Measurement errors
- Short-term unexpected factors

### Additive vs. Multiplicative Decomposition

**Additive Model:**
```
Y(t) = Trend(t) + Seasonality(t) + Cycle(t) + Irregular(t)
```
Used when seasonal variations are relatively constant over time.

**Multiplicative Model:**
```
Y(t) = Trend(t) × Seasonality(t) × Cycle(t) × Irregular(t)
```
Used when seasonal variations increase or decrease proportionally with the level of the series.

## Time Series in Supply Chain Management

### Applications in Supply Chain
- **Demand Forecasting**: Predicting future customer demand
- **Inventory Optimization**: Determining optimal stock levels
- **Production Planning**: Scheduling manufacturing activities
- **Logistics Planning**: Optimizing transportation and distribution
- **Supplier Performance Analysis**: Monitoring supplier delivery times and quality
- **Price Optimization**: Adjusting prices based on temporal patterns

### Challenges in Supply Chain Time Series
- **Variable Lead Times**: Fluctuations in supplier delivery times
- **Bullwhip Effect**: Amplification of demand variability moving up the supply chain
- **Product Lifecycles**: Changing demand patterns as products age
- **External Factors**: Economic conditions, competitor actions, regulatory changes
- **Stockouts and Backorders**: Incomplete historical data due to unfulfilled demand
- **Promotions and Events**: Irregular spikes due to marketing activities

### Data Collection Considerations
- **Granularity**: Determining appropriate time intervals (hourly, daily, weekly)
- **Aggregation Level**: Product categories vs. individual SKUs
- **Data Quality**: Handling missing values and outliers
- **Integration**: Combining data from multiple sources (ERP, CRM, POS)
- **Latency**: Accounting for delays in data reporting

## Forecasting Techniques for Supply Chain

### Statistical Methods

#### Moving Averages
Simple technique that uses the average of past observations to predict future values.

**Simple Moving Average (SMA):**
```
SMA(t) = (Y(t-1) + Y(t-2) + ... + Y(t-n)) / n
```

**Weighted Moving Average (WMA):**
```
WMA(t) = (w₁ × Y(t-1) + w₂ × Y(t-2) + ... + wₙ × Y(t-n)) / Σw
```

**Exponential Smoothing Methods:**
- **Simple Exponential Smoothing**: For data with no trend or seasonality
  ```
  S(t) = α × Y(t) + (1-α) × S(t-1)
  ```
  
- **Holt's Linear Method**: For data with trend but no seasonality
  ```
  Level: L(t) = α × Y(t) + (1-α) × (L(t-1) + T(t-1))
  Trend: T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
  Forecast: F(t+h) = L(t) + h × T(t)
  ```
  
- **Holt-Winters Method**: For data with both trend and seasonality
  ```
  Level: L(t) = α × (Y(t)/S(t-s)) + (1-α) × (L(t-1) + T(t-1))
  Trend: T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
  Seasonal: S(t) = γ × (Y(t)/L(t)) + (1-γ) × S(t-s)
  Forecast: F(t+h) = (L(t) + h × T(t)) × S(t-s+h)
  ```

#### ARIMA Models
**AutoRegressive Integrated Moving Average (ARIMA)** models are denoted as ARIMA(p,d,q):
- p: Order of autoregression
- d: Degree of differencing
- q: Order of moving average

**SARIMA** extends ARIMA by incorporating seasonality with additional parameters (P,D,Q,s).

### Machine Learning Approaches

#### Regression-Based Methods
- Linear Regression
- Decision Trees
- Random Forests
- Gradient Boosting Machines

#### Neural Network Approaches
- **Recurrent Neural Networks (RNN)**: Specialized for sequential data
- **Long Short-Term Memory (LSTM)**: Better at capturing long-term dependencies
- **Temporal Convolutional Networks (TCN)**: Efficient parallel processing of time series
- **Prophet**: Facebook's forecasting tool that handles seasonality and holidays

### Hybrid Methods
Combining statistical and machine learning approaches can leverage the strengths of each method:
- Statistical methods for interpretability and handling of seasonality
- Machine learning for capturing complex patterns and incorporating external variables

## Deseasonalization Methods

### Why Deseasonalize?
Deseasonalization removes seasonal effects from time series data to:
- Identify underlying trends more clearly
- Compare different time periods on an equivalent basis
- Improve forecasting accuracy for models that don't handle seasonality well
- Isolate irregular components for anomaly detection

### Methods for Deseasonalization

#### 1. Seasonal Indices Method
1. Calculate the average value for each season (e.g., month) across multiple years
2. Compute the overall average across all seasons
3. Divide seasonal averages by the overall average to get seasonal indices
4. Divide original data by corresponding seasonal indices

#### 2. Moving Average Method
1. Apply a centered moving average with a window equal to the seasonal period
2. The resulting series represents the trend-cycle component
3. Divide the original series by the trend-cycle component to isolate seasonal and irregular components
4. Average these ratios/differences by season to estimate the seasonal factors
5. Divide original data by seasonal factors

#### 3. Seasonal Decomposition of Time Series (STL)
STL is a versatile and robust method that decomposes a time series into:
- Seasonal component
- Trend component
- Remainder (irregular) component

Advantages include handling non-linear trends and changing seasonality.

#### 4. X-11/X-13 ARIMA
Advanced decomposition methods used by many statistical agencies:
1. Initial estimation of trend-cycle using moving averages
2. Estimation of seasonal component from detrended series
3. Identification and removal of outliers
4. Iterative refinement of components

#### 5. Census Method (X-12-ARIMA)
An extension of X-11 that:
1. Applies ARIMA modeling prior to decomposition
2. Adjusts for trading days, holidays, and outliers
3. Uses advanced methods for seasonal adjustment

### Steps in Deseasonalization Process

1. **Identify Seasonal Pattern**:
   - Determine if seasonality is present (visual inspection, autocorrelation)
   - Identify the seasonal period (monthly, quarterly, weekly)

2. **Choose Appropriate Model**:
   - Additive model if seasonal variation is constant
   - Multiplicative model if seasonal variation changes with level

3. **Calculate Seasonal Components**:
   - Using one of the methods described above

4. **Remove Seasonal Component**:
   - For additive model: Original - Seasonal Component
   - For multiplicative model: Original ÷ Seasonal Component

5. **Verify Results**:
   - Check if seasonality has been effectively removed
   - Examine autocorrelation of deseasonalized series

### Example: Deseasonalizing Monthly Sales Data

For multiplicative model:
1. Calculate 12-month centered moving average (trend-cycle)
2. Divide original data by trend-cycle to get seasonal-irregular ratios
3. Average these ratios by month to get seasonal indices
4. Divide original data by seasonal indices to get deseasonalized series

## Practical Implementation

### Data Preparation
1. **Handle Missing Values**:
   - Linear interpolation
   - Mean/median imputation
   - Forward/backward fill
   - Advanced imputation methods (MICE, KNN)

2. **Outlier Detection and Treatment**:
   - Z-score approach
   - IQR method
   - DBSCAN for temporal data
   - Domain-specific rules

3. **Resampling**:
   - Upsampling: Increasing frequency (daily to hourly)
   - Downsampling: Decreasing frequency (daily to weekly)

4. **Feature Engineering**:
   - Lag features
   - Rolling statistics (mean, std, min, max)
   - Date-based features (day of week, month, quarter)

### Implementation in Python

#### Libraries for Time Series Analysis
- **pandas**: Data manipulation and time series functionality
- **statsmodels**: Statistical models including ARIMA, exponential smoothing
- **scikit-learn**: Machine learning algorithms
- **Prophet**: Facebook's forecasting tool
- **pmdarima**: Auto ARIMA implementation
- **LSTM/TensorFlow**: Deep learning approaches
- **tslearn**: Time series specific machine learning tools

#### Basic Decomposition Example
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data
df = pd.read_csv('supply_chain_data.csv', parse_dates=['date'], index_col='date')

# Perform decomposition
result = seasonal_decompose(df['sales'], model='multiplicative', period=12)

# Plot results
fig = plt.figure(figsize=(12, 10))
fig = result.plot()
plt.tight_layout()
plt.show()

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Create deseasonalized series
deseasonalized = df['sales'] / seasonal
```

#### SARIMA Forecasting Example
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Fit SARIMA model
model = SARIMAX(df['sales'], 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit()

# Forecast
forecast = results.get_forecast(steps=24)
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(df['sales'], label='Historical')
plt.plot(forecast.predicted_mean, label='Forecast')
plt.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=0.2)
plt.legend()
plt.title('SARIMA Sales Forecast')
plt.show()
```

#### Prophet Example
```python
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Prepare data for Prophet
df_prophet = df.reset_index()
df_prophet = df_prophet.rename(columns={'date': 'ds', 'sales': 'y'})

# Fit model
model = Prophet(yearly_seasonality=True, 
                weekly_seasonality=True,
                daily_seasonality=False)
model.fit(df_prophet)

# Make future dataframe for predictions
future = model.make_future_dataframe(periods=52, freq='W')

# Forecast
forecast = model.predict(future)

# Plot components
fig = model.plot_components(forecast)
plt.show()

# Plot forecast
fig = model.plot(forecast)
plt.show()
```

## Evaluation Metrics

### Accuracy Metrics
- **Mean Absolute Error (MAE)**:
  ```
  MAE = (1/n) × Σ|Actual - Forecast|
  ```
  
- **Mean Absolute Percentage Error (MAPE)**:
  ```
  MAPE = (100%/n) × Σ|Actual - Forecast|/|Actual|
  ```
  
- **Root Mean Square Error (RMSE)**:
  ```
  RMSE = √((1/n) × Σ(Actual - Forecast)²)
  ```

- **Weighted MAPE (WMAPE)**:
  ```
  WMAPE = Σ|Actual - Forecast| / Σ|Actual|
  ```

### Supply Chain Specific Metrics
- **Service Level**: Percentage of demand met on time
- **Stock-out Rate**: Frequency of inventory depletion
- **Inventory Turnover**: Rate at which inventory is used and replaced
- **Fill Rate**: Portion of customer demand satisfied from available inventory
- **Days of Supply**: Number of days inventory will last at current usage rate

### Cross-Validation for Time Series
- **Forward Chaining**: Train on increasing window of data
- **Rolling Window**: Fixed-size moving training window
- **K-fold with Temporal Constraints**: Ensure training data precedes validation data

## Advanced Topics

### Hierarchical Forecasting
Managing forecasts across multiple levels of aggregation:
- Top-down: Forecast at aggregate level and distribute
- Bottom-up: Forecast at detailed level and aggregate
- Middle-out: Forecast at intermediate level and distribute up/down
- Reconciliation methods: Statistical techniques to ensure consistency

### Intermittent Demand Forecasting
Methods for sparse, irregular demand patterns:
- Croston's method
- Syntetos-Boylan Approximation (SBA)
- TSB method (Teunter-Syntetos-Babai)

### Causal Factors in Supply Chain Forecasting
Incorporating external variables:
- Promotions and marketing activities
- Price changes
- Competitor actions
- Economic indicators
- Weather conditions
- Special events and holidays

### Collaborative Planning, Forecasting, and Replenishment (CPFR)
Framework for sharing forecasting information across the supply chain:
- Joint business planning
- Sales forecasting
- Order planning/forecasting
- Order generation/delivery

## Resources for Further Learning

### Books
- "Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos
- "Business Forecasting" by J. Holton Wilson and Barry Keating
- "Supply Chain Management: Strategy, Planning, and Operation" by Sunil Chopra and Peter Meindl
- "Fundamentals of Supply Chain Theory" by Lawrence V. Snyder and Zuo-Jun Max Shen

### Online Courses
- "Practical Time Series Analysis" on Coursera
- "Demand Planning: Forecasting and Inventory Management" on edX
- "Supply Chain Analytics" specializations on major platforms

### Software and Tools
- **R packages**: forecast, tseries, prophet
- **Python libraries**: statsmodels, prophet, pmdarima, sktime
- **Commercial Tools**: SAP IBP, Oracle Demantra, JDA, Kinaxis

### Communities and Conferences
- International Symposium on Forecasting
- INFORMS
- Council of Supply Chain Management Professionals (CSCMP)
- M5, M6 Forecasting Competitions