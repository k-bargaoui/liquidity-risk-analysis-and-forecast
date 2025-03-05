from common_imports import *

def compute_return_and_volatility(df, window=10):
    """
    Computes rolling volatility of the 'Close' price using a specified window size.
    
    Parameters:
        df (DataFrame): DataFrame containing the 'Close' price column.
        window (int): Rolling window size for volatility estimation (default is 10).
    
    Returns:
        DataFrame: Original DataFrame with an added 'Volatility' column.
    """
    df["Return"] = df["Close"].pct_change().dropna()  # Compute daily returns
    df["Volatility"] = df["Return"].rolling(window=window).std() 


def add_ewma_volatility(df, span=10):
    """
    Adds an EWMA volatility column to the DataFrame based on the Spread column.
    
    Parameters:
        df (DataFrame): DataFrame containing 'Spread' column.
        span (int): The span for the EWMA calculation (higher values give more smoothing).
    
    Returns:
        DataFrame: The original DataFrame with the added 'EWMA Volatility' column.
    """
    # Calculate the daily return of spreads
    df['Return'] = df['Spread'].pct_change()
    
    # Compute EWMA volatility (rolling standard deviation of the return)
    df['EWMA Vol'] = df['Return'].ewm(span=span, min_periods=span).std()

    # Drop NaN values from EWMA volatility (due to the first few rows)
    df.dropna(subset=["EWMA Vol"], inplace=True)    


def compute_average_spread(df):
    return (df["High"] - df["Low"]).mean()


def compute_amihud_illiquidity(df):
    returns = df["Return"]
    illiquidity = (abs(returns) / df["Volume"]).mean()
    return illiquidity


def compute_kyle_lambda(df):    
    # Drop NaN values that result from pct_change()
    df.dropna(subset=["Return", "Volume"], inplace=True)
    # Set up the regression: Return as the dependent variable and Volume as the independent variable
    X = df["Volume"]
    y = df["Return"]
    # Add a constant term for the intercept
    X = add_constant(X)
    # Fit the OLS regression model
    model = OLS(y, X).fit()
    # Kyle's Lambda is the slope of the regression
    lambda_value = model.params["Volume"]
    return lambda_value


def compute_market_impact(df):
    df["Market Impact (Linear)"] = df["Spread"] * df["Volume"] * 0.001
    df["Market Impact (Non-Linear)"] = df["Spread"] * np.sqrt(df["Volume"]) * 0.001
    df["Market Impact (Almgren-Chriss)"] = df["Spread"] * (df["Volume"] ** 0.6) * 0.001
    return df[["Market Impact (Linear)", "Market Impact (Non-Linear)", "Market Impact (Almgren-Chriss)"]].mean()


def compute_execution_cost(df):
    df["Impact Cost"] = df["Spread"] * df["Volume"] * 0.01  
    df["Total Execution Cost"] = df["Spread"] + df["Impact Cost"]
    return df["Total Execution Cost"].mean()


def roll_impact(df):
    """Estimate Roll’s Impact using autocovariance of price changes, 
    with handling for positive and negative autocovariance."""
    
    # Calculate price changes (difference between consecutive close prices)
    df["PriceChange"] = df["Close"].diff()
    
    # Drop NaN values before calculating autocovariance
    price_changes = df["PriceChange"].dropna()
    shifted_price_changes = price_changes.shift(1).dropna()
    
    # Ensure the lengths match for covariance calculation
    if len(price_changes) != len(shifted_price_changes):
        # If they are not equal in length, truncate the longer one
        price_changes = price_changes.iloc[:-1]
    
    # Calculate autocovariance between consecutive price changes
    gamma = np.cov(price_changes, shifted_price_changes)[0, 1]
    
    # If autocovariance is negative, calculate Roll's Impact
    if gamma < 0:
        price_impact = 2 * np.sqrt(-gamma)
        return price_impact
    # If autocovariance is positive, return a different measure (e.g., average price change)
    elif gamma > 0:
        # This indicates a trending market where autocovariance is positive
        avg_price_change = np.abs(df["PriceChange"].dropna()).mean()
        return f"Positive autocovariance detected. Avg Price Change: {avg_price_change:.4f}"
    # If autocovariance is zero or undefined, handle appropriately
    else:
        return "Autocovariance is zero or undefined, indicating no significant price relationship."


def corwin_schultz_spread(df):
    """Compute Corwin-Schultz spread estimator."""
    df["HighLowRatio"] = (df["High"] / df["Low"]).apply(np.log) ** 2
    beta = df["HighLowRatio"].rolling(window=2).sum().mean()
    alpha = np.sqrt(beta) / np.sqrt(2) - np.sqrt(beta) / 2
    return 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))


def pastor_stambaugh(df):
    """Compute Pastor-Stambaugh Liquidity Measure based on returns and volume."""
    df["RetDiff"] = df["Return"].diff()
    df["PastorStambaugh"] = df["RetDiff"] / df["Volume"]
    return df["PastorStambaugh"].mean()


def turnover_ratio(data):
    """Turnover Ratio (Volume / Market Cap)"""
    avg_volume = data["Volume"].mean()
    market_cap = data["Close"].mean() * avg_volume
    return avg_volume / market_cap if market_cap > 0 else None


def hasbrouck_lambda(data):
    """Hasbrouck's Lambda: More refined price impact measure."""
    # Calculate returns
    returns = data['Close'].pct_change()

    # Concatenate returns and volume into a DataFrame and drop NaN values
    aligned_data = pd.concat([returns, data['Volume']], axis=1).dropna()

    # Ensure aligned_data has at least two data points
    if len(aligned_data) < 2:
        raise ValueError("Insufficient data points to compute Hasbrouck's Lambda.")

    returns_aligned = aligned_data.iloc[:, 0]
    volume_aligned = aligned_data.iloc[:, 1]

    # Calculate covariance and variance
    covariance = np.cov(returns_aligned, volume_aligned)[0, 1]
    variance = np.var(volume_aligned)

    # Check for zero variance to avoid division by zero
    if variance == 0:
        raise ValueError("Variance of volume is zero, cannot compute Hasbrouck's Lambda.")

    # Compute Hasbrouck's Lambda
    lambda_value = covariance / variance
    return lambda_value


def order_book_imbalance(order_book):
    """Computes order book imbalance ratio."""
    
    # Get the total bid and ask volumes
    bid_volume = order_book['Bid Volume'].sum()
    ask_volume = order_book['Ask Volume'].sum()
    
    # Calculate the order book imbalance ratio
    return (bid_volume - ask_volume) / (bid_volume + ask_volume)


def compute_market_resilience(order_book_df):
    """Measures resilience by checking how fast spreads close after widening."""
    
    # Calculate the order book imbalance (this is a global metric)
    depth_ratio = order_book_imbalance(order_book_df)
    
    # Calculate resilience: depth ratio divided by spread at each time step
    order_book_df["Resilience"] = depth_ratio / order_book_df["Spread"]
    
    return order_book_df["Resilience"]



def compute_hurst_exponent(df, column_name="Spread"):
    """Compute the Hurst exponent of a time series from a DataFrame."""
    # Extract the time series from the DataFrame and drop NaN values
    time_series = df[column_name].dropna().values

    # Check if the time series is empty or has insufficient data
    if len(time_series) < 10:
        raise ValueError("Time series is too short to compute Hurst exponent.")

    # Compute the Hurst exponent
    n = len(time_series)
    rs_values = []

    for i in range(2, n // 2 + 1):
        sub_series = [time_series[j:j + i] for j in range(0, n - i + 1, i)]
        r_sub = []
        s_sub = []

        for sub in sub_series:
            if len(sub) > 1:  # Ensure sub-series has more than one element
                cumsum_sub = np.cumsum(sub - np.mean(sub))
                r_sub.append(np.max(cumsum_sub) - np.min(cumsum_sub))
                s_sub.append(np.std(sub))

        # Avoid division by zero in sub-series
        if any(s == 0 for s in s_sub):
            continue  # Skip this sub-series length if any standard deviation is zero

        rs_sub = np.mean(np.divide(r_sub, s_sub))
        rs_values.append((np.log(i), np.log(rs_sub)))

    if not rs_values:
        raise ValueError("No valid sub-series lengths found for Hurst exponent calculation.")

    rs_values = np.array(rs_values)

    # Check for NaN values in rs_values
    if np.any(np.isnan(rs_values)):
        raise ValueError("NaN values encountered in rs_values.")

    hurst_exponent = np.polyfit(rs_values[:, 0], rs_values[:, 1], 1)[0]
    return hurst_exponent


def liquidity_summary(df):
    if df is not None:
        compute_return_and_volatility(df)
        add_ewma_volatility(df)
        spread = compute_average_spread(df)
        amihud = compute_amihud_illiquidity(df)
        kyle = compute_kyle_lambda(df)
        corwin = corwin_schultz_spread(df)
        roll = roll_impact(df)
        pastor = pastor_stambaugh(df)
        turnover = turnover_ratio(df)
        hasbrouck = hasbrouck_lambda(df)
        hurst = compute_hurst_exponent(df)
        execution_cost = compute_execution_cost(df)
        market_impact = compute_market_impact(df)
        average_resilience = compute_market_resilience(df)

        summary = {
            "Average Spread": spread,
            "Amihud Illiquidity": amihud,
            "Kyle Lambda": kyle,
            "Corwin-Schultz Spread": corwin,
            "Roll Impact": roll,
            "Pastor-Stambaugh Liquidity Measure": pastor,
            "Turnover Ratio": turnover,
            "Hasbrouck Lambda": hasbrouck,
            "Hurst Exponent": hurst,
            "Execution Cost": execution_cost,
            "Market Impact": market_impact,
            "Average Resilience": average_resilience
        }

        return summary




