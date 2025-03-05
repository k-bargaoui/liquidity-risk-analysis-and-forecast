from common_imports import *

def plot_forecast_vs_actual(df):
    df.sort_index(inplace=True)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['actual'], label='Actual', color='blue', linewidth=2)

    # Plot the predicted values
    plt.plot(df.index, df['predicted'], label='Predicted', color='red', linestyle='--', linewidth=2)

    # Adding title and labels
    plt.title("Actual vs Predicted Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")

    # Show the legend
    plt.legend()

    # Display the plot
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()
    plt.show()


def plot_liquidity_trends(df):
    """
    Plot bid-ask spread and liquidity metrics over time.
    
    Args:
        df (DataFrame): Historical stock data.
        ticker (str): Stock ticker.
    """
    sns.set(style="whitegrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Bid-Ask Spread Over Time
    axes[0].plot(df.index, df["Spread"], label="Bid-Ask Spread", color="blue")
    axes[0].set_title(f"Bid-Ask Spread Over Time ")
    axes[0].set_ylabel("Spread (€)")
    axes[0].legend()

    # Market Impact Ratio Over Time
    df["market_impact"] = df["Spread"] / df["Volume"]
    axes[1].plot(df.index, df["market_impact"], label="Market Impact Ratio", color="red")
    axes[1].set_title(f"Market Impact Over Time ")
    axes[1].set_ylabel("Impact Ratio")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_acf_and_pacf(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ACF
    plot_acf(df['Spread'].dropna(), lags=50, ax=axes[0])
    axes[0].set_title("ACF of Spreads")

    # Plot PACF
    plot_pacf(df['Spread'].dropna(), lags=50, ax=axes[1])
    axes[1].set_title("PACF of Spreads")

    plt.show()


def plot_spread_cdf(df):
    """
    Plots the Cumulative Distribution Function (CDF) of the 'Spread' column.
    
    Parameters:
        df (DataFrame): DataFrame containing the 'Spread' column.
    """
    sorted_spread = np.sort(df['Spread'].dropna())  # Remove NaNs and sort
    cdf = np.arange(1, len(sorted_spread) + 1) / len(sorted_spread)  # Compute CDF
    
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_spread, cdf, marker='.', linestyle='none', color='blue', label="Empirical CDF")
    plt.xlabel("Spread")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution Function (CDF) of Spread")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_spread_hist(df):
    """
    Plots a histogram of the 'Spread' column.
    
    Parameters:
        df (DataFrame): DataFrame containing 'Spread' column.
    """
    # Plotting histogram for the 'Spread' column
    plt.figure(figsize=(10, 6))
    plt.hist(df['Spread'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Spread')
    plt.ylabel('Frequency')
    plt.title('Histogram of Spread')
    plt.grid(True)
    plt.show


def plot_market_resilience(df, resilience_column='Resilience'):
    """Plot the market resilience over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[resilience_column], label='Market Resilience')
    plt.title('Market Resilience Over Time')
    plt.xlabel('Date')
    plt.ylabel('Resilience')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_spreads_and_volatility(df):
    """
    Plots Spread and Rolling Volatility on the same graph for high-frequency data (15-minute intervals).
    
    Parameters:
        df (DataFrame): DataFrame containing 'Spread' and 'Volatility' indexed by timestamp.
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Adjust if needed
        df.set_index("Timestamp", inplace=True)

    # Check if Volatility exists; if not, compute it
    if "Volatility" not in df.columns:
        print("Volatility' column is missing! Computing a 10-period rolling volatility...")
        df["Return"] = df["Spread"].pct_change()
        df["Volatility"] = df["Return"].rolling(10).std()

    df = df.dropna(subset=["Volatility"])  # Drop initial NaN values

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Spread on primary y-axis
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Spread", color="tab:blue")
    ax1.plot(df.index, df["Spread"], color="tab:blue", label="Spread", alpha=0.7)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create a second y-axis for volatility
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volatility", color="tab:red")
    ax2.plot(df.index, df["Volatility"], color="tab:red", linestyle="dashed", label="Volatility", alpha=0.7)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Format x-axis for better readability
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show every 5th day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))  # Format as 'Month Day'
    plt.xticks(rotation=45)

    # Add title and legend
    plt.title("Spread and Volatility Over Time")
    fig.tight_layout()
    plt.show()


def plot_spread_heatmap(df):
    """
    Plots a heatmap of spread values across time, with dates as rows and times of day as columns.

    Parameters:
        df (DataFrame): DataFrame containing 'Timestamp' (datetime index) and 'Spread'.
    """
    # Ensure Timestamp is datetime and set as index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Adjust column name if needed
        df.set_index("Timestamp", inplace=True)

    # Extract date and time components
    df["Date"] = df.index.date
    df["Time"] = df.index.time

    # Pivot table: rows = Date, columns = Time, values = Spread
    pivot_table = df.pivot_table(values="Spread", index="Date", columns="Time", aggfunc="mean")

    # Compute better vmin/vmax for better contrast
    vmin, vmax = np.percentile(pivot_table.dropna().values, [5, 95])  # Ignore extremes for better contrast

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap="coolwarm_r", linewidths=0.1, linecolor="gray", 
                cbar=True, robust=True, vmin=vmin, vmax=vmax)

    # Labels and title
    plt.xlabel("Time of Day")
    plt.ylabel("Date")
    plt.title("Heatmap of Spread Over Time (15-Min Intervals)")
    plt.xticks(rotation=45)

    plt.show()


def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix between Spread and Volatility.
    """
    correlation_matrix = df[["Return", "Spread", "Volume","Volatility", "Mid Price", "Bid Price", "Ask Price", "EWMA Vol"]].corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation matrix")
    plt.show()        