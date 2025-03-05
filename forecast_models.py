from common_imports import *
plt.style.use("ggplot")

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_xgboost_model_CV(df_input):
    df = df_input.copy()
    # Feature Engineering
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Liquidity Factor"] = df["Volume"] / df["Spread"]

    # Additional features
    df["Lag Volatility"] = df["Volatility"].shift(1)
    df["Lag Liquidity"] = df["Liquidity Factor"].shift(1)

    # Time-based features
    df['Day of Week'] = df.index.dayofweek  
    df['Week of Year'] = df.index.isocalendar().week  

    # Lag features for Spread
    for lag in range(1, 6):
        df[f'Lag Spread {lag}'] = df['Spread'].shift(lag)

    # Moving averages
    df['Spread MA 5'] = df['Spread'].rolling(5).mean()
    df['Spread MA 10'] = df['Spread'].rolling(10).mean()

    # Interaction terms
    df["Volatility * Liquidity"] = df["Volatility"] * df["Liquidity Factor"]

    # Log Transformation (both features and target)
    df["Log Volume"] = np.log(df["Volume"] + 1)
    df["Log Spread"] = np.log(df["Spread"] + 1)

    # Target Variable
    df["Next Spread"] = np.log(df["Spread"].shift(-1) + 1)

    # Drop NaNs
    df.dropna(inplace=True)

    # Define Features and Target
    features = [
        "Volume", "Return", "Volatility", "Liquidity Factor", "Lag Volatility","EWMA Vol", "Lag Liquidity",
        "Day of Week", "Week of Year",
        "Lag Spread 1", "Lag Spread 2", "Lag Spread 3", "Lag Spread 4", "Lag Spread 5",
        "Spread MA 5", "Spread MA 10", "Volatility * Liquidity"
    ]
    target = "Next Spread"

    # Split Data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Model with Randomized Search for Hyperparameter Tuning
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    model = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(model, param_dist, scoring='neg_mean_absolute_error', n_iter=50, cv=10, verbose=1, random_state=42)
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Predict and Evaluate
    predictions = best_model.predict(X_test)
    predictions = np.exp(predictions) - 1  # Inverse transformation of log(target)
    y_test_original = np.exp(y_test) - 1  # Inverse transformation of log(target)

    mae = mean_absolute_error(y_test_original, predictions)
    mape = mean_absolute_percentage_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    # Feature Importance Analysis
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # SHAP Analysis
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test)  # Visualize SHAP summary

    # Predicted vs Actual Comparison
    pred_vs_actual = pd.DataFrame({
        "actual": y_test_original,
        "predicted": predictions
    })

    return mae, mape, r2, pred_vs_actual



def train_linear_regression_model(df_input):
    df = df_input.copy()
    
    # Feature Engineering
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()
    df["Liquidity Factor"] = df["Volume"] / df["Spread"]

    # Additional features
    df["Lag Volatility"] = df["Volatility"].shift(1)
    df["Lag Liquidity"] = df["Liquidity Factor"].shift(1)

    # Time-based features
    df['Day of Week'] = df.index.dayofweek  
    df['Week of Year'] = df.index.isocalendar().week  

    # Lag features for Spread
    for lag in range(1, 6):
        df[f'Lag Spread {lag}'] = df['Spread'].shift(lag)

    # Moving averages
    df['Spread MA 5'] = df['Spread'].rolling(5).mean()
    df['Spread MA 10'] = df['Spread'].rolling(10).mean()

    # Interaction terms
    df["Volatility * Liquidity"] = df["Volatility"] * df["Liquidity Factor"]

    # Log Transformation (both features and target)
    df["Log Volume"] = np.log(df["Volume"] + 1)
    df["Log Spread"] = np.log(df["Spread"] + 1)

    # Target Variable
    df["Next Spread"] = np.log(df["Spread"].shift(-1) + 1)

    # Drop NaNs
    df.dropna(inplace=True)

    # Define Features and Target
    features = [
        "Volume", "Return", "Volatility", "Liquidity Factor", "Lag Volatility","EWMA Vol",
        "Week of Year",
        "Lag Spread 1",  "Lag Spread 3", "Lag Spread 4", "Lag Spread 5",
        "Spread MA 5", "Spread MA 10"
    ]
    target = "Next Spread"

    # Split Data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and Evaluate
    predictions = model.predict(X_test_scaled)
    predictions = np.exp(predictions) - 1  # Inverse transformation of log(target)
    y_test_original = np.exp(y_test) - 1  # Inverse transformation of log(target)

    mae = mean_absolute_error(y_test_original, predictions)
    mape = mean_absolute_percentage_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    # Predicted vs Actual Comparison
    pred_vs_actual = pd.DataFrame({
        "actual": y_test_original,
        "predicted": predictions
    })

    return mae, mape, r2, pred_vs_actual