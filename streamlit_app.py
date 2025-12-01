import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats

# ---------------------------------------------------------

# Utility functions

# ---------------------------------------------------------

def simple_linear_regression(X_train, y_train):
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
w = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y_train)
return w

def simple_ridge_regression(X_train, y_train, alpha=1.0):
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
I = np.eye(X_b.shape[1])
I[0,0] = 0
w = np.linalg.pinv(X_b.T @ X_b + alpha * I) @ (X_b.T @ y_train)
return w

def predict(w, X):
X_b = np.c_[np.ones((X.shape[0], 1)), X]
return X_b.dot(w)

def mse(y_true, y_pred):
return np.mean((y_true - y_pred) ** 2)

def icc(values):
n, k = values.shape
MSB = np.var(values.mean(axis=1), ddof=1) * k
MSW = np.mean(np.var(values, axis=1, ddof=1))
ICC1 = (MSB - MSW) / (MSB + (k - 1) * MSW)
ICC2 = (MSB - MSW) / MSB
return ICC1, ICC2

# ---------------------------------------------------------

# App layout

# ---------------------------------------------------------

st.title("Basic Science Exercise 3 — Enhanced Regression Demo (No Sklearn)")

st.sidebar.header("Instructions")
st.sidebar.write("""

1. Upload a CSV file
2. Select target variable
3. Choose model type
4. View performance, residuals, correlations, ICC
   """)
   st.sidebar.warning("Do not upload private or sensitive data.")

# ---------------------------------------------------------

# File upload

# ---------------------------------------------------------

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
df = pd.read_csv(uploaded)
st.write("### Data Preview")
st.dataframe(df)

```
# Cleaning Options
st.write("### Data Cleaning")
if st.checkbox("Drop rows with missing values"):
    df = df.dropna()

if st.checkbox("Standardize numeric columns (z-score)"):
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = (df[numeric] - df[numeric].mean()) / df[numeric].std()

# Feature Selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target = st.selectbox("Select Target", numeric_cols)

if target:
    features = [c for c in numeric_cols if c != target]
    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # Train/Test Split
    test_size = st.slider("Test Split (%)", 10, 50, 20)
    split = int(len(df) * (1 - test_size/100))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model Selection
    model_choice = st.selectbox("Model Type", ["Linear Regression", "Ridge Regression"])
    alpha = None
    if model_choice == "Ridge Regression":
        alpha = st.slider("Ridge Alpha", 0.1, 10.0, 1.0)

    # Train
    if model_choice == "Linear Regression":
        w = simple_linear_regression(X_train, y_train)
    else:
        w = simple_ridge_regression(X_train, y_train, alpha)

    # Predictions
    y_pred = predict(w, X_test)
    error = mse(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"**MSE:** {error:.5f}")

    # Feature Importance
    st.write("### Feature Importance (Absolute Weights)")
    importance = pd.DataFrame({
        "Feature": ["Intercept"] + features,
        "Weight": w.flatten()
    })
    st.bar_chart(importance.set_index("Feature"))

    # Prediction vs Actual
    st.write("### Predictions vs Actual")
    comp_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
    st.line_chart(comp_df)

    # Residuals
    st.write("### Residual Plot")
    residuals = y_test.flatten() - y_pred.flatten()
    st.line_chart(pd.DataFrame({"Residuals": residuals}))

    # Correlation Matrix
    st.write("### Correlation Matrix")
    corr = df[numeric_cols].corr()
    st.dataframe(corr.style.background_gradient(cmap="Blues"))

    # ICC Section
    st.write("### ICC (If multiple columns represent repeated measures)")
    icc_cols = st.multiselect("Select columns for ICC", numeric_cols)

    if len(icc_cols) > 1:
        matrix = df[icc_cols].dropna().values
        ICC1, ICC2 = icc(matrix)
        st.write(f"**ICC(1):** {ICC1:.4f}")
        st.write(f"**ICC(2):** {ICC2:.4f}")
```

else:
st.info("Upload a file to begin.")
