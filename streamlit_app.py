import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def simple_regression_model(X_train, y_train):
    X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    w = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
    return w

def predict(w, X):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(w)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ---------------------------------------------------------
# App layout
# ---------------------------------------------------------

st.title("Basic Science Exercise 3 — Regression Demo")
st.write("Use sample data or upload your own CSV. Choose a target, model type, and noise settings for sample data. Do not upload private or sensitive data.")

# ---------------------------------------------------------
# Data Source
# ---------------------------------------------------------

source = st.radio("Choose data source", ["Use sample data", "Upload CSV"], horizontal=True)

df = None

if source == "Use sample data":
    num_samples = st.slider("Number of samples", 50, 500, 100, step=10)
    noise_level = st.slider("Noise level (std deviation)", 0.0, 5.0, 1.0, step=0.1)
    x = np.linspace(0, 10, num_samples)
    noise = np.random.normal(0, noise_level, num_samples)

    df = pd.DataFrame({
        "x": x,
        "noise": noise,
    })
    df["y"] = 3 * x + 5 + noise  # linear relation

    st.write("### Sample Data Preview")
    st.dataframe(df)

else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Uploaded Data Preview")
        st.dataframe(df)

# ---------------------------------------------------------
# Modeling Controls
# ---------------------------------------------------------

if df is not None:

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    target = st.selectbox("Select target column", numeric_cols)
    features = [c for c in numeric_cols if c != target]

    if not features:
        st.error("No numeric feature columns available.")
        st.stop()

    X = df[features].values
    y = df[target].values.reshape(-1, 1)

    # Model selection
    st.write("### Model Selection")
    model_type = st.selectbox("Choose model", [
        "Simple Linear Regression",
        "Polynomial Regression (Degree 2)",
        "Polynomial Regression (Degree 3)"
    ])

    # Split
    test_size = st.slider("Test size (%)", 10, 50, 20)
    split = int(len(df) * (1 - test_size / 100))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---------------------------------------------------------
    # Model training
    # ---------------------------------------------------------

    if model_type == "Simple Linear Regression":
        X_train_model = X_train
        X_test_model = X_test

    elif model_type == "Polynomial Regression (Degree 2)":
        X_train_model = np.c_[X_train, X_train ** 2]
        X_test_model = np.c_[X_test, X_test ** 2]

    elif model_type == "Polynomial Regression (Degree 3)":
        X_train_model = np.c_[X_train, X_train ** 2, X_train ** 3]
        X_test_model = np.c_[X_test, X_test ** 2, X_test ** 3]

    w = simple_regression_model(X_train_model, y_train)
    y_pred = predict(w, X_test_model)

    mse = mean_squared_error(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")

    # Linear vs Polynomial comparison plot for sample data
    if source == "Use sample data":
        st.write("### Linear vs Polynomial Comparison")
        y_pred_linear = predict(simple_regression_model(X_train, y_train), X_test)
        comparison_df = pd.DataFrame({
            "Actual": y_test.flatten(),
            "Linear Predicted": y_pred_linear.flatten(),
            "Poly Predicted": y_pred.flatten()
        })
        st.line_chart(comparison_df)

    # Regular prediction chart
    st.write("### Prediction Comparison")
    chart_df = pd.DataFrame({
        "Actual": y_test.flatten(),
        "Predicted": y_pred.flatten()
    })
    st.line_chart(chart_df)

    # Correlation matrix (no styling, lightweight)
    st.write("### Correlation Matrix")
    corr = df[numeric_cols].corr()
    st.dataframe(corr)

else:
    st.info("Upload a CSV or choose sample data to continue.")
