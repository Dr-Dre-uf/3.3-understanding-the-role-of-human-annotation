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

st.title("Basic Science Exercise 3 — Regression Demo (No Sklearn)")

st.write("Upload a dataset, choose your target, and run a lightweight regression model. Do not upload private or sensitive data.")

# ---------------------------------------------------------

# File upload

# ---------------------------------------------------------

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
df = pd.read_csv(uploaded)
st.write("### Data Preview")
st.dataframe(df)

```
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in this file.")
else:
    target = st.selectbox("Select target column", numeric_cols)

    if target:
        features = [c for c in numeric_cols if c != target]

        if not features:
            st.error("No numeric feature columns available.")
        else:
            X = df[features].values
            y = df[target].values.reshape(-1, 1)

            test_size = st.slider("Test size (%)", 10, 50, 20)
            split = int(len(df) * (1 - test_size / 100))

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            w = simple_regression_model(X_train, y_train)

            y_pred = predict(w, X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.write("### Model Performance")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")

            chart_df = pd.DataFrame({
                "Actual": y_test.flatten(),
                "Predicted": y_pred.flatten()
            })

            st.write("### Prediction Comparison")
            st.line_chart(chart_df)

            st.write("### Correlation Matrix")
            corr = df[numeric_cols].corr()
            st.dataframe(corr.style.background_gradient(cmap="Blues"))
```

else:
st.info("Upload a CSV file to begin.")
