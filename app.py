import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Regresi", layout="wide")

st.title("📊 Dashboard Regresi Sederhana & Berganda")

# ======================
# DATA DEFAULT (DUMMY)
# ======================
def generate_dummy_data():
    np.random.seed(42)
    n = 50
    X1 = np.random.rand(n) * 100
    X2 = np.random.rand(n) * 50
    noise = np.random.randn(n) * 10
    Y = 5 + 2*X1 + 3*X2 + noise
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y
    })
    return df

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("⚙️ Input Data")
mode = st.sidebar.radio("Pilih Sumber Data:", ["Data Dummy", "Input Manual"])

if mode == "Data Dummy":
    df = generate_dummy_data()
else:
    st.sidebar.write("Masukkan data (pisahkan dengan koma)")
    x1_input = st.sidebar.text_input("X1", "10,20,30,40")
    x2_input = st.sidebar.text_input("X2", "5,10,15,20")
    y_input = st.sidebar.text_input("Y", "30,60,90,120")

    try:
        X1 = list(map(float, x1_input.split(',')))
        X2 = list(map(float, x2_input.split(',')))
        Y = list(map(float, y_input.split(',')))
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})
    except:
        st.error("Format input salah!")
        st.stop()

st.subheader("📄 Dataset")
st.dataframe(df)

# ======================
# REGRESI SEDERHANA
# ======================
st.subheader("📈 Regresi Sederhana (X1 → Y)")

X_simple = df[['X1']]
y = df['Y']

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

y_pred_simple = model_simple.predict(X_simple)

st.write(f"Intercept: {model_simple.intercept_:.2f}")
st.write(f"Koefisien X1: {model_simple.coef_[0]:.2f}")

fig1, ax1 = plt.subplots()
ax1.scatter(df['X1'], y)
ax1.plot(df['X1'], y_pred_simple)
ax1.set_xlabel("X1")
ax1.set_ylabel("Y")
st.pyplot(fig1)

# ======================
# REGRESI BERGANDA
# ======================
st.subheader("📊 Regresi Berganda (X1, X2 → Y)")

X_multi = df[['X1', 'X2']]

model_multi = LinearRegression()
model_multi.fit(X_multi, y)

st.write(f"Intercept: {model_multi.intercept_:.2f}")
st.write(f"Koefisien X1: {model_multi.coef_[0]:.2f}")
st.write(f"Koefisien X2: {model_multi.coef_[1]:.2f}")

# Prediksi
pred = model_multi.predict(X_multi)

df['Prediksi'] = pred

st.subheader("📋 Hasil Prediksi")
st.dataframe(df)

# ======================
# EVALUASI MODEL
# ======================
st.subheader("📉 Evaluasi Model")

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y, pred)
mse = mean_squared_error(y, pred)

st.write(f"R² Score: {r2:.3f}")
st.write(f"MSE: {mse:.3f}")

st.success("Dashboard siap digunakan 🚀")
