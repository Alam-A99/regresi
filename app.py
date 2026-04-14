import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

st.set_page_config(page_title="Dashboard Regresi", layout="wide")

st.title("📊 Dashboard Regresi Fleksibel (hingga 7 Variabel Bebas)")

# ======================
# DATA DUMMY
# ======================
def generate_dummy_data():
    np.random.seed(42)
    n = 60
    data = {}
    for i in range(1,8):
        data[f'X{i}'] = np.random.rand(n)*100
    noise = np.random.randn(n)*10
    Y = 5 + sum((i+1)*data[f'X{i}'] for i in range(1,4)) + noise
    data['Y'] = Y
    return pd.DataFrame(data)

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("⚙️ Input Data")
mode = st.sidebar.radio("Sumber Data:", ["Dummy", "Manual"])

if mode == "Dummy":
    df = generate_dummy_data()
else:
    df = pd.DataFrame()
    for i in range(1,8):
        val = st.sidebar.text_input(f"X{i}", "10,20,30,40")
        df[f'X{i}'] = list(map(float, val.split(',')))
    y_val = st.sidebar.text_input("Y", "30,60,90,120")
    df['Y'] = list(map(float, y_val.split(',')))

st.subheader("📄 Dataset")
st.dataframe(df)

# ======================
# PILIH VARIABEL
# ======================
st.sidebar.header("🎯 Pilih Variabel")
all_vars = [col for col in df.columns if col.startswith('X')]
selected_X = st.sidebar.multiselect("Variabel Bebas (X)", all_vars, default=all_vars[:2])

if len(selected_X) == 0:
    st.warning("Pilih minimal 1 variabel bebas")
    st.stop()

X = df[selected_X]
y = df['Y']

# ======================
# REGRESI
# ======================
model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)
df['Prediksi'] = pred

intercept = model.intercept_
coefs = model.coef_

# Persamaan
persamaan = f"Y = {intercept:.2f}"
for i, var in enumerate(selected_X):
    persamaan += f" + {coefs[i]:.2f}{var}"

st.subheader("📈 Persamaan Regresi")
st.latex(persamaan)

# ======================
# EVALUASI
# ======================
r2 = r2_score(y, pred)
mse = mean_squared_error(y, pred)

st.subheader("📉 Evaluasi")
st.write(f"R²: {r2:.3f}")
st.write(f"MSE: {mse:.3f}")

# ======================
# UJI ASUMSI
# ======================
st.subheader("🧠 Uji Asumsi Klasik")
residual = y - pred

# Normalitas
stat, p = stats.shapiro(residual)
st.write("Normalitas (Shapiro)", p)

# VIF
X_const = sm.add_constant(X)
vif = pd.DataFrame()
vif['Variabel'] = X_const.columns
vif['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
st.dataframe(vif)

# Heteroskedastisitas
bp = het_breuschpagan(residual, X_const)
st.write("Breusch-Pagan p-value:", bp[1])

# Autokorelasi
dw = sm.stats.stattools.durbin_watson(residual)
st.write("Durbin-Watson:", dw)

st.subheader("📋 Data + Prediksi")
st.dataframe(df)

st.success("Model regresi fleksibel siap 🚀")
