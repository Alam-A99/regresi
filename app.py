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

st.title("📊 Dashboard Regresi Sederhana & Berganda + Uji Asumsi Klasik")

# ======================
# DATA DUMMY
# ======================
def generate_dummy_data():
    np.random.seed(42)
    n = 50
    X1 = np.random.rand(n) * 100
    X2 = np.random.rand(n) * 50
    noise = np.random.randn(n) * 10
    Y = 5 + 2*X1 + 3*X2 + noise
    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("⚙️ Input Data")
mode = st.sidebar.radio("Pilih Data:", ["Dummy", "Manual"])

if mode == "Dummy":
    df = generate_dummy_data()
else:
    x1 = list(map(float, st.sidebar.text_input("X1", "10,20,30,40").split(',')))
    x2 = list(map(float, st.sidebar.text_input("X2", "5,10,15,20").split(',')))
    y = list(map(float, st.sidebar.text_input("Y", "30,60,90,120").split(',')))
    df = pd.DataFrame({'X1': x1, 'X2': x2, 'Y': y})

st.subheader("📄 Dataset")
st.dataframe(df)

# ======================
# REGRESI
# ======================
X = df[['X1','X2']]
y = df['Y']

model = LinearRegression()
model.fit(X,y)

pred = model.predict(X)
df['Prediksi'] = pred

intercept = model.intercept_
b1, b2 = model.coef_

st.subheader("📈 Persamaan Regresi Berganda")
st.latex(f"Y = {intercept:.2f} + {b1:.2f}X_1 + {b2:.2f}X_2")

# ======================
# VISUALISASI
# ======================
fig, ax = plt.subplots()
ax.scatter(y, pred)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
st.pyplot(fig)

# ======================
# EVALUASI
# ======================
r2 = r2_score(y, pred)
mse = mean_squared_error(y, pred)

st.subheader("📉 Evaluasi Model")
st.write(f"R²: {r2:.3f}")
st.write(f"MSE: {mse:.3f}")

# ======================
# UJI ASUMSI KLASIK
# ======================
st.subheader("🧠 Uji Asumsi Klasik")

# Residual
residual = y - pred

# 1. Normalitas (Shapiro-Wilk)
stat, p = stats.shapiro(residual)
st.write("**Uji Normalitas (Shapiro-Wilk)**")
st.write(f"p-value: {p:.4f}")
if p > 0.05:
    st.success("Residual berdistribusi normal")
else:
    st.error("Residual tidak normal")

# 2. Multikolinearitas (VIF)
st.write("**Uji Multikolinearitas (VIF)**")
X_const = sm.add_constant(X)
vif = pd.DataFrame()
vif['Variabel'] = X_const.columns
vif['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
st.dataframe(vif)

# 3. Heteroskedastisitas (Breusch-Pagan)
st.write("**Uji Heteroskedastisitas (Breusch-Pagan)**")
bp_test = het_breuschpagan(residual, X_const)
pval_bp = bp_test[1]
st.write(f"p-value: {pval_bp:.4f}")
if pval_bp > 0.05:
    st.success("Tidak terjadi heteroskedastisitas")
else:
    st.error("Terjadi heteroskedastisitas")

# 4. Autokorelasi (Durbin-Watson)
st.write("**Uji Autokorelasi (Durbin-Watson)**")
dw = sm.stats.stattools.durbin_watson(residual)
st.write(f"Nilai DW: {dw:.3f}")

st.subheader("📋 Data + Prediksi")
st.dataframe(df)

st.success("Dashboard lengkap dengan uji asumsi klasik 🚀")
