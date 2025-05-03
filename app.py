import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

st.markdown("""
<style>
h1 {
  font-size:26px !important;
  margin-bottom: 0.5rem;
}
h1 span {
  font-size:16px !important;
}
</style>
<h1>トラウト成長予測<span>(GompertzいわきRASモデル)</span></h1>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("batch_growth_data.csv")

data = load_data()

# 最も成長が早かった・遅かったバッチを除外
batch_summary = data.groupby("batch").agg(final_weight=("weight", "max"))
fastest_batch = batch_summary["final_weight"].idxmax()
slowest_batch = batch_summary["final_weight"].idxmin()
filtered_data = data[~data["batch"].isin([fastest_batch, slowest_batch])]

# Gompertz成長関数定義
def gompertz(t, A, B, K):
    return A * np.exp(-np.exp(B - K * t))

# パラメータフィッティング
X_all = filtered_data["day"].values
y_all = filtered_data["weight"].values
initial_guess = [300, 5, 0.03]
weights = np.where(X_all < 100, 2.0, 1.0)
params, _ = curve_fit(gompertz, X_all, y_all, p0=initial_guess, sigma=1/weights, absolute_sigma=False, maxfev=10000)
A, B, K = params

# 目標体重の範囲
min_weight = 10
max_weight = 350

# ▼ 日数から予測体重
st.header("🧮 日数から予測体重")
input_day = st.slider("日数を入力（0〜300日）", min_value=0, max_value=300, value=100)
predicted_weight = gompertz(input_day, A, B, K)
st.markdown(f"### 🐟 予測体重：**{predicted_weight:.2f} g**（{input_day}日目）")

# ▼ 目標体重まであと何日？
st.header("📈 目標体重まであと何日？")
current_day = st.slider("現在の飼育日数（日）", min_value=0, max_value=300, value=50)
current_weight = st.slider("現在の体重（g）", min_value=0.0, max_value=350.0, value=50.0, step=0.5)
target_weight = st.slider("目標体重（g）", min_value=min_weight, max_value=max_weight, value=200, step=10)

# 補正カーブのパラメータ
def adjusted_gompertz_params(current_day, current_weight, A, B, K):
    try:
        expected_weight = gompertz(current_day, A, B, K)
        ratio = current_weight / expected_weight if expected_weight > 0 else 1.0
        growth_factor = min(1 + (ratio - 1) * 0.8, 1.5)
        K_adj = K * growth_factor

        log_term = -np.log(current_weight / A)
        B_adj = np.log(log_term) + K_adj * current_day

        return A, B_adj, K_adj
    except:
        return A, B, K

# 到達日数の推定
def find_target_day(current_day, current_weight, target_weight, A, B, K):
    future_days = np.arange(current_day, 301, 0.1)
    predicted_weights = gompertz(future_days, A, B, K)
    mask = predicted_weights >= target_weight
    if np.any(mask):
        return future_days[mask][0]
    else:
        return future_days[-1]

A_adj, B_adj, K_adj = adjusted_gompertz_params(current_day, current_weight, A, B, K)
target_day = find_target_day(current_day, current_weight, target_weight, A_adj, B_adj, K_adj)
remaining_days = target_day - current_day

if current_weight >= target_weight:
    st.markdown(f"### ✅ すでに{target_weight}gを超えています！")
else:
    st.markdown(f"### ⏳ 到達予測：**{target_day:.1f}日目**（あと **{remaining_days:.1f}日**）")

# グラフ描画
future_days = np.arange(0, 301, 1)
predicted_curve = gompertz(future_days, A, B, K)
future_days_adj = np.arange(0, 301, 1)
predicted_curve_adj_full = gompertz(future_days_adj, A_adj, B_adj, K_adj)


# 500g未満のデータだけ使用
mask_under_500 = predicted_curve_adj_full <= 500

# 表示のための補正カーブの時間軸と値（現在の体重に合わせる）
index_now = np.where(future_days_adj == current_day)[0]
if index_now.size > 0:
    predicted_curve_adj_full[index_now[0]] = current_weight


# 表示用に500g未満のデータだけ使用
future_days_adj = future_days_adj[mask_under_500]
predicted_curve_adj = predicted_curve_adj_full[mask_under_500]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data["day"], filtered_data["weight"], 'o', alpha=0.3, label="Measured values")
ax.plot(future_days, predicted_curve, 'r-', label="Weighted fit (B)")
if current_weight <= 500:
    ax.plot(future_days_adj, predicted_curve_adj, 'g--', label="Adjusted to current")
ax.axvline(input_day, color='orange', linestyle='--', label=f"Input day: {input_day}d")
ax.axhline(predicted_weight, color='blue', linestyle=':', label=f"Predicted weight: {predicted_weight:.2f}g")
ax.scatter(current_day, current_weight, color='green', label="Current weight")
if current_weight < target_weight:
    ax.axvline(target_day, color='purple', linestyle='--', label=f"Target {target_weight}g: {target_day:.1f}d")
    ax.axhline(target_weight, color='gray', linestyle=':', label=f"Target weight: {target_weight}g")

ax.set_xlabel("Days")
ax.set_ylabel("Weight (g)")
ax.legend()
ax.grid(True)
st.pyplot(fig, use_container_width=True)
