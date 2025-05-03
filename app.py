import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'

st.title("トラウト成長サイズ予測アプリ（日数 → 体重）")

@st.cache_data
def load_data():
    return pd.read_csv("batch_growth_data.csv")

data = load_data()

# 最も成長が早かった・遅かったバッチを除外
batch_summary = data.groupby("batch").agg(final_weight=("weight", "max"))
fastest_batch = batch_summary["final_weight"].idxmax()
slowest_batch = batch_summary["final_weight"].idxmin()
filtered_data = data[~data["batch"].isin([fastest_batch, slowest_batch])]

# 成長モデル学習
X_all = filtered_data[["day"]]
y_all = filtered_data["weight"]
model = LinearRegression()
model.fit(X_all, y_all)

# ▼ 機能①：日数から予測体重
st.header("🧮 日数から予測体重")
input_day = st.slider("日数を入力（0〜300日）", min_value=0, max_value=300, value=100)
predicted_weight = model.predict(np.array([[input_day]]))[0]
st.markdown(f"### 🐟 予測体重：**{predicted_weight:.2f} g**（{input_day}日目）")

# ▼ 機能②：現在の体重から目標体重までの日数予測
st.header("📈 現在の状況から目標体重到達までの予測")
current_day = st.slider("現在の飼育日数（日）", min_value=0, max_value=300, value=50)
current_weight = st.slider("現在の体重（g）", min_value=0.0, max_value=300.0, value=50.0, step=0.5)
target_weight = st.slider("目標体重（g）", min_value=100, max_value=300, value=200, step=10)

a = model.coef_[0]
b_adjusted = current_weight - a * current_day

target_day = (target_weight - b_adjusted) / a
remaining_days = target_day - current_day

if remaining_days < 0:
    st.markdown(f"### ✅ すでに{target_weight}gを超えています！")
else:
    st.markdown(f"### ⏳ 予測到達日数：**{target_day:.1f}日目**（あと **{remaining_days:.1f}日**）")

# グラフ描画
future_days = np.arange(0, 301).reshape(-1, 1)
predicted_curve = model.predict(future_days)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data["day"], filtered_data["weight"], 'o', alpha=0.3, label="実測値（外れ値除外）")
ax.plot(future_days, predicted_curve, 'r-', label="平均成長予測線")
ax.axvline(input_day, color='orange', linestyle='--', label=f"入力日数: {input_day}日")
ax.axhline(predicted_weight, color='blue', linestyle=':', label=f"予測体重: {predicted_weight:.2f}g")
ax.scatter(current_day, current_weight, color='green', label="現在の体重")
if remaining_days > 0:
    ax.axvline(target_day, color='purple', linestyle='--', label=f"{target_weight}g予測到達日: {target_day:.1f}日")
    ax.axhline(target_weight, color='gray', linestyle=':', label=f"目標体重: {target_weight}g")
ax.set_title("平均成長曲線（両端除外）")
ax.set_xlabel("日数")
ax.set_ylabel("体重 (g)")
ax.legend()
ax.grid(True)
st.pyplot(fig)