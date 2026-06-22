"""
宁德时代多因子量化择时策略 — Streamlit Dashboard
运行: streamlit run dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="CATL量化策略", page_icon="📊", layout="wide")

# ==================== Sidebar ====================
st.sidebar.title("📊 策略参数")
st.sidebar.markdown("---")

ma_short = st.sidebar.slider("短期均线", 3, 15, 5)
ma_long = st.sidebar.slider("长期均线", 15, 60, 20)
rsi_low = st.sidebar.slider("RSI下限", 20, 50, 40)
rsi_high = st.sidebar.slider("RSI上限", 50, 85, 75)
sent_threshold = st.sidebar.slider("情感阈值", 0.45, 0.65, 0.52, 0.01)
test_ratio = st.sidebar.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("💡 调整参数后点击下方按钮重新计算")
if st.sidebar.button("🔄 重新回测", type="primary"):
    st.rerun()

# ==================== Main ====================
st.title("📱 宁德时代(300750) 多因子量化择时策略")
st.markdown("*技术面 + 情绪面 + 资金面 三维过滤体系*")
st.markdown("---")

# ==================== Simulate Data ====================
@st.cache_data
def simulate_data(seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', '2026-05-30', freq='B')
    n = len(dates)
    trend = np.linspace(0, 1, n) * 120 + 80
    seasonal = np.sin(np.linspace(0, 8*np.pi, n)) * 25
    noise = np.cumsum(np.random.randn(n) * 3)
    close = np.maximum(trend + seasonal + noise, 50)
    volume = np.random.randint(5000000, 30000000, n)
    return pd.DataFrame({'date': dates, 'close': close, 'volume': volume}).set_index('date')

df = simulate_data()

# ==================== Compute Indicators ====================
df['MA5'] = df['close'].rolling(ma_short).mean()
df['MA20'] = df['close'].rolling(ma_long).mean()
exp12 = df['close'].ewm(span=12, adjust=False).mean()
exp26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD_HIST'] = 2 * ((exp12 - exp26) - (exp12 - exp26).ewm(span=9, adjust=False).mean())
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))
df['sentiment'] = (0.5 + 0.12 * (df['close'] - df['close'].rolling(60).mean()) /
                    df['close'].rolling(60).std().replace(0, 1)).clip(0, 1)
df['sent_MA5'] = df['sentiment'].rolling(5).mean()
money_z = (np.cumsum(np.random.randn(len(df)) * 0.5) * 10000 - np.cumsum(
    np.random.randn(len(df)) * 0.5).rolling(20).mean() * 10000) / (
    np.cumsum(np.random.randn(len(df)) * 0.5).rolling(20).std().replace(0, 1) * 10000)
df['money_z'] = money_z.fillna(0)

# ==================== Signals ====================
df['sig_trend'] = (df['MA5'] > df['MA20']).astype(int)
df['sig_macd'] = (df['MACD_HIST'] > 0).astype(int)
df['sig_rsi'] = ((df['RSI'] > rsi_low) & (df['RSI'] < rsi_high)).astype(int)
df['sig_sent'] = (df['sent_MA5'] > sent_threshold).astype(int)
df['sig_money'] = (df['money_z'] > -0.5).astype(int)
df['signal'] = (df['sig_trend'] & df['sig_macd'] & df['sig_rsi'] &
                df['sig_sent'] & df['sig_money']).astype(int)
df['ret'] = df['close'].pct_change()
df['strat_ret'] = df['signal'].shift(1) * df['ret']
df['bench_ret'] = df['ret']

# ==================== Performance ====================
df['strat_nav'] = (1 + df['strat_ret'].fillna(0)).cumprod()
df['bench_nav'] = (1 + df['bench_ret'].fillna(0)).cumprod()
strat_cum = (df['strat_nav'].iloc[-1] - 1) * 100
bench_cum = (df['bench_nav'].iloc[-1] - 1) * 100
strat_dd = ((df['strat_nav'] - df['strat_nav'].cummax()) / df['strat_nav'].cummax()).min() * 100
bench_dd = ((df['bench_nav'] - df['bench_nav'].cummax()) / df['bench_nav'].cummax()).min() * 100
win_rate = (df['strat_ret'][df['strat_ret'] != 0] > 0).mean() * 100
n_trades = (df['strat_ret'] != 0).sum()

# ==================== Dashboard Cards ====================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("累计收益", f"{strat_cum:+.1f}%", f"vs BH {bench_cum:+.1f}%")
col2.metric("最大回撤", f"{strat_dd:.1f}%", f"vs BH {bench_dd:.1f}%")
col3.metric("胜率", f"{win_rate:.1f}%")
col4.metric("交易次数", str(n_trades))
col5.metric("开仓占比", f"{df['signal'].mean()*100:.1f}%")

st.markdown("---")

# ==================== Charts ====================
tab1, tab2, tab3 = st.tabs(["📈 净值曲线", "📊 技术指标", "📋 绩效明细"])

with tab1:
    st.subheader("累计净值对比")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(df.index, df['strat_nav'], 'r-', lw=2, label=f"复合策略 ({strat_cum:+.1f}%)")
    ax1.plot(df.index, df['bench_nav'], 'gray', lw=1, alpha=0.5, label=f"买入持有 ({bench_cum:+.1f}%)")
    ax1.fill_between(df.index, 1, df['strat_nav'], alpha=0.1, color='red')
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylabel('累计净值')

    dd = (df['strat_nav'] - df['strat_nav'].cummax()) / df['strat_nav'].cummax()
    bdd = (df['bench_nav'] - df['bench_nav'].cummax()) / df['bench_nav'].cummax()
    ax2.fill_between(df.index, 0, dd, color='red', alpha=0.4, label=f"复合策略 (max {strat_dd:.1f}%)")
    ax2.fill_between(df.index, 0, bdd, color='gray', alpha=0.2, label=f"买入持有 (max {bench_dd:.1f}%)")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylabel('回撤'); ax2.set_xlabel('日期')
    ax2.set_ylim(min(dd.min(), bdd.min()) * 1.1, 0.02)
    st.pyplot(fig)

with tab2:
    st.subheader("技术指标面板")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    axes[0].plot(df.index, df['close'], 'k-', lw=0.8, alpha=0.5, label='Close')
    axes[0].plot(df.index, df['MA5'], 'b-', lw=1, label=f'MA{ma_short}')
    axes[0].plot(df.index, df['MA20'], 'orange', lw=1, label=f'MA{ma_long}')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3); axes[0].set_ylabel('Price')

    axes[1].bar(df.index, df['MACD_HIST'], color=['red' if v > 0 else 'green' for v in df['MACD_HIST']], alpha=0.5, width=1)
    axes[1].axhline(0, color='black', lw=0.5); axes[1].set_ylabel('MACD'); axes[1].grid(alpha=0.3)

    axes[2].plot(df.index, df['RSI'], 'purple', lw=1)
    axes[2].axhline(rsi_high, color='r', ls='--', alpha=0.5); axes[2].axhline(rsi_low, color='g', ls='--', alpha=0.5)
    axes[2].set_ylabel('RSI'); axes[2].set_ylim(0, 100); axes[2].grid(alpha=0.3)

    axes[3].fill_between(df.index, 0, df['signal'], step='post', color='red', alpha=0.4, label='Signal')
    axes[3].set_ylabel('Signal'); axes[3].set_ylim(0, 1.2); axes[3].legend(fontsize=7); axes[3].grid(alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.subheader("年度收益对比")
    df['year'] = df.index.year
    annual = df.groupby('year').apply(lambda g: pd.Series({
        'strat': (1 + g['strat_ret'].fillna(0)).prod() - 1,
        'bench': (1 + g['bench_ret'].fillna(0)).prod() - 1
    })).reset_index()
    st.dataframe(annual.style.format("{:.2%}", subset=['strat', 'bench']), use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.caption("宁德时代(300750) 多因子量化择时策略 | 财经数据分析综合实践 | 刘桂超 | 2026-06-16")
