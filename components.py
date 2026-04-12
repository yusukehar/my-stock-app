# components.py：描画専用モジュール

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import streamlit as st

# components.py

def create_stock_chart(symbol, df_all):
    # --- 既存のデータ抽出 ---
    s_close = df_all['Close'][symbol]
    s_open  = df_all['Open'][symbol]
    s_high  = df_all['High'][symbol]
    s_low   = df_all['Low'][symbol]
    s_vol   = df_all['Volume'][symbol]

    # --- 移動平均線の計算を追加 ---
    ma5  = s_close.rolling(window=5).mean()
    ma25 = s_close.rolling(window=25).mean()
    ma75 = s_close.rolling(window=75).mean()

    # サブプロットの作成（既存通り）
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.7, 0.3], 
        vertical_spacing=0.05
    )

    # 1. ローソク足の追加（既存通り）
    fig.add_trace(
        go.Candlestick(
            x=df_all.index, 
            open=s_open, high=s_high, low=s_low, close=s_close, 
            name='株価'
        ), 
        row=1, col=1
    )

    # 2. 移動平均線の追加（ここを追記）
    fig.add_trace(go.Scatter(x=df_all.index, y=ma5,  name='5MA',  line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_all.index, y=ma25, name='25MA', line=dict(color='red',    width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_all.index, y=ma75, name='75MA', line=dict(color='blue',   width=1)), row=1, col=1)

    # 3. 出来高の追加（既存通り）
    fig.add_trace(
        go.Bar(x=df_all.index, y=s_vol, name='出来高', marker_color='gray', opacity=0.5), 
        row=2, col=1
    )

    # レイアウト調整（スライダーを非表示にするとMAが見やすくなります）
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    
    return fig

@st.dialog("銘柄詳細チャート", width="large")
def show_chart_dialog(symbol, df_all):
    """
    Streamlitのダイアログとしてチャートを表示
    """
    fig = create_stock_chart(symbol, df_all)
    st.plotly_chart(fig, use_container_width=True)


@st.dialog("銘柄詳細", width="large")
def show_stock_detail_dialog(row, closes, full_stock_data):
    """
    銘柄の詳細情報をダイアログで表示する（リスト行の「詳細」ボタンから呼び出す）
    """
    display_stock_card_content(row, closes, full_stock_data)

def display_valuation_metrics(row):
    """
    銘柄のバリュエーション（PER/PBR）をきれいに表示するコンポーネント
    """
    col1, col2, col3 = st.columns(3)
    
    # PERの表示
    per_color = "inverse" if row['PER_判定'] == "Under" else "normal"
    col1.metric("PER", f"{row['PeRatio']:.1f}", f"業種比: {row['PER_判定']}", delta_color=per_color)
    
    # PBRの表示
    pbr_color = "inverse" if row['PBR_判定'] == "Under" else "normal"
    col2.metric("PBR", f"{row['PbRatio']:.1f}", f"業種比: {row['PBR_判定']}", delta_color=pbr_color)
    
    # その他情報
    col3.write(f"**市場:** {row['class']}")
    col3.write(f"**業種:** {row['industry']}")


def display_valuation_chart(row):
    """
    銘柄のPER/PBRと業種平均を比較するチャートを表示する
    """
    # PERの比較グラフ
    fig_per = go.Figure()
    fig_per.add_trace(go.Bar(
        y=['業種平均', '銘柄'],
        x=[row['Average_PER'], row['PeRatio']],
        orientation='h',
        marker_color=['lightgrey', 'blue' if row['PER_判定'] == 'Under' else 'orange'],
        text=[f"{row['Average_PER']:.1f}", f"{row['PeRatio']:.1f}"],
        textposition='auto',
    ))
    fig_per.update_layout(
        title="PER 比較",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="倍率")
    )
    
    # PBRの比較グラフ
    fig_pbr = go.Figure()
    fig_pbr.add_trace(go.Bar(
        y=['業種平均', '銘柄'],
        x=[row['Average_PBR'], row['PbRatio']],
        orientation='h',
        marker_color=['lightgrey', 'blue' if row['PBR_判定'] == 'Under' else 'orange'],
        text=[f"{row['Average_PBR']:.1f}", f"{row['PbRatio']:.1f}"],
        textposition='auto',
    ))
    fig_pbr.update_layout(
        title="PBR 比較",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title="倍率")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_per, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pbr, use_container_width=True)

# components.py

def display_financial_trends_chart(row):
    """
    銘柄の売上高と営業利益の推移を棒グラフで表示する
    row には logic.py から取得した財務データが含まれている想定
    """
    # logic.py で保存した財務データ（リスト形式など）を取り出す
    # 変数名は既存の logic.py の取得形式に合わせる必要があります
    years = row.get('financial_years', [])
    revenue = row.get('financial_revenue', [])
    profit = row.get('financial_profit', [])

    if not years:
        st.caption("財務推移データがありません。")
        return

    fig = go.Figure()

    # 売上高の棒グラフ
    fig.add_trace(go.Bar(
        x=years, y=revenue, name="売上高",
        marker_color="rgb(55, 83, 109)"
    ))
    
    # 営業利益の棒グラフ
    fig.add_trace(go.Bar(
        x=years, y=profit, name="営業利益",
        marker_color="rgb(26, 118, 255)"
    ))

    fig.update_layout(
        title="財務推移 (直近3期)",
        xaxis_tickfont_size=12,
        yaxis=dict(title="金額"),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        barmode='group',
        bargap=0.15,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# components.py

def display_performance_metrics(row, closes):
    """
    指定された期間（5, 20, 60日）の株価上昇率を表示する
    """
    symbol = row['code']
    if symbol not in closes.columns:
        st.write("株価データが見つかりません")
        return

    # 対象銘柄の終値シリーズを取得
    s_close = closes[symbol].dropna()
    if len(s_close) < 61: # 60日分のデータがない場合
        st.caption("十分な期間の株価データがありません")
        return

    # 現在の価格と過去の価格
    current_p = s_close.iloc[-1]
    p_5  = s_close.iloc[-6]  # 5日前
    p_20 = s_close.iloc[-21] # 20日前
    p_60 = s_close.iloc[-61] # 60日前

    # 上昇率計算
    change_5  = (current_p - p_5) / p_5 * 100
    change_20 = (current_p - p_20) / p_20 * 100
    change_60 = (current_p - p_60) / p_60 * 100

    # UI表示
    st.markdown("##### 騰落率 (%)")
    c1, c2, c3 = st.columns(3)
    
    def format_delta(val):
        return f"{'+' if val > 0 else ''}{val:.2f}%"

    c1.metric("5日", f"{current_p:.1f}", format_delta(change_5))
    c2.metric("20日", f"{current_p:.1f}", format_delta(change_20))
    c3.metric("60日", f"{current_p:.1f}", format_delta(change_60))

# components.py

def display_stock_card_content(row, closes, full_stock_data):
    """
    4つのブロック構成:
    1. 銘柄属性 | 2. テクニカル分析 | 3. バリュエーション | 4. 財務・業績
    """
    symbol = row['code']
    
    # --- 1. 銘柄属性ブロック ---
    st.markdown(f"#### 🏢 銘柄情報: {row['name']} ({symbol})")
    st.write(f"**市場:** {row['class']}  |  **業種:** {row['industry']}")
    st.divider()

    # --- 2. テクニカル分析ブロック (CV, 騰落率, 出来高) ---
    st.markdown("#### ⚡ テクニカル分析")
    col_cv, col_perf, col_vol = st.columns([1, 2, 2])
    
    with col_cv:
        st.write("**安定性指標**")
        st.metric("CV値", f"{st.session_state.cv_list.get(symbol, 0):.4f}")

    # 株価・出来高データの取得
    s_close = closes[symbol].dropna()
    s_vol = full_stock_data['Volume'][symbol].dropna() if 'Volume' in full_stock_data else pd.Series()
    
    with col_perf:
        st.write("**株価騰落率**")
        if len(s_close) >= 61:
            c5 = (s_close.iloc[-1] - s_close.iloc[-6]) / s_close.iloc[-6] * 100
            c20 = (s_close.iloc[-1] - s_close.iloc[-21]) / s_close.iloc[-21] * 100
            c60 = (s_close.iloc[-1] - s_close.iloc[-61]) / s_close.iloc[-61] * 100
            
            p1, p2, p3 = st.columns(3)
            p1.metric("5日", f"{c5:+.2f}%")
            p2.metric("20日", f"{c20:+.2f}%")
            p3.metric("60日", f"{c60:+.2f}%")

    with col_vol:
        st.write("**出来高変化 (5日平均比)**")
        if len(s_vol) >= 61:
            v5_avg = s_vol.iloc[-5:].mean()
            v20_avg = s_vol.iloc[-20:].mean()
            v60_avg = s_vol.iloc[-60:].mean()
            
            v_ratio_20 = (v5_avg / v20_avg - 1) * 100 if v20_avg > 0 else 0
            v_ratio_60 = (v5_avg / v60_avg - 1) * 100 if v60_avg > 0 else 0
            
            v1, v2 = st.columns(2)
            # 出来高急増は赤（注目）、減少は緑などで表現
            v1.metric("vs 20日平均", f"{v_ratio_20:+.1f}%", help="直近5日間の平均出来高が20日平均よりどれだけ増減したか")
            v2.metric("vs 60日平均", f"{v_ratio_60:+.1f}%")
    
    st.divider()

    # --- 3. バリュエーション分析ブロック (既存の横並び構成) ---
    st.markdown("#### 💰 バリュエーション分析")
    col_v_num, col_v_graph = st.columns([1, 2])
    with col_v_num:
        st.write("**【PER】**")
        st.write(f"銘柄: {row['PeRatio']:.1f} 倍 / 業種: {row['Average_PER']:.1f} 倍")
        st.write(f"判定: **{row['PER_判定']}**")
        st.write("**【PBR】**")
        st.write(f"銘柄: {row['PbRatio']:.1f} 倍 / 業種: {row['Average_PBR']:.1f} 倍")
        st.write(f"判定: **{row['PBR_判定']}**")
    with col_v_graph:
        display_valuation_chart(row)

    st.divider()

    # --- 4. 財務分析ブロック (既存の横並び構成) ---
    st.markdown("#### 📈 財務・業績分析")
    col_f_num, col_f_graph = st.columns([1, 2])
    with col_f_num:
        st.write(f"財務トレンド: {'✅ 3期連続増収増益' if row['財務トレンド'] else '⚠️ 未達'}")
        if row.get('financial_years'):
            for y, r in zip(row['financial_years'], row['financial_revenue']):
                st.caption(f"{y}期 売上: {r:,.0f}")
    with col_f_graph:
        display_financial_trends_chart(row)

    # --- 5. チャートブロック（常にインライン表示）---
    st.divider()
    st.markdown("#### 📈 株価チャート")
    if full_stock_data is not None and symbol in full_stock_data.get('Close', pd.DataFrame()).columns:
        fig = create_stock_chart(symbol, full_stock_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("株価データが読み込まれていません。STEP 1 でデータを選択してください。")