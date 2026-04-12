import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import logic       # 自作モジュール
import components  # 自作モジュール

# --- 1. 設定値 ---
CONFIG = {
    "PATH_META": "meta_data.xls",
    "PATH_INDUSTRY": "perpbr202511_編集済み.csv"
}

# --- 2. 初期設定・データロード ---
st.set_page_config(layout="wide", page_title="Stock Screener Pro")

@st.cache_data
def load_base_data():
    """
    起動時に読み込む静的データ（銘柄メタデータ・業種PER）のみ。
    株価時系列データはSTEP0/STEP1で選択して読み込む。
    """
    # A. 銘柄メタデータ
    meta = pd.read_excel(CONFIG["PATH_META"])
    meta = meta.drop(["日付", "33業種コード", "17業種コード", "17業種区分", "規模区分"], axis=1)
    meta = meta[meta["33業種区分"] != "-"]

    market_map = {
        'プライム（内国株式）':'プライム', 'グロース（内国株式）':'グロース',
        'PRO Market':'PROMarket', 'スタンダード（内国株式）':'スタンダード',
        'プライム（外国株式）':'プライム', 'スタンダード（外国株式）':'スタンダード',
        'グロース（外国株式）':'グロース'
    }
    meta["市場・商品区分"] = meta["市場・商品区分"].map(market_map)
    meta.columns = ['code', 'name', 'class', 'industry', 'size_code']
    meta['code'] = meta['code'].astype(str) + ".T"

    # B. 業種PERデータ
    industry = pd.read_csv(CONFIG["PATH_INDUSTRY"], encoding='cp932')
    market_mapping = {'Prime': 'プライム', 'Growth': 'グロース', 'Standard': 'スタンダード', 'PRO Market': 'PROMarket'}
    industry['kubun'] = industry['kubun'].map(market_mapping)
    industry['Average_PER'] = pd.to_numeric(industry['Average_PER'], errors='coerce')
    industry['Average_PBR'] = pd.to_numeric(industry['Average_PBR'], errors='coerce')
    industry = industry.rename(columns={'kubun': 'class'})

    return meta, industry

# 静的データロード
stock_meta, industry_per = load_base_data()
symbols = stock_meta['code'].tolist()

# --- 3. セッション状態の初期化 ---
if "phase" not in st.session_state:
    st.session_state.update({
        "phase": "ready",
        "primary_hits": [],
        "cv_list": None,
        "valuation_results": pd.DataFrame(),
        "selected_financial_timestamp": None,
        # 株価時系列データ（STEP0/STEP1で読み込む）
        "full_stock_data": None,
        "closes": None,
        "selected_stock_timestamp": None,
    })

# --- 4. メインUI構築 ---
st.title("📈 Stock Screener Pro")

tab0, tab1, tab2 = st.tabs(["STEP 0: データ取得", "STEP 1: テクニカル分析", "STEP 2: 財務・バリュエーション"])

# ==========================================
# STEP 0: データの一括取得（株価 + 財務）
# ==========================================
with tab0:

    # ── 株価時系列データ ──────────────────────────
    st.subheader("📊 株価時系列データの取得")
    st.write(
        f"yfinance を使って全 **{len(symbols)}** 銘柄の株価時系列データを取得し、CSVに保存します。"
        "保存済みのデータは STEP 1 で利用できます。"
    )

    # 保存済みデータ一覧
    st.markdown("#### 保存済みデータ一覧")
    saved_stock_datasets = logic.list_saved_stock_datasets()
    if saved_stock_datasets:
        stock_list_df = pd.DataFrame(saved_stock_datasets)[["label", "size_mb"]]
        stock_list_df.columns = ["取得日時", "ファイルサイズ (MB)"]
        st.dataframe(stock_list_df, use_container_width=True, hide_index=True)
    else:
        st.info("保存済みデータはありません。")

    st.divider()

    # 取得期間の指定
    col_s, col_e = st.columns(2)
    with col_s:
        stock_fetch_start = st.date_input("取得開始日", value=date(2025, 1, 1), key="step0_stock_start")
    with col_e:
        stock_fetch_end = st.date_input("取得終了日", value=date.today(), key="step0_stock_end")

    if st.button("📥 全銘柄の株価データを取得", type="primary", key="btn_fetch_stock"):
        if stock_fetch_start >= stock_fetch_end:
            st.error("開始日は終了日より前にしてください。")
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
            status_text = st.empty()
            progress_bar = st.progress(0)

            with st.spinner("yfinance からデータを取得中..."):
                stock_df = logic.fetch_all_stock_prices(
                    symbols,
                    stock_fetch_start,
                    stock_fetch_end,
                    batch_size=300,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

            if stock_df.empty:
                st.error("データを取得できませんでした。ネットワーク接続を確認してください。")
            else:
                logic.save_stock_data(stock_df, timestamp_str)
                status_text.empty()
                progress_bar.empty()
                # 取得できた銘柄数を表示
                if isinstance(stock_df.columns, pd.MultiIndex):
                    n_tickers = len(stock_df.columns.get_level_values(1).unique())
                else:
                    n_tickers = len(stock_df.columns)
                dt_label = datetime.strptime(timestamp_str, "%Y%m%d%H%M").strftime("%Y/%m/%d %H:%M")
                st.success(
                    f"取得完了！ {n_tickers} 銘柄分のデータを保存しました。"
                    f"（取得日時: {dt_label}）"
                )
                st.rerun()

    st.divider()

    # ── 財務データ ────────────────────────────────
    st.subheader("💹 財務データの取得")
    st.write(
        f"Yahoo Finance API から全 **{len(symbols)}** 銘柄の財務データを取得し、CSVに保存します。"
        "保存済みのデータは STEP 2 で利用できます。"
    )

    # 保存済みデータ一覧
    st.markdown("#### 保存済みデータ一覧")
    saved_fin_datasets = logic.list_saved_financial_datasets()
    if saved_fin_datasets:
        fin_list_df = pd.DataFrame(saved_fin_datasets)[["label", "income_rows", "valuation_rows"]]
        fin_list_df.columns = ["取得日時", "損益計算書（行数）", "バリュエーション（行数）"]
        st.dataframe(fin_list_df, use_container_width=True, hide_index=True)
    else:
        st.info("保存済みデータはありません。")

    st.divider()

    if st.button("📥 全銘柄の財務データを取得", type="primary", key="btn_fetch_financial"):
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
        status_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Yahoo Finance API からデータを取得中..."):
            income_df, valuation_df = logic.fetch_all_financial_data(
                symbols,
                batch_size=200,
                progress_bar=progress_bar,
                status_text=status_text,
            )

        if income_df.empty and valuation_df.empty:
            st.error("データを取得できませんでした。ネットワーク接続を確認してください。")
        else:
            logic.save_financial_data(income_df, valuation_df, timestamp_str)
            status_text.empty()
            progress_bar.empty()
            dt_label = datetime.strptime(timestamp_str, "%Y%m%d%H%M").strftime("%Y/%m/%d %H:%M")
            st.success(
                f"取得完了！ 損益計算書: {len(income_df)} 件、"
                f"バリュエーション: {len(valuation_df)} 件を保存しました。"
                f"（取得日時: {dt_label}）"
            )
            st.rerun()


# ==========================================
# STEP 1: テクニカルスクリーニング
# ==========================================
with tab1:

    # ── データソース選択 ──────────────────────────
    st.subheader("1. 株価データの選択")

    saved_stock_datasets = logic.list_saved_stock_datasets()
    has_saved_stock = len(saved_stock_datasets) > 0

    stock_source_options = ["Yahoo Finance (yfinance) からリアルタイム取得"]
    if has_saved_stock:
        stock_source_options.append("保存済みデータを使用")

    stock_data_source = st.radio(
        "株価データソース",
        options=stock_source_options,
        index=1 if has_saved_stock else 0,   # 保存済みがあればデフォルトで選択
        label_visibility="collapsed",
    )

    # 現在ロード済みのデータを表示
    if st.session_state.selected_stock_timestamp:
        dt = datetime.strptime(st.session_state.selected_stock_timestamp, "%Y%m%d%H%M")
        st.caption(f"✅ 現在のデータ: {dt.strftime('%Y/%m/%d %H:%M')} 取得分")

    st.divider()

    use_saved_stock = stock_data_source == "保存済みデータを使用"

    if not use_saved_stock:
        # ── リアルタイム取得 ──
        col_s1, col_e1 = st.columns(2)
        with col_s1:
            s1_start = st.date_input("開始日", value=date(2025, 1, 1), key="step1_start")
        with col_e1:
            s1_end = st.date_input("終了日", value=date.today(), key="step1_end")

        if st.button("📥 株価データを取得して STEP 1 で使用", type="primary", key="btn_step1_fetch"):
            if s1_start >= s1_end:
                st.error("開始日は終了日より前にしてください。")
            else:
                timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
                status_text = st.empty()
                progress_bar = st.progress(0)

                stock_df = logic.fetch_all_stock_prices(
                    symbols, s1_start, s1_end,
                    batch_size=300,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

                if stock_df.empty:
                    st.error("データを取得できませんでした。")
                else:
                    logic.save_stock_data(stock_df, timestamp_str)
                    closes = stock_df['Close'].apply(pd.to_numeric, errors='coerce')
                    closes.columns = [str(c).strip() for c in closes.columns]
                    st.session_state.full_stock_data = stock_df
                    st.session_state.closes = closes
                    st.session_state.selected_stock_timestamp = timestamp_str
                    # データ変更時はスクリーニング結果をリセット
                    st.session_state.phase = "ready"
                    st.session_state.primary_hits = []
                    st.session_state.valuation_results = pd.DataFrame()
                    status_text.empty()
                    progress_bar.empty()
                    st.success("取得完了！下のスクリーニングを実行できます。")
                    st.rerun()

    else:
        # ── 保存済みデータを使用 ──
        dataset_labels = [d["label"] for d in saved_stock_datasets]
        selected_stock_label = st.selectbox(
            "使用するデータセットを選択",
            options=dataset_labels,
            key="step1_stock_select",
        )
        selected_stock_ds = next(d for d in saved_stock_datasets if d["label"] == selected_stock_label)
        st.caption(f"ファイルサイズ: {selected_stock_ds['size_mb']} MB")

        if st.button("📂 このデータを読み込む", type="primary", key="btn_step1_load"):
            with st.spinner("データを読み込み中..."):
                stock_df = logic.load_stock_data(selected_stock_ds["timestamp"])
                closes = stock_df['Close'].apply(pd.to_numeric, errors='coerce')
                closes.columns = [str(c).strip() for c in closes.columns]
                st.session_state.full_stock_data = stock_df
                st.session_state.closes = closes
                st.session_state.selected_stock_timestamp = selected_stock_ds["timestamp"]
                # データ変更時はスクリーニング結果をリセット
                st.session_state.phase = "ready"
                st.session_state.primary_hits = []
                st.session_state.valuation_results = pd.DataFrame()
            st.success(f"「{selected_stock_label}」のデータを読み込みました。")
            st.rerun()

    # ── スクリーニング（データが読み込まれた場合のみ表示） ──
    if st.session_state.closes is None:
        st.info("⬆️ 上で株価データを取得または選択すると、スクリーニングを実行できます。")
    else:
        closes = st.session_state.closes
        screening_symbols = closes.columns.tolist()

        st.divider()
        st.subheader("2. トレンドと安定性の判定")

        col_input, col_desc = st.columns([1, 1])

        with col_input:
            st.markdown("##### パラメータ設定")
            u_d = st.number_input("d (比較間隔日)", value=5, min_value=1)
            u_i = st.slider("i (継続回数)", min_value=2, max_value=10, value=2)

            use_cv = st.checkbox("CV値による安定性判定を有効にする", value=True)

            if use_cv:
                u_buffer = st.number_input("buffer (最高値判定範囲)", value=15, min_value=1)
                u_N = st.number_input("N (CV計算対象)", value=3, min_value=2)
                cv_threshold = st.slider("CV しきい値", 0.0, 0.1, 0.03, step=0.005, format="%.3f")
            else:
                st.info("CV判定はスキップされ、上昇トレンドのみで抽出します。")

        with col_desc:
            logic_text = f"1. **上昇トレンド:** {u_d}日ごとの株価上昇が {u_i}回 連続しているか。"
            if use_cv:
                logic_text += f"\n2. **安定性(CV):** CV値が {cv_threshold} 以下か。"
            st.info(f"**スクリーニングロジック:**\n\n{logic_text}")

            if st.button("テクニカルスクリーニング実行", type="primary", use_container_width=True):
                with st.spinner("分析中..."):
                    is_hit = pd.Series(True, index=screening_symbols)
                    for j in range(u_i):
                        current_slice = closes.iloc[-1 - (j * u_d)]
                        prev_slice    = closes.iloc[-1 - ((j + 1) * u_d)]
                        is_hit &= (current_slice > prev_slice)

                    cv_list = None
                    if use_cv:
                        cv_list = logic.calculate_dynamic_cv(screening_symbols, closes, u_buffer, u_N)
                        is_hit &= (cv_list <= cv_threshold)
                    else:
                        cv_list = pd.Series(0.0, index=screening_symbols)

                    hit_symbols = is_hit[is_hit].index.tolist()
                    st.session_state.primary_hits = hit_symbols
                    st.session_state.cv_list = cv_list
                    st.session_state.phase = "primary"
                    st.session_state.valuation_results = pd.DataFrame()

                    if hit_symbols:
                        st.success(f"一次判定完了！ {len(hit_symbols)} 銘柄が通過しました。")
                    else:
                        st.error("条件に一致する銘柄が見つかりませんでした。")

        # 一次判定結果の簡易表示
        if st.session_state.phase in ["primary", "final"] and st.session_state.primary_hits:
            st.divider()
            st.write(f"📊 **一次通過リスト ({len(st.session_state.primary_hits)}件):**")
            st.caption("詳細はSTEP 2へ進んでください。")
            st.code(", ".join(st.session_state.primary_hits))


# ==========================================
# STEP 2: 財務・バリュエーション
# ==========================================
with tab2:
    st.subheader("2. 財務健全性と割安度の詳細分析")

    saved_datasets = logic.list_saved_financial_datasets()
    has_saved = len(saved_datasets) > 0
    has_step1 = bool(st.session_state.primary_hits)

    # --- データソース選択 ---
    st.markdown("#### データソースを選択")

    data_source_options = ["Yahoo Finance API からリアルタイム取得"]
    if has_saved:
        data_source_options.append("保存済みデータを使用")

    data_source = st.radio(
        "データソース",
        options=data_source_options,
        index=1 if has_saved else 0,   # 保存済みがあればデフォルトで選択
        label_visibility="collapsed",
    )

    use_saved = data_source == "保存済みデータを使用"

    st.divider()

    # --- APIモード ---
    if not use_saved:
        if not has_step1:
            st.warning("⚠️ まずは [STEP 1] タブでテクニカルスクリーニングを実行してください。")
        else:
            st.write(
                f"一次通過した **{len(st.session_state.primary_hits)}** 銘柄に対し、"
                "Yahoo Finance API を用いて財務データを取得・判定します。"
            )
            if st.button("詳細分析（財務・割安度）を開始", type="primary"):
                with st.spinner("Yahoo API からデータを取得中..."):
                    res = logic.get_valuation_and_trends(
                        st.session_state.primary_hits,
                        industry_per,
                        stock_meta
                    )
                    st.session_state.valuation_results = res
                    st.session_state.phase = "final"

    # --- 保存済みデータモード ---
    else:
        dataset_labels = [d["label"] for d in saved_datasets]
        selected_label = st.selectbox("使用するデータセットを選択", options=dataset_labels)
        selected_dataset = next(d for d in saved_datasets if d["label"] == selected_label)

        # STEP1通過銘柄フィルタ
        filter_by_step1 = st.checkbox(
            "STEP 1 通過銘柄のみを対象にする",
            value=has_step1,
            disabled=not has_step1,
            help="チェックを外すと保存データの全銘柄を対象にします。STEP 1 未実行の場合は全銘柄対象になります。",
        )

        target_symbols = st.session_state.primary_hits if (filter_by_step1 and has_step1) else None
        target_desc = f"STEP 1 通過 {len(st.session_state.primary_hits)} 銘柄" if target_symbols else "保存データの全銘柄"
        st.caption(f"対象: {target_desc} ／ データセット: {selected_label}")

        if st.button("詳細分析（財務・割安度）を開始", type="primary", key="btn_saved_analysis"):
            with st.spinner("保存済みデータを読み込み中..."):
                income_df, valuation_df = logic.load_financial_data(selected_dataset["timestamp"])
                res = logic.process_financial_data(
                    income_df, valuation_df, target_symbols, industry_per, stock_meta
                )
                st.session_state.valuation_results = res
                st.session_state.phase = "final"
                st.session_state.selected_financial_timestamp = selected_dataset["timestamp"]

    # --- 分析完了後の表示ロジック ---
    if st.session_state.phase == "final" and not st.session_state.valuation_results.empty:
        # 株価データが未ロードの場合の警告
        closes = st.session_state.closes
        full_stock_data = st.session_state.full_stock_data
        if closes is None:
            st.warning("⚠️ 株価データが読み込まれていません。STEP 1 で株価データを選択してください。株価・チャートの表示が制限される場合があります。")

        df_filtered = st.session_state.valuation_results.copy()

        with st.expander("🔍 表示条件を絞り込む", expanded=True):
            col_m, col_i = st.columns(2)
            col_pe, col_pb, col_tr = st.columns(3)

            with col_m:
                f_market = st.multiselect("市場", options=df_filtered['class'].unique())
            with col_i:
                f_industry = st.multiselect("業種", options=sorted(df_filtered['industry'].unique()))
            with col_pe:
                f_per = st.multiselect("PER業種比", options=["Under", "Over", "Fair", "N/A"])
            with col_pb:
                f_pbr = st.multiselect("PBR業種比", options=["Under", "Over", "Fair", "N/A"])
            with col_tr:
                f_trend_only = st.checkbox("増収増益（3期連続）のみ表示")

        if f_market:
            df_filtered = df_filtered[df_filtered['class'].isin(f_market)]
        if f_industry:
            df_filtered = df_filtered[df_filtered['industry'].isin(f_industry)]
        if f_per:
            df_filtered = df_filtered[df_filtered['PER_判定'].isin(f_per)]
        if f_pbr:
            df_filtered = df_filtered[df_filtered['PBR_判定'].isin(f_pbr)]
        if f_trend_only:
            df_filtered = df_filtered[df_filtered['財務トレンド'] == True]

        df_filtered['スコア'] = (df_filtered['財務トレンド'] == True) & \
                               ((df_filtered['PER_判定'] == 'Under') | (df_filtered['PBR_判定'] == 'Under'))

        # --- 表示用メトリクスの計算（騰落率・出来高変化率）---
        metrics_df = logic.compute_display_metrics(
            df_filtered['code'].tolist(), closes, full_stock_data
        )
        if not metrics_df.empty:
            df_filtered = df_filtered.join(metrics_df, on='code')
        else:
            for col in ['change_5', 'change_20', 'change_60', 'vol_ratio_20', 'vol_ratio_60']:
                df_filtered[col] = float('nan')

        # --- ソート設定（種類 × 順番の2段構成）---
        st.markdown("#### 🔽 ソート")
        SORT_METRICS = {
            "デフォルト（財務スコア順）": None,
            "株価騰落率 5日":             "change_5",
            "株価騰落率 20日":            "change_20",
            "株価騰落率 60日":            "change_60",
            "出来高変化 vs 20日平均":     "vol_ratio_20",
            "出来高変化 vs 60日平均":     "vol_ratio_60",
        }
        sort_col_s, sort_dir_s = st.columns([2, 1])
        with sort_col_s:
            sort_metric_label = st.selectbox(
                "ソート項目",
                options=list(SORT_METRICS.keys()),
            )
        sort_metric_col = SORT_METRICS[sort_metric_label]

        if sort_metric_col is None:
            # デフォルト：財務スコア順（固定）
            final_df = df_filtered.sort_values(["財務トレンド", "スコア"], ascending=[False, False])
        else:
            with sort_dir_s:
                sort_dir = st.radio(
                    "順番",
                    options=["▲ 高い順", "▽ 低い順"],
                    horizontal=True,
                )
            ascending = sort_dir == "▽ 低い順"
            final_df = df_filtered.sort_values(sort_metric_col, ascending=ascending, na_position='last')

        total_filtered = len(final_df)

        if final_df.empty:
            st.warning("条件に一致する銘柄がありません。フィルターを緩めてください。")
        else:
            col_count, _ = st.columns([1, 3])
            with col_count:
                items_per_page = st.selectbox(
                    "1ページあたりの表示件数",
                    options=[10, 20, 50, 100],
                    index=0,
                    key="items_per_page"
                )

            if "list_start_idx" not in st.session_state:
                st.session_state.list_start_idx = 0

            start_idx = min(st.session_state.list_start_idx, max(0, total_filtered - 1))
            end_idx = min(start_idx + items_per_page, total_filtered)

            current_page = start_idx // items_per_page + 1
            total_pages = max(1, (total_filtered - 1) // items_per_page + 1)

            st.write(f"✅ 表示件数: {start_idx + 1}件 〜 {end_idx}件 / 全 {total_filtered} 件")

            def set_start(idx):
                st.session_state.list_start_idx = max(0, min(idx, total_filtered - 1))

            if total_pages > 1:
                st.write("")

                max_page_buttons = 5
                start_page = max(1, current_page - 2)
                end_page = min(total_pages, start_page + max_page_buttons - 1)

                if end_page - start_page + 1 < max_page_buttons:
                    start_page = max(1, end_page - max_page_buttons + 1)

                page_numbers = list(range(start_page, end_page + 1))
                cols = st.columns([2] + [1] * (len(page_numbers) + 2) + [2])

                with cols[1]:
                    st.button("前へ", key="prev_btn", disabled=(current_page == 1),
                              on_click=set_start, args=(start_idx - items_per_page,))

                for i, p in enumerate(page_numbers):
                    with cols[i + 2]:
                        btn_type = "primary" if p == current_page else "secondary"
                        st.button(str(p), key=f"page_btn_{p}", type=btn_type,
                                  on_click=set_start, args=((p - 1) * items_per_page,))

                with cols[len(page_numbers) + 2]:
                    st.button("次へ", key="next_btn", disabled=(current_page == total_pages),
                              on_click=set_start, args=(start_idx + items_per_page,))

                st.markdown(
                    f"<p style='text-align: center; color: gray; font-size: 0.9em;'>"
                    f"{current_page}/{total_pages}ページ</p>",
                    unsafe_allow_html=True
                )
                st.divider()

            display_df = final_df.iloc[start_idx:end_idx]

            # --- ヘッダー行 ---
            # 列幅: 銘柄名(2.5) 市場(0.7) 株価(0.8) 5日%(0.65) 20日%(0.65) 60日%(0.65) 出来高/20日(0.85) 出来高/60日(0.85) 詳細(0.55)
            COL_W = [2.5, 0.7, 0.8, 0.65, 0.65, 0.65, 0.85, 0.85, 0.55]
            h = st.columns(COL_W)
            h[0].markdown("**銘柄**")
            h[1].markdown("**市場**")
            h[2].markdown("**株価**")
            h[3].markdown("**5日%**")
            h[4].markdown("**20日%**")
            h[5].markdown("**60日%**")
            h[6].markdown("**出来高/20日**")
            h[7].markdown("**出来高/60日**")
            h[8].markdown("")
            st.divider()

            def _pct_html(val):
                """騰落率・出来高変化率を色付きHTMLで返す"""
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "<span style='color:gray'>N/A</span>"
                sign = "+" if val > 0 else ""
                color = "#2ecc71" if val > 0 else "#e74c3c" if val < 0 else "gray"
                return f"<span style='color:{color};font-weight:bold'>{sign}{val:.1f}%</span>"

            for _, row in display_df.iterrows():
                code = row['code']
                price = (
                    closes[code].dropna().iloc[-1]
                    if (closes is not None and code in closes.columns)
                    else None
                )
                price_str = f"¥{price:,.0f}" if price is not None else "N/A"

                c = st.columns(COL_W)
                c[0].markdown(f"**{code}**<br><small>{row['name']}</small>", unsafe_allow_html=True)
                c[1].write(row['class'])
                c[2].write(price_str)
                c[3].markdown(_pct_html(row.get('change_5')),    unsafe_allow_html=True)
                c[4].markdown(_pct_html(row.get('change_20')),   unsafe_allow_html=True)
                c[5].markdown(_pct_html(row.get('change_60')),   unsafe_allow_html=True)
                c[6].markdown(_pct_html(row.get('vol_ratio_20')), unsafe_allow_html=True)
                c[7].markdown(_pct_html(row.get('vol_ratio_60')), unsafe_allow_html=True)
                if c[8].button("詳細", key=f"detail_{code}", use_container_width=True):
                    components.show_stock_detail_dialog(row, closes, full_stock_data)
                st.divider()
