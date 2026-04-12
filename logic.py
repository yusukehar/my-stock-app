# logic.py：計算とデータ処理の心臓部

import os
import glob
import time
import pandas as pd
import numpy as np
from datetime import datetime
from yahooquery import Ticker
import warnings

# --- 1. データ加工・計算ロジック ---

def check_max_in_range(series, buffer):
    """前後buffer件の範囲で最高値かどうかを判定"""
    results = []
    vals = series.values
    length = len(series)
    for i in range(length):
        start = max(0, i - buffer)
        end = min(length, i + buffer + 1)
        results.append(vals[i] == vals[start:end].max())
    return results

def calculate_dynamic_cv(symbols, closes, buffer, N_points):
    """変動係数(CV)を計算"""
    is_highest_flag = closes.copy()
    for ticker in symbols:
        is_highest_flag[ticker] = check_max_in_range(closes[ticker], buffer)

    cv_list_result = pd.Series(index=symbols, dtype=float)
    for ticker in symbols:
        sym_close = closes[ticker]
        sym_highest = is_highest_flag[ticker].fillna(False).astype(bool)
        high_day_closes = sym_close[sym_highest].dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if len(high_day_closes) >= 2:
                n_largest = high_day_closes.nlargest(N_points)
                mean_val = n_largest.mean()
                cv_list_result[ticker] = n_largest.std() / mean_val if mean_val != 0 else np.nan
            else:
                cv_list_result[ticker] = np.nan
    return cv_list_result


# --- 2. 株価時系列データの一括取得・保存・読み込み ---

STOCK_DATA_DIR = "stock_data"


def fetch_all_stock_prices(symbols, start_date, end_date, batch_size=300, progress_bar=None, status_text=None):
    """
    全銘柄の株価時系列データをyfinanceでバッチ取得する。
    エラーが出たバッチはスキップし、取得できた分だけ返す。
    """
    import yfinance as yf

    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    n_batches = len(batches)
    all_frames = []

    for i, batch in enumerate(batches):
        if status_text:
            status_text.text(f"株価取得中... バッチ {i + 1}/{n_batches} ({batch[0]} 〜 {batch[-1]})")
        try:
            data = yf.download(
                tickers=batch,
                start=str(start_date),
                end=str(end_date),
                interval="1d",
                progress=False,
                auto_adjust=False,
            )
            if not data.empty:
                all_frames.append(data)
        except Exception:
            pass

        if progress_bar:
            progress_bar.progress((i + 1) / n_batches)

        if i < n_batches - 1:
            time.sleep(1)  # レート制限への配慮

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, axis=1)
    # 重複列を除去（同一銘柄が複数バッチに入った場合など）
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def save_stock_data(df, timestamp_str, save_dir=STOCK_DATA_DIR):
    """株価時系列データをCSVファイルに保存する"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"stock_data_{timestamp_str}.csv")
    df.to_csv(path)


def load_stock_data(timestamp_str, save_dir=STOCK_DATA_DIR):
    """保存済み株価時系列データをCSVから読み込む"""
    path = os.path.join(save_dir, f"stock_data_{timestamp_str}.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


def list_saved_stock_datasets(save_dir=STOCK_DATA_DIR):
    """
    保存済みの株価時系列データ一覧を返す。
    戻り値: [{"timestamp": str, "label": str, "size_mb": float}, ...]
    """
    if not os.path.isdir(save_dir):
        return []

    files = glob.glob(os.path.join(save_dir, "stock_data_*.csv"))
    datasets = []

    for f in files:
        basename = os.path.basename(f)
        ts_str = basename.replace("stock_data_", "").replace(".csv", "")
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M")
            label = dt.strftime("%Y/%m/%d %H:%M")
            size_mb = round(os.path.getsize(f) / (1024 * 1024), 1)
        except Exception:
            continue

        datasets.append({
            "timestamp": ts_str,
            "label": label,
            "size_mb": size_mb,
        })

    datasets.sort(key=lambda x: x["timestamp"], reverse=True)
    return datasets


# --- 3. リスト表示用メトリクスの一括計算 ---

def compute_display_metrics(code_list, closes, full_stock_data):
    """
    銘柄リストの表示・ソート用メトリクス（騰落率・出来高変化率）を一括計算する。
    closes や full_stock_data が None の場合は空のDataFrameを返す。
    戻り値: code をインデックスとした DataFrame
    """
    records = []

    vol_df = None
    if full_stock_data is not None and 'Volume' in full_stock_data:
        vol_df = full_stock_data['Volume']

    for code in code_list:
        row = {
            'code': code,
            'change_5': np.nan,
            'change_20': np.nan,
            'change_60': np.nan,
            'vol_ratio_20': np.nan,
            'vol_ratio_60': np.nan,
        }

        # 株価騰落率
        if closes is not None and code in closes.columns:
            s = closes[code].dropna()
            n = len(s)
            if n >= 6:
                row['change_5']  = (s.iloc[-1] - s.iloc[-6])  / s.iloc[-6]  * 100
            if n >= 21:
                row['change_20'] = (s.iloc[-1] - s.iloc[-21]) / s.iloc[-21] * 100
            if n >= 61:
                row['change_60'] = (s.iloc[-1] - s.iloc[-61]) / s.iloc[-61] * 100

        # 出来高変化率
        if vol_df is not None and code in vol_df.columns:
            sv = vol_df[code].dropna()
            n = len(sv)
            if n >= 5:
                v5 = sv.iloc[-5:].mean()
                if n >= 20:
                    v20 = sv.iloc[-20:].mean()
                    if v20 > 0:
                        row['vol_ratio_20'] = (v5 / v20 - 1) * 100
                if n >= 60:
                    v60 = sv.iloc[-60:].mean()
                    if v60 > 0:
                        row['vol_ratio_60'] = (v5 / v60 - 1) * 100

        records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index('code')


# --- 4. 財務データの一括取得・保存・読み込み ---

FINANCIAL_DATA_DIR = "financial_data"

def fetch_all_financial_data(symbols, batch_size=200, progress_bar=None, status_text=None):
    """
    全銘柄の財務データをバッチ処理でYahoo Finance APIから取得する。
    エラーが出たバッチはスキップし、取得できた分だけ返す。
    """
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    n_batches = len(batches)

    all_income = []
    all_valuation = []

    for i, batch in enumerate(batches):
        if status_text:
            status_text.text(f"取得中... バッチ {i + 1}/{n_batches} ({batch[0]} 〜 {batch[-1]})")

        try:
            tickers = Ticker(batch, asynchronous=True)
            income = tickers.income_statement(frequency='a', trailing=False)
            valuation = tickers.valuation_measures

            if isinstance(income, pd.DataFrame) and not income.empty:
                all_income.append(income)
            if isinstance(valuation, pd.DataFrame) and not valuation.empty:
                all_valuation.append(valuation)
        except Exception:
            pass  # バッチ単位でエラーをスキップ

        if progress_bar:
            progress_bar.progress((i + 1) / n_batches)

    income_df = pd.concat(all_income) if all_income else pd.DataFrame()
    valuation_df = pd.concat(all_valuation) if all_valuation else pd.DataFrame()

    return income_df, valuation_df


def save_financial_data(income_df, valuation_df, timestamp_str, save_dir=FINANCIAL_DATA_DIR):
    """財務データをCSVファイルに保存する"""
    os.makedirs(save_dir, exist_ok=True)
    income_df.to_csv(os.path.join(save_dir, f"income_{timestamp_str}.csv"))
    valuation_df.to_csv(os.path.join(save_dir, f"valuation_{timestamp_str}.csv"))


def load_financial_data(timestamp_str, save_dir=FINANCIAL_DATA_DIR):
    """保存済み財務データをCSVから読み込む"""
    income_df = pd.read_csv(
        os.path.join(save_dir, f"income_{timestamp_str}.csv"),
        index_col=0
    )
    valuation_df = pd.read_csv(
        os.path.join(save_dir, f"valuation_{timestamp_str}.csv"),
        index_col=0
    )
    return income_df, valuation_df


def list_saved_financial_datasets(save_dir=FINANCIAL_DATA_DIR):
    """
    保存済みの財務データセット一覧を返す。
    income_*.csv と valuation_*.csv のペアを抽出し、新しい順にソートする。
    戻り値: [{"timestamp": str, "label": str, "income_rows": int, "valuation_rows": int}, ...]
    """
    if not os.path.isdir(save_dir):
        return []

    income_files = glob.glob(os.path.join(save_dir, "income_*.csv"))
    datasets = []

    for income_path in income_files:
        basename = os.path.basename(income_path)
        # "income_yyyymmddhhmm.csv" からタイムスタンプを抽出
        ts_str = basename.replace("income_", "").replace(".csv", "")
        valuation_path = os.path.join(save_dir, f"valuation_{ts_str}.csv")

        if not os.path.exists(valuation_path):
            continue  # ペアが揃っていない場合はスキップ

        try:
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M")
            label = dt.strftime("%Y/%m/%d %H:%M")
            income_rows = sum(1 for _ in open(income_path)) - 1
            valuation_rows = sum(1 for _ in open(valuation_path)) - 1
        except Exception:
            continue

        datasets.append({
            "timestamp": ts_str,
            "label": label,
            "income_rows": income_rows,
            "valuation_rows": valuation_rows,
        })

    # 新しい順にソート
    datasets.sort(key=lambda x: x["timestamp"], reverse=True)
    return datasets


# --- 3. 財務データ処理ロジック（API/保存済みデータ共通） ---

def process_financial_data(income_df, valuation_df, symbol_list, industry_per_data, stock_meta_data):
    """
    income_df と valuation_df から財務トレンド・バリュエーション判定を行い結果DataFrameを返す。
    symbol_list が None の場合は全銘柄対象。指定がある場合はフィルタリングする。
    """
    if income_df.empty or valuation_df.empty:
        return pd.DataFrame()

    # --- 財務トレンド判定 & グラフ用データ抽出 ---
    trends_dict = {}
    financial_history = {}

    income_work = income_df.copy()

    # reset_index が必要かどうかを判定（symbol列がindexの場合とcolumnの場合に対応）
    if 'symbol' not in income_work.columns:
        income_work = income_work.reset_index()

    # asOfDate を datetime に変換
    if 'asOfDate' in income_work.columns:
        income_work['asOfDate'] = pd.to_datetime(income_work['asOfDate'], errors='coerce')

    # symbol_list でフィルタリング
    if symbol_list is not None and 'symbol' in income_work.columns:
        income_work = income_work[income_work['symbol'].isin(symbol_list)]

    def process_financials(g):
        g = g.sort_values('asOfDate', ascending=True)
        years = g['asOfDate'].dt.year.astype(str).tolist()[-3:]
        revenue = g['TotalRevenue'].tolist()[-3:]
        profit = g['OperatingIncome'].tolist()[-3:]

        g_desc = g.sort_values('asOfDate', ascending=False)
        is_growing = False
        if len(g_desc) >= 3:
            is_growing = (
                g_desc['TotalRevenue'].iloc[0] > g_desc['TotalRevenue'].iloc[1] > g_desc['TotalRevenue'].iloc[2]
            ) and (
                g_desc['OperatingIncome'].iloc[0] > g_desc['OperatingIncome'].iloc[1] > g_desc['OperatingIncome'].iloc[2]
            )
        return {'is_growing': is_growing, 'years': years, 'revenue': revenue, 'profit': profit}

    if 'symbol' in income_work.columns and 'asOfDate' in income_work.columns:
        for sym, group in income_work.groupby('symbol'):
            financial_history[sym] = process_financials(group)
            trends_dict[sym] = financial_history[sym]['is_growing']

    # --- PER/PBRバリュエーション判定 ---
    valuation_work = valuation_df.copy()

    if 'symbol' not in valuation_work.columns:
        valuation_work = valuation_work.reset_index()

    if 'asOfDate' in valuation_work.columns:
        valuation_work['asOfDate'] = pd.to_datetime(valuation_work['asOfDate'], errors='coerce')

    if symbol_list is not None and 'symbol' in valuation_work.columns:
        valuation_work = valuation_work[valuation_work['symbol'].isin(symbol_list)]

    if valuation_work.empty:
        return pd.DataFrame()

    latest_ratios = (
        valuation_work.sort_values(by='asOfDate', ascending=False)
        .groupby('symbol').first().reset_index()
    )

    # MarketCap を含める（列が存在する場合のみ）
    val_cols = ['symbol', 'PeRatio', 'PbRatio']
    if 'MarketCap' in latest_ratios.columns:
        val_cols.append('MarketCap')

    res = pd.merge(
        latest_ratios[val_cols],
        stock_meta_data, left_on='symbol', right_on='code', how='left'
    )
    res = pd.merge(res, industry_per_data, on=['class', 'industry'], how='left')

    def classify(stock_val, avg_val):
        if pd.isna(stock_val) or pd.isna(avg_val) or avg_val == 0:
            return 'N/A'
        if stock_val > avg_val * 1.2:
            return 'Over'
        if stock_val < avg_val / 1.2:
            return 'Under'
        return 'Fair'

    res['PER_判定'] = res.apply(lambda r: classify(r['PeRatio'], r['Average_PER']), axis=1)
    res['PBR_判定'] = res.apply(lambda r: classify(r['PbRatio'], r['Average_PBR']), axis=1)
    res['財務トレンド'] = res['symbol'].map(trends_dict).fillna(False)
    res['financial_years'] = res['symbol'].map(lambda x: financial_history.get(x, {}).get('years', []))
    res['financial_revenue'] = res['symbol'].map(lambda x: financial_history.get(x, {}).get('revenue', []))
    res['financial_profit'] = res['symbol'].map(lambda x: financial_history.get(x, {}).get('profit', []))

    return res


# --- 4. API・外部データ連携ロジック（後方互換ラッパー） ---

def get_valuation_and_trends(symbol_list, industry_per_data, stock_meta_data):
    """財務トレンド、履歴データ、バリュエーション判定を一括取得（後方互換）"""
    if not symbol_list:
        return pd.DataFrame()

    tickers = Ticker(symbol_list, asynchronous=True)
    income_df = tickers.income_statement(frequency='a', trailing=False)
    valuation_df = tickers.valuation_measures

    if not isinstance(income_df, pd.DataFrame):
        income_df = pd.DataFrame()
    if not isinstance(valuation_df, pd.DataFrame):
        valuation_df = pd.DataFrame()

    return process_financial_data(income_df, valuation_df, symbol_list, industry_per_data, stock_meta_data)
