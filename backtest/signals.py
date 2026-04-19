# signals.py: エントリー・買い増し・決済シグナルの計算

import pandas as pd
import numpy as np


def check_entry_signal(price_series: pd.Series, iloc: int, d: int, i: int) -> bool:
    """
    d日ごとにi回連続上昇しているかを判定する（エントリーシグナル）。

    Parameters
    ----------
    price_series : 終値の時系列（全期間）
    iloc         : 判定基準日の整数インデックス位置
    d            : 比較間隔（営業日数）
    i            : 連続上昇回数

    Returns
    -------
    bool : シグナルが立つ場合 True
    """
    required_start = iloc - i * d
    if required_start < 0:
        return False

    # prices[0] = 直近, prices[1] = d日前, ..., prices[i] = i*d日前
    prices = []
    for k in range(i + 1):
        val = price_series.iloc[iloc - k * d]
        if pd.isna(val) or val <= 0:
            return False
        prices.append(float(val))

    # 連続上昇チェック: prices[0] > prices[1] > ... > prices[i]
    for j in range(i):
        if prices[j] <= prices[j + 1]:
            return False

    return True


def check_addon_signal(
    current_price: float,
    initial_entry_price: float,
    add_x: float
) -> bool:
    """
    買い増し条件を判定する。

    現在価格が初回エントリー価格の +add_x% を上回っていれば True。

    Parameters
    ----------
    current_price        : 現在の株価
    initial_entry_price  : 初回エントリー時の株価
    add_x                : 買い増し閾値（%）
    """
    if initial_entry_price <= 0:
        return False
    return current_price > initial_entry_price * (1 + add_x / 100)


def compute_stock_metrics(
    close_series: pd.Series,
    open_series: pd.Series,
    volume_series: pd.Series,
    date_iloc: int,
    lookback_return: int = 20,
    lookback_candles: int = 10,
) -> dict:
    """
    スワップ判定に使うスコア指標を計算する。

    Parameters
    ----------
    close_series    : 終値の時系列（全期間）
    open_series     : 始値の時系列（全期間）
    volume_series   : 出来高の時系列（全期間）
    date_iloc       : 判定基準日の整数インデックス位置
    lookback_return : 株価騰落率を計算する参照期間（営業日）
    lookback_candles: 陽線比率を計算する直近営業日数

    Returns
    -------
    dict with keys:
        return_rate   : 過去 lookback_return 日間の株価騰落率（%）
        bullish_ratio : 直近 lookback_candles 日間の陽線比率（0.0〜1.0）
        trading_value : 過去 lookback_return 日間の平均日次売買代金（円）
    戻せない場合はすべて None
    """
    if date_iloc < 1:
        return {'return_rate': None, 'bullish_ratio': None, 'trading_value': None}

    # --- ③ 株価騰落率 ---
    cur_close = close_series.iloc[date_iloc]
    if pd.isna(cur_close) or cur_close <= 0:
        return {'return_rate': None, 'bullish_ratio': None, 'trading_value': None}

    start_iloc = max(0, date_iloc - lookback_return)
    past_close = close_series.iloc[start_iloc]
    return_rate = None
    if not pd.isna(past_close) and past_close > 0:
        return_rate = (float(cur_close) - float(past_close)) / float(past_close) * 100

    # --- ④ 陽線比率 ---
    bullish_ratio = None
    if open_series is not None:
        candle_start = max(0, date_iloc - lookback_candles + 1)
        c_slice = close_series.iloc[candle_start: date_iloc + 1]
        o_slice = open_series.iloc[candle_start: date_iloc + 1]
        valid_mask = c_slice.notna() & o_slice.notna() & (c_slice > 0) & (o_slice > 0)
        n_valid = valid_mask.sum()
        if n_valid > 0:
            n_bullish = ((c_slice[valid_mask].values) > (o_slice[valid_mask].values)).sum()
            bullish_ratio = float(n_bullish) / float(n_valid)

    # --- ⑤ 平均売買代金 ---
    trading_value = None
    if volume_series is not None:
        tv_start = max(0, date_iloc - lookback_return + 1)
        c_sl = close_series.iloc[tv_start: date_iloc + 1]
        v_sl = volume_series.iloc[tv_start: date_iloc + 1]
        tv_sl = c_sl * v_sl
        valid_tv = tv_sl[tv_sl.notna() & (tv_sl > 0)]
        if len(valid_tv) > 0:
            trading_value = float(valid_tv.mean())

    return {
        'return_rate'  : return_rate,
        'bullish_ratio': bullish_ratio,
        'trading_value': trading_value,
    }


def check_exit_signal(
    current_price: float,
    initial_entry_price: float,
    exit_x: float
) -> bool:
    """
    決済条件（損切り）を判定する。

    現在価格が初回エントリー価格の -exit_x% を下回っていれば True。

    Parameters
    ----------
    current_price        : 現在の株価
    initial_entry_price  : 初回エントリー時の株価
    exit_x               : 損切り閾値（%）
    """
    if initial_entry_price <= 0:
        return False
    return current_price < initial_entry_price * (1 - exit_x / 100)
