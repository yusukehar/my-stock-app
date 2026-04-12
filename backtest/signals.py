# signals.py: エントリー・買い増し・決済シグナルの計算

import pandas as pd


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
