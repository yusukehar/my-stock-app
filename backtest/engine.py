# engine.py: 日次シミュレーションエンジン・ポジション管理

import math
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Dict

from signals import check_entry_signal

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================
# データクラス
# ============================================================

@dataclass
class Lot:
    """1ロット分のエントリー情報"""
    entry_date: pd.Timestamp
    entry_price: float
    amount: float        # 投資金額（円）

    @property
    def shares(self) -> float:
        """保有株数（小数OK・売買単位は考慮しない）"""
        return self.amount / self.entry_price


@dataclass
class Position:
    """1銘柄のポジション全体"""
    symbol: str
    initial_entry_price: float      # 初回エントリー価格（損切り判定基準）
    initial_entry_date: pd.Timestamp
    lots: List[Lot] = field(default_factory=list)

    # 最終エントリー情報（買い増し・初回利確の判定に使用）
    last_entry_price: float = 0.0
    last_entry_date: pd.Timestamp = None

    # 利確管理
    tp_count: int = 0               # 0: TP1待ち  /  1: TP1済み・TP2待ち
    tp1_price: float = 0.0          # 初回利確時の株価（TP2の -Y% 基準）
    tp1_check_start: pd.Timestamp = None   # この日以降 TP1 チェック開始

    # 損切り管理
    stop_check_start: pd.Timestamp = None  # この日以降 損切りチェック開始（猶予期間）

    # 買い増しタイミング
    next_addon_check: pd.Timestamp = None

    @property
    def total_invested(self) -> float:
        return sum(lot.amount for lot in self.lots)


# ============================================================
# 内部ユーティリティ
# ============================================================

def _close_lot(trades: list, symbol: str, lot: Lot,
               exit_date: pd.Timestamp, exit_price: float,
               exit_reason: str,
               initial_entry_date: pd.Timestamp = None) -> float:
    """
    1ロットを決済し、取引履歴に追加する。
    initial_entry_date: ポジション全体の初回エントリー日（集計キーとして使用）
    戻り値: 回収金額（proceeds）= 株数 × 決済価格
    """
    proceeds = lot.shares * exit_price
    pnl      = lot.shares * (exit_price - lot.entry_price)
    pnl_pct  = (exit_price - lot.entry_price) / lot.entry_price * 100
    trades.append({
        'symbol'             : symbol,
        'initial_entry_date' : initial_entry_date if initial_entry_date is not None else lot.entry_date,
        'entry_date'         : lot.entry_date,
        'exit_date'          : exit_date,
        'entry_price'        : round(lot.entry_price, 2),
        'exit_price'         : round(exit_price, 2),
        'amount'             : lot.amount,
        'proceeds'           : round(proceeds, 0),
        'pnl'                : round(pnl, 0),
        'pnl_pct'            : round(pnl_pct, 2),
        'exit_reason'        : exit_reason,
    })
    return proceeds


def _close_all_lots(trades, symbol, pos, date, price, reason):
    """全ロットを決済し、合計回収金額と合計損益を返す"""
    total_proceeds = 0.0
    total_pnl      = 0.0
    for lot in pos.lots:
        total_proceeds += _close_lot(trades, symbol, lot, date, price, reason,
                                     pos.initial_entry_date)
        total_pnl      += lot.shares * (price - lot.entry_price)
    return total_proceeds, total_pnl


def _calc_avg_cost(lots) -> float:
    """全ロットの加重平均購入単価を返す"""
    total_shares = sum(lot.shares for lot in lots)
    if total_shares <= 0:
        return 0.0
    return sum(lot.amount for lot in lots) / total_shares


def _sell_txn(symbol, date, price, lots_to_sell, avg_cost, tx_type) -> dict:
    """売り取引レコードを生成する（transactions リスト用）"""
    shares   = sum(lot.shares for lot in lots_to_sell)
    proceeds = round(shares * price, 0)
    cost     = shares * avg_cost
    pnl      = round(proceeds - cost, 0)
    pnl_pct  = round(pnl / cost * 100, 2) if cost > 0 else 0.0
    return {
        'date'     : date,
        'symbol'   : symbol,
        'type'     : tx_type,
        'price'    : round(price, 2),
        'avg_cost' : round(avg_cost, 2),
        'shares'   : round(shares, 4),
        'amount'   : None,
        'proceeds' : proceeds,
        'pnl'      : pnl,
        'pnl_pct'  : pnl_pct,
    }


# ============================================================
# バックテスト実行
# ============================================================

def run_backtest(
    closes: pd.DataFrame,
    params: dict,
    start_date: str,
    end_date: str,
    daily_universe: Dict[pd.Timestamp, frozenset] = None,
) -> tuple:
    """
    バックテストのメイン実行関数。

    Parameters
    ----------
    closes         : 終値DataFrame（行=日付, 列=銘柄コード）
    params         : パラメータ辞書
    start_date     : バックテスト開始日（'YYYY-MM-DD'）
    end_date       : バックテスト終了日（'YYYY-MM-DD'）
    daily_universe : 日付→その日のuniverse銘柄セット。None の場合は全銘柄対象。

    Returns
    -------
    trades        : list[dict] 取引履歴（1ロット決済ごとに1レコード）
    daily_records : list[dict] 日次ポートフォリオ状況
    """
    d                    = params['d']
    i                    = params['i']
    add_n                = params['add_n']
    add_x                = params['add_x']
    tp1_n                = params['tp1_n']
    tp1_x                = params.get('tp1_x', 0.0)   # 初回利確の最低利益率（%）
    tp2_y                = params['tp2_y']
    stop_x               = params['stop_x']
    stop_n               = params['stop_n']
    lot_amount           = params['lot_amount']
    max_amount_per_stock = params['max_amount_per_stock']
    total_budget         = params['total_budget']
    max_positions        = params.get('max_positions', None)  # 最大保有銘柄数（None=制限なし）

    remaining_budget = float(total_budget)

    all_dates    = closes.index
    symbols      = closes.columns.tolist()
    start_ts     = pd.Timestamp(start_date)
    end_ts       = pd.Timestamp(end_date)
    target_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]

    if len(target_dates) == 0:
        print("対象期間のデータがありません。START_DATE / END_DATE を確認してください。")
        return [], []

    date_to_iloc: Dict[pd.Timestamp, int] = {
        d_: idx for idx, d_ in enumerate(all_dates)
    }

    positions: Dict[str, Position] = {}
    trades         = []
    transactions   = []   # 全売買アクション（買い・売り）を時系列で記録
    daily_records  = []
    cumulative_pnl = 0.0

    iterator = (
        tqdm(target_dates, desc="バックテスト実行中", ncols=80)
        if TQDM_AVAILABLE else target_dates
    )

    for date in iterator:
        date_iloc      = date_to_iloc[date]
        realized_today = 0.0

        # ============================================================
        # 1. 既存ポジションの処理
        #    優先順位: 損切り > TP2 > TP1 > 買い増し
        # ============================================================
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            raw = closes[symbol].iloc[date_iloc]
            if pd.isna(raw) or raw <= 0:
                continue
            price = float(raw)

            # ---- 損切り（最優先・全額即時決済、猶予期間経過後のみ）----
            if date >= pos.stop_check_start and price < pos.initial_entry_price * (1 - stop_x / 100):
                transactions.append(_sell_txn(symbol, date, price, pos.lots,
                                              _calc_avg_cost(pos.lots), '損切り'))
                proceeds, pnl = _close_all_lots(trades, symbol, pos, date, price, '損切り')
                realized_today   += pnl
                remaining_budget += proceeds
                del positions[symbol]
                continue

            # ---- TP2（tp_count==1 のとき毎日チェック）----
            if pos.tp_count == 1:
                if price < pos.tp1_price * (1 - tp2_y / 100):
                    transactions.append(_sell_txn(symbol, date, price, pos.lots,
                                                  _calc_avg_cost(pos.lots), '2回目利確'))
                    proceeds, pnl = _close_all_lots(trades, symbol, pos, date, price, '2回目利確')
                    realized_today   += pnl
                    remaining_budget += proceeds
                    del positions[symbol]
                    continue

            # ---- TP1（tp_count==0 かつ tp1_n日経過後、毎日チェック）----
            # 発動条件:
            #   ① 現在価格 < 最終エントリー価格（モメンタム喪失）
            #   ② 現在価格 >= 初回エントリー価格 × (1 + tp1_x%) （最低利益ライン超え）
            skip_addon = False
            if pos.tp_count == 0 and date >= pos.tp1_check_start:
                tp1_threshold = pos.initial_entry_price * (1 + tp1_x / 100)
                if price < pos.last_entry_price and price >= tp1_threshold:
                    # 金額ベース切り上げ → ceil(lot数/2) 本を FIFO で決済
                    n_exit = math.ceil(len(pos.lots) / 2)
                    avg_cost = _calc_avg_cost(pos.lots)
                    transactions.append(_sell_txn(symbol, date, price,
                                                  pos.lots[:n_exit], avg_cost, '初回利確'))
                    for _ in range(n_exit):
                        if pos.lots:
                            lot      = pos.lots.pop(0)
                            proceeds = _close_lot(trades, symbol, lot, date, price, '初回利確',
                                                  pos.initial_entry_date)
                            realized_today   += lot.shares * (price - lot.entry_price)
                            remaining_budget += proceeds

                    if not pos.lots:
                        # 全ロット決済済み（1ロットのみ保有だった場合など）
                        del positions[symbol]
                        continue

                    pos.tp_count        = 1
                    pos.tp1_price       = price
                    skip_addon          = True   # 同日の買い増しはスキップ

            # ---- 買い増しチェック ----
            if not skip_addon and symbol in positions:
                if date >= pos.next_addon_check:
                    can_add = (
                        pos.total_invested + lot_amount <= max_amount_per_stock
                        and remaining_budget >= lot_amount
                    )
                    if can_add:
                        if pos.tp_count == 0:
                            # 通常買い増し: 最終エントリー価格の +add_x% 超
                            if price > pos.last_entry_price * (1 + add_x / 100):
                                pos.lots.append(Lot(date, price, lot_amount))
                                pos.last_entry_price  = price
                                pos.last_entry_date   = date
                                pos.tp1_check_start   = date + timedelta(days=tp1_n)
                                pos.next_addon_check  = date + timedelta(days=add_n)  # 買い増し後はN日待機
                                remaining_budget      -= lot_amount
                                transactions.append({
                                    'date': date, 'symbol': symbol, 'type': '買い増し',
                                    'price': round(price, 2), 'avg_cost': None,
                                    'shares': round(lot_amount / price, 4),
                                    'amount': lot_amount, 'proceeds': None,
                                    'pnl': None, 'pnl_pct': None,
                                })

                        elif pos.tp_count == 1:
                            # 利確後買い増し: 最終エントリー価格の +add_x% 超 → リセット
                            if price > pos.last_entry_price * (1 + add_x / 100):
                                pos.lots.append(Lot(date, price, lot_amount))
                                pos.last_entry_price  = price
                                pos.last_entry_date   = date
                                pos.tp1_check_start   = date + timedelta(days=tp1_n)
                                pos.tp_count          = 0     # 利確カウントリセット
                                pos.tp1_price         = 0.0
                                pos.next_addon_check  = date + timedelta(days=add_n)  # 買い増し後はN日待機
                                remaining_budget      -= lot_amount
                                transactions.append({
                                    'date': date, 'symbol': symbol, 'type': '買い増し',
                                    'price': round(price, 2), 'avg_cost': None,
                                    'shares': round(lot_amount / price, 4),
                                    'amount': lot_amount, 'proceeds': None,
                                    'pnl': None, 'pnl_pct': None,
                                })

                    # 買い増し未成立の場合は翌日も毎日チェック（N日待機は不要）
                    if date >= pos.next_addon_check:
                        pos.next_addon_check = date + timedelta(days=1)

        # ============================================================
        # 2. 新規エントリー判定
        # ============================================================
        symbols_today = daily_universe.get(date, frozenset()) if daily_universe is not None else symbols
        for symbol in symbols_today:
            _dbg = (symbol == '285A.T')
            if _dbg: print(f"[DEBUG] {date} 285A.T ループ到達")

            if symbol in positions:
                if _dbg: print(f"[DEBUG] {date} 285A.T → スキップ: 既にポジションあり")
                continue
            if remaining_budget < lot_amount:
                if _dbg: print(f"[DEBUG] {date} 285A.T → スキップ: 予算不足 残{remaining_budget:,.0f}円")
                continue
            if max_positions is not None and len(positions) >= max_positions:
                if _dbg: print(f"[DEBUG] {date} 285A.T → スキップ: max_positions到達({len(positions)}社)")
                break

            raw = closes[symbol].iloc[date_iloc]
            if pd.isna(raw) or raw <= 0:
                if _dbg: print(f"[DEBUG] {date} 285A.T → スキップ: 株価データなし")
                continue
            price = float(raw)

            _signal = check_entry_signal(closes[symbol], date_iloc, d, i)
            if _dbg:
                if _signal: print(f"[DEBUG] {date} 285A.T → シグナル発火！ 価格:{price}")
                else:        print(f"[DEBUG] {date} 285A.T → シグナル不発 価格:{price}")

            if _signal:
                positions[symbol] = Position(
                    symbol              = symbol,
                    initial_entry_price = price,
                    initial_entry_date  = date,
                    lots                = [Lot(date, price, lot_amount)],
                    last_entry_price    = price,
                    last_entry_date     = date,
                    tp_count            = 0,
                    tp1_check_start     = date + timedelta(days=tp1_n),
                    stop_check_start    = date + timedelta(days=stop_n),
                    next_addon_check    = date + timedelta(days=add_n),
                )
                remaining_budget -= lot_amount
                transactions.append({
                    'date': date, 'symbol': symbol, 'type': '新規エントリー',
                    'price': round(price, 2), 'avg_cost': None,
                    'shares': round(lot_amount / price, 4),
                    'amount': lot_amount, 'proceeds': None,
                    'pnl': None, 'pnl_pct': None,
                })

        # ============================================================
        # 3. 日次記録
        # ============================================================
        cumulative_pnl += realized_today

        unrealized_pnl = 0.0
        for sym, pos in positions.items():
            cp = closes[sym].iloc[date_iloc]
            if not pd.isna(cp) and cp > 0:
                for lot in pos.lots:
                    unrealized_pnl += lot.shares * (float(cp) - lot.entry_price)

        daily_records.append({
            'date'                    : date,
            'realized_pnl'            : round(realized_today, 0),
            'cumulative_realized_pnl' : round(cumulative_pnl, 0),
            'unrealized_pnl'          : round(unrealized_pnl, 0),
            'total_pnl'               : round(cumulative_pnl + unrealized_pnl, 0),
            'open_positions'          : len(positions),
            'remaining_budget'        : round(remaining_budget, 0),
        })

    return trades, daily_records, transactions
