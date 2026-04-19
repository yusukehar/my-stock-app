# engine.py: 日次シミュレーションエンジン・ポジション管理

import math
import pandas as pd
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Dict, Optional

from signals import check_entry_signal, compute_stock_metrics

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
    total_proceeds = 0.0
    total_pnl      = 0.0
    for lot in pos.lots:
        total_proceeds += _close_lot(trades, symbol, lot, date, price, reason,
                                     pos.initial_entry_date)
        total_pnl      += lot.shares * (price - lot.entry_price)
    return total_proceeds, total_pnl


def _calc_avg_cost(lots) -> float:
    total_shares = sum(lot.shares for lot in lots)
    if total_shares <= 0:
        return 0.0
    return sum(lot.amount for lot in lots) / total_shares


def _sell_txn(symbol, date, price, lots_to_sell, avg_cost, tx_type) -> dict:
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
# スワップ関連ヘルパー
# ============================================================

def _compute_metrics_for_symbol(
    symbol: str,
    closes: pd.DataFrame,
    opens: Optional[pd.DataFrame],
    volume: Optional[pd.DataFrame],
    date_iloc: int,
    lookback_return: int,
    lookback_candles: int,
) -> dict:
    """1銘柄のメトリクスを計算して返す"""
    open_s   = opens[symbol]  if opens  is not None and symbol in opens.columns  else None
    volume_s = volume[symbol] if volume is not None and symbol in volume.columns else None
    return compute_stock_metrics(
        closes[symbol], open_s, volume_s,
        date_iloc, lookback_return, lookback_candles,
    )


def _get_trading_value(
    symbol: str,
    closes: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    date_iloc: int,
    lookback: int,
) -> float:
    """売買代金（終値×出来高）の lookback 期間平均を返す。計算不可なら 0.0。"""
    if volume is None or symbol not in closes.columns or symbol not in volume.columns:
        return 0.0
    tv_start = max(0, date_iloc - lookback + 1)
    c_sl = closes[symbol].iloc[tv_start: date_iloc + 1]
    v_sl = volume[symbol].iloc[tv_start: date_iloc + 1]
    tv_sl = c_sl * v_sl
    valid_tv = tv_sl[tv_sl.notna() & (tv_sl > 0)]
    return float(valid_tv.mean()) if len(valid_tv) > 0 else 0.0


def _composite_score(m: dict) -> float:
    """
    スワップ比率チェック用の複合スコア。
    ④bullish_ratio（重み3） + ③return_rate（重み2） + ⑤trading_value（重み1）
    """
    br = m['bullish_ratio'] or 0
    rr = (m['return_rate']  or 0) / 100   # % → 比率に正規化
    tv = (m['trading_value'] or 0) / 1e8  # 億円単位に正規化
    return br * 3 + rr * 2 + tv * 1


def _find_weakest_position(
    positions: Dict[str, 'Position'],
    closes: pd.DataFrame,
    opens: Optional[pd.DataFrame],
    volume: Optional[pd.DataFrame],
    date_iloc: int,
    lookback_return: int,
    lookback_candles: int,
    protect_gain_pct: float,
    lot_amount: float,
    needed_budget: float,
) -> Optional[str]:
    """
    スワップ対象として最も弱いポジションを探す。

    - 含み益 >= protect_gain_pct% のポジションは除外
    - 残りを ③return_rate・④bullish_ratio・⑤trading_value でそれぞれ昇順ランク付け（低い方が弱い）
    - ランク和が最大（最も弱い）銘柄を返す
    - 部分売却コスト計算: needed_budget を回収できる最小ロット数を売れば足りるか確認
    """
    candidates = {}
    for sym, pos in positions.items():
        raw = closes[sym].iloc[date_iloc]
        if pd.isna(raw) or raw <= 0:
            continue
        price = float(raw)
        # 含み益チェック
        gain_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100
        if gain_pct >= protect_gain_pct:
            continue
        metrics = _compute_metrics_for_symbol(
            sym, closes, opens, volume, date_iloc, lookback_return, lookback_candles
        )
        if any(v is None for v in metrics.values()):
            continue
        candidates[sym] = metrics

    if not candidates:
        return None

    syms = list(candidates.keys())

    # 各指標について昇順ランク（同率は平均ランク）
    def rank_vals(vals):
        """昇順ランク（小さい方が弱い=ランク低い）→ ランク和が大きい=弱い"""
        sorted_idx = sorted(range(len(vals)), key=lambda x: vals[x])
        ranks = [0] * len(vals)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = rank + 1
        return ranks

    rr_vals  = [candidates[s]['return_rate']   for s in syms]
    br_vals  = [candidates[s]['bullish_ratio']  for s in syms]
    tv_vals  = [candidates[s]['trading_value']  for s in syms]

    rr_ranks  = rank_vals(rr_vals)
    br_ranks  = rank_vals(br_vals)
    tv_ranks  = rank_vals(tv_vals)

    rank_sums = {syms[k]: rr_ranks[k] + br_ranks[k] + tv_ranks[k] for k in range(len(syms))}
    weakest   = max(rank_sums, key=lambda s: rank_sums[s])
    return weakest


def _build_swap_candidates(
    symbols_today,
    score_cache: dict,
    closes: pd.DataFrame,
    date_iloc: int,
    d: int,
    swap_i: int,
    swap_top_pct: float,
    positions: dict,
    pending_buys: dict,
) -> list:
    """
    スワップ候補銘柄リストを構築して返す。

    フィルター順序
    ① universe内で ⑤trading_value 上位 swap_top_pct%
    ② ④bullish_ratio >= 0.5 のものを絞り込み
    ③ swap_i 回連続 d 日上昇シグナルを満たすものを絞り込み
    ④ ③return_rate 降順でソート

    Returns
    -------
    list of symbol strings（return_rate 降順）
    """
    # 対象: positions・pending にない、キャッシュにある銘柄
    pool = {
        s: score_cache[s]
        for s in symbols_today
        if s in score_cache
        and s not in positions
        and s not in pending_buys
        and s in closes.columns
    }
    if not pool:
        return []

    # ① trading_value 上位 swap_top_pct%
    n_top = max(1, math.ceil(len(pool) * swap_top_pct))
    sorted_by_tv = sorted(pool, key=lambda s: pool[s]['trading_value'] or 0, reverse=True)
    top_tv = sorted_by_tv[:n_top]

    # ② bullish_ratio >= 0.5
    bullish_filtered = [
        s for s in top_tv
        if (pool[s]['bullish_ratio'] or 0) >= 0.5
    ]

    # ③ swap_i 回連続 d 日上昇シグナル
    signal_filtered = [
        s for s in bullish_filtered
        if check_entry_signal(closes[s], date_iloc, d, swap_i)
    ]

    # ④ return_rate 降順ソート
    signal_filtered.sort(key=lambda s: pool[s]['return_rate'] or 0, reverse=True)
    return signal_filtered


# ============================================================
# バックテスト実行
# ============================================================

def run_backtest(
    closes: pd.DataFrame,
    params: dict,
    start_date: str,
    end_date: str,
    daily_universe: Dict[pd.Timestamp, frozenset] = None,
    opens: pd.DataFrame = None,
    volume: pd.DataFrame = None,
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
    opens          : 始値DataFrame（スワップ・次日始値エントリーに使用）
    volume         : 出来高DataFrame（スワップスコア計算に使用）

    Returns
    -------
    trades        : list[dict] 取引履歴（1ロット決済ごとに1レコード）
    daily_records : list[dict] 日次ポートフォリオ状況
    transactions  : list[dict] 全売買アクション時系列
    """
    d                    = params['d']
    i                    = params['i']
    add_n                = params['add_n']
    add_x                = params['add_x']
    tp1_n                = params['tp1_n']
    tp1_x                = params.get('tp1_x', 0.0)
    tp2_y                = params['tp2_y']
    stop_x               = params['stop_x']
    stop_n               = params['stop_n']
    lot_amount           = params['lot_amount']
    max_amount_per_stock = params['max_amount_per_stock']
    total_budget         = params['total_budget']
    max_positions        = params.get('max_positions', None)

    # --- スワップパラメータ ---
    swap_enabled      = params.get('swap_enabled', False)
    swap_i            = params.get('swap_i', i + 1)       # スワップ用連続上昇回数
    swap_top_pct      = params.get('swap_top_pct', 0.05)  # 上位何%をスワップ候補とするか
    swap_ratio        = params.get('swap_ratio', 1.5)     # 新規スコアが弱ポジの何倍必要か
    protect_gain_pct  = params.get('protect_gain_pct', 20.0)  # 含み益%以上は売らない
    score_update_days = params.get('score_update_days', 5)    # スコアを更新する間隔（営業日）
    lookback_return   = params.get('lookback_return', 20)     # 騰落率の参照期間
    lookback_candles  = params.get('lookback_candles', 10)    # 陽線比率の参照期間

    remaining_budget = float(total_budget)

    all_dates    = closes.index
    symbols      = closes.columns.tolist()
    start_ts     = pd.Timestamp(start_date)
    end_ts       = pd.Timestamp(end_date)
    target_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]

    if len(target_dates) == 0:
        print("対象期間のデータがありません。START_DATE / END_DATE を確認してください。")
        return [], [], []

    date_to_iloc: Dict[pd.Timestamp, int] = {
        d_: idx for idx, d_ in enumerate(all_dates)
    }

    positions: Dict[str, Position] = {}
    trades         = []
    transactions   = []
    daily_records  = []
    cumulative_pnl = 0.0

    # --- pending_buys: 翌日始値でエントリーする銘柄 ---
    # {symbol: {'entry_type': '新規エントリー'|'スワップ', 'signal_date': date}}
    pending_buys: Dict[str, dict] = {}

    # --- スワップ用スコアキャッシュ ---
    # score_cache[symbol] = {'return_rate': ..., 'bullish_ratio': ..., 'trading_value': ...}
    score_cache: Dict[str, dict] = {}
    score_cache_date: Optional[pd.Timestamp] = None  # 最後にキャッシュを更新した日
    score_update_count = 0  # 経過営業日カウント（5日ごとに更新）

    iterator = (
        tqdm(target_dates, desc="バックテスト実行中", ncols=80)
        if TQDM_AVAILABLE else target_dates
    )

    for date in iterator:
        date_iloc      = date_to_iloc[date]
        realized_today = 0.0

        # ============================================================
        # 0. 前日の pending_buys を本日の始値でエントリー
        # ============================================================
        if pending_buys and opens is not None:
            for sym, info in list(pending_buys.items()):
                if sym in positions:
                    # すでにポジションあり（同日中に別ルートで入った等）→スキップ
                    del pending_buys[sym]
                    continue
                if remaining_budget < lot_amount:
                    del pending_buys[sym]
                    continue
                if max_positions is not None and len(positions) >= max_positions:
                    del pending_buys[sym]
                    continue
                raw_open = opens[sym].iloc[date_iloc] if sym in opens.columns else float('nan')
                if pd.isna(raw_open) or raw_open <= 0:
                    # 始値が取れない場合は終値で代替
                    raw_open = closes[sym].iloc[date_iloc]
                if pd.isna(raw_open) or raw_open <= 0:
                    del pending_buys[sym]
                    continue
                open_price = float(raw_open)

                positions[sym] = Position(
                    symbol              = sym,
                    initial_entry_price = open_price,
                    initial_entry_date  = date,
                    lots                = [Lot(date, open_price, lot_amount)],
                    last_entry_price    = open_price,
                    last_entry_date     = date,
                    tp_count            = 0,
                    tp1_check_start     = date + timedelta(days=tp1_n),
                    stop_check_start    = date + timedelta(days=stop_n),
                    next_addon_check    = date + timedelta(days=add_n),
                )
                remaining_budget -= lot_amount
                transactions.append({
                    'date': date, 'symbol': sym,
                    'type': info.get('entry_type', '新規エントリー'),
                    'price': round(open_price, 2), 'avg_cost': None,
                    'shares': round(lot_amount / open_price, 4),
                    'amount': lot_amount, 'proceeds': None,
                    'pnl': None, 'pnl_pct': None,
                })
                del pending_buys[sym]

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
            skip_addon = False
            if pos.tp_count == 0 and date >= pos.tp1_check_start:
                tp1_threshold = pos.initial_entry_price * (1 + tp1_x / 100)
                if price < pos.last_entry_price and price >= tp1_threshold:
                    n_exit   = math.ceil(len(pos.lots) / 2)
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
                        del positions[symbol]
                        continue

                    pos.tp_count        = 1
                    pos.tp1_price       = price
                    skip_addon          = True

            # ---- 買い増しチェック ----
            if not skip_addon and symbol in positions:
                if date >= pos.next_addon_check:
                    can_add = (
                        pos.total_invested + lot_amount <= max_amount_per_stock
                        and remaining_budget >= lot_amount
                    )
                    if can_add:
                        if pos.tp_count == 0:
                            if price > pos.last_entry_price * (1 + add_x / 100):
                                pos.lots.append(Lot(date, price, lot_amount))
                                pos.last_entry_price  = price
                                pos.last_entry_date   = date
                                pos.tp1_check_start   = date + timedelta(days=tp1_n)
                                pos.next_addon_check  = date + timedelta(days=add_n)
                                remaining_budget      -= lot_amount
                                transactions.append({
                                    'date': date, 'symbol': symbol, 'type': '買い増し',
                                    'price': round(price, 2), 'avg_cost': None,
                                    'shares': round(lot_amount / price, 4),
                                    'amount': lot_amount, 'proceeds': None,
                                    'pnl': None, 'pnl_pct': None,
                                })
                        elif pos.tp_count == 1:
                            if price > pos.last_entry_price * (1 + add_x / 100):
                                pos.lots.append(Lot(date, price, lot_amount))
                                pos.last_entry_price  = price
                                pos.last_entry_date   = date
                                pos.tp1_check_start   = date + timedelta(days=tp1_n)
                                pos.tp_count          = 0
                                pos.tp1_price         = 0.0
                                pos.next_addon_check  = date + timedelta(days=add_n)
                                remaining_budget      -= lot_amount
                                transactions.append({
                                    'date': date, 'symbol': symbol, 'type': '買い増し',
                                    'price': round(price, 2), 'avg_cost': None,
                                    'shares': round(lot_amount / price, 4),
                                    'amount': lot_amount, 'proceeds': None,
                                    'pnl': None, 'pnl_pct': None,
                                })

                    if date >= pos.next_addon_check:
                        pos.next_addon_check = date + timedelta(days=1)

        # ============================================================
        # 2. スコアキャッシュ更新（swap_enabled 時のみ・5営業日ごと）
        # ============================================================
        if swap_enabled:
            if score_cache_date is None or score_update_count >= score_update_days:
                symbols_today = sorted(
                    daily_universe.get(date, frozenset())
                    if daily_universe is not None else symbols
                )
                new_cache = {}
                for sym in symbols_today:
                    if sym not in closes.columns:
                        continue
                    m = _compute_metrics_for_symbol(
                        sym, closes, opens, volume,
                        date_iloc, lookback_return, lookback_candles
                    )
                    if not any(v is None for v in m.values()):
                        new_cache[sym] = m
                score_cache      = new_cache
                score_cache_date = date
                score_update_count = 0
            else:
                score_update_count += 1

        # ============================================================
        # 3. 新規エントリー判定
        # ============================================================
        symbols_today = sorted(
            daily_universe.get(date, frozenset()) if daily_universe is not None else symbols
        )
        swap_done_today = False  # 1日1スワップ制限

        # --- 3a. 通常エントリー（i 回連続上昇シグナル）---
        # 売買代金の大きい順にソートして流動性の高い銘柄を優先
        def _tv_key(sym: str) -> float:
            if sym in score_cache:
                return score_cache[sym].get('trading_value') or 0.0
            return _get_trading_value(sym, closes, volume, date_iloc, lookback_return)

        for symbol in sorted(symbols_today, key=_tv_key, reverse=True):
            if symbol in positions or symbol in pending_buys:
                continue
            if remaining_budget < lot_amount:
                continue
            if max_positions is not None and len(positions) + len(pending_buys) >= max_positions:
                continue

            raw = closes[symbol].iloc[date_iloc]
            if pd.isna(raw) or raw <= 0:
                continue

            if check_entry_signal(closes[symbol], date_iloc, d, i):
                pending_buys[symbol] = {'entry_type': '新規エントリー', 'signal_date': date}

        # --- 3b. スワップロジック ---
        # 予算不足 or max_positions 到達のときにのみ実行
        if swap_enabled and not swap_done_today and positions:
            budget_full    = remaining_budget >= lot_amount
            positions_full = (max_positions is not None
                              and len(positions) + len(pending_buys) >= max_positions)

            if not budget_full or positions_full:
                # スワップ候補リストを構築（フィルターチェーン）
                swap_candidates = _build_swap_candidates(
                    symbols_today, score_cache, closes,
                    date_iloc, d, swap_i, swap_top_pct,
                    positions, pending_buys,
                )

                for new_sym in swap_candidates:
                    if swap_done_today:
                        break

                    # 最弱ポジションを探す
                    weakest_sym = _find_weakest_position(
                        positions, closes, opens, volume,
                        date_iloc, lookback_return, lookback_candles,
                        protect_gain_pct, lot_amount, lot_amount,
                    )
                    if weakest_sym is None:
                        break

                    # スワップ比率チェック: 新規の複合スコアが最弱の swap_ratio 倍以上か
                    new_metrics  = score_cache.get(new_sym)
                    if new_metrics is None:
                        continue
                    weak_metrics = _compute_metrics_for_symbol(
                        weakest_sym, closes, opens, volume,
                        date_iloc, lookback_return, lookback_candles
                    )
                    if any(v is None for v in weak_metrics.values()):
                        continue

                    new_score  = _composite_score(new_metrics)
                    weak_score = _composite_score(weak_metrics)

                    if weak_score <= 0 or new_score < weak_score * swap_ratio:
                        continue

                    # --- スワップ実行 ---
                    pos_weak   = positions[weakest_sym]
                    weak_price = float(closes[weakest_sym].iloc[date_iloc])

                    # 必要回収額: lot_amount - remaining_budget（負の場合は0）
                    needed = max(0.0, lot_amount - remaining_budget)
                    if needed <= 0:
                        # 予算は足りている（max_positions 制限のみ）→ 最小1ロット売却
                        lots_to_sell = pos_weak.lots[:1]
                    else:
                        lots_to_sell = []
                        recovered    = 0.0
                        for lot in pos_weak.lots:
                            lots_to_sell.append(lot)
                            recovered += lot.shares * weak_price
                            if recovered >= needed:
                                break

                    avg_cost = _calc_avg_cost(pos_weak.lots)
                    transactions.append(_sell_txn(
                        weakest_sym, date, weak_price, lots_to_sell, avg_cost, 'スワップ売り'
                    ))
                    for lot in lots_to_sell:
                        proceeds = _close_lot(
                            trades, weakest_sym, lot, date, weak_price, 'スワップ売り',
                            pos_weak.initial_entry_date
                        )
                        realized_today   += lot.shares * (weak_price - lot.entry_price)
                        remaining_budget += proceeds
                        pos_weak.lots.remove(lot)

                    if not pos_weak.lots:
                        del positions[weakest_sym]

                    # 新規銘柄を翌日始値で買い（pending登録）
                    pending_buys[new_sym] = {'entry_type': 'スワップ買い', 'signal_date': date}
                    swap_done_today = True

        # ============================================================
        # 4. 日次記録
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

    return trades, daily_records, transactions, positions
