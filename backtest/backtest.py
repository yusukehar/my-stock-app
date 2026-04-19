# backtest.py: メイン実行スクリプト
# 使い方: python backtest.py

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from typing import Optional, Dict
from engine import run_backtest
from report import (
    compute_summary, compute_max_drawdown,
    print_summary, save_trades_csv, save_daily_csv, plot_equity_curve,
    save_transactions_csv, save_open_positions_csv,
)


# ============================================================
# ★ パラメータ設定（ここを変えて条件を検証してください）
# ============================================================

PARAMS = {
    # --- エントリー条件 ---
    'd': 4,            # d 営業日ごとに株価を比較する
    'i': 2,            # i 回連続上昇でシグナル発生

    # --- 買い増し条件 ---
    'add_n': 5,        # 最終エントリー日から N 日経過後、毎日チェック開始
    'add_x': 5.0,      # 最終エントリー価格の +X% 超で買い増し

    # --- 初回利確条件 ---
    'tp1_n': 20,       # 最終エントリー日から N 日経過後、毎日チェック開始
                       # 条件①: 現在価格 < 最終エントリー価格（モメンタム喪失）
                       # 条件②: 現在価格 >= 初回エントリー価格 + tp1_x%（最低利益ライン）
                       # → 両条件を満たしたとき、投資金額の半分（切り上げ）を決済
    'tp1_x': 3.0,      # 初回利確の最低利益率（%）。この水準未満では TP1 を発動しない

    # --- 2回目利確条件 ---
    'tp2_y': 5.0,      # 初回利確時の株価から -Y% 割れで残り全額決済

    # --- 損切り条件（最優先・全額即時）---
    'stop_x': 10.0,    # 初回エントリー価格から -X% 割れで全ロット即時決済
    'stop_n': 3,       # エントリーから N 日間は損切りチェックをしない（猶予期間）

    # --- ポジション管理 ---
    'lot_amount'           :  70_000,    # 1 ロットあたりの投資金額（円）
    'max_amount_per_stock' : 800_000,    # 1 銘柄あたりの最大保有金額（円）
    'max_positions'        :    15,    # 同時保有できる最大銘柄数（None = 制限なし）

    # --- 予算管理 ---
    'total_budget'         : 4_000_000,  # 総予算（円）。利確・損切りの回収金額は全額返還

    # --- スワップロジック ---
    'swap_enabled'     : True,   # スワップロジックを使用するか
    'swap_i'           : 3,      # スワップ候補の連続上昇回数（通常エントリーより1回多く）
    'swap_top_pct'     : 0.05,   # 上位何%をスワップ候補とするか（5%）
    'swap_ratio'       : 1.5,    # 新規スコアが最弱ポジの何倍以上でスワップ実行
    'protect_gain_pct' : 20.0,   # 含み益がこの%以上のポジションはスワップ売り対象外
    'score_update_days': 5,      # スコアキャッシュを更新する間隔（営業日）
    'lookback_return'  : 20,     # 株価騰落率の参照期間（営業日）
    'lookback_candles' : 10,     # 陽線比率の参照期間（営業日）
}

# ============================================================
# ★ バックテスト期間
# ============================================================

START_DATE = '2025-01-06'  # 日経平均：33,288円
END_DATE   = '2026-04-09'  # 日経平均：50,339円（+51.2%）

# ============================================================
# ★ ユニバース（対象銘柄）設定
#    月次リバランス + 新規上場の動的追加でルックアヘッドバイアスなく構築する。
# ============================================================

UNIVERSE_SIZE     = 700   # 上位何社を対象にするか
UNIVERSE_LOOKBACK =  20   # 売買代金を算出する過去営業日数（≒1ヶ月）

# ============================================================
# ★ 株価データのパス
#    既存アプリ（my-stock-app）の stock_data フォルダ内の CSV を指定してください。
#    例: '/Applications/my-stock-app/stock_data/stock_data_202603210855.csv'
# ============================================================

STOCK_DATA_PATH = os.path.expanduser(
    '~/my-stock-app/stock_data/stock_data_202604111404.csv'
)
# セッション内フォールバック（Claudeが実行する場合）
if not os.path.exists(STOCK_DATA_PATH):
    _fallback = '/sessions/beautiful-bold-johnson/mnt/my-stock-app/stock_data/stock_data_202603210855.csv'
    if os.path.exists(_fallback):
        STOCK_DATA_PATH = _fallback

# 出力先フォルダ（backtest/results/ に保存されます）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ============================================================
# データ読み込み
# ============================================================

def load_data(path: str):
    """
    yfinance 形式（MultiIndex 列）の CSV を読み込み、全銘柄の closes・opens・volume を返す。
    universe の絞り込みはしない（build_daily_universe で行う）。

    Returns
    -------
    closes : 終値 DataFrame（行=日付, 列=銘柄コード）
    opens  : 始値 DataFrame（同形式）。データがない場合は None。
    volume : 出来高 DataFrame（同形式）。データがない場合は None。
    """
    print(f"データ読み込み中: {path}")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        closes = df['Close']
        opens  = df['Open']   if 'Open'   in level0 else None
        volume = df['Volume'] if 'Volume' in level0 else None
    else:
        closes = df
        opens  = None
        volume = None

    closes = closes.sort_index().dropna(axis=1, how='all')
    if opens is not None:
        opens  = opens.reindex(columns=closes.columns).sort_index()
    if volume is not None:
        volume = volume.reindex(columns=closes.columns).sort_index()

    print(f"  → {len(closes)} 日分 × {len(closes.columns)} 銘柄を読み込みました")
    return closes, opens, volume


def build_daily_universe(
    closes: pd.DataFrame,
    volume,
    start_date: str,
    end_date: str,
    lookback: int = 20,
    size: int = 700,
):
    """
    各営業日の universe を事前計算する（ルックアヘッドバイアスなし）。

    ルール
    ------
    - 初回  : START_DATE 直前の過去 lookback 営業日で上位 size 銘柄
    - 月次リバランス : 毎月第1営業日に直近 lookback 営業日で再選定
    - 新規上場 : バックテスト期間中に初登場した銘柄を追跡し、
                 lookback 日分のデータが揃った時点で上位 size 位以内なら即時追加。
                 圏外なら毎日チェックを継続（次の月次リバランスで再評価）。

    Returns
    -------
    daily_universe : Dict[pd.Timestamp, frozenset]  日付 → universe 銘柄セット
    initial_stats  : pd.DataFrame  初回 universe の統計情報（universe.csv 用）
    """
    all_dates = closes.index
    start_ts  = pd.Timestamp(start_date)
    end_ts    = pd.Timestamp(end_date)
    target_dates = all_dates[(all_dates >= start_ts) & (all_dates <= end_ts)]

    if volume is None:
        print("  ※ Volume データなし: 全銘柄を universe とします")
        univ = frozenset(closes.columns)
        empty_stats = pd.DataFrame(columns=['symbol', 'avg_close_jpy', 'data_days', 'universe_rank'])
        return {d: univ for d in target_dates}, empty_stats

    turnover = closes * volume  # 日次売買代金

    def top_n_at(as_of_date: pd.Timestamp) -> set:
        """as_of_date 直前の過去 lookback 営業日の平均売買代金で上位 size 銘柄を返す"""
        avail  = all_dates[all_dates < as_of_date]
        recent = avail[-lookback:] if len(avail) >= lookback else avail
        if len(recent) == 0:
            return set()
        avg = turnover.loc[recent].mean(axis=0).dropna()
        return set(avg.nlargest(size).index)

    # 各銘柄の初登場日（closes に初めて有効値が出た日）
    first_seen: Dict[str, pd.Timestamp] = {}
    for sym in closes.columns:
        valid = closes[sym].dropna()
        if not valid.empty:
            first_seen[sym] = valid.index[0]

    # バックテスト期間中に初登場する銘柄（新規上場候補）
    new_listings: Dict[str, pd.Timestamp] = {
        sym: d for sym, d in first_seen.items()
        if start_ts <= d <= end_ts
    }
    print(f"  → バックテスト期間中の新規上場候補: {len(new_listings)} 銘柄")

    # 初回 universe（START_DATE 直前の過去 lookback 日）
    current_universe: set = top_n_at(start_ts)

    # 初回 universe の統計情報（CSV 用）
    avail_pre    = all_dates[all_dates < start_ts]
    recent_pre   = avail_pre[-lookback:] if len(avail_pre) >= lookback else avail_pre
    init_syms    = list(current_universe)
    avg_to       = turnover[init_syms].loc[recent_pre].mean(axis=0).round(0).astype(int)
    avg_cl       = closes[init_syms].loc[recent_pre].mean(axis=0).round(1)
    data_days    = closes[init_syms].loc[recent_pre].notna().sum(axis=0).astype(int)
    initial_stats = (
        pd.DataFrame({
            'symbol'           : init_syms,
            'avg_turnover_jpy' : avg_to.values,
            'avg_close_jpy'    : avg_cl.values,
            'data_days'        : data_days.values,
        })
        .sort_values('avg_turnover_jpy', ascending=False)
        .reset_index(drop=True)
    )
    initial_stats['universe_rank'] = range(1, len(initial_stats) + 1)

    # 評価待ち新規上場銘柄: {symbol: listing_date}
    pending: Dict[str, pd.Timestamp] = {}

    current_month = None
    daily_universe: Dict[pd.Timestamp, frozenset] = {}
    newly_added = set()

    for date in target_dates:

        # ---- 月次リバランス（毎月第1営業日） ----
        if date.month != current_month:
            current_month  = date.month
            current_universe = top_n_at(date)

            # pending を再構築: まだ lookback 日未満の新規上場銘柄のみ残す
            pending = {}
            for sym, listing_date in new_listings.items():
                if sym in current_universe:
                    continue
                days = len(all_dates[(all_dates >= listing_date) & (all_dates < date)])
                if days < lookback:
                    pending[sym] = listing_date

        # ---- 新規上場の初登場検出（当日が listing_date の銘柄） ----
        for sym, listing_date in new_listings.items():
            if listing_date == date and sym not in current_universe and sym not in pending:
                pending[sym] = listing_date

        # ---- pending の日次評価（lookback 日分のデータが揃ったもの） ----
        for sym, listing_date in list(pending.items()):
            days = len(all_dates[(all_dates >= listing_date) & (all_dates <= date)])
            if days < lookback:
                continue  # まだデータ不足
            recent = all_dates[all_dates <= date][-lookback:]
            t_sym  = turnover[sym].loc[recent].mean()
            if pd.isna(t_sym):
                continue
            t_all = turnover.loc[recent].mean(axis=0).dropna()
            rank  = int((t_all > t_sym).sum()) + 1  # 1-based（小さいほど上位）
            if rank <= size:
                current_universe.add(sym)
                newly_added.add(sym)
                del pending[sym]
            # 圏外の場合は pending に残し翌日も評価

        daily_universe[date] = frozenset(current_universe)

    print(f"  → 月次リバランス外で動的追加された新規上場銘柄: {len(newly_added)} 銘柄"
          + (f" ({', '.join(sorted(newly_added))})" if newly_added else ""))

    return daily_universe, initial_stats


# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':

    # --- データ存在チェック ---
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"\n[エラー] 株価データが見つかりません。")
        print(f"  パス: {STOCK_DATA_PATH}")
        print("  backtest.py の STOCK_DATA_PATH を正しいパスに変更してください。\n")
        sys.exit(1)

    # --- データ読み込み（全銘柄・universe絞り込みなし）---
    closes, opens, volume = load_data(STOCK_DATA_PATH)

    # --- 日次 universe 構築（ルックアヘッドバイアスなし）---
    print(f"\nuniverse構築中（lookback={UNIVERSE_LOOKBACK}日, size={UNIVERSE_SIZE}）...")
    daily_universe, universe_df = build_daily_universe(
        closes, volume,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback=UNIVERSE_LOOKBACK,
        size=UNIVERSE_SIZE,
    )

    # --- ユニバースリスト（初回）に銘柄名・業種を付加して保存 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _meta_path = os.path.normpath(
        os.path.join(os.path.dirname(STOCK_DATA_PATH), '..', 'meta_data.xls')
    )
    if os.path.exists(_meta_path) and not universe_df.empty:
        try:
            import sys as _sys
            _venv_sp = os.path.join(os.path.dirname(_meta_path), 'venv',
                                    'lib', 'python3.9', 'site-packages')
            if os.path.isdir(_venv_sp):
                _sys.path.insert(0, _venv_sp)
            import xlrd
            _wb  = xlrd.open_workbook(_meta_path)
            _ws  = _wb.sheets()[0]
            _hdr = [_ws.cell_value(0, c) for c in range(_ws.ncols)]
            _ci, _ni, _ii = _hdr.index('コード'), _hdr.index('銘柄名'), _hdr.index('33業種区分')
            _name_map = {}
            for _r in range(1, _ws.nrows):
                _cv   = _ws.cell_value(_r, _ci)
                _code = str(int(_cv)) if isinstance(_cv, float) and _cv > 0 else str(_cv).strip()
                if _code not in _name_map:
                    _name_map[_code] = (_ws.cell_value(_r, _ni), _ws.cell_value(_r, _ii))
            universe_df['_code'] = universe_df['symbol'].str.replace('.T', '', regex=False)
            universe_df.insert(1, '銘柄名', universe_df['_code'].map(
                lambda c: _name_map.get(c, ('', ''))[0]))
            universe_df.insert(2, '業種',   universe_df['_code'].map(
                lambda c: _name_map.get(c, ('', ''))[1]))
            universe_df.drop(columns=['_code'], inplace=True)
        except Exception as _e:
            print(f"  ※ 銘柄名の付加に失敗しました: {_e}")
    universe_path = os.path.join(OUTPUT_DIR, 'universe.csv')
    universe_df.to_csv(universe_path, index=False, encoding='utf-8-sig')
    print(f"初回ユニバースリスト保存: {universe_path}  ({len(universe_df)} 銘柄)")

    # --- バックテスト実行 ---
    print(f"\nバックテスト期間: {START_DATE} 〜 {END_DATE}")
    trades, daily_records, transactions, open_positions = run_backtest(
        closes, PARAMS, START_DATE, END_DATE,
        daily_universe=daily_universe,
        opens=opens,
        volume=volume,
    )
    print(f"完了: {len(trades)} 件の取引が発生しました\n")

    # --- 集計 ---
    summary = compute_summary(trades)
    max_dd  = compute_max_drawdown(daily_records)

    # --- インデックスデータ取得（日経平均・TOPIX） ---
    index_df = None
    try:
        import yfinance as yf
        print("インデックスデータ取得中（日経平均・TOPIX）...")
        index_df = yf.download(
            ['^N225', '^TOPIX'],
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
        )['Close']
        # MultiIndex になる場合はフラット化
        if isinstance(index_df.columns, pd.MultiIndex):
            index_df.columns = index_df.columns.get_level_values(0)
        index_df = index_df.dropna(how='all')
        print(f"  → {len(index_df)} 日分のインデックスデータを取得しました")
    except Exception as e:
        print(f"  ※ インデックスデータの取得に失敗しました（比較グラフはスキップ）: {e}")

    # --- 出力 ---
    print_summary(summary, PARAMS, START_DATE, END_DATE, max_dd, daily_records, index_df)
    print("ファイル出力:")
    save_trades_csv(trades, OUTPUT_DIR)
    save_daily_csv(daily_records, OUTPUT_DIR)
    plot_equity_curve(daily_records, OUTPUT_DIR,
                      total_budget=PARAMS['total_budget'], index_df=index_df)
    # 銘柄名付き全売買記録
    _meta = os.path.normpath(os.path.join(os.path.dirname(STOCK_DATA_PATH), '..', 'meta_data.xls'))
    save_transactions_csv(transactions, OUTPUT_DIR, meta_path=_meta)
    # 保有中ポジション一覧
    save_open_positions_csv(open_positions, closes, END_DATE, OUTPUT_DIR, meta_path=_meta)
    print(f"\n結果の出力先: {OUTPUT_DIR}\n")

