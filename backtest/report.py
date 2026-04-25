# report.py: 結果集計・コンソール表示・CSV/グラフ出力

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib  # 日本語フォント対応（pip install japanize-matplotlib）
except ImportError:
    pass  # 未インストールでも動作はする（グラフラベルは英語にフォールバック）


# ============================================================
# パフォーマンス指標の計算
# ============================================================

def compute_summary(trades: list) -> dict:
    """取引リストからパフォーマンスサマリーを計算する"""
    empty = {
        'total_trades': 0, 'win_rate': 0.0,
        'avg_profit_pct': 0.0, 'avg_loss_pct': 0.0,
        'profit_factor': 0.0, 'total_pnl': 0.0,
    }
    if not trades:
        return empty

    df = pd.DataFrame(trades)
    total   = len(df)
    winners = df[df['pnl'] > 0]
    losers  = df[df['pnl'] <= 0]

    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
    gross_loss   = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.0

    return {
        'total_trades'   : total,
        'win_rate'       : round(len(winners) / total * 100, 1),
        'avg_profit_pct' : round(winners['pnl_pct'].mean(), 2) if len(winners) > 0 else 0.0,
        'avg_loss_pct'   : round(losers['pnl_pct'].mean(), 2)  if len(losers) > 0  else 0.0,
        'profit_factor'  : round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf'),
        'total_pnl'      : round(df['pnl'].sum(), 0),
    }


def compute_max_drawdown(daily_records: list) -> float:
    """資産曲線（total_pnl）から最大ドローダウン（円）を計算する"""
    if not daily_records:
        return 0.0
    equity = pd.DataFrame(daily_records)['total_pnl'].values
    peak, max_dd = equity[0], 0.0
    for val in equity:
        peak   = max(peak, val)
        max_dd = max(max_dd, peak - val)
    return round(max_dd, 0)


# ============================================================
# コンソール出力
# ============================================================

def print_summary(
    summary: dict,
    params: dict,
    start_date: str,
    end_date: str,
    max_dd: float,
    daily_records: list = None,
    index_df: 'pd.DataFrame | None' = None,
):
    """サマリーをコンソールに整形して表示"""
    # 日次記録の最終行から評価損益・合計損益を取得
    final_unrealized = 0.0
    final_total      = 0.0
    if daily_records:
        last = daily_records[-1]
        final_unrealized = float(last.get('unrealized_pnl', 0))
        final_total      = float(last.get('total_pnl', 0))

    W = 58  # 区切り線の幅
    print("\n" + "=" * W)
    print("  バックテスト結果")
    print("=" * W)
    print(f"  期間              : {start_date} 〜 {end_date}")
    print(f"  エントリー条件    : d={params['d']}日, i={params['i']}回連続上昇")
    print(f"  買い増し条件      : {params['add_n']}日ごと / 初回比 +{params['add_x']}% 超")
    print(f"  初回利確条件      : {params['tp1_n']}日経過後 / 最終エントリー価格割れ（半額決済）")
    print(f"  2回目利確条件     : 初回利確価格比 -{params['tp2_y']}% 割れ（全額決済）")
    print(f"  損切り条件        : 初回比 -{params['stop_x']}% 割れ（全額即時・最優先）、猶予 {params['stop_n']} 日")
    print(f"  1ロット金額       : {params['lot_amount']:,} 円")
    print(f"  1銘柄上限金額     : {params['max_amount_per_stock']:,} 円")
    max_pos = params.get('max_positions')
    print(f"  最大保有銘柄数    : {max_pos if max_pos is not None else '制限なし'}")
    print(f"  総予算            : {params['total_budget']:,} 円")
    print("-" * W)
    if summary['total_trades'] == 0:
        print("  取引が発生しませんでした。")
        print("  パラメータや期間の設定を見直してください。")
    else:
        pf_str = (
            f"{summary['profit_factor']}"
            if summary['profit_factor'] != float('inf')
            else "∞（損失トレードなし）"
        )
        realized = summary['total_pnl']
        total_budget = params['total_budget']

        print(f"  総トレード数              : {summary['total_trades']}")
        print(f"  勝率                      : {summary['win_rate']} %")
        print(f"  平均利益率                : +{summary['avg_profit_pct']} %")
        print(f"  平均損失率                : {summary['avg_loss_pct']} %")
        print(f"  プロフィットファクター    : {pf_str}")
        print("-" * W)
        print(f"  累積実現損益              : {realized:+,.0f} 円")
        print(f"  期末評価損益（含み損益）  : {final_unrealized:+,.0f} 円")
        print(f"  ★ 合計損益（実現＋評価） : {final_total:+,.0f} 円"
              f"  ({final_total / total_budget * 100:+.1f} %)")
        print("-" * W)
        print(f"  最大ドローダウン          : -{max_dd:,.0f} 円"
              f"  ({-max_dd / total_budget * 100:.1f} %)")

    # インデックス比較
    if index_df is not None and not index_df.empty:
        print("-" * W)
        label_map = {'^N225': '日経平均', '^TOPIX': 'TOPIX'}
        for col in index_df.columns:
            series = index_df[col].dropna()
            if len(series) < 2:
                continue
            start_val = series.iloc[0]
            end_val   = series.iloc[-1]
            pct       = (end_val - start_val) / start_val * 100
            label     = label_map.get(col, col)
            print(f"  {label}（同期間）              :              {pct:+.1f} %")

    print("=" * W + "\n")


# ============================================================
# ファイル出力
# ============================================================

def save_trades_csv(trades: list, output_dir: str):
    """
    取引履歴をポジション単位（初回エントリー〜完全決済）に集計して CSV 保存。
    同一ポジション内の複数ロット/複数決済を1行に合算する。
    """
    os.makedirs(output_dir, exist_ok=True)
    if not trades:
        print("  取引履歴なし（CSV スキップ）")
        return

    df = pd.DataFrame(trades)
    # 株数 = 投資金額 / エントリー価格（ロットごと）
    df['shares'] = df['amount'] / df['entry_price']

    rows = []
    for (symbol, init_date), grp in df.groupby(['symbol', 'initial_entry_date'], sort=False):
        total_invested = grp['amount'].sum()
        total_proceeds = grp['proceeds'].sum()
        total_pnl      = total_proceeds - total_invested
        total_shares   = grp['shares'].sum()
        pnl_pct        = total_pnl / total_invested * 100 if total_invested > 0 else 0.0

        # 平均購入単価 = 総投資額 / 総株数
        avg_entry = total_invested / total_shares if total_shares > 0 else 0.0
        # 平均決済価格 = 総回収額 / 総株数
        avg_exit  = total_proceeds / total_shares if total_shares > 0 else 0.0

        reasons = grp['exit_reason'].unique().tolist()

        rows.append({
            '銘柄コード'      : symbol,
            '初回エントリー日' : pd.Timestamp(init_date),
            '最終決済日'       : grp['exit_date'].max(),
            '平均購入単価（円）': round(avg_entry, 2),
            '平均決済価格（円）': round(avg_exit, 2),
            '投資金額（円）'   : round(total_invested, 0),
            '回収金額（円）'   : round(total_proceeds, 0),
            '損益（円）'       : round(total_pnl, 0),
            '損益率（%）'      : round(pnl_pct, 2),
            '決済理由'         : ' / '.join(reasons),
        })

    result = pd.DataFrame(rows).sort_values('初回エントリー日').reset_index(drop=True)
    path = os.path.join(output_dir, 'trades.csv')
    result.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  取引履歴 → {path}")


def save_transactions_csv(transactions: list, output_dir: str, meta_path: str = None):
    """
    全売買アクション（買い・売り）を時系列で CSV 保存。
    買い行: 価格・株数・投資金額を記載。決済関連列は空欄。
    売り行: 決済価格・平均購入単価・回収金額・損益を記載。
    meta_path を指定すると銘柄名・業種列を付加する。
    """
    os.makedirs(output_dir, exist_ok=True)
    if not transactions:
        print("  取引アクションなし（transactions CSV スキップ）")
        return

    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 銘柄名・業種を付加（meta_path が指定された場合）
    if meta_path and os.path.exists(meta_path):
        try:
            import sys
            venv_sp = os.path.join(os.path.dirname(meta_path), 'venv',
                                   'lib', 'python3.9', 'site-packages')
            if os.path.isdir(venv_sp):
                sys.path.insert(0, venv_sp)
            import xlrd
            wb  = xlrd.open_workbook(meta_path)
            ws  = wb.sheets()[0]
            hdr = [ws.cell_value(0, c) for c in range(ws.ncols)]
            ci, ni, ii = hdr.index('コード'), hdr.index('銘柄名'), hdr.index('33業種区分')
            name_map = {}
            for r in range(1, ws.nrows):
                cv = ws.cell_value(r, ci)
                code = str(int(cv)) if isinstance(cv, float) and cv > 0 else str(cv).strip()
                if code not in name_map:
                    name_map[code] = (ws.cell_value(r, ni), ws.cell_value(r, ii))
            df['_code'] = df['symbol'].str.replace('.T', '', regex=False)
            df['銘柄名'] = df['_code'].map(lambda c: name_map.get(c, ('', ''))[0])
            df['業種']   = df['_code'].map(lambda c: name_map.get(c, ('', ''))[1])
            df.drop(columns=['_code'], inplace=True)
        except Exception:
            df['銘柄名'] = ''
            df['業種']   = ''
    else:
        df['銘柄名'] = ''
        df['業種']   = ''

    df = df.rename(columns={
        'date'     : '日付',
        'symbol'   : '銘柄コード',
        'type'     : '取引種別',
        'price'    : '価格（円）',
        'avg_cost' : '平均購入単価（円）',
        'shares'   : '株数',
        'amount'   : '投資金額（円）',
        'proceeds' : '回収金額（円）',
        'pnl'      : '損益（円）',
        'pnl_pct'  : '損益率（%）',
    })

    col_order = ['日付', '銘柄コード', '銘柄名', '業種', '取引種別',
                 '価格（円）', '平均購入単価（円）', '株数',
                 '投資金額（円）', '回収金額（円）', '損益（円）', '損益率（%）']
    df = df[[c for c in col_order if c in df.columns]]

    path = os.path.join(output_dir, 'transactions.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  全取引記録 → {path}")


def save_daily_csv(daily_records: list, output_dir: str):
    """日次ポートフォリオ状況を CSV で保存"""
    os.makedirs(output_dir, exist_ok=True)
    if not daily_records:
        return
    df = pd.DataFrame(daily_records).rename(columns={
        'date'                    : '日付',
        'realized_pnl'            : '当日実現損益（円）',
        'cumulative_realized_pnl' : '累積実現損益（円）',
        'unrealized_pnl'          : '含み損益（円）',
        'total_pnl'               : '合計損益（円）',
        'open_positions'          : '保有銘柄数',
        'remaining_budget'        : '残予算（円）',
    })
    path = os.path.join(output_dir, 'daily.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  日次記録  → {path}")


def save_open_positions_csv(
    positions: dict,
    closes: 'pd.DataFrame',
    end_date: str,
    output_dir: str,
    meta_path: str = None,
):
    """
    バックテスト終了時点で保有中の銘柄一覧を CSV で保存する。

    出力列
    ------
    銘柄コード, 銘柄名（任意）, 業種（任意）,
    初回エントリー日, 保有ロット数, 平均取得単価（円）,
    期末株価（円）, 保有額（円）, 含み損益（円）, 含み損益率（%）
    """
    os.makedirs(output_dir, exist_ok=True)
    if not positions:
        print("  保有ポジションなし（open_positions CSV スキップ）")
        return

    end_ts = pd.Timestamp(end_date)

    # end_date 以前で最も新しいデータ行を取得
    avail_dates = closes.index[closes.index <= end_ts]
    if avail_dates.empty:
        print("  期末株価データなし（open_positions CSV スキップ）")
        return
    last_date = avail_dates[-1]

    rows = []
    for symbol, pos in positions.items():
        # 期末株価
        if symbol not in closes.columns:
            last_price = float('nan')
        else:
            raw = closes[symbol].loc[last_date]
            last_price = float(raw) if not pd.isna(raw) and raw > 0 else float('nan')

        # ロット集計
        total_invested = pos.total_invested
        total_shares   = sum(lot.shares for lot in pos.lots)
        avg_cost       = total_invested / total_shares if total_shares > 0 else float('nan')

        if pd.isna(last_price):
            market_value  = float('nan')
            unrealized    = float('nan')
            unrealized_pct = float('nan')
        else:
            market_value   = round(total_shares * last_price, 0)
            unrealized     = round(total_shares * (last_price - avg_cost), 0)
            unrealized_pct = round((last_price - avg_cost) / avg_cost * 100, 2) if avg_cost > 0 else float('nan')

        rows.append({
            '銘柄コード'        : symbol,
            '初回エントリー日'   : pos.initial_entry_date,
            '保有ロット数'       : len(pos.lots),
            '平均取得単価（円）' : round(avg_cost, 2) if not pd.isna(avg_cost) else None,
            '期末株価（円）'     : round(last_price, 2) if not pd.isna(last_price) else None,
            '保有額（円）'       : market_value,
            '含み損益（円）'     : unrealized,
            '含み損益率（%）'    : unrealized_pct,
        })

    df = pd.DataFrame(rows).sort_values('初回エントリー日').reset_index(drop=True)

    # 銘柄名・業種を付加（meta_path が指定された場合）
    if meta_path and os.path.exists(meta_path):
        try:
            import sys
            venv_sp = os.path.join(os.path.dirname(meta_path), 'venv',
                                   'lib', 'python3.9', 'site-packages')
            if os.path.isdir(venv_sp):
                sys.path.insert(0, venv_sp)
            import xlrd
            wb  = xlrd.open_workbook(meta_path)
            ws  = wb.sheets()[0]
            hdr = [ws.cell_value(0, c) for c in range(ws.ncols)]
            ci, ni, ii = hdr.index('コード'), hdr.index('銘柄名'), hdr.index('33業種区分')
            name_map = {}
            for r in range(1, ws.nrows):
                cv   = ws.cell_value(r, ci)
                code = str(int(cv)) if isinstance(cv, float) and cv > 0 else str(cv).strip()
                if code not in name_map:
                    name_map[code] = (ws.cell_value(r, ni), ws.cell_value(r, ii))
            df['_code'] = df['銘柄コード'].str.replace('.T', '', regex=False)
            df.insert(1, '銘柄名', df['_code'].map(lambda c: name_map.get(c, ('', ''))[0]))
            df.insert(2, '業種',   df['_code'].map(lambda c: name_map.get(c, ('', ''))[1]))
            df.drop(columns=['_code'], inplace=True)
        except Exception as e:
            print(f"  ※ 銘柄名の付加に失敗しました: {e}")

    path = os.path.join(output_dir, 'open_positions.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  保有銘柄  → {path}  ({len(df)} 銘柄)")


def save_per_stock_pnl_csv(
    trades: list,
    open_positions: dict,
    closes: 'pd.DataFrame',
    end_date: str,
    output_dir: str,
    meta_path: str = None,
):
    """
    銘柄ごとの損益サマリーを CSV で保存する。

    出力列
    ------
    銘柄コード, 銘柄名（任意）, 業種（任意）,
    実現損益（円）, 実現取引ロット数,
    含み損益（円）, 含み損益率（%）,  ← 保有中のみ。未保有は 0
    合計損益（円）,
    ステータス        ← '保有中' / '決済済み'
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 実現損益の集計 ─────────────────────────────────────────────
    realized_map: dict = {}   # symbol -> {'pnl': float, 'lots': int, 'invested': float}
    if trades:
        for t in trades:
            sym = t['symbol']
            if sym not in realized_map:
                realized_map[sym] = {'pnl': 0.0, 'lots': 0, 'invested': 0.0}
            realized_map[sym]['pnl']      += t['pnl']
            realized_map[sym]['lots']     += 1
            realized_map[sym]['invested'] += t['amount']

    # ── 含み損益の集計 ─────────────────────────────────────────────
    end_ts      = pd.Timestamp(end_date)
    avail_dates = closes.index[closes.index <= end_ts]
    last_date   = avail_dates[-1] if not avail_dates.empty else None

    unrealized_map: dict = {}  # symbol -> {'unrealized_pnl': float, 'invested': float, 'last_price': float}
    for sym, pos in open_positions.items():
        total_shares   = sum(lot.shares for lot in pos.lots)
        total_invested = pos.total_invested
        avg_cost       = total_invested / total_shares if total_shares > 0 else 0.0

        last_price = float('nan')
        if last_date is not None and sym in closes.columns:
            raw = closes[sym].loc[last_date]
            if not pd.isna(raw) and raw > 0:
                last_price = float(raw)

        if not pd.isna(last_price):
            unrealized = total_shares * (last_price - avg_cost)
            unr_pct    = (last_price - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0.0
        else:
            unrealized = float('nan')
            unr_pct    = float('nan')

        unrealized_map[sym] = {
            'unrealized_pnl'     : unrealized,
            'unrealized_pct'     : unr_pct,
            'invested'           : total_invested,
            'last_price'         : last_price,
        }

    # ── 全銘柄を統合 ──────────────────────────────────────────────
    all_symbols = set(realized_map) | set(unrealized_map)
    rows = []
    for sym in sorted(all_symbols):
        r = realized_map.get(sym, {'pnl': 0.0, 'lots': 0, 'invested': 0.0})
        u = unrealized_map.get(sym)

        realized_pnl  = r['pnl']
        realized_lots = r['lots']
        is_open       = u is not None

        if is_open:
            unr_pnl  = u['unrealized_pnl']
            unr_pct  = u['unrealized_pct']
        else:
            unr_pnl  = 0.0
            unr_pct  = None

        total_pnl = realized_pnl + (unr_pnl if not pd.isna(unr_pnl) else 0.0)

        rows.append({
            '銘柄コード'        : sym,
            '実現損益（円）'     : round(realized_pnl, 0),
            '実現取引ロット数'   : realized_lots,
            '含み損益（円）'     : round(unr_pnl, 0) if not pd.isna(unr_pnl) else None,
            '含み損益率（%）'    : round(unr_pct, 2) if unr_pct is not None and not pd.isna(unr_pct) else None,
            '合計損益（円）'     : round(total_pnl, 0),
            'ステータス'         : '保有中' if is_open else '決済済み',
        })

    df = pd.DataFrame(rows).sort_values('合計損益（円）', ascending=False).reset_index(drop=True)

    # ── 銘柄名・業種を付加 ────────────────────────────────────────
    if meta_path and os.path.exists(meta_path):
        try:
            import sys
            venv_sp = os.path.join(os.path.dirname(meta_path), 'venv',
                                   'lib', 'python3.9', 'site-packages')
            if os.path.isdir(venv_sp):
                sys.path.insert(0, venv_sp)
            import xlrd
            wb  = xlrd.open_workbook(meta_path)
            ws  = wb.sheets()[0]
            hdr = [ws.cell_value(0, c) for c in range(ws.ncols)]
            ci, ni, ii = hdr.index('コード'), hdr.index('銘柄名'), hdr.index('33業種区分')
            name_map = {}
            for r_i in range(1, ws.nrows):
                cv   = ws.cell_value(r_i, ci)
                code = str(int(cv)) if isinstance(cv, float) and cv > 0 else str(cv).strip()
                if code not in name_map:
                    name_map[code] = (ws.cell_value(r_i, ni), ws.cell_value(r_i, ii))
            df['_code'] = df['銘柄コード'].str.replace('.T', '', regex=False)
            df.insert(1, '銘柄名', df['_code'].map(lambda c: name_map.get(c, ('', ''))[0]))
            df.insert(2, '業種',   df['_code'].map(lambda c: name_map.get(c, ('', ''))[1]))
            df.drop(columns=['_code'], inplace=True)
        except Exception as e:
            print(f"  ※ 銘柄名の付加に失敗しました: {e}")

    path = os.path.join(output_dir, 'per_stock_pnl.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  銘柄別損益 → {path}  ({len(df)} 銘柄)")


def plot_equity_curve(
    daily_records: list,
    output_dir: str,
    total_budget: float = None,
    index_df: 'pd.DataFrame | None' = None,
):
    """資産推移グラフ（4段構成）を PNG で保存"""
    os.makedirs(output_dir, exist_ok=True)
    if not daily_records:
        print("  データなし（グラフスキップ）")
        return

    df = pd.DataFrame(daily_records)
    df['date'] = pd.to_datetime(df['date'])

    # 万円単位に変換して見やすく
    df['total_pnl_man']        = df['total_pnl'] / 10_000
    df['realized_pnl_man']     = df['cumulative_realized_pnl'] / 10_000
    df['unrealized_pnl_man']   = df['unrealized_pnl'] / 10_000
    df['remaining_budget_man'] = df['remaining_budget'] / 10_000

    # バックテストのリターン率（%）
    if total_budget and total_budget > 0:
        df['backtest_pct'] = df['total_pnl'] / total_budget * 100
    else:
        df['backtest_pct'] = None

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle('Backtest Result', fontsize=14, fontweight='bold')

    # --- 1段目: リターン率比較（バックテスト vs 日経平均 vs TOPIX） ---
    ax = axes[0]

    if df['backtest_pct'].notna().any():
        ax.plot(df['date'], df['backtest_pct'],
                color='steelblue', lw=1.5, label='Backtest')
        ax.fill_between(df['date'], df['backtest_pct'], 0,
                        where=df['backtest_pct'] >= 0, alpha=0.15, color='steelblue')
        ax.fill_between(df['date'], df['backtest_pct'], 0,
                        where=df['backtest_pct'] < 0,  alpha=0.15, color='red')
    else:
        ax.plot(df['date'], df['total_pnl_man'], color='steelblue', lw=1.5, label='Total P&L')

    # インデックスを重ね描き
    if index_df is not None and not index_df.empty and df['backtest_pct'].notna().any():
        label_map  = {'^N225': '日経平均 (N225)', '^TOPIX': 'TOPIX'}
        color_map  = {'^N225': 'crimson',        '^TOPIX': 'darkorange'}
        start_date = df['date'].iloc[0]
        for col in index_df.columns:
            series = index_df[col].dropna()
            if series.empty:
                continue
            # バックテスト開始日以降に絞り、始値を基準に %化
            series = series[series.index >= start_date]
            if series.empty:
                continue
            base  = series.iloc[0]
            pct   = (series - base) / base * 100
            ax.plot(pct.index, pct.values,
                    color=color_map.get(col, 'gray'), lw=1.2, ls='--',
                    label=label_map.get(col, col))

        ax.set_ylabel('Return (%)')
        ax.set_title('Cumulative Return Comparison: Backtest vs Index')
    else:
        ax.set_ylabel('Man-yen (10k JPY)' if df['backtest_pct'].isna().all() else 'Return (%)')
        ax.set_title('Cumulative P&L (Realized + Unrealized)')

    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --- 2段目: 累積実現損益のみ ---
    ax = axes[1]
    ax.plot(df['date'], df['realized_pnl_man'], color='green', lw=1.5, label='Realized P&L')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.fill_between(df['date'], df['realized_pnl_man'], 0,
                    where=df['realized_pnl_man'] >= 0, alpha=0.2, color='green')
    ax.fill_between(df['date'], df['realized_pnl_man'], 0,
                    where=df['realized_pnl_man'] < 0,  alpha=0.2, color='red')
    ax.set_ylabel('Man-yen (10k JPY)')
    ax.set_title('Cumulative Realized P&L')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --- 3段目: 残り予算（現金） ---
    ax = axes[2]
    ax.plot(df['date'], df['remaining_budget_man'], color='purple', lw=1.5, label='Remaining Cash')
    ax.fill_between(df['date'], df['remaining_budget_man'],
                    df['remaining_budget_man'].min() * 0.98,
                    alpha=0.15, color='purple')
    ax.set_ylabel('Man-yen (10k JPY)')
    ax.set_title('Remaining Cash (Available Budget)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # --- 4段目: 同時保有銘柄数 ---
    ax = axes[3]
    ax.bar(df['date'], df['open_positions'], color='orange', alpha=0.7, width=1.2)
    ax.set_ylabel('# of Positions')
    ax.set_title('Open Positions')
    ax.grid(True, alpha=0.3)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    path = os.path.join(output_dir, 'equity_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  グラフ    → {path}")
