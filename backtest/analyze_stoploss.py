# analyze_stoploss.py: 損切り後の株価追跡・取り逃がし分析
# 使い方: python backtest/analyze_stoploss.py
# ※ 既存コードは一切変更しない。results/ の trades.csv と株価データを読むだけ。

import os
import sys
import math

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import japanize_matplotlib  # noqa: F401  日本語フォント

# ============================================================
# 設定
# ============================================================

TRADES_CSV      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'trades.csv')
STOCK_DATA_PATH = os.path.expanduser('~/my-stock-app/stock_data/stock_data_202604111404.csv')
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# 追跡する経過営業日数
HORIZONS = [30, 60, 90, 180, 360]

# 「取り逃がし」判定: 各期間の日経平均リターン + この値（%）以上なら取り逃がし
NIKKEI_ALPHA = 30.0


# ============================================================
# 株価データ読み込み（Close のみ）
# ============================================================

def load_closes(path: str) -> pd.DataFrame:
    print(f"株価データ読み込み中: {path}")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        closes = df['Close']
    else:
        closes = df
    closes = closes.sort_index().dropna(axis=1, how='all')
    print(f"  → {len(closes)} 日分 × {len(closes.columns)} 銘柄")
    return closes


# ============================================================
# 日経平均データ取得
# ============================================================

def load_nikkei(start_date: str, end_date: str) -> pd.Series:
    """
    yfinance で ^N225 を取得して終値 Series を返す。
    取得失敗時は空の Series を返す（閾値を NIKKEI_ALPHA 固定にフォールバック）。
    """
    try:
        import yfinance as yf
        print("日経平均データ取得中（^N225）...")
        df = yf.download('^N225', start=start_date, end=end_date,
                         progress=False, auto_adjust=True)
        close = df['Close'].squeeze().dropna()
        close.index = pd.to_datetime(close.index)
        print(f"  → {len(close)} 日分取得")
        return close
    except Exception as e:
        print(f"  ※ 日経平均の取得に失敗（フォールバック: 閾値={NIKKEI_ALPHA}%固定）: {e}")
        return pd.Series(dtype=float)


def nikkei_return_after(nikkei: pd.Series, from_date: pd.Timestamp, n_bdays: int) -> 'float | None':
    """
    from_date 以降の日経平均について、N 営業日後のリターン（%）を返す。
    データが不足している場合は None を返す。
    """
    if nikkei.empty:
        return None
    future = nikkei[nikkei.index > from_date]
    if future.empty:
        return None
    base_price = float(future.iloc[0])           # 損切り翌営業日の価格を基準にする
    if len(future) >= n_bdays:
        target_price = float(future.iloc[n_bdays - 1])
    else:
        target_price = float(future.iloc[-1])    # データ末尾で代替
    if base_price <= 0:
        return None
    return (target_price - base_price) / base_price * 100


# ============================================================
# 再エントリー済み銘柄の検出
# ============================================================

def build_reentry_map(trades_df: pd.DataFrame) -> dict:
    """
    losses（損切り）後に同銘柄への再エントリーが発生しているかを確認する。

    Returns
    -------
    reentry_map : {(symbol, stop_date) -> earliest_reentry_date}
        損切り後に再エントリーした最初の日付。再エントリーがない場合はキーに含まれない。
    """
    reentry_map = {}
    stop_rows = trades_df[trades_df['決済理由'] == '損切り']

    for _, row in stop_rows.iterrows():
        symbol    = row['銘柄コード']
        stop_date = pd.Timestamp(row['最終決済日'])

        # 同銘柄で初回エントリー日が損切り日より後のレコードを探す
        later = trades_df[
            (trades_df['銘柄コード'] == symbol) &
            (pd.to_datetime(trades_df['初回エントリー日']) > stop_date)
        ]
        if not later.empty:
            earliest = pd.to_datetime(later['初回エントリー日']).min()
            reentry_map[(symbol, stop_date)] = earliest

    return reentry_map


# ============================================================
# 損切り後の追跡
# ============================================================

def track_after_stoploss(
    stops_df: pd.DataFrame,
    closes: pd.DataFrame,
    nikkei: pd.Series,
    reentry_map: dict,
) -> pd.DataFrame:
    """
    各損切りトレードについて、損切り日以降の株価推移を追跡する。

    追加列
    ------
    reentry_date   : 損切り後に同銘柄へ再エントリーした日（なければ NaT）
    future_N       : 損切り日から N 営業日後の終値
    ret_N          : 損切り価格に対するリターン（%）
    nikkei_ret_N   : 同期間の日経平均リターン（%）
    threshold_N    : 取り逃がし判定閾値（nikkei_ret_N + NIKKEI_ALPHA）
    missed_N       : その期間単独で取り逃がし条件を満たすか
    missed         : いずれかの期間で取り逃がし条件を満たすか（再エントリー済みは False）
    verdict        : "取り逃がし" / "損切り正解" / "再エントリー済み" / "データなし"
    entry_loss_pct : エントリー価格に対する損切り時の損失率（%）
    hold_days      : エントリーから損切りまでのカレンダー日数
    """
    rows      = []
    all_dates = closes.index

    for _, row in stops_df.iterrows():
        symbol      = row['銘柄コード']
        stop_date   = pd.Timestamp(row['最終決済日'])
        stop_price  = float(row['平均決済価格（円）'])
        entry_price = float(row['平均購入単価（円）'])
        entry_date  = pd.Timestamp(row['初回エントリー日'])

        reentry_date = reentry_map.get((symbol, stop_date), pd.NaT)
        is_reentered = not pd.isna(reentry_date)

        rec = {
            'symbol'         : symbol,
            'entry_date'     : entry_date,
            'stop_date'      : stop_date,
            'entry_price'    : entry_price,
            'stop_price'     : stop_price,
            'entry_loss_pct' : round((stop_price - entry_price) / entry_price * 100, 2),
            'hold_days'      : (stop_date - entry_date).days,
            'reentry_date'   : reentry_date,
        }

        if symbol not in closes.columns:
            for h in HORIZONS:
                rec[f'future_{h}']      = None
                rec[f'ret_{h}']         = None
                rec[f'nikkei_ret_{h}']  = None
                rec[f'threshold_{h}']   = None
                rec[f'missed_{h}']      = False
            rec['missed']  = False
            rec['verdict'] = 'データなし'
            rows.append(rec)
            continue

        future_dates = all_dates[all_dates > stop_date]
        prices_after = closes[symbol].reindex(future_dates)

        any_missed = False
        for h in HORIZONS:
            nk_ret    = nikkei_return_after(nikkei, stop_date, h)
            threshold = (nk_ret + NIKKEI_ALPHA) if nk_ret is not None else NIKKEI_ALPHA

            if len(future_dates) >= h:
                future_price  = prices_after.iloc[h - 1]
            elif len(future_dates) > 0:
                future_price  = prices_after.iloc[-1]
            else:
                rec[f'future_{h}']     = None
                rec[f'ret_{h}']        = None
                rec[f'nikkei_ret_{h}'] = round(nk_ret, 2) if nk_ret is not None else None
                rec[f'threshold_{h}']  = round(threshold, 2)
                rec[f'missed_{h}']     = False
                continue

            if pd.isna(future_price) or future_price <= 0:
                rec[f'future_{h}']     = None
                rec[f'ret_{h}']        = None
                rec[f'nikkei_ret_{h}'] = round(nk_ret, 2) if nk_ret is not None else None
                rec[f'threshold_{h}']  = round(threshold, 2)
                rec[f'missed_{h}']     = False
                continue

            ret       = (float(future_price) - stop_price) / stop_price * 100
            missed_h  = (ret >= threshold)

            rec[f'future_{h}']     = round(float(future_price), 2)
            rec[f'ret_{h}']        = round(ret, 2)
            rec[f'nikkei_ret_{h}'] = round(nk_ret, 2) if nk_ret is not None else None
            rec[f'threshold_{h}']  = round(threshold, 2)
            rec[f'missed_{h}']     = missed_h

            if missed_h:
                any_missed = True

        # 再エントリー済みは取り逃がしから除外
        if is_reentered:
            rec['missed']  = False
            rec['verdict'] = '再エントリー済み'
        elif any_missed:
            rec['missed']  = True
            rec['verdict'] = '取り逃がし'
        else:
            rec['missed']  = False
            rec['verdict'] = '損切り正解'

        rows.append(rec)

    return pd.DataFrame(rows)


# ============================================================
# 集計サマリー表示
# ============================================================

def print_summary(result: pd.DataFrame):
    total         = len(result)
    missed_df     = result[result['verdict'] == '取り逃がし']
    correct_df    = result[result['verdict'] == '損切り正解']
    reentry_df    = result[result['verdict'] == '再エントリー済み']
    no_data_df    = result[result['verdict'] == 'データなし']

    print("\n" + "=" * 65)
    print("  損切り後の追跡分析 サマリー")
    print(f"  取り逃がし判定条件: 日経平均リターン + {NIKKEI_ALPHA:.0f}% 以上")
    print("=" * 65)
    print(f"  損切り総件数        : {total} 件")
    print(f"  取り逃がし          : {len(missed_df)} 件  ({len(missed_df)/total*100:.1f}%)")
    print(f"  損切り正解          : {len(correct_df)} 件  ({len(correct_df)/total*100:.1f}%)")
    print(f"  再エントリー済み    : {len(reentry_df)} 件  ({len(reentry_df)/total*100:.1f}%)")
    if len(no_data_df) > 0:
        print(f"  データなし          : {len(no_data_df)} 件")

    if not missed_df.empty:
        missed_loss = missed_df['entry_loss_pct'].apply(
            lambda x: abs(x) / 100 * 70_000
        ).sum()
        print(f"\n  取り逃がし時の確定損失合計（概算）: {missed_loss:,.0f} 円")

    # 各期間での統計（全件）
    print(f"\n  ── 損切り後のリターン分布（全件・日経平均との比較）──")
    print(f"  {'期間':>8}  {'株価中央値':>10}  {'日経中央値':>10}  {'超過中央値':>10}  {'閾値超え件数':>12}")
    for h in HORIZONS:
        ret_col  = f'ret_{h}'
        nk_col   = f'nikkei_ret_{h}'
        thr_col  = f'threshold_{h}'
        miss_col = f'missed_{h}'

        ret_vals = result[ret_col].dropna()
        nk_vals  = result[nk_col].dropna()
        thr_ok   = result[miss_col].sum() if miss_col in result.columns else 0

        excess_vals = (result[ret_col] - result[nk_col]).dropna()

        if ret_vals.empty:
            continue
        print(
            f"  {h:>5}日後  "
            f"{ret_vals.median():>+9.1f}%  "
            f"{nk_vals.median():>+9.1f}%  "
            f"{excess_vals.median():>+9.1f}%  "
            f"{thr_ok:>9}件"
        )

    # 取り逃がし銘柄リスト
    if not missed_df.empty:
        print(f"\n  ── 取り逃がし銘柄一覧 ──")
        show_cols = (
            ['symbol', 'stop_date', 'entry_loss_pct', 'hold_days'] +
            [f'ret_{h}' for h in HORIZONS] +
            [f'threshold_{h}' for h in HORIZONS]
        )
        show_cols = [c for c in show_cols if c in missed_df.columns]
        # 最長期間のリターンで降順
        sort_col = f'ret_{HORIZONS[-1]}'
        disp = missed_df.sort_values(sort_col, ascending=False) if sort_col in missed_df.columns else missed_df
        print(disp[show_cols].to_string(index=False))

    # 再エントリー済み銘柄リスト
    if not reentry_df.empty:
        print(f"\n  ── 再エントリー済み銘柄（除外対象）──")
        show_cols = ['symbol', 'stop_date', 'entry_loss_pct', 'reentry_date'] + \
                    [f'ret_{h}' for h in HORIZONS]
        show_cols = [c for c in show_cols if c in reentry_df.columns]
        print(reentry_df[show_cols].to_string(index=False))

    print("=" * 65)


# ============================================================
# 可視化
# ============================================================

def plot_results(result: pd.DataFrame, output_dir: str):
    missed_df  = result[result['verdict'] == '取り逃がし']
    correct_df = result[result['verdict'] == '損切り正解']
    labels     = [f'{h}日後' for h in HORIZONS]

    # ── 図1: 各期間のリターン分布（ボックスプロット + 日経平均オーバーレイ）─
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_horizon = [result[f'ret_{h}'].dropna().values for h in HORIZONS]
    bp = ax.boxplot(data_by_horizon, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#d9534f' if d.mean() < 0 else '#5cb85c' for d in data_by_horizon]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # 日経平均中央値ライン
    nk_medians = [result[f'nikkei_ret_{h}'].dropna().median() for h in HORIZONS]
    thr_medians = [nk + NIKKEI_ALPHA for nk in nk_medians]
    ax.plot(range(1, len(HORIZONS) + 1), nk_medians, 'b--', linewidth=1.2,
            label='日経平均リターン（中央値）')
    ax.plot(range(1, len(HORIZONS) + 1), thr_medians, color='orange', linestyle=':',
            linewidth=1.5, label=f'取り逃がし閾値（日経+{NIKKEI_ALPHA:.0f}%）')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.6)

    ax.set_title('損切り後のリターン分布（損切り価格起点）')
    ax.set_ylabel('リターン（%）')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    out = os.path.join(output_dir, 'stoploss_return_boxplot.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図2: 損切り後の平均リターン推移（グループ別）+ 日経平均・閾値 ────
    fig, ax = plt.subplots(figsize=(10, 5))
    for grp, label, color in [
        (missed_df,  '取り逃がし',   '#e67e22'),
        (correct_df, '損切り正解',   '#2980b9'),
        (result,     '全件',         '#7f8c8d'),
    ]:
        if grp.empty:
            continue
        means = [grp[f'ret_{h}'].mean() for h in HORIZONS]
        ax.plot(labels, means, marker='o', label=f'{label} (n={len(grp)})',
                color=color, linewidth=2)

    ax.plot(labels, nk_medians, 'b--', linewidth=1.2, label='日経平均リターン（中央値）')
    ax.plot(labels, thr_medians, color='orange', linestyle=':', linewidth=1.5,
            label=f'取り逃がし閾値（日経+{NIKKEI_ALPHA:.0f}%）')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.6)
    ax.set_title('損切り後の平均リターン推移（グループ別）')
    ax.set_ylabel('平均リターン（%）')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    out = os.path.join(output_dir, 'stoploss_avg_return.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図3: 個別銘柄ヒートマップ（株価リターン - 日経リターン = 超過リターン）
    excess_cols = []
    for h in HORIZONS:
        col = f'excess_{h}'
        result[col] = result[f'ret_{h}'] - result[f'nikkei_ret_{h}']
        excess_cols.append(col)

    # 同一銘柄が複数回登場するためユニークな行ラベルを作る（銘柄+損切り日）
    row_labels  = (result['symbol'] + '\n' + result['stop_date'].astype(str).str[:10]).tolist()
    verdicts    = result['verdict'].tolist()
    hmap_values = result[excess_cols].values

    n = len(result)
    fig_h = max(5, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(hmap_values, aspect='auto', cmap='RdYlGn', vmin=-40, vmax=40)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=7)

    for i in range(n):
        for j in range(len(excess_cols)):
            val = hmap_values[i, j]
            if val == val:  # not NaN
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < 30 else 'white')

    plt.colorbar(im, ax=ax, label='超過リターン（%）')
    ax.set_title(f'各銘柄の超過リターン（株価リターン - 日経リターン）\n'
                 f'橙色=取り逃がし / 青色=再エントリー済み')

    color_map = {'取り逃がし': '#e67e22', '損切り正解': '#2c3e50',
                 '再エントリー済み': '#2980b9', 'データなし': '#95a5a6'}
    for i, verdict in enumerate(verdicts):
        c = color_map.get(verdict, 'black')
        ax.get_yticklabels()[i].set_color(c)
        if verdict == '取り逃がし':
            ax.get_yticklabels()[i].set_fontweight('bold')

    plt.tight_layout()
    out = os.path.join(output_dir, 'stoploss_heatmap.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図4: 損失率 vs 損切り後360日間リターン（散布図）────────────────
    last_h    = HORIZONS[-1]
    x_col     = 'entry_loss_pct'
    y_col     = f'ret_{last_h}'
    thr_col   = f'threshold_{last_h}'
    plot_data = result[[x_col, y_col, thr_col, 'symbol', 'verdict']].dropna()

    fig, ax = plt.subplots(figsize=(9, 6))
    color_map2 = {'取り逃がし': '#e67e22', '損切り正解': '#2980b9',
                  '再エントリー済み': '#27ae60', 'データなし': '#bdc3c7'}
    for v, c in color_map2.items():
        sub = plot_data[plot_data['verdict'] == v]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub[y_col], c=c, label=v, alpha=0.8, s=70)
        for _, r in sub.iterrows():
            ax.annotate(r['symbol'], (r[x_col], r[y_col]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords='offset points')

    # 閾値は銘柄ごとに異なるので、中央値を参考ラインとして表示
    median_thr = plot_data[thr_col].median()
    ax.axhline(median_thr, color='orange', linestyle=':', linewidth=1.5,
               label=f'取り逃がし閾値（中央値 +{median_thr:.1f}%）')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6)
    ax.set_xlabel('エントリーからの損失率（%）')
    ax.set_ylabel(f'損切り後{last_h}日後リターン（%）')
    ax.set_title(f'損切り損失率 vs 損切り後{last_h}日後リターン')
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    out = os.path.join(output_dir, 'stoploss_scatter.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # 超過リターン列は一時的なので削除
    result.drop(columns=excess_cols, inplace=True, errors='ignore')


# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':
    # --- ファイル存在チェック ---
    for path, name in [(TRADES_CSV, 'trades.csv'), (STOCK_DATA_PATH, '株価データ')]:
        if not os.path.exists(path):
            print(f"[エラー] {name} が見つかりません: {path}")
            sys.exit(1)

    # --- trades.csv 読み込み ---
    trades_df = pd.read_csv(TRADES_CSV, encoding='utf-8-sig')
    stops_df  = trades_df[trades_df['決済理由'] == '損切り'].copy()
    print(f"損切りトレード: {len(stops_df)} 件 / 全 {len(trades_df)} 件")

    if stops_df.empty:
        print("損切りトレードがありません。")
        sys.exit(0)

    # --- 再エントリー済みマップ構築 ---
    reentry_map = build_reentry_map(trades_df)
    print(f"損切り後に再エントリーした銘柄: {len(reentry_map)} 件")

    # --- 株価データ読み込み ---
    closes = load_closes(STOCK_DATA_PATH)

    # --- 日経平均取得（バックテスト期間の前後をカバー）---
    all_stops_min = pd.to_datetime(stops_df['最終決済日']).min()
    all_stops_max = pd.to_datetime(stops_df['最終決済日']).max()
    nk_start = (all_stops_min - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    nk_end   = (all_stops_max + pd.Timedelta(days=400)).strftime('%Y-%m-%d')
    nikkei = load_nikkei(nk_start, nk_end)

    # --- 損切り後の追跡 ---
    print("\n損切り後の株価を追跡中...")
    result = track_after_stoploss(stops_df, closes, nikkei, reentry_map)

    # --- サマリー表示 ---
    print_summary(result)

    # --- CSV 出力 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, 'stoploss_analysis.csv')
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n詳細CSV保存: {out_csv}")

    # --- グラフ出力 ---
    print("\nグラフ生成中...")
    try:
        plot_results(result, OUTPUT_DIR)
    except Exception as e:
        print(f"  ※ グラフ生成中にエラー: {e}")

    print(f"\n完了。出力先: {OUTPUT_DIR}\n")
