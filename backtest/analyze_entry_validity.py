# analyze_entry_validity.py: エントリー条件の妥当性検証
# 使い方: python backtest/analyze_entry_validity.py
#
# ロジック:
#   1. バックテストと同じユニバース（全期間のunion）を構築
#   2. 各銘柄の総リターンと下方偏差（年率）を算出
#   3. 下方偏差の上位25%（高ボラ）を除外
#   4. 残りの中で総リターン上位10% = 「本来捕捉すべき銘柄（勝者）」
#   5. 実際にエントリーした銘柄と照合してキャプチャ率を算出

import os
import sys
import math
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import japanize_matplotlib  # noqa: F401

# backtest.py のある backtest/ ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    load_data, build_daily_universe,
    STOCK_DATA_PATH, START_DATE, END_DATE,
    UNIVERSE_SIZE, UNIVERSE_LOOKBACK,
)

# ============================================================
# 設定
# ============================================================

TRANSACTIONS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', 'transactions.csv'
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

WINNER_TOP_PCT       = 0.10   # 上位何%を「勝者」とするか
HIGH_VOL_CUTOFF_PCT  = 0.75   # 下方偏差の上位何%をカットするか（0.75 = 上位25%を除外）
MIN_DATA_DAYS        = 30     # 分析対象とする最低データ日数
MAX_RETURN_CAP       = 2000.0 # 総リターンの上限（%）: これを超える銘柄はデータ異常とみなし除外

ENTRY_TYPES = {'新規エントリー', 'スワップ買い'}


# ============================================================
# 下方偏差の計算
# ============================================================

def downside_deviation_annual(prices: pd.Series) -> float:
    """
    バックテスト期間中の終値から日次リターンを計算し、
    負のリターンのみを使って年率換算した下方偏差（%）を返す。
    サンプル不足の場合は NaN を返す。
    """
    daily_ret = prices.pct_change().dropna()
    neg_rets  = daily_ret[daily_ret < 0]
    if len(neg_rets) < 5:
        return float('nan')
    return float(neg_rets.std() * math.sqrt(252) * 100)


# ============================================================
# 各銘柄の総リターンと下方偏差を算出
# ============================================================

def compute_stock_stats(
    union_universe: set,
    closes: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    ユニバース全銘柄の総リターンと下方偏差を返す。

    Returns
    -------
    DataFrame: symbol, total_return_pct, downside_dev_annual, data_days
    """
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)

    rows = []
    for sym in sorted(union_universe):
        if sym not in closes.columns:
            continue
        prices = closes[sym].loc[start_ts:end_ts].dropna()
        if len(prices) < MIN_DATA_DAYS:
            continue

        first_price = float(prices.iloc[0])
        last_price  = float(prices.iloc[-1])
        if first_price <= 0:
            continue

        total_ret = (last_price - first_price) / first_price * 100
        if total_ret > MAX_RETURN_CAP:  # データ異常（超低価格の新規上場等）を除外
            continue
        dd_annual = downside_deviation_annual(prices)

        rows.append({
            'symbol'             : sym,
            'total_return_pct'   : round(total_ret, 2),
            'downside_dev_annual': round(dd_annual, 2) if not math.isnan(dd_annual) else None,
            'data_days'          : len(prices),
            'first_price'        : round(first_price, 2),
            'last_price'         : round(last_price, 2),
        })

    return pd.DataFrame(rows)


# ============================================================
# 実際にエントリーした銘柄を取得
# ============================================================

def load_entered_symbols(transactions_csv: str) -> set:
    """
    transactions.csv からエントリー銘柄のセットを返す。
    新規エントリーとスワップ買いの両方を含む。
    """
    df = pd.read_csv(transactions_csv, encoding='utf-8-sig')
    entered = df[df['取引種別'].isin(ENTRY_TYPES)]['銘柄コード'].unique()
    return set(entered)


# ============================================================
# 妥当性検証のメイン処理
# ============================================================

def validate_entry(stats_df: pd.DataFrame, entered_syms: set) -> dict:
    """
    勝者の定義とキャプチャ率を計算する。

    Returns
    -------
    dict with:
        stats_df_with_flags : 全銘柄にフラグを付けた DataFrame
        winners             : 勝者 DataFrame
        high_vol_threshold  : 高ボラカット閾値（年率%）
        winner_threshold    : 勝者判定の最低リターン（%）
        capture_rate        : 勝者のうちエントリーできた割合
        precision           : エントリーしたうちの勝者割合
    """
    df = stats_df.dropna(subset=['downside_dev_annual']).copy()

    # ── 高ボラティリティのカット ──────────────────────────────────────
    high_vol_threshold = df['downside_dev_annual'].quantile(HIGH_VOL_CUTOFF_PCT)
    df['is_high_vol'] = df['downside_dev_annual'] > high_vol_threshold

    low_vol_df = df[~df['is_high_vol']].copy()

    # ── 勝者（総リターン上位10%）────────────────────────────────────
    winner_threshold = low_vol_df['total_return_pct'].quantile(1 - WINNER_TOP_PCT)
    low_vol_df['is_winner'] = low_vol_df['total_return_pct'] >= winner_threshold

    # ── エントリーフラグ ─────────────────────────────────────────────
    low_vol_df['is_entered'] = low_vol_df['symbol'].isin(entered_syms)

    winners    = low_vol_df[low_vol_df['is_winner']]
    w_entered  = winners[winners['is_entered']]
    non_winner_entered = low_vol_df[~low_vol_df['is_winner'] & low_vol_df['is_entered']]

    capture_rate = len(w_entered) / len(winners) * 100 if len(winners) > 0 else 0.0
    precision    = len(w_entered) / len(low_vol_df[low_vol_df['is_entered']]) * 100 \
                   if len(low_vol_df[low_vol_df['is_entered']]) > 0 else 0.0

    # 全銘柄DFにもフラグを付与（グラフ用）
    df = df.copy()
    df['is_winner']  = df['symbol'].isin(winners['symbol'])
    df['is_entered'] = df['symbol'].isin(entered_syms)

    return {
        'stats_df'           : low_vol_df,
        'all_df'             : df,
        'winners'            : winners,
        'w_entered'          : w_entered,
        'non_winner_entered' : non_winner_entered,
        'high_vol_threshold' : high_vol_threshold,
        'winner_threshold'   : winner_threshold,
        'capture_rate'       : capture_rate,
        'precision'          : precision,
    }


# ============================================================
# サマリー表示
# ============================================================

def print_summary(res: dict, entered_syms: set, stats_df: pd.DataFrame):
    lv      = res['stats_df']        # 低ボラ銘柄全体
    winners = res['winners']
    we      = res['w_entered']
    nwe     = res['non_winner_entered']

    # エントリーした銘柄のうち高ボラで除外されたもの
    all_df         = res['all_df']
    high_vol_syms  = set(all_df[all_df['is_high_vol']]['symbol'])
    entered_highvol = entered_syms & high_vol_syms

    print("\n" + "=" * 65)
    print("  エントリー条件 妥当性検証")
    print(f"  勝者定義: 低ボラ銘柄の総リターン上位{WINNER_TOP_PCT*100:.0f}%")
    print(f"  高ボラ除外: 下方偏差 > {res['high_vol_threshold']:.1f}%/年（上位25%）")
    print("=" * 65)

    print(f"\n  ── ユニバース銘柄数の内訳 ──")
    print(f"  ユニバース総銘柄数（十分なデータあり） : {len(stats_df):>5} 銘柄")
    print(f"  うち 高ボラ除外（下方偏差上位25%）    : {len(all_df[all_df['is_high_vol']]):>5} 銘柄")
    print(f"  うち 低ボラ対象銘柄                   : {len(lv):>5} 銘柄")
    print(f"  うち 勝者（総リターン上位{WINNER_TOP_PCT*100:.0f}%）       : {len(winners):>5} 銘柄"
          f"  （閾値: +{res['winner_threshold']:.1f}%以上）")

    print(f"\n  ── エントリー結果 ──")
    n_entered_lowvol = lv['is_entered'].sum()
    print(f"  エントリーした銘柄（低ボラ内）         : {n_entered_lowvol:>5} 銘柄")
    print(f"  エントリーした銘柄（高ボラ・除外対象） : {len(entered_highvol):>5} 銘柄")
    print(f"  ✅ 勝者をエントリーできた（捕捉）      : {len(we):>5} 銘柄")
    print(f"  ❌ 勝者をエントリーできなかった（見逃）: {len(winners) - len(we):>5} 銘柄")
    print(f"  ⚠️  勝者ではないのにエントリーした      : {len(nwe):>5} 銘柄")

    print(f"\n  ── 精度指標 ──")
    print(f"  キャプチャ率（再現率）  : {res['capture_rate']:>5.1f}%"
          f"  ← 勝者のうちエントリーできた割合")
    print(f"  精度（Precision）       : {res['precision']:>5.1f}%"
          f"  ← エントリーしたうち実際に勝者だった割合")

    # 見逃した勝者リスト
    missed_winners = winners[~winners['is_entered']].sort_values(
        'total_return_pct', ascending=False
    )
    if not missed_winners.empty:
        print(f"\n  ── 見逃した勝者（リターン降順）──")
        print(missed_winners[['symbol', 'total_return_pct', 'downside_dev_annual',
                               'data_days']].to_string(index=False))

    # 捕捉できた勝者リスト
    print(f"\n  ── 捕捉できた勝者（リターン降順）──")
    print(we.sort_values('total_return_pct', ascending=False)[
        ['symbol', 'total_return_pct', 'downside_dev_annual', 'data_days']
    ].to_string(index=False))

    print("=" * 65)


# ============================================================
# 可視化
# ============================================================

def plot_results(res: dict, output_dir: str):
    lv      = res['stats_df']
    all_df  = res['all_df']
    labels  = ['見逃した勝者', '捕捉できた勝者', '非勝者エントリー', '未エントリー低ボラ', '高ボラ除外']

    # ── 図1: 下方偏差の分布とカット閾値 ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    vals = all_df['downside_dev_annual'].dropna()
    ax.hist(vals, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(res['high_vol_threshold'], color='red', linestyle='--', linewidth=1.5,
               label=f'高ボラ閾値（{res["high_vol_threshold"]:.1f}%）')
    ax.set_xlabel('下方偏差（年率 %）')
    ax.set_ylabel('銘柄数')
    ax.set_title('下方偏差の分布（ユニバース全銘柄）')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, 'entry_validity_vol_dist.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図2: 総リターンの分布（低ボラ銘柄）+ 勝者閾値 ───────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lv['total_return_pct'], bins=60, color='steelblue', alpha=0.7,
            edgecolor='white', label='低ボラ銘柄')
    ax.axvline(res['winner_threshold'], color='orange', linestyle='--', linewidth=1.5,
               label=f'勝者閾値（+{res["winner_threshold"]:.1f}%）')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.6)
    ax.set_xlabel('総リターン（%）')
    ax.set_ylabel('銘柄数')
    ax.set_title(f'総リターン分布（低ボラ銘柄・上位{WINNER_TOP_PCT*100:.0f}%が勝者）')
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    out = os.path.join(output_dir, 'entry_validity_return_dist.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図3: 散布図（総リターン vs 下方偏差、カテゴリ別に色分け）──────
    def _cat(row):
        if row['is_high_vol']:
            return '高ボラ除外'
        if row['is_winner'] and row['is_entered']:
            return '捕捉できた勝者'
        if row['is_winner'] and not row['is_entered']:
            return '見逃した勝者'
        if not row['is_winner'] and row['is_entered']:
            return '非勝者エントリー'
        return '未エントリー低ボラ'

    all_df = all_df.copy()
    all_df['category'] = all_df.apply(_cat, axis=1)

    color_map = {
        '捕捉できた勝者'  : '#27ae60',
        '見逃した勝者'    : '#e74c3c',
        '非勝者エントリー': '#e67e22',
        '未エントリー低ボラ': '#bdc3c7',
        '高ボラ除外'      : '#95a5a6',
    }
    size_map = {
        '捕捉できた勝者'  : 60,
        '見逃した勝者'    : 80,
        '非勝者エントリー': 50,
        '未エントリー低ボラ': 15,
        '高ボラ除外'      : 10,
    }
    alpha_map = {
        '捕捉できた勝者'  : 0.9,
        '見逃した勝者'    : 0.9,
        '非勝者エントリー': 0.8,
        '未エントリー低ボラ': 0.25,
        '高ボラ除外'      : 0.15,
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    for cat in ['高ボラ除外', '未エントリー低ボラ', '非勝者エントリー',
                '見逃した勝者', '捕捉できた勝者']:
        sub = all_df[all_df['category'] == cat]
        if sub.empty:
            continue
        ax.scatter(sub['total_return_pct'], sub['downside_dev_annual'],
                   c=color_map[cat], s=size_map[cat], alpha=alpha_map[cat],
                   label=f'{cat}（{len(sub)}銘柄）', zorder=3 if cat in {'捕捉できた勝者', '見逃した勝者'} else 1)

    # 銘柄ラベル（勝者のみ表示）
    for _, row in all_df[all_df['is_winner']].iterrows():
        ax.annotate(row['symbol'],
                    (row['total_return_pct'], row['downside_dev_annual']),
                    fontsize=6, alpha=0.8, xytext=(3, 3),
                    textcoords='offset points')

    ax.axvline(res['winner_threshold'], color='orange', linestyle=':', linewidth=1.2,
               label=f'勝者閾値（+{res["winner_threshold"]:.1f}%）')
    ax.axhline(res['high_vol_threshold'], color='red', linestyle=':', linewidth=1.2,
               label=f'高ボラ閾値（{res["high_vol_threshold"]:.1f}%）')
    ax.set_xlabel('総リターン（%）')
    ax.set_ylabel('下方偏差（年率 %）')
    ax.set_title('総リターン vs 下方偏差（カテゴリ別）')
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    out = os.path.join(output_dir, 'entry_validity_scatter.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")

    # ── 図4: 混同行列（2×2）───────────────────────────────────────────
    tp = len(res['w_entered'])
    fn = len(res['winners']) - tp
    fp = len(res['non_winner_entered'])
    tn = len(lv) - tp - fn - fp

    matrix = np.array([[tp, fn], [fp, tn]])
    row_labels_m = ['勝者（実際）', '非勝者（実際）']
    col_labels_m = ['エントリーした', 'エントリーしなかった']

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels_m)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(row_labels_m)
    cell_labels = [['TP（捕捉）', 'FN（見逃）'], ['FP（空振り）', 'TN（正しく回避）']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{cell_labels[i][j]}\n{matrix[i, j]}銘柄',
                    ha='center', va='center', fontsize=10,
                    color='white' if matrix[i, j] > matrix.max() * 0.6 else 'black')
    ax.set_title(
        f'エントリー判定 混同行列（低ボラ銘柄 {len(lv)}銘柄）\n'
        f'キャプチャ率={res["capture_rate"]:.1f}%  精度={res["precision"]:.1f}%'
    )
    plt.tight_layout()
    out = os.path.join(output_dir, 'entry_validity_confusion.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  保存: {out}")


# ============================================================
# メイン
# ============================================================

if __name__ == '__main__':
    for path, name in [(STOCK_DATA_PATH, '株価データ'), (TRANSACTIONS_CSV, 'transactions.csv')]:
        if not os.path.exists(path):
            print(f"[エラー] {name} が見つかりません: {path}")
            sys.exit(1)

    # --- データ読み込み ---
    closes, opens, volume = load_data(STOCK_DATA_PATH)

    # --- ユニバース構築 ---
    print(f"\nuniverse構築中（lookback={UNIVERSE_LOOKBACK}日, size={UNIVERSE_SIZE}）...")
    daily_universe, _ = build_daily_universe(
        closes, volume,
        start_date=START_DATE, end_date=END_DATE,
        lookback=UNIVERSE_LOOKBACK, size=UNIVERSE_SIZE,
    )

    # バックテスト期間に登場したすべての銘柄を union で取得
    union_universe = set()
    for syms in daily_universe.values():
        union_universe |= syms
    print(f"ユニバースのunion: {len(union_universe)} 銘柄")

    # --- 各銘柄の統計算出 ---
    print("\n各銘柄の総リターン・下方偏差を算出中...")
    stats_df = compute_stock_stats(union_universe, closes, START_DATE, END_DATE)
    print(f"  → 有効銘柄数: {len(stats_df)} 銘柄（{MIN_DATA_DAYS}日以上データあり）")

    # --- エントリー済み銘柄の取得 ---
    entered_syms = load_entered_symbols(TRANSACTIONS_CSV)
    print(f"エントリーした銘柄数: {len(entered_syms)} 銘柄")

    # --- 妥当性検証 ---
    print("\n妥当性検証中...")
    res = validate_entry(stats_df, entered_syms)

    # --- サマリー表示 ---
    print_summary(res, entered_syms, stats_df)

    # --- CSV 出力 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, 'entry_validity.csv')
    res['stats_df'].to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n詳細CSV保存: {out_csv}")

    # --- グラフ出力 ---
    print("\nグラフ生成中...")
    try:
        plot_results(res, OUTPUT_DIR)
    except Exception as e:
        import traceback
        print(f"  ※ グラフ生成中にエラー: {e}")
        traceback.print_exc()

    print(f"\n完了。出力先: {OUTPUT_DIR}\n")
