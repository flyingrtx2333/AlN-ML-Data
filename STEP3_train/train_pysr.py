import argparse
import json
import pathlib
import time
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import sympy
from loguru import logger
from pysr import PySRRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sympy import lambdify

from config import selected_features, selected_features_after
from train import load_and_clean
from utils.plot_style import (
    PAPER_AXIS_LABEL_STYLE,
    PAPER_FIGURE_BG,
    PAPER_PLOT_STYLE_CONTEXT,
    PAPER_TICK_LABEL_STYLE,
    PAPER_TITLE_STYLE,
)

logger.add("logs/runtime.log")

data_dir = pathlib.Path(__file__).parent / "data"
json_file = data_dir / "results_preprocessed.json"
default_folds_file = data_dir / "cv5_folds.json"
plt.rcParams["font.family"] = "Times New Roman"


def _style_pysr_axis(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#D1FAE5", linewidth=1.0, alpha=0.95, zorder=1)
    ax.tick_params(
        axis="both",
        labelsize=PAPER_TICK_LABEL_STYLE["fontsize"],
        direction="in",
        length=5,
        width=1.2,
        colors=PAPER_TICK_LABEL_STYLE["color"],
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(PAPER_TICK_LABEL_STYLE["fontfamily"])
        label.set_fontweight(PAPER_TICK_LABEL_STYLE["fontweight"])
        label.set_color(PAPER_TICK_LABEL_STYLE["color"])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#CBD5E1")
        ax.spines[side].set_linewidth(1.6)


def load_predefined_folds(folds_file):
    with open(folds_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    folds = payload.get("folds", [])
    if not folds:
        raise ValueError(f"No folds found in {folds_file}")
    return payload.get("meta", {}), folds


def build_name_to_idx(composition_map):
    return {str(v): int(k) for k, v in composition_map.items()}


def resolve_fold_train_indices(fold_entry, name_to_idx):
    train_names = fold_entry.get("train_names", [])
    train_idx = []
    missing = []
    for n in train_names:
        if n in name_to_idx:
            train_idx.append(name_to_idx[n])
        else:
            missing.append(n)
    if len(train_idx) == 0:
        raise ValueError(f"Fold {fold_entry.get('fold', '?')} has no mapped training samples")
    return np.array(sorted(train_idx), dtype=int), missing


def rank_equations(equations_df, top_k):
    df = equations_df.copy()
    if "loss" not in df.columns:
        raise ValueError("Cannot sort PySR equations_ without a loss column")
    eq_col = "sympy_format" if "sympy_format" in df.columns else "equation"
    df["eqn_str"] = df[eq_col].astype(str)
    df["is_constant"] = df["eqn_str"].apply(
        lambda s: int(len(sympy.sympify(s).free_symbols) == 0) if isinstance(s, str) and s else 0
    )
    complexity_col = "complexity" if "complexity" in df.columns else None
    sort_cols = ["loss", "is_constant"]
    ascending = [True, True]
    if complexity_col is not None:
        sort_cols.append(complexity_col)
        ascending.append(True)
    ranked = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return ranked.head(max(1, int(top_k)))


def evaluate_equation_on_train(eqn_str, X_train, y_train):
    try:
        feature_names = list(X_train.columns)
        symbols = sympy.symbols(feature_names)
        expr = sympy.sympify(str(eqn_str))
        if len(expr.free_symbols - set(symbols)) > 0:
            return np.nan, np.nan
        fn = lambdify(symbols, expr, "numpy")
        y_pred = np.asarray(fn(*[X_train[c].values for c in feature_names]), dtype=float).reshape(-1)
        if len(y_pred) != len(y_train) or np.any(~np.isfinite(y_pred)):
            return np.nan, np.nan
        return float(r2_score(y_train, y_pred)), float(mean_absolute_error(y_train, y_pred))
    except Exception:
        return np.nan, np.nan


def predict_equation(eqn_str, X_eval):
    feature_names = list(X_eval.columns)
    symbols = sympy.symbols(feature_names)
    expr = sympy.sympify(str(eqn_str))
    fn = lambdify(symbols, expr, "numpy")
    y_pred = np.asarray(fn(*[X_eval[c].values for c in feature_names]), dtype=float).reshape(-1)
    return y_pred


def run_fold_discovery(X_train, y_train, fold_id, niterations, output_root, top_k):
    model = PySRRegressor(
        niterations=niterations,
        populations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "exp", "log"],
        maxsize=100,
        complexity_of_operators={"*": 1, "+": 1, "exp": 2, "log": 2, "square": 2},
        model_selection="accuracy",
        procs=0,
        multithreading=True,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(X_train, y_train)
    equations_df = model.equations_.copy()

    ts = time.strftime("%Y%m%d%H%M%S")
    fold_dir = pathlib.Path(output_root) / f"{ts}-F{int(fold_id):02d}-FIT"
    fold_dir.mkdir(parents=True, exist_ok=True)

    equations_csv = fold_dir / "pysr_discovery_d33.csv"
    equations_df.to_csv(equations_csv, index=False, encoding="utf-8-sig")

    top_df = rank_equations(equations_df, top_k=top_k).copy()
    r2_list, mae_list = [], []
    for _, row in top_df.iterrows():
        r2_val, mae_val = evaluate_equation_on_train(row["eqn_str"], X_train, y_train)
        r2_list.append(r2_val)
        mae_list.append(mae_val)
    top_df["train_r2"] = r2_list
    top_df["train_mae"] = mae_list
    top_df = top_df.sort_values(["train_r2", "train_mae", "loss"], ascending=[False, True, True]).reset_index(drop=True)

    top_csv = fold_dir / "pysr_top_formulas_with_metrics.csv"
    top_df.to_csv(top_csv, index=False, encoding="utf-8-sig")
    top_txt = fold_dir / "pysr_top_formulas.txt"
    with open(top_txt, "w", encoding="utf-8") as f:
        f.write(f"Fold {fold_id} Top {len(top_df)} formulas\n")
        f.write("=" * 80 + "\n")
        for i, row in top_df.iterrows():
            loss_val = float(row["loss"]) if pd.notna(row["loss"]) else np.nan
            cmp_val = row["complexity"] if "complexity" in row.index else "NA"
            f.write(f"Rank {i + 1}\n")
            f.write(f"loss: {loss_val:.6f}\n")
            f.write(f"complexity: {cmp_val}\n")
            f.write(f"is_constant: {int(row['is_constant'])}\n")
            f.write(f"train_r2: {float(row['train_r2']):.6f}\n" if pd.notna(row["train_r2"]) else "train_r2: NaN\n")
            f.write(f"train_mae: {float(row['train_mae']):.6f}\n" if pd.notna(row["train_mae"]) else "train_mae: NaN\n")
            f.write(f"equation:\n{row['eqn_str']}\n")
            f.write("-" * 80 + "\n")

    return {
        "fold": int(fold_id),
        "n_train_samples": int(len(X_train)),
        "output_dir": str(fold_dir),
        "discovery_csv": str(equations_csv),
        "top_with_metrics_csv": str(top_csv),
        "top_formulas_txt": str(top_txt),
        "top_formulas": top_df["eqn_str"].tolist(),
        "best_formula": str(top_df.iloc[0]["eqn_str"]) if len(top_df) > 0 else "",
        "best_formula_train_r2": float(top_df.iloc[0]["train_r2"]) if len(top_df) > 0 and pd.notna(top_df.iloc[0]["train_r2"]) else np.nan,
        "best_formula_train_mae": float(top_df.iloc[0]["train_mae"]) if len(top_df) > 0 and pd.notna(top_df.iloc[0]["train_mae"]) else np.nan,
    }


def export_fold_scatter_grid(df, composition_map, folds, discovery_records, output_root):
    name_to_idx = build_name_to_idx(composition_map)
    rec_by_fold = {int(r["fold"]): r for r in discovery_records}
    n_folds = len(folds)
    ncols = 3
    nrows = int(np.ceil(n_folds / ncols))
    with plt.rc_context(PAPER_PLOT_STYLE_CONTEXT):
        fig = plt.figure(figsize=(5.4 * ncols, 4.9 * nrows), facecolor=PAPER_FIGURE_BG)
        fig.patch.set_facecolor(PAPER_FIGURE_BG)
        outer_gs = fig.add_gridspec(nrows, ncols)
        axes = []

        full_rows = n_folds // ncols
        remainder = n_folds % ncols

        for row in range(full_rows):
            for col in range(ncols):
                axes.append(fig.add_subplot(outer_gs[row, col]))

        if remainder > 0:
            tail_row = 0 if nrows == 1 else (nrows - 1)
            if remainder == 1:
                tail_gs = outer_gs[tail_row, :].subgridspec(
                    1,
                    3,
                    width_ratios=[0.55, 1.0, 0.55],
                    wspace=0.0,
                )
                axes.append(fig.add_subplot(tail_gs[0, 1]))
            elif remainder == 2:
                tail_gs = outer_gs[tail_row, :].subgridspec(
                    1,
                    5,
                    width_ratios=[0.36, 1.0, 0.08, 1.0, 0.36],
                    wspace=0.0,
                )
                axes.append(fig.add_subplot(tail_gs[0, 1]))
                axes.append(fig.add_subplot(tail_gs[0, 3]))
            else:
                tail_gs = outer_gs[tail_row, :].subgridspec(1, remainder)
                for col in range(remainder):
                    axes.append(fig.add_subplot(tail_gs[0, col]))

        for i, (ax, fold) in enumerate(zip(axes, folds)):
            fold_id = int(fold.get("fold", i + 1))
            rec = rec_by_fold.get(fold_id, {})
            eq = rec.get("best_formula", "")
            train_names = fold.get("train_names", [])
            idx = [name_to_idx[n] for n in train_names if n in name_to_idx]
            if len(idx) == 0 or not eq:
                ax.axis("off")
                continue
            X_fold = df.iloc[idx].drop(columns=["target"])
            y_fold = df.iloc[idx]["target"].values
            try:
                y_pred = predict_equation(eq, X_fold)
                r2_val = r2_score(y_fold, y_pred)
                mae_val = mean_absolute_error(y_fold, y_pred)
                lo = min(float(np.min(y_fold)), float(np.min(y_pred)))
                hi = max(float(np.max(y_fold)), float(np.max(y_pred)))
                margin = (hi - lo) * 0.08 if hi > lo else 1.0
                lower = lo - margin
                upper = hi + margin

                ax.scatter(
                    y_fold,
                    y_pred,
                    s=28,
                    alpha=0.82,
                    color="#0EA5A4",
                    edgecolors="#1F2937",
                    linewidth=0.35,
                    zorder=3,
                )
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)
                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    color="#CF4457",
                    linestyle="--",
                    linewidth=1.8,
                    zorder=2,
                )
                ax.set_aspect("equal", adjustable="box")
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_major_locator(MultipleLocator(5))
                _style_pysr_axis(ax)
                title_style = PAPER_TITLE_STYLE.copy()
                title_style["fontsize"] = 13
                ax.set_title(f"Fold {fold_id}", pad=8, **title_style)
                ax.set_xlabel(r"Observed $\mathit{d}_{33}$ (pC/N)", **PAPER_AXIS_LABEL_STYLE)
                ax.set_ylabel(r"Predicted $\mathit{d}_{33}$ (pC/N)", **PAPER_AXIS_LABEL_STYLE)
                ax.text(
                    0.70,
                    0.08,
                    f"R$^2$ = {r2_val:.3f}\nMAE = {mae_val:.2f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=PAPER_TICK_LABEL_STYLE["fontsize"],
                    fontfamily=PAPER_TICK_LABEL_STYLE["fontfamily"],
                    fontweight=PAPER_TICK_LABEL_STYLE["fontweight"],
                    color=PAPER_TICK_LABEL_STYLE["color"],
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#FFFFFF",
                        edgecolor="none",
                        alpha=0.82,
                    ),
                    zorder=4,
                )
            except Exception:
                ax.axis("off")

        fig.tight_layout()
        ts = time.strftime("%Y%m%d%H%M%S")
        out_path = pathlib.Path(output_root) / f"pysr_fold_scatter_grid_{ts}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=PAPER_FIGURE_BG)
        plt.close(fig)
    return out_path


def export_selected_formula_template(discovery_payload, output_path):
    fold_map = {}
    for item in discovery_payload.get("fold_discovery", []):
        fold_key = str(item["fold"])
        fold_map[fold_key] = item.get("top_formulas", [])[:3]

    payload = {
        "description": "Editable list of symbolic formulas for feature engineering in each fold",
        "source_discovery_file": discovery_payload.get("summary_json", ""),
        "fold_selected_formulas": fold_map,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def extract_features_from_formula(eqn_str, allowed_features):
    try:
        expr = sympy.sympify(str(eqn_str))
        symbols = {str(s) for s in expr.free_symbols}
        return sorted([s for s in symbols if s in allowed_features])
    except Exception:
        return []


def export_fold_feature_lists(discovery_payload, output_path, allowed_features, top_n_formulas=3):
    allowed = set(allowed_features)
    fold_feature_lists = {}
    for item in discovery_payload.get("fold_discovery", []):
        fold_id = str(item["fold"])
        formulas = item.get("top_formulas", [])[:max(1, int(top_n_formulas))]
        feat_set = set()
        for eq in formulas:
            feat_set.update(extract_features_from_formula(eq, allowed))
        fold_feature_lists[fold_id] = sorted(feat_set)

    payload = {
        "description": "Draft feature lists extracted from each fold's top formulas. Review them before training.",
        "source_discovery_file": discovery_payload.get("summary_json", ""),
        "top_n_formulas_per_fold": int(top_n_formulas),
        "fold_feature_lists": fold_feature_lists,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def export_formula_feature_evidence(discovery_payload, output_csv, output_md, allowed_features, top_n_formulas=5):
    allowed = set(allowed_features)
    rows = []
    formula_examples = []
    for item in discovery_payload.get("fold_discovery", []):
        fold_id = int(item["fold"])
        formulas = item.get("top_formulas", [])[:max(1, int(top_n_formulas))]
        for rank, eq in enumerate(formulas, start=1):
            feats = extract_features_from_formula(eq, allowed)
            formula_examples.append(
                {
                    "fold": fold_id,
                    "rank": rank,
                    "features": ", ".join(feats),
                    "formula": eq,
                }
            )
            for feat in feats:
                rows.append({"feature": feat, "fold": fold_id, "rank": rank})

    if rows:
        usage = pd.DataFrame(rows)
        summary = (
            usage.groupby("feature")
            .agg(
                occurrence_count=("feature", "size"),
                fold_count=("fold", "nunique"),
                best_rank=("rank", "min"),
                mean_rank=("rank", "mean"),
            )
            .reset_index()
            .sort_values(["fold_count", "occurrence_count", "best_rank"], ascending=[False, False, True])
        )
    else:
        summary = pd.DataFrame(columns=["feature", "occurrence_count", "fold_count", "best_rank", "mean_rank"])

    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# PySR feature evidence\n\n")
        f.write(f"Source: `{discovery_payload.get('summary_json', '')}`\n\n")
        f.write(f"Top formulas per fold scanned: {int(top_n_formulas)}\n\n")
        f.write("## Recurring features\n\n")
        f.write("| feature | occurrences | folds | best rank | mean rank |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for _, row in summary.iterrows():
            f.write(
                f"| `{row['feature']}` | {int(row['occurrence_count'])} | "
                f"{int(row['fold_count'])} | {int(row['best_rank'])} | {float(row['mean_rank']):.2f} |\n"
            )
        f.write("\n## Representative formulas\n\n")
        for ex in formula_examples[: min(15, len(formula_examples))]:
            f.write(f"### Fold {ex['fold']} rank {ex['rank']}\n\n")
            f.write(f"Features: {ex['features']}\n\n")
            f.write("```text\n")
            f.write(str(ex["formula"]))
            f.write("\n```\n\n")

    return output_csv, output_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit PySR and export candidate formulas for each fold without evaluating the test sets.")
    parser.add_argument("--no-swap", dest="use_swap", action="store_false", help="Disable A/B swap augmentation")
    parser.add_argument("--threshold", type=float, default=45, help="Upper d33 cutoff (default: 45, lower cutoff: 3)")
    parser.add_argument("--niterations", type=int, default=500, help="PySR iterations per fold (default: 500)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of candidate formulas to export per fold (default: 20)")
    parser.add_argument("--topn-feature-formulas", type=int, default=3, help="Number of top formulas per fold used to extract features (default: 3)")
    parser.add_argument("--folds-file", type=str, default=str(default_folds_file), help="Path to the predefined five-fold split")
    parser.add_argument("--feature-set", choices=["before", "after"], default="before", help="Feature set to use for PySR")
    args = parser.parse_args()

    active_features = selected_features_after if args.feature_set == "after" else selected_features

    df, composition_map, _ = load_and_clean(
        json_file,
        max_threshold=args.threshold,
        min_threshold=3,
        use_swap=args.use_swap,
        selected_features=active_features,
    )
    X = df.drop(columns=["target"])
    y = df["target"]

    folds_meta, folds = load_predefined_folds(args.folds_file)
    name_to_idx = build_name_to_idx(composition_map)
    logger.info(f"Loaded predefined folds: file={args.folds_file}, n_folds={len(folds)}, n_samples={len(df)}")

    output_root = data_dir / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    fold_records = []
    for fold in folds:
        fold_id = int(fold.get("fold", len(fold_records) + 1))
        train_idx, missing = resolve_fold_train_indices(fold, name_to_idx)
        if missing:
            logger.warning(f"[Fold {fold_id}] Skipping {len(missing)} training samples absent from the current feature table")
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        logger.info(f"[Fold {fold_id}] Fitting PySR on {len(X_train)} training samples")
        record = run_fold_discovery(X_train, y_train, fold_id, args.niterations, output_root, args.top_k)
        fold_records.append(record)
        logger.info(f"[Fold {fold_id}] Exported formulas to {record['discovery_csv']}")

    ts = time.strftime("%Y%m%d%H%M%S")
    summary_json = output_root / f"pysr_fold_discovery_{ts}.json"
    summary_payload = {
        "meta": {
            "created_at": ts,
            "folds_file": str(args.folds_file),
            "folds_meta": folds_meta,
            "use_swap": bool(args.use_swap),
            "threshold": float(args.threshold),
            "niterations": int(args.niterations),
            "top_k": int(args.top_k),
            "n_total_samples": int(len(df)),
            "feature_set": args.feature_set,
            "selected_features": active_features,
        },
        "fold_discovery": fold_records,
        "summary_json": str(summary_json),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"Exported the formula summary for all folds to {summary_json}")

    selected_template = data_dir / f"pysr_{args.feature_set}_selected_formulas.json"
    export_selected_formula_template(summary_payload, selected_template)
    print(f"Exported the formula selection template to {selected_template}")

    fold_feature_file = data_dir / f"pysr_{args.feature_set}_fold_feature_lists.json"
    export_fold_feature_lists(
        summary_payload,
        fold_feature_file,
        active_features,
        top_n_formulas=args.topn_feature_formulas,
    )
    print(f"Exported the optional draft feature lists to {fold_feature_file}")

    evidence_csv = data_dir / f"pysr_{args.feature_set}_feature_evidence.csv"
    evidence_md = data_dir / f"pysr_{args.feature_set}_feature_evidence.md"
    export_formula_feature_evidence(
        summary_payload,
        evidence_csv,
        evidence_md,
        active_features,
        top_n_formulas=args.topn_feature_formulas,
    )
    print(f"Exported the PySR feature evidence table to {evidence_csv}")
    print(f"Exported the PySR feature evidence report to {evidence_md}")

    scatter_grid = export_fold_scatter_grid(df, composition_map, folds, fold_records, output_root)
    print(f"Exported the five-fold scatter plots to {scatter_grid}")
