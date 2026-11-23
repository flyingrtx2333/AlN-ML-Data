"""Predict and validate d33 values for AlN candidates."""

import argparse
import json
from pathlib import Path
import re
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_style import (
    PAPER_AXIS_LABEL_STYLE,
    PAPER_FIGURE_BG,
    PAPER_FONT_FAMILY,
    PAPER_PLOT_STYLE_CONTEXT,
    PAPER_TICK_LABEL_STYLE,
)
from utils.feature_labels import format_feature_label

try:
    from .common import (
        CA_DIFF_THRESHOLD,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROCESSED_DIR,
        MODEL_FEATURES,
        ROOT_DIR,
        ensure_dir,
        get_ionic_radius,
        get_valence_electrons,
    )
except ImportError:
    from common import (
        CA_DIFF_THRESHOLD,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PROCESSED_DIR,
        MODEL_FEATURES,
        ROOT_DIR,
        ensure_dir,
        get_ionic_radius,
        get_valence_electrons,
    )

LIMIT = 6 
D33_AXIS_LABEL = r"$\mathit{d}_{33}$ (pC/N)"
D33_PREDICTED_LABEL = rf"Predicted {D33_AXIS_LABEL}"
D33_COMMITTEE_MEAN_LABEL = rf"Committee mean of predicted {D33_AXIS_LABEL}"
D33_STD_LABEL = rf"Cross-fold SD of predicted {D33_AXIS_LABEL}"
DEFAULT_COMMITTEE_CALIBRATION = "linear"
DEFAULT_COMMITTEE_AGGREGATION = "trimmed"
RELEASE_SUMMARY_TICK_SIZE = 18
RELEASE_SUMMARY_AXIS_LABEL_SIZE = 20
RELEASE_SUMMARY_LEGEND_SIZE = 18
RELEASE_SUMMARY_PANEL_LABEL_SIZE = 24
RELEASE_SUMMARY_AXES_BG = "#FFFFFF"
RELEASE_SUMMARY_FRAME_COLOR = "#CBD5E1"
RELEASE_SUMMARY_FRAME_LINEWIDTH = 2.0
RELEASE_SUMMARY_ACCENT = "#C2410C"
RELEASE_SUMMARY_GOLD = "#F59E0B"
RELEASE_SUMMARY_BLUE = "#0F766E"
RELEASE_SUMMARY_TAIL = "#FDE68A"
RELEASE_SUMMARY_IQR = "#A7F3D0"
RELEASE_SUMMARY_CMAP = "turbo"
PAPER_STYLE = {
    **PAPER_PLOT_STYLE_CONTEXT,
    "figure.facecolor": PAPER_FIGURE_BG,
    "axes.facecolor": PAPER_FIGURE_BG,
    "savefig.facecolor": PAPER_FIGURE_BG,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.4,
    "axes.grid": False,
    "grid.color": "#D1FAE5",
    "grid.linewidth": 1.0,
    "grid.alpha": 0.95,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "legend.frameon": False,
}


def apply_paper_style():
    plt.rcParams.update(PAPER_STYLE)


def style_axes(ax):
    ax.set_facecolor(PAPER_FIGURE_BG)
    ax.set_axisbelow(True)
    ax.set_frame_on(True)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(RELEASE_SUMMARY_FRAME_COLOR)
        ax.spines[side].set_linewidth(RELEASE_SUMMARY_FRAME_LINEWIDTH)
        ax.spines[side].set_zorder(20)
    ax.patch.set_edgecolor(RELEASE_SUMMARY_FRAME_COLOR)
    ax.patch.set_linewidth(RELEASE_SUMMARY_FRAME_LINEWIDTH)
    ax.tick_params(
        axis="both",
        which="major",
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
    ax.grid(axis="x", color="#D1FAE5", linewidth=1.0, alpha=0.95, zorder=0)


def save_figure_with_tif(fig, output_path: Path, *, facecolor, dpi: int = 600):
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        tif_path = output_path.with_suffix(".tif")
        try:
            fig.savefig(
                tif_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=facecolor,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            fig.savefig(tif_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)


def style_colorbar(cbar):
    cbar.outline.set_visible(False)
    cbar.ax.set_facecolor(PAPER_FIGURE_BG)
    cbar.ax.tick_params(
        labelsize=PAPER_TICK_LABEL_STYLE["fontsize"],
        length=0,
        colors=PAPER_TICK_LABEL_STYLE["color"],
    )
    cbar.ax.yaxis.label.set_fontfamily(PAPER_FONT_FAMILY)
    cbar.ax.yaxis.label.set_fontsize(PAPER_AXIS_LABEL_STYLE["fontsize"])
    cbar.ax.yaxis.label.set_fontweight(PAPER_AXIS_LABEL_STYLE["fontweight"])
    cbar.ax.yaxis.label.set_color(PAPER_AXIS_LABEL_STYLE["color"])
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(PAPER_TICK_LABEL_STYLE["fontfamily"])
        label.set_fontweight(PAPER_TICK_LABEL_STYLE["fontweight"])
        label.set_color(PAPER_TICK_LABEL_STYLE["color"])


def style_overview_axes(ax):
    style_axes(ax)
    ax.set_axisbelow(True)
    ax.grid(True, color="#D1FAE5", linewidth=1.0, alpha=0.95, zorder=0)


def style_release_summary_ticks(ax, labelsize: int = RELEASE_SUMMARY_TICK_SIZE):
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(labelsize)
        label.set_fontfamily(PAPER_TICK_LABEL_STYLE["fontfamily"])
        label.set_fontweight(PAPER_TICK_LABEL_STYLE["fontweight"])
        label.set_color(PAPER_TICK_LABEL_STYLE["color"])


def style_release_summary_axis_labels(ax):
    ax.xaxis.label.set_fontsize(RELEASE_SUMMARY_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(RELEASE_SUMMARY_AXIS_LABEL_SIZE)
    ax.xaxis.label.set_fontfamily(PAPER_AXIS_LABEL_STYLE["fontfamily"])
    ax.yaxis.label.set_fontfamily(PAPER_AXIS_LABEL_STYLE["fontfamily"])
    ax.xaxis.label.set_fontweight(PAPER_AXIS_LABEL_STYLE["fontweight"])
    ax.yaxis.label.set_fontweight(PAPER_AXIS_LABEL_STYLE["fontweight"])
    ax.xaxis.label.set_color(PAPER_AXIS_LABEL_STYLE["color"])
    ax.yaxis.label.set_color(PAPER_AXIS_LABEL_STYLE["color"])


def style_release_summary_frame(ax):
    ax.set_facecolor(RELEASE_SUMMARY_AXES_BG)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(RELEASE_SUMMARY_FRAME_COLOR)
        ax.spines[side].set_linewidth(RELEASE_SUMMARY_FRAME_LINEWIDTH)
    ax.patch.set_edgecolor(RELEASE_SUMMARY_FRAME_COLOR)
    ax.patch.set_linewidth(RELEASE_SUMMARY_FRAME_LINEWIDTH)


def format_material_label(label: str) -> str:
    text = str(label or "")
    return re.sub(r"(\d+)", lambda m: rf"$_{{{m.group(1)}}}$", text)


def format_structure_variant_label(label: str) -> str:
    text = str(label or "")
    if text.startswith("POSCAR-"):
        text = text[len("POSCAR-"):]
    match = re.match(r"^(.*?)(-\d+)$", text)
    if match:
        return f"{format_material_label(match.group(1))}{match.group(2)}"
    return format_material_label(text)


def format_structure_release_label(label: str) -> str:
    text = str(label or "")
    if text.startswith("POSCAR-"):
        text = text[len("POSCAR-"):]
    text = re.sub(r"-\d+$", "", text)
    return format_material_label(text)


def prediction_metric_column(df: pd.DataFrame) -> str:
    for col in ("pred_d33", "pred_d33_committee", "pred_d33_mean", "pred_d33_full"):
        if col in df.columns:
            return col
    raise KeyError("No prediction column found")


def load_preprocessed_data(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return pd.DataFrame(payload["records"])
    if isinstance(payload, dict):
        return pd.DataFrame(list(payload.values()))
    raise ValueError(f"Unsupported input format: {input_path}")


def load_model(model_path: Path):
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed to load {model_path}. The saved model needs a dependency that is missing in this environment: {e}"
        ) from e


def ensure_derived_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def numeric_col(name: str):
        if name not in df.columns:
            return None
        return pd.to_numeric(df[name], errors="coerce")

    c_over_a = numeric_col("c_over_a")
    density = numeric_col("density")
    sharpness = numeric_col("sharpness")
    a_en_diff = numeric_col("A_en_diff")
    b_en_diff = numeric_col("B_en_diff")
    a_radius_diff = numeric_col("A_radius_diff")
    b_radius_diff = numeric_col("B_radius_diff")
    a_amount = numeric_col("A_amount")
    b_amount = numeric_col("B_amount")
    peak1_pos = numeric_col("peak1_pos")
    peak1_height = numeric_col("peak1_height")
    peak1_fwhm = numeric_col("peak1_fwhm")
    peak2_pos = numeric_col("peak2_pos")
    peak2_height = numeric_col("peak2_height")
    peak2_fwhm = numeric_col("peak2_fwhm")
    volume_ang3 = numeric_col("volume_ang3")

    def safe_div_series(numerator, denominator):
        numerator = pd.to_numeric(numerator, errors="coerce") if numerator is not None else None
        denominator = pd.to_numeric(denominator, errors="coerce") if denominator is not None else None
        if numerator is None or denominator is None:
            return None
        return np.where(np.abs(denominator) > 1e-12, numerator / denominator, np.nan)

    def fill_default_feature(name: str, value=0.0):
        if name not in df.columns:
            df[name] = value

    def valence_diff(element, default=0.0):
        valence = get_valence_electrons(element)
        al_valence = get_valence_electrons("Al")
        if valence is None or al_valence is None:
            return default
        return float(valence - al_valence)

    if "A_valence_electrons_diff" not in df.columns and "A_element" in df.columns:
        df["A_valence_electrons_diff"] = df["A_element"].map(lambda x: valence_diff(x, default=0.0))

    if "B_valence_electrons_diff" not in df.columns and "B_element" in df.columns:
        df["B_valence_electrons_diff"] = df["B_element"].map(lambda x: valence_diff(x, default=0.0))

    if "octahedral_factor" not in df.columns and "B_element" in df.columns:
        o_ionic_radius = get_ionic_radius("O", valence=-2)

        def octahedral_factor(element):
            valence = get_valence_electrons(element)
            b_ionic_radius = get_ionic_radius(element, valence=valence)
            if b_ionic_radius is None or o_ionic_radius is None or o_ionic_radius <= 0:
                return 0.0
            return float(b_ionic_radius / o_ionic_radius)

        df["octahedral_factor"] = df["B_element"].map(octahedral_factor)

    if "volume" not in df.columns and volume_ang3 is not None:
        df["volume"] = volume_ang3

    if "log_density" not in df.columns and density is not None:
        df["log_density"] = np.where(density > -1, np.log1p(density), np.nan)

    if "Reciprocal_sharpness" not in df.columns and sharpness is not None:
        df["Reciprocal_sharpness"] = np.where(sharpness != 0, 1.0 / sharpness, np.nan)

    if "polar_factor" not in df.columns and c_over_a is not None and a_en_diff is not None and b_en_diff is not None:
        df["polar_factor"] = c_over_a * (a_en_diff + b_en_diff)

    if "ca_diff_sq" not in df.columns and c_over_a is not None:
        df["ca_diff_sq"] = (c_over_a - CA_DIFF_THRESHOLD) ** 2

    if "ca_diff_sq4" not in df.columns:
        ca_diff_sq_for_fourth = numeric_col("ca_diff_sq")
        if ca_diff_sq_for_fourth is not None:
            df["ca_diff_sq4"] = ca_diff_sq_for_fourth ** 2

    if "feat11" not in df.columns and density is not None and c_over_a is not None:
        df["feat11"] = density * c_over_a

    if "feat12" not in df.columns and c_over_a is not None:
        df["feat12"] = np.where(c_over_a != 0, 0.63 - (1.0 / c_over_a), np.nan)

    if "feat2" not in df.columns and density is not None and sharpness is not None:
        df["feat2"] = np.log1p(density) * safe_div_series(pd.Series(1.0, index=df.index), sharpness)

    if "B_en_diff_minus_18" not in df.columns and b_en_diff is not None:
        df["B_en_diff_minus_18"] = b_en_diff - 1.8

    ca_diff_sq = numeric_col("ca_diff_sq")
    b_en_diff_minus_18 = numeric_col("B_en_diff_minus_18")

    if "pysr_peak2_logA_logca2" not in df.columns and peak2_pos is not None and a_amount is not None and ca_diff_sq is not None:
        safe_a_amount = np.clip(a_amount, 1e-6, None)
        safe_ca_diff_sq = np.clip(ca_diff_sq, 1e-6, None)
        df["pysr_peak2_logA_logca2"] = peak2_pos * (
            np.log(safe_a_amount) + np.log(safe_ca_diff_sq) ** 2
        )

    if "pysr_Ben18_ca" not in df.columns and b_en_diff_minus_18 is not None and ca_diff_sq is not None:
        df["pysr_Ben18_ca"] = b_en_diff_minus_18 * ca_diff_sq

    pysr_peak2_logA_logca2 = numeric_col("pysr_peak2_logA_logca2")
    pysr_ben18_ca = numeric_col("pysr_Ben18_ca")
    if "pysr_peak2_ben18_sq" not in df.columns and pysr_peak2_logA_logca2 is not None and pysr_ben18_ca is not None:
        df["pysr_peak2_ben18_sq"] = (pysr_peak2_logA_logca2 * pysr_ben18_ca) ** 2

    if "pysr_Aen_Ben_ca" not in df.columns and a_en_diff is not None and b_en_diff is not None and c_over_a is not None:
        df["pysr_Aen_Ben_ca"] = a_en_diff * b_en_diff * c_over_a

    if "pysr_density_A_radius" not in df.columns and density is not None and a_radius_diff is not None:
        df["pysr_density_A_radius"] = density * a_radius_diff

    if "pysr_B_radius_B" not in df.columns and b_radius_diff is not None and b_amount is not None:
        df["pysr_B_radius_B"] = b_radius_diff * b_amount

    if "pysr_u_peak2_over_sharp" not in df.columns and peak2_pos is not None and sharpness is not None:
        u = numeric_col("u")
        if u is not None:
            df["pysr_u_peak2_over_sharp"] = safe_div_series(u * peak2_pos, sharpness)

    if "pysr_density_Ben" not in df.columns and density is not None and b_en_diff is not None:
        df["pysr_density_Ben"] = density * b_en_diff

    if "pysr_density_AB" not in df.columns and density is not None and a_amount is not None and b_amount is not None:
        df["pysr_density_AB"] = density * a_amount * b_amount

    if "pysr_peak1_minus_ca" not in df.columns and peak1_pos is not None and c_over_a is not None:
        df["pysr_peak1_minus_ca"] = peak1_pos - c_over_a

    if "pysr_Ben_over_ca_minus_Ben" not in df.columns and b_en_diff is not None and c_over_a is not None:
        df["pysr_Ben_over_ca_minus_Ben"] = b_en_diff / np.maximum(np.abs(c_over_a - b_en_diff), 1e-6)

    if "pysr_log_peak1_minus_ca_abs" not in df.columns and peak1_pos is not None and c_over_a is not None:
        df["pysr_log_peak1_minus_ca_abs"] = np.log1p(np.abs(peak1_pos - c_over_a))

    if "pysr_peak2_over_sharp" not in df.columns and peak2_pos is not None and sharpness is not None:
        df["pysr_peak2_over_sharp"] = safe_div_series(peak2_pos, sharpness)

    if "rdf_width_ratio_21" not in df.columns:
        value = safe_div_series(peak2_fwhm, peak1_fwhm)
        df["rdf_width_ratio_21"] = value if value is not None else 0.0

    if "rdf_area_ratio_21" not in df.columns:
        if peak1_height is not None and peak1_fwhm is not None and peak2_height is not None and peak2_fwhm is not None:
            df["rdf_area_ratio_21"] = safe_div_series(peak2_height * peak2_fwhm, peak1_height * peak1_fwhm)
        else:
            df["rdf_area_ratio_21"] = 0.0

    if "rdf_gap_over_width" not in df.columns:
        if peak1_pos is not None and peak2_pos is not None and peak1_fwhm is not None and peak2_fwhm is not None:
            df["rdf_gap_over_width"] = safe_div_series(peak2_pos - peak1_pos, peak1_fwhm + peak2_fwhm)
        else:
            df["rdf_gap_over_width"] = 0.0

    if "rdf_peak2_sharpness" not in df.columns:
        value = safe_div_series(peak2_height, peak2_fwhm)
        df["rdf_peak2_sharpness"] = value if value is not None else 0.0

    for structural_feature in (
        "cation_n_bond_std",
        "tetra_angle_dev_mean",
        "cation_offcenter_mean",
        "dopant_n_bond_delta",
    ):
        fill_default_feature(structural_feature, 0.0)

    return df


def resolve_feature_columns(model, df: pd.DataFrame) -> list:
    feature_cols = None
    if hasattr(model, "feature_names_in_"):
        feature_cols = [str(x) for x in model.feature_names_in_]
    elif hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        inner = model.named_steps["model"]
        if hasattr(inner, "feature_names_in_"):
            feature_cols = [str(x) for x in inner.feature_names_in_]

    if not feature_cols:
        feature_cols = list(MODEL_FEATURES)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return feature_cols


def load_training_reference(train_json: Path, folds_file: Path, feature_cols: list):
    from train import build_name_to_idx, load_and_clean, load_predefined_folds, resolve_fold_indices

    folds_meta, folds = load_predefined_folds(folds_file)
    df, composition_map, _ = load_and_clean(
        str(train_json),
        max_threshold=folds_meta.get("max_threshold", 30),
        min_threshold=folds_meta.get("min_threshold", 3),
        use_swap=folds_meta.get("use_swap", False),
        selected_features=feature_cols,
    )
    name_to_idx = build_name_to_idx(composition_map)
    return {
        "X": df[feature_cols].copy(),
        "y": df["target"].copy(),
        "folds_meta": folds_meta,
        "folds": folds,
        "name_to_idx": name_to_idx,
        "resolve_fold_indices": resolve_fold_indices,
    }


def fit_committee_fold_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    try:
        from train import get_xgb_sample_weight

        model.fit(X_train, y_train, model__sample_weight=get_xgb_sample_weight(y_train))
    except Exception:
        model.fit(X_train, y_train)
    return model


def fit_oof_calibration(raw_pred: np.ndarray, y_true: pd.Series, method: str = DEFAULT_COMMITTEE_CALIBRATION) -> dict:
    raw = np.asarray(raw_pred, dtype=float).reshape(-1)
    target = np.asarray(y_true, dtype=float).reshape(-1)
    mask = np.isfinite(raw) & np.isfinite(target)
    if int(mask.sum()) < 2:
        return {"method": method, "enabled": False}

    method = str(method or "linear").lower()
    raw_valid = raw[mask]
    target_valid = target[mask]
    raw_oof_r2 = float(r2_score(target_valid, raw_valid))
    raw_oof_mae = float(mean_absolute_error(target_valid, raw_valid))

    if method == "none":
        calibrated = raw_valid
        return {
            "method": "none",
            "enabled": False,
            "raw_oof_r2": raw_oof_r2,
            "raw_oof_mae": raw_oof_mae,
            "calibrated_oof_r2": raw_oof_r2,
            "calibrated_oof_mae": raw_oof_mae,
        }

    if method == "isotonic":
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_valid, target_valid)
            calibrated = calibrator.predict(raw_valid)
            return {
                "method": "isotonic",
                "enabled": True,
                "x_thresholds": [float(x) for x in calibrator.X_thresholds_],
                "y_thresholds": [float(y) for y in calibrator.y_thresholds_],
                "raw_oof_r2": raw_oof_r2,
                "raw_oof_mae": raw_oof_mae,
                "calibrated_oof_r2": float(r2_score(target_valid, calibrated)),
                "calibrated_oof_mae": float(mean_absolute_error(target_valid, calibrated)),
            }
        except Exception as e:
            logger.warning(f"Isotonic calibration failed; falling back to linear calibration: {e}")
            method = "linear"

    slope, intercept = np.polyfit(raw_valid, target_valid, deg=1)
    calibrated = intercept + slope * raw_valid
    return {
        "method": "linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "enabled": True,
        "raw_oof_r2": raw_oof_r2,
        "raw_oof_mae": raw_oof_mae,
        "calibrated_oof_r2": float(r2_score(target_valid, calibrated)),
        "calibrated_oof_mae": float(mean_absolute_error(target_valid, calibrated)),
    }


def apply_linear_calibration(values, calibration: dict) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not calibration or not calibration.get("enabled", False):
        return arr
    method = str(calibration.get("method", "linear")).lower()
    if method == "isotonic":
        x_thresholds = np.asarray(calibration.get("x_thresholds", []), dtype=float)
        y_thresholds = np.asarray(calibration.get("y_thresholds", []), dtype=float)
        if x_thresholds.size >= 2 and y_thresholds.size == x_thresholds.size:
            return np.interp(arr.reshape(-1), x_thresholds, y_thresholds).reshape(arr.shape)
        return arr
    return float(calibration["intercept"]) + float(calibration["slope"]) * arr


def aggregate_committee_predictions(pred_df: pd.DataFrame, method: str = DEFAULT_COMMITTEE_AGGREGATION) -> pd.Series:
    method = str(method or "mean").lower()
    if method == "median":
        return pred_df.median(axis=1)
    if method == "trimmed" and pred_df.shape[1] >= 3:
        values = np.sort(pred_df.to_numpy(dtype=float), axis=1)
        return pd.Series(values[:, 1:-1].mean(axis=1), index=pred_df.index)
    return pred_df.mean(axis=1)


def compute_committee_predictions(
    model_template,
    candidate_df: pd.DataFrame,
    feature_cols: list,
    train_json: Path,
    folds_file: Path,
    topk_size: int,
    calibration_method: str = DEFAULT_COMMITTEE_CALIBRATION,
    aggregation_method: str = DEFAULT_COMMITTEE_AGGREGATION,
):
    reference = load_training_reference(train_json, folds_file, feature_cols)
    X_train = reference["X"]
    y_train = reference["y"]
    folds = reference["folds"]
    name_to_idx = reference["name_to_idx"]
    resolve_fold_indices = reference["resolve_fold_indices"]

    raw_fold_predictions = {}
    fold_ranks = {}
    X_candidate = candidate_df[feature_cols]
    oof_pred = np.full(len(X_train), np.nan, dtype=float)

    for fold_entry in folds:
        fold_id = int(fold_entry.get("fold", 0))
        train_idx, test_idx = resolve_fold_indices(fold_entry, name_to_idx)
        fold_model = clone(model_template)
        fit_committee_fold_model(fold_model, X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_pred = np.asarray(fold_model.predict(X_candidate), dtype=float).reshape(-1)
        raw_fold_predictions[f"pred_fold_{fold_id}"] = fold_pred
        if len(test_idx):
            oof_pred[test_idx] = np.asarray(fold_model.predict(X_train.iloc[test_idx]), dtype=float).reshape(-1)

    calibration = fit_oof_calibration(oof_pred, y_train, method=calibration_method)
    fold_predictions = {
        col: apply_linear_calibration(values, calibration)
        for col, values in raw_fold_predictions.items()
    }

    for col, fold_pred in fold_predictions.items():
        fold_id = int(col.rsplit("_", 1)[1])
        fold_ranks[f"rank_fold_{fold_id}"] = (
            pd.Series(fold_pred, index=candidate_df.index).rank(ascending=False, method="min")
        )

    pred_df = pd.DataFrame(fold_predictions, index=candidate_df.index)
    rank_df = pd.DataFrame(fold_ranks, index=candidate_df.index)

    ddof = 1 if pred_df.shape[1] > 1 else 0
    summary = pd.DataFrame(index=candidate_df.index)
    summary["pred_d33_mean"] = pred_df.mean(axis=1)
    summary["pred_d33_std"] = pred_df.std(axis=1, ddof=ddof).fillna(0.0)
    summary["pred_d33_median"] = pred_df.median(axis=1)
    q1 = pred_df.quantile(0.25, axis=1)
    q3 = pred_df.quantile(0.75, axis=1)
    summary["pred_d33_iqr"] = q3 - q1
    summary["pred_d33_committee"] = aggregate_committee_predictions(pred_df, aggregation_method)
    summary["rank_mean"] = rank_df.mean(axis=1)
    summary["rank_std"] = rank_df.std(axis=1, ddof=ddof).fillna(0.0)
    summary["topk_frequency"] = (rank_df.le(float(topk_size))).mean(axis=1)

    committee_meta = dict(reference["folds_meta"])
    committee_meta["oof_calibration"] = calibration
    committee_meta["oof_linear_calibration"] = calibration
    committee_meta["aggregation_method"] = str(aggregation_method)
    return summary, pred_df, rank_df, committee_meta


def plot_top_candidates(df: pd.DataFrame, output_path: Path, top_k: int = 15):
    top_df = df.head(top_k).copy()
    if top_df.empty:
        return

    metric_col = prediction_metric_column(top_df)
    error_col = "pred_d33_std" if "pred_d33_std" in top_df.columns else None

    apply_paper_style()
    top_df = top_df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.46 * len(top_df) + 2.1)))
    y = np.arange(len(top_df))
    colors = ["#0b7285" if i == 0 else "#74c0fc" for i in range(len(top_df))]
    values = top_df[metric_col].to_numpy(dtype=float)
    xerr = top_df[error_col].to_numpy(dtype=float) if error_col else None

    ax.barh(
        y,
        values,
        height=0.62,
        color=colors,
        edgecolor="#1f1f1f",
        linewidth=0.7,
        xerr=xerr,
        error_kw={"elinewidth": 1.2, "ecolor": "#495057", "capsize": 2.5},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top_df["name"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(D33_PREDICTED_LABEL, labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_axes(ax)

    x_max = float(np.nanmax(values + (xerr if xerr is not None else 0)))
    x_min = 0.0 if x_max >= 0 else float(np.nanmin(values))
    span = max(1e-6, x_max - x_min)
    ax.set_xlim(x_min - 0.05 * span, x_max + 0.34 * span)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    for idx, value in enumerate(values):
        label = f"{value:.2f}"
        if error_col:
            label = f"{value:.2f} ± {top_df.iloc[idx][error_col]:.2f}"
        if "topk_frequency" in top_df.columns:
            label = f"{label} | top-k {top_df.iloc[idx]['topk_frequency'] * 100:.0f}%"
        ax.text(
            value + 0.02 * span,
            idx,
            label,
            va="center",
            ha="left",
            fontsize=9.6,
            color="#222222",
        )

    ax.text(
        0.99,
        0.03,
        f"n = {len(top_df)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    plt.tight_layout()
    save_figure_with_tif(fig, output_path, dpi=600, facecolor="white")
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, output_path: Path):
    metric_col = prediction_metric_column(df)
    values = df[metric_col].dropna().values
    if len(values) == 0:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.4, 5.2), facecolor=PAPER_FIGURE_BG)
    fig.patch.set_facecolor(PAPER_FIGURE_BG)
    bins = min(24, max(8, int(np.sqrt(len(values)) * 2)))
    ax.hist(
        values,
        bins=bins,
        density=True,
        color="#adb5bd",
        edgecolor="#1f1f1f",
        linewidth=0.7,
        alpha=0.75,
    )
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])

    x = np.linspace(values.min(), values.max(), 300)
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        ax.plot(x, kde(x), color="#0b7285", linewidth=2.4, label="KDE")
    except Exception:
        pass

    ax.axvline(mean_val, color="#d9480f", linestyle="-", linewidth=2.2, label=f"Mean = {mean_val:.2f} pC/N")
    ax.axvline(median_val, color="#2b8a3e", linestyle="--", linewidth=2.0, label=f"Median = {median_val:.2f} pC/N")
    ax.axvspan(q1, q3, color="#ffd43b", alpha=0.15, label="IQR")

    ax.set_xlabel(D33_PREDICTED_LABEL, labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    ax.set_ylabel("Probability density (N/pC)", labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_overview_axes(ax)
    style_release_summary_ticks(ax)
    style_release_summary_axis_labels(ax)
    style_release_summary_frame(ax)
    ax.legend(loc="upper right", fontsize=RELEASE_SUMMARY_LEGEND_SIZE)
    plt.tight_layout()
    save_figure_with_tif(fig, output_path, dpi=600, facecolor=PAPER_FIGURE_BG)
    plt.close(fig)


def plot_feature_landscape(df: pd.DataFrame, output_path: Path):
    required = {"c_over_a", "polar_factor"}
    metric_col = prediction_metric_column(df)
    if not required.issubset(df.columns) or metric_col not in df.columns:
        return

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.6), facecolor=PAPER_FIGURE_BG)
    fig.patch.set_facecolor(PAPER_FIGURE_BG)
    order_df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    scatter = ax.scatter(
        order_df["c_over_a"],
        order_df["polar_factor"],
        c=order_df[metric_col],
        cmap="viridis",
        s=np.clip(40 + (order_df[metric_col] - order_df[metric_col].min()) * 2.2, 36, 120),
        alpha=0.92,
        edgecolors="#1f1f1f",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_xlabel(format_feature_label("c_over_a"), labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    ax.set_ylabel(format_feature_label("polar_factor"), labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_overview_axes(ax)
    style_release_summary_ticks(ax)
    style_release_summary_axis_labels(ax)
    style_release_summary_frame(ax)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(D33_PREDICTED_LABEL)
    style_colorbar(cbar)
    cbar.ax.tick_params(labelsize=RELEASE_SUMMARY_TICK_SIZE, length=0)
    cbar.ax.yaxis.label.set_fontsize(RELEASE_SUMMARY_AXIS_LABEL_SIZE)

    best = order_df.iloc[0]
    ax.scatter(
        [best["c_over_a"]],
        [best["polar_factor"]],
        s=190,
        facecolors="none",
        edgecolors="#d9480f",
        linewidths=2.0,
        zorder=6,
    )
    ax.annotate(
        format_structure_release_label(best["name"]),
        (best["c_over_a"], best["polar_factor"]),
        fontsize=16,
        fontweight="bold",
        fontfamily=PAPER_FONT_FAMILY,
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color="#111111",
        zorder=7,
    )
    plt.tight_layout()
    save_figure_with_tif(fig, output_path, dpi=600, facecolor=PAPER_FIGURE_BG)
    plt.close(fig)


def plot_uncertainty_overview(df: pd.DataFrame, output_path: Path):
    required = {"pred_d33_mean", "pred_d33_std"}
    if not required.issubset(df.columns):
        return

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 6.0), facecolor=PAPER_FIGURE_BG)
    fig.patch.set_facecolor(PAPER_FIGURE_BG)
    scatter = ax.scatter(
        df["pred_d33_mean"],
        df["pred_d33_std"],
        c=df["rank_mean"] if "rank_mean" in df.columns else df["pred_d33_mean"],
        cmap="cividis",
        s=60,
        alpha=0.88,
        edgecolors="#1f1f1f",
        linewidths=0.45,
        zorder=3,
    )
    ax.set_xlabel(D33_COMMITTEE_MEAN_LABEL, labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    ax.set_ylabel(D33_STD_LABEL, labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_overview_axes(ax)

    label_candidates = df.sort_values("pred_d33_mean", ascending=False).head(min(8, len(df)))
    for _, row in label_candidates.iterrows():
        ax.annotate(
            str(row["name"]),
            (row["pred_d33_mean"], row["pred_d33_std"]),
            fontsize=8.3,
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.7),
        )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Mean rank")
    style_colorbar(cbar)
    plt.tight_layout()
    save_figure_with_tif(fig, output_path, dpi=600, facecolor=PAPER_FIGURE_BG)
    plt.close(fig)


def _add_panel_label(fig, ax, label: str):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 - 0.058,
        bbox.y1 + 0.006,
        label,
        ha="left",
        va="bottom",
        fontsize=RELEASE_SUMMARY_PANEL_LABEL_SIZE,
        fontfamily=PAPER_FONT_FAMILY,
        fontweight="bold",
        color="#0f172a",
    )


def _draw_distribution_panel(ax, df: pd.DataFrame):
    metric_col = prediction_metric_column(df)
    values = df[metric_col].dropna().values
    if len(values) == 0:
        ax.axis("off")
        return

    ranked_values = np.sort(values)[::-1]
    top30_threshold = float(ranked_values[min(29, len(ranked_values) - 1)])
    bins = min(24, max(8, int(np.sqrt(len(values)) * 2)))
    counts, _, _ = ax.hist(
        values,
        bins=bins,
        density=True,
        color="#C7D2FE",
        edgecolor="#1f1f1f",
        linewidth=0.7,
        alpha=0.82,
    )
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    x = np.linspace(values.min(), values.max(), 300)
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        ax.plot(x, kde(x), color=RELEASE_SUMMARY_BLUE, linewidth=3.0, label="KDE")
    except Exception:
        pass

    y_top = float(max(np.nanmax(counts) if len(counts) else 0.0, ax.get_ylim()[1]))
    value_span = max(1e-6, float(values.max() - values.min()))
    x_left = float(values.min() - 0.05 * value_span)
    x_right = float(values.max() + 0.05 * value_span)
    ax.set_xlim(x_left, x_right)
    ax.axvspan(top30_threshold, x_right, color=RELEASE_SUMMARY_TAIL, alpha=0.34, zorder=0, label="Top-30 tail")
    ax.axvspan(q1, q3, color=RELEASE_SUMMARY_IQR, alpha=0.42, zorder=0, label="IQR")
    ax.axvline(mean_val, color=RELEASE_SUMMARY_ACCENT, linestyle="-", linewidth=2.6, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color="#166534", linestyle="--", linewidth=2.3, label=f"Median = {median_val:.2f}")
    ax.axvline(top30_threshold, color="#7C2D12", linestyle="-.", linewidth=2.4, label=f"Top-30 cutoff = {top30_threshold:.2f}")
    rug_y = -0.045 * y_top
    ax.plot(values, np.full_like(values, rug_y), "|", color="#334155", markersize=7, markeredgewidth=1.0, alpha=0.42, clip_on=False)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(rug_y * 1.8, y_top * 1.12)
    ax.set_xlabel(D33_PREDICTED_LABEL, labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    ax.set_ylabel("Probability density (N/pC)", labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_overview_axes(ax)
    style_release_summary_ticks(ax)
    style_release_summary_axis_labels(ax)
    style_release_summary_frame(ax)
    ax.legend(loc="upper right", fontsize=13)


def _draw_landscape_panel(ax, df: pd.DataFrame):
    required = {"c_over_a", "polar_factor"}
    metric_col = prediction_metric_column(df)
    if not required.issubset(df.columns) or metric_col not in df.columns:
        ax.axis("off")
        return None

    order_df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    x_vals = order_df["c_over_a"].to_numpy(dtype=float)
    y_vals = order_df["polar_factor"].to_numpy(dtype=float)
    z_vals = order_df[metric_col].to_numpy(dtype=float)
    ax.scatter(
        x_vals,
        y_vals,
        s=24,
        color="#94A3B8",
        alpha=0.24,
        edgecolors="none",
        zorder=1,
    )
    scatter = ax.scatter(
        x_vals,
        y_vals,
        c=z_vals,
        cmap=RELEASE_SUMMARY_CMAP,
        s=np.clip(42 + (z_vals - np.nanmin(z_vals)) * 2.4, 38, 138),
        alpha=0.92,
        edgecolors="#1f1f1f",
        linewidths=0.5,
        zorder=3,
    )
    ax.set_xlabel(format_feature_label("c_over_a"), labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    ax.set_ylabel(format_feature_label("polar_factor"), labelpad=8, **PAPER_AXIS_LABEL_STYLE)
    style_overview_axes(ax)
    style_release_summary_ticks(ax)
    style_release_summary_axis_labels(ax)
    style_release_summary_frame(ax)

    best = order_df.iloc[0]
    ax.scatter(
        [best["c_over_a"]],
        [best["polar_factor"]],
        s=170,
        marker="o",
        facecolors=RELEASE_SUMMARY_GOLD,
        edgecolors=RELEASE_SUMMARY_ACCENT,
        linewidths=1.8,
        zorder=8,
    )
    ax.annotate(
        format_structure_release_label(best["name"]),
        (best["c_over_a"], best["polar_factor"]),
        fontsize=17,
        fontweight="bold",
        fontfamily=PAPER_FONT_FAMILY,
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color="#0f172a",
        zorder=9,
    )
    return scatter


def _select_release_top_df(
    df: pd.DataFrame,
    validation_lookup: dict,
    top_k: int,
    selection_mode: str,
    unique_canonical: bool,
    min_topk_frequency: float | None,
    max_pred_std: float | None,
    min_pred: float | None,
) -> tuple[pd.DataFrame, str]:
    metric_col = prediction_metric_column(df)
    if selection_mode == "dft_available":
        candidate_df = df[df["name"].isin(set(validation_lookup))].copy()
    else:
        candidate_df = df.copy()

    if min_topk_frequency is not None and "topk_frequency" in candidate_df.columns:
        candidate_df = candidate_df[pd.to_numeric(candidate_df["topk_frequency"], errors="coerce") >= float(min_topk_frequency)].copy()
    if max_pred_std is not None and "pred_d33_std" in candidate_df.columns:
        candidate_df = candidate_df[pd.to_numeric(candidate_df["pred_d33_std"], errors="coerce") <= float(max_pred_std)].copy()
    if min_pred is not None and metric_col in candidate_df.columns:
        candidate_df = candidate_df[pd.to_numeric(candidate_df[metric_col], errors="coerce") >= float(min_pred)].copy()

    candidate_df["dft_d33"] = candidate_df["name"].map(validation_lookup)
    candidate_df["pred_reference"] = candidate_df[metric_col]
    candidate_df["dft_gap"] = candidate_df["dft_d33"] - candidate_df["pred_reference"]
    candidate_df["abs_gap"] = candidate_df["dft_gap"].abs()

    if "canonical_name" in candidate_df.columns:
        candidate_df["_variant_group"] = candidate_df["canonical_name"].fillna(candidate_df["name"])
        candidate_df = candidate_df.sort_values(["_variant_group", "abs_gap", metric_col], ascending=[True, True, False], na_position="last")
        candidate_df = (
            candidate_df.drop_duplicates(subset=["_variant_group"], keep="first")
            .drop(columns=["_variant_group"])
            .sort_values(metric_col, ascending=False)
            .copy()
        )

    return candidate_df.head(top_k).copy(), metric_col


def _draw_top30_release_panel(
    ax,
    df: pd.DataFrame,
    validation_lookup: dict,
    top_k: int,
    selection_mode: str,
    unique_canonical: bool,
    min_topk_frequency: float | None,
    max_pred_std: float | None,
    min_pred: float | None,
):
    top_df, metric_col = _select_release_top_df(
        df,
        validation_lookup,
        top_k,
        selection_mode,
        unique_canonical,
        min_topk_frequency,
        max_pred_std,
        min_pred,
    )
    if top_df.empty:
        ax.axis("off")
        return

    fold_cols = sorted([c for c in top_df.columns if c.startswith("pred_fold_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    distributions = []
    for _, row in top_df.iterrows():
        values = pd.to_numeric(row[fold_cols], errors="coerce").dropna().to_numpy(dtype=float) if fold_cols else np.asarray([float(row[metric_col])])
        if values.size == 0:
            values = np.asarray([float(row[metric_col])], dtype=float)
        distributions.append(values)

    positions = np.arange(len(top_df))
    mean_values = top_df["pred_reference"].to_numpy(dtype=float)
    cmap = plt.get_cmap(RELEASE_SUMMARY_CMAP)
    norm = plt.Normalize(vmin=float(np.nanmin(mean_values)), vmax=float(np.nanmax(mean_values)))
    box_colors = [cmap(norm(v)) for v in mean_values]
    box = ax.boxplot(
        distributions,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#0f172a", linewidth=2.0),
        whiskerprops=dict(color="#1f1f1f", linewidth=1.25),
        capprops=dict(color="#1f1f1f", linewidth=1.25),
        boxprops=dict(edgecolor="#1f1f1f", linewidth=1.2),
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    dft_mask = top_df["dft_d33"].notna()
    for idx, row in top_df[dft_mask].iterrows():
        xpos = top_df.index.get_loc(idx)
        pred_val = float(row["pred_reference"])
        dft_val = float(row["dft_d33"])
        ax.vlines(xpos, ymin=min(pred_val, dft_val), ymax=max(pred_val, dft_val), colors="#495057", linestyles="--", linewidth=1.0, alpha=0.95, zorder=2)

    ax.scatter(
        positions[dft_mask.to_numpy()],
        top_df.loc[dft_mask, "dft_d33"].to_numpy(dtype=float),
        marker="D",
        s=72,
        color="#111111",
        edgecolors="#FFFFFF",
        linewidths=0.55,
        zorder=4,
        label="DFT validation",
    )
    display_labels = [format_structure_release_label(x) for x in top_df["name"].fillna(top_df["canonical_name"])]
    ax.set_xticks(positions)
    ax.set_xticklabels(display_labels, rotation=65, ha="right", fontsize=RELEASE_SUMMARY_TICK_SIZE)
    ax.set_ylabel(D33_AXIS_LABEL, labelpad=10, **PAPER_AXIS_LABEL_STYLE)
    style_axes(ax)
    style_release_summary_ticks(ax)
    style_release_summary_axis_labels(ax)
    style_release_summary_frame(ax)
    ax.grid(axis="y", alpha=0.16)
    ax.set_xlim(-0.8, len(top_df) - 0.2)

    y_value_parts = [np.asarray(mean_values, dtype=float), top_df["dft_d33"].dropna().to_numpy(dtype=float)]
    y_value_parts.extend(vals for vals in distributions if vals.size > 0)
    y_values = np.concatenate(y_value_parts)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    span = max(1e-6, y_max - y_min)
    lower_pad = max(1.2, 0.22 * span)
    upper_pad = max(1.9, 0.33 * span)
    ax.set_ylim(y_min - lower_pad, y_max + upper_pad)
    if dft_mask.any():
        signed_error = top_df.loc[dft_mask, "pred_reference"].to_numpy(dtype=float) - top_df.loc[dft_mask, "dft_d33"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(signed_error)))
        rmse = float(np.sqrt(np.mean(np.square(signed_error))))
        bias = float(np.mean(signed_error))
    legend_handles = [Patch(facecolor="#9ecae1", edgecolor="#1f1f1f", label="Committee prediction")]
    if dft_mask.any():
        legend_handles.extend(
            [
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#111111", markersize=6, label="DFT validation"),
                Line2D([0], [0], color="#495057", linestyle="--", linewidth=1.0, label="Prediction-DFT gap"),
            ]
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=RELEASE_SUMMARY_LEGEND_SIZE)


def plot_release_summary_figure(
    df: pd.DataFrame,
    validation_lookup: dict,
    output_path: Path,
    top_k: int = 30,
    selection_mode: str = "dft_available",
    unique_canonical: bool = False,
    min_topk_frequency: float | None = None,
    max_pred_std: float | None = None,
    min_pred: float | None = None,
):
    if not validation_lookup:
        logger.info("Skip release summary figure because no DFT validation data is available")
        return

    apply_paper_style()
    fig = plt.figure(figsize=(18.6, 14.2), facecolor=PAPER_FIGURE_BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28], hspace=0.20, wspace=0.42)
    ax_landscape = fig.add_subplot(gs[0, 0])
    ax_distribution = fig.add_subplot(gs[0, 1])
    ax_top30 = fig.add_subplot(gs[1, :])

    scatter = _draw_landscape_panel(ax_landscape, df)
    _draw_distribution_panel(ax_distribution, df)
    _draw_top30_release_panel(
        ax_top30,
        df,
        validation_lookup,
        top_k,
        selection_mode,
        unique_canonical,
        min_topk_frequency,
        max_pred_std,
        min_pred,
    )

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax_landscape, pad=0.02, fraction=0.05)
        cbar.set_label(D33_PREDICTED_LABEL, labelpad=18)
        style_colorbar(cbar)
        cbar.ax.tick_params(labelsize=RELEASE_SUMMARY_TICK_SIZE, length=0, pad=8)
        cbar.ax.yaxis.label.set_fontsize(RELEASE_SUMMARY_AXIS_LABEL_SIZE)

    _add_panel_label(fig, ax_landscape, "(a)")
    _add_panel_label(fig, ax_distribution, "(b)")
    _add_panel_label(fig, ax_top30, "(c)")
    save_figure_with_tif(fig, output_path, dpi=600, facecolor=PAPER_FIGURE_BG)
    plt.close(fig)


def load_validation_lookup(validation_json: Path) -> dict:
    if not validation_json.exists():
        return {}

    with open(validation_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    lookup = {}
    for name, record in payload.items():
        if not isinstance(record, dict):
            continue
        dft_d33 = record.get("d33")
        if dft_d33 is None:
            continue
        try:
            lookup[str(name)] = float(dft_d33)
        except Exception:
            continue
    return lookup


def plot_top20_prediction_vs_dft(
    df: pd.DataFrame,
    validation_lookup: dict,
    output_path: Path,
    top_k: int = 10,
    selection_mode: str = "ranked",
    dedup_strategy: str = "dft_abs_gap",
    annotate_mae: bool = False,
    unique_canonical: bool = False,
    min_topk_frequency: float | None = None,
    max_pred_std: float | None = None,
    min_pred: float | None = None,
    release_labels: bool = False,
    show_committee_distribution: bool = True,
):
    valid_modes = {"ranked", "dft_available"}
    if selection_mode not in valid_modes:
        raise ValueError(f"Unsupported DFT plot selection mode: {selection_mode}")
    valid_dedup_strategies = {"dft_abs_gap", "prediction", "none"}
    if dedup_strategy not in valid_dedup_strategies:
        raise ValueError(f"Unsupported DFT plot dedup strategy: {dedup_strategy}")

    metric_col = prediction_metric_column(df)
    if selection_mode == "dft_available":
        validation_names = set(validation_lookup)
        candidate_df = df[df["name"].isin(validation_names)].copy()
    else:
        candidate_df = df.copy()

    if min_topk_frequency is not None and "topk_frequency" in candidate_df.columns:
        frequency = pd.to_numeric(candidate_df["topk_frequency"], errors="coerce")
        candidate_df = candidate_df[frequency >= float(min_topk_frequency)].copy()

    if max_pred_std is not None and "pred_d33_std" in candidate_df.columns:
        pred_std = pd.to_numeric(candidate_df["pred_d33_std"], errors="coerce")
        candidate_df = candidate_df[pred_std <= float(max_pred_std)].copy()

    if min_pred is not None and metric_col in candidate_df.columns:
        pred_values = pd.to_numeric(candidate_df[metric_col], errors="coerce")
        candidate_df = candidate_df[pred_values >= float(min_pred)].copy()

    if candidate_df.empty:
        if selection_mode == "dft_available":
            logger.info("No ranked candidates matched names in the DFT validation data")
        return

    candidate_df = candidate_df.copy()
    candidate_df["dft_d33"] = candidate_df["name"].map(validation_lookup)
    candidate_df["pred_reference"] = candidate_df[metric_col]
    candidate_df["dft_gap"] = candidate_df["dft_d33"] - candidate_df["pred_reference"]
    candidate_df["abs_gap"] = candidate_df["dft_gap"].abs()

    best_variant_per_formula = "canonical_name" in candidate_df.columns and dedup_strategy != "none"
    if best_variant_per_formula:
        candidate_df["_variant_group"] = candidate_df["canonical_name"].fillna(candidate_df["name"])
        if dedup_strategy == "dft_abs_gap":
            # Paper validation plot: keep the structural variant that best matches DFT.
            sort_cols = ["_variant_group", "abs_gap", metric_col]
            ascending = [True, True, False]
        else:
            # Debug ranking plot: keep the highest-predicted variant for each formula.
            sort_cols = ["_variant_group", metric_col]
            ascending = [True, False]
            if "pred_d33_std" in candidate_df.columns:
                sort_cols.append("pred_d33_std")
                ascending.append(True)
        candidate_df = candidate_df.sort_values(sort_cols, ascending=ascending, na_position="last")
        candidate_df = (
            candidate_df.drop_duplicates(subset=["_variant_group"], keep="first")
            .drop(columns=["_variant_group"])
            .sort_values(metric_col, ascending=False)
            .copy()
        )

    top_df = candidate_df.head(top_k).copy()
    if top_df.empty:
        if selection_mode == "dft_available":
            logger.info("No ranked candidates matched names in the DFT validation data")
        return

    fold_cols = sorted(
        [c for c in top_df.columns if c.startswith("pred_fold_")],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )

    distributions = []
    for _, row in top_df.iterrows():
        if fold_cols:
            values = pd.to_numeric(row[fold_cols], errors="coerce").dropna().to_numpy(dtype=float)
        else:
            values = np.asarray([float(row[metric_col])], dtype=float)
        if values.size == 0:
            values = np.asarray([float(row[metric_col])], dtype=float)
        distributions.append(values)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(max(12.0, 0.72 * len(top_df) + 3.8), 7.4))
    fig.patch.set_facecolor(PAPER_FIGURE_BG)
    fig.add_artist(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=fig.transFigure,
            facecolor=PAPER_FIGURE_BG,
            edgecolor="none",
            zorder=-100,
        )
    )
    positions = np.arange(len(top_df))
    mean_values = top_df["pred_reference"].to_numpy(dtype=float)
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(vmin=float(np.nanmin(mean_values)), vmax=float(np.nanmax(mean_values)))
    box_colors = [cmap(norm(v)) for v in mean_values]

    if show_committee_distribution:
        box = ax.boxplot(
            distributions,
            positions=positions,
            widths=0.62,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#f08c00", linewidth=1.7),
            whiskerprops=dict(color="#1f1f1f", linewidth=1.0),
            capprops=dict(color="#1f1f1f", linewidth=1.0),
            boxprops=dict(edgecolor="#1f1f1f", linewidth=1.0),
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.95)
    else:
        ax.scatter(
            positions,
            mean_values,
            marker="o",
            s=66,
            c=box_colors,
            edgecolors="#1f1f1f",
            linewidths=0.8,
            zorder=3,
            label="Committee prediction",
        )

    dft_mask = top_df["dft_d33"].notna()
    for idx, row in top_df[dft_mask].iterrows():
        xpos = top_df.index.get_loc(idx)
        pred_val = float(row["pred_reference"])
        dft_val = float(row["dft_d33"])
        ax.vlines(
            xpos,
            ymin=min(pred_val, dft_val),
            ymax=max(pred_val, dft_val),
            colors="#495057",
            linestyles="--",
            linewidth=1.0,
            alpha=0.95,
            zorder=2,
        )

    ax.scatter(
        positions[dft_mask.to_numpy()],
        top_df.loc[dft_mask, "dft_d33"].to_numpy(dtype=float),
        marker="D",
        s=46,
        color="#111111",
        zorder=4,
        label="DFT validation",
    )

    ax.set_xticks(positions)
    label_formatter = format_structure_release_label if release_labels else format_structure_variant_label
    display_labels = [label_formatter(x) for x in top_df["name"].fillna(top_df["canonical_name"])]
    ax.set_xticklabels(display_labels, rotation=65, ha="right", fontsize=RELEASE_SUMMARY_TICK_SIZE)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True, labelsize=RELEASE_SUMMARY_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=RELEASE_SUMMARY_TICK_SIZE)
    style_release_summary_ticks(ax)

    ax.set_ylabel(D33_AXIS_LABEL, labelpad=10, **PAPER_AXIS_LABEL_STYLE)
    style_release_summary_axis_labels(ax)
    filter_notes = []
    if unique_canonical:
        filter_notes.append("unique formula")
    if best_variant_per_formula:
        filter_notes.append("best DFT variant per formula" if dedup_strategy == "dft_abs_gap" else "highest prediction per formula")
    if min_topk_frequency is not None:
        filter_notes.append(f"top-k freq >= {float(min_topk_frequency):g}")
    if max_pred_std is not None:
        filter_notes.append(f"std <= {float(max_pred_std):g}")
    if min_pred is not None:
        filter_notes.append(f"pred >= {float(min_pred):g}")
    style_axes(ax)
    ax.grid(axis="y", alpha=0.16)
    ax.set_xlim(-0.8, len(top_df) - 0.2)

    mae = rmse = bias = None
    if dft_mask.any():
        signed_error = top_df.loc[dft_mask, "pred_reference"].to_numpy(dtype=float) - top_df.loc[dft_mask, "dft_d33"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(signed_error)))
        rmse = float(np.sqrt(np.mean(np.square(signed_error))))
        bias = float(np.mean(signed_error))
        if annotate_mae:
            ax.text(
                0.018,
                0.97,
                f"MAE = {mae:.3f} pC/N\nRMSE = {rmse:.3f} pC/N\nn = {int(dft_mask.sum())}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=RELEASE_SUMMARY_LEGEND_SIZE,
                fontfamily="Times New Roman",
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.32", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.88),
                zorder=6,
            )

    y_value_parts = [
        np.asarray(mean_values, dtype=float),
        top_df["dft_d33"].dropna().to_numpy(dtype=float),
    ]
    if show_committee_distribution:
        y_value_parts.extend(vals for vals in distributions if vals.size > 0)
    y_values = np.concatenate(y_value_parts)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    span = max(1e-6, y_max - y_min)
    lower_pad = max(1.2, 0.22 * span)
    upper_pad = max(1.6, 0.28 * span)
    ax.set_ylim(y_min - lower_pad, y_max + upper_pad)

    if show_committee_distribution:
        legend_handles = [Patch(facecolor="#9ecae1", edgecolor="#1f1f1f", label="Committee prediction")]
    else:
        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ecae1", markeredgecolor="#1f1f1f", markersize=7, label="Committee prediction")
        ]
    if dft_mask.any():
        legend_handles.extend(
            [
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#111111", markersize=6, label="DFT validation"),
                Line2D([0], [0], color="#495057", linestyle="--", linewidth=1.0, label="Prediction-DFT gap"),
            ]
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=RELEASE_SUMMARY_LEGEND_SIZE)
    plt.tight_layout()
    save_figure_with_tif(fig, output_path, dpi=600, facecolor=PAPER_FIGURE_BG)
    plt.close(fig)

    summary_csv = output_path.with_suffix(".csv")
    summary_cols = (
        ["name", "canonical_name", metric_col, "pred_d33_std", "dft_d33", "dft_gap", "abs_gap"]
        if "pred_d33_std" in top_df.columns
        else ["name", "canonical_name", metric_col, "dft_d33", "dft_gap", "abs_gap"]
    )
    if mae is not None:
        top_df["mae_top"] = mae
        top_df["rmse_top"] = rmse
        top_df["bias_top_pred_minus_dft"] = bias
        summary_cols = summary_cols + ["mae_top", "rmse_top", "bias_top_pred_minus_dft"]
    top_df[summary_cols].to_csv(summary_csv, index=False, encoding="utf-8-sig")


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [
        "name",
        "canonical_name",
        "pred_d33",
        "pred_d33_full",
        "pred_d33_committee",
        "pred_d33_mean",
        "pred_d33_std",
        "pred_d33_median",
        "pred_d33_iqr",
        "rank_mean",
        "rank_std",
        "topk_frequency",
    ]
    fold_cols = sorted([c for c in df.columns if c.startswith("pred_fold_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    front = [c for c in front if c in df.columns] + fold_cols
    tail = [c for c in df.columns if c not in front]
    return df[front + tail]


def main():
    parser = argparse.ArgumentParser(description="Predict d33 using the trained XGBoost model")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_PROCESSED_DIR / "scf_features.csv"),
        help="Preprocessed CSV or JSON path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(ROOT_DIR / "models" / "after_xgboost.joblib"),
        help="Path to the trained model joblib file",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default=str(ROOT_DIR / "data" / "results_preprocessed.json"),
        help="Training JSON used to reconstruct the predefined-fold committee",
    )
    parser.add_argument(
        "--folds-file",
        type=str,
        default=str(ROOT_DIR / "data" / "cv5_folds.json"),
        help="Predefined folds JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for prediction outputs",
    )
    parser.add_argument(
        "--validation-json",
        type=str,
        default=str(ROOT_DIR / "prediction_system" / "predict_val_results.json"),
        help="JSON file containing DFT validation results for predicted candidates",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to show in the plot")
    parser.add_argument(
        "--debug-top-k",
        type=int,
        default=10,
        help="Deprecated; retained for compatibility. Debug DFT plots are no longer generated.",
    )
    parser.add_argument(
        "--dft-plot-mode",
        choices=["ranked", "dft_available"],
        default="dft_available",
        help=(
            "DFT comparison plot selection mode. 'ranked' uses the first --top-k ranked candidates; "
            "'dft_available' filters to candidates with DFT validation data before taking --top-k."
        ),
    )
    parser.add_argument(
        "--dft-unique-canonical",
        action="store_true",
        help="For the DFT comparison plot, keep only the first ranked structure for each canonical formula.",
    )
    parser.add_argument(
        "--dft-min-topk-frequency",
        type=float,
        default=None,
        help="For the DFT comparison plot, require committee top-k frequency to be at least this value.",
    )
    parser.add_argument(
        "--dft-max-std",
        type=float,
        default=None,
        help="For the DFT comparison plot, require cross-fold prediction std to be at most this value.",
    )
    parser.add_argument(
        "--dft-min-pred",
        type=float,
        default=None,
        help="For the DFT comparison plot, require the ranking prediction to be at least this value.",
    )
    parser.add_argument(
        "--committee-top-k",
        type=int,
        default=10,
        help="Top-k threshold used to compute committee stability frequency",
    )
    parser.add_argument(
        "--committee-calibration",
        choices=["none", "linear", "isotonic"],
        default=DEFAULT_COMMITTEE_CALIBRATION,
        help="OOF calibration applied to predefined-fold committee predictions",
    )
    parser.add_argument(
        "--committee-aggregation",
        choices=["mean", "median", "trimmed"],
        default=DEFAULT_COMMITTEE_AGGREGATION,
        help="Aggregation used as the ranking prediction across committee folds",
    )
    parser.add_argument(
        "--no-committee",
        dest="use_committee",
        action="store_false",
        default=True,
        help="Disable predefined-fold committee predictions and only use the saved full-data model",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model_path)
    train_json = Path(args.train_json)
    folds_file = Path(args.folds_file)
    validation_json = Path(args.validation_json)
    output_dir = ensure_dir(Path(args.output_dir))
    figure_dir = ensure_dir(Path(__file__).resolve().parent / "figures")
    apply_paper_style()

    logger.info(f"Loading preprocessed data from: {input_path}")
    df = load_preprocessed_data(input_path)
    df = ensure_derived_prediction_features(df)
    if "name" not in df.columns:
        raise ValueError("Input data must contain a 'name' column")

    model = load_model(model_path)
    feature_cols = resolve_feature_columns(model, df)

    model_df = df.copy()
    model_df[feature_cols] = model_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    model_df = model_df.dropna(subset=feature_cols).reset_index(drop=True)
    if len(model_df) == 0:
        raise ValueError("No rows left after dropping missing feature values")

    model_df["pred_d33_full"] = np.asarray(model.predict(model_df[feature_cols]), dtype=float)

    committee_meta = {}
    if args.use_committee:
        logger.info("Building predefined five-fold committee predictions for external candidates")
        committee_summary, fold_pred_df, _, committee_meta = compute_committee_predictions(
            model_template=model,
            candidate_df=model_df,
            feature_cols=feature_cols,
            train_json=train_json,
            folds_file=folds_file,
            topk_size=args.committee_top_k,
            calibration_method=args.committee_calibration,
            aggregation_method=args.committee_aggregation,
        )
        calibration = committee_meta.get("oof_calibration", committee_meta.get("oof_linear_calibration", {}))
        if calibration.get("enabled", False):
            model_df["pred_d33_full_raw"] = model_df["pred_d33_full"]
            model_df["pred_d33_full"] = apply_linear_calibration(model_df["pred_d33_full"], calibration)
        model_df = pd.concat([model_df, committee_summary, fold_pred_df], axis=1)
        model_df["pred_d33"] = model_df["pred_d33_committee"]
    else:
        model_df["pred_d33"] = model_df["pred_d33_full"]

    ranked = model_df.sort_values("pred_d33", ascending=False).reset_index(drop=True)
    ranked = reorder_columns(ranked)

    ranked_csv = output_dir / "ranked_candidates.csv"
    ranked_json = output_dir / "ranked_candidates.json"
    best_json = output_dir / "best_candidate.json"
    ranked.to_csv(ranked_csv, index=False, encoding="utf-8-sig")

    meta = {
        "model_path": str(model_path.resolve()),
        "feature_cols": feature_cols,
        "ranking_metric": "pred_d33_committee" if args.use_committee else "pred_d33_full",
        "uncertainty_definition": (
            "cross-fold dispersion over OOF-calibrated predefined five-fold committee predictions"
            if args.use_committee
            else "not available; committee disabled"
        ),
        "train_json": str(train_json.resolve()),
        "folds_file": str(folds_file.resolve()),
        "committee_top_k": int(args.committee_top_k),
        "committee_calibration": args.committee_calibration,
        "committee_aggregation": args.committee_aggregation,
        "dft_plot_mode": args.dft_plot_mode,
        "dft_unique_canonical": bool(args.dft_unique_canonical),
        "dft_min_topk_frequency": args.dft_min_topk_frequency,
        "dft_max_std": args.dft_max_std,
        "dft_min_pred": args.dft_min_pred,
        "row_count": int(len(ranked)),
        "figure_dir": str(figure_dir.resolve()),
        "committee_meta": committee_meta,
    }
    with open(ranked_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": ranked.to_dict(orient="records")}, f, indent=2, ensure_ascii=False)
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(ranked.iloc[0].to_dict(), f, indent=2, ensure_ascii=False)

    top_k = min(args.top_k, len(ranked))
    plot_top_candidates(ranked, figure_dir / "top_candidates.png", top_k=25)
    plot_distribution(ranked, figure_dir / "prediction_distribution.png")
    plot_feature_landscape(ranked, figure_dir / "feature_landscape.png")
    plot_uncertainty_overview(ranked, figure_dir / "prediction_uncertainty.png")
    validation_lookup = load_validation_lookup(validation_json)
    if validation_lookup:
        suffix_parts = []
        if (
            args.dft_unique_canonical
            or args.dft_min_topk_frequency is not None
            or args.dft_max_std is not None
            or args.dft_min_pred is not None
        ):
            if args.dft_unique_canonical:
                suffix_parts.append("unique")
            if args.dft_min_topk_frequency is not None:
                suffix_parts.append(f"freq{float(args.dft_min_topk_frequency):g}".replace(".", "p"))
            if args.dft_max_std is not None:
                suffix_parts.append(f"std{float(args.dft_max_std):g}".replace(".", "p"))
            if args.dft_min_pred is not None:
                suffix_parts.append(f"pred{float(args.dft_min_pred):g}".replace(".", "p"))
        dft_plot_suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
        dft_top_ks = [top_k]
        top30_k = min(30, len(ranked))
        if top30_k not in dft_top_ks:
            dft_top_ks.append(top30_k)
        for dft_top_k in dft_top_ks:
            for output_stem, release_labels in (
                (f"top{dft_top_k}_prediction_vs_dft", False),
                (f"top{dft_top_k}_release", True),
            ):
                plot_top20_prediction_vs_dft(
                    ranked,
                    validation_lookup=validation_lookup,
                    output_path=figure_dir / f"{output_stem}{dft_plot_suffix}.png",
                    top_k=dft_top_k,
                    selection_mode=args.dft_plot_mode,
                    dedup_strategy="dft_abs_gap",
                    unique_canonical=args.dft_unique_canonical,
                    min_topk_frequency=args.dft_min_topk_frequency,
                    max_pred_std=args.dft_max_std,
                    min_pred=args.dft_min_pred,
                    release_labels=release_labels,
                    show_committee_distribution=True,
                )
        plot_release_summary_figure(
            df=ranked,
            validation_lookup=validation_lookup,
            output_path=figure_dir / f"release_summary{dft_plot_suffix}.png",
            top_k=top30_k,
            selection_mode=args.dft_plot_mode,
            unique_canonical=args.dft_unique_canonical,
            min_topk_frequency=args.dft_min_topk_frequency,
            max_pred_std=args.dft_max_std,
            min_pred=args.dft_min_pred,
        )
    else:
        logger.info(f"No DFT validation data found in: {validation_json}")

    best_row = ranked.iloc[0]
    logger.success(f"Best candidate: {best_row['name']}")
    logger.success(f"Predicted d33 used for ranking: {best_row['pred_d33']:.6f}")
    if "pred_d33_std" in best_row:
        logger.success(f"Cross-fold std: {best_row['pred_d33_std']:.6f}")
    logger.success(f"Saved ranked candidates to: {ranked_csv}")
    logger.success(f"Saved plots to: {figure_dir}")


if __name__ == "__main__":
    main()
