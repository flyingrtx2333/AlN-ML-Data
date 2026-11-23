#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified machine learning training script.
Supported models: Ridge, Random Forest, SVR, GPR, XGBoost, AdaBoost, and RFR.
Swap-based data augmentation is optional.
When swap augmentation is enabled, GroupKFold keeps related samples together to prevent data leakage.
Predicted vs. actual plots label points with an absolute error greater than 5 by chemical composition for debugging.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import warnings
import re
import traceback
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb
from tqdm import tqdm
from loguru import logger
from config import selected_features, selected_features_after, gen_row
from utils.DrawHeatmap import CORRELATION_CMAP, plot_all_models_heatmap, plot_per_model_heatmaps
from utils.DrawLandscape import plot_structure_property_landscape
from utils.DrawPrediction import plot_predictions
from utils.DrawReleaseFigure import plot_landscape_shap_release_figure
from utils.DrawShap import save_shap_summary_plot
from utils.feature_labels import apply_feature_label_text_style, format_feature_labels
from utils.figure_io import save_figure_with_tif
from utils.plot_style import (
    PAPER_AXIS_LABEL_STYLE,
    PAPER_FEATURE_LABEL_STYLE,
    PAPER_FIGURE_BG,
    PAPER_FONT_FAMILY,
    PAPER_PANEL_LABEL_STYLE,
    PAPER_PLOT_STYLE_CONTEXT,
    PAPER_TICK_LABEL_STYLE,
    PAPER_TITLE_STYLE,
    center_last_row_axes,
)


def prefixed_filename(filename_prefix, filename):
    if not filename_prefix:
        return filename
    return f"{filename_prefix}_{filename}"


def prefix_saved_file(path, filename_prefix):
    if not path or not filename_prefix:
        return path
    path = Path(path)
    if not path.exists():
        return str(path)
    prefixed_path = path.with_name(prefixed_filename(filename_prefix, path.name))
    path.replace(prefixed_path)
    tif_path = path.with_suffix(".tif")
    if tif_path.exists():
        tif_path.replace(prefixed_path.with_suffix(".tif"))
    return str(prefixed_path)


def safe_metric_pct_change(before_value, delta_value):
    if before_value is None or pd.isna(before_value) or abs(before_value) < 1e-12:
        return np.nan
    return delta_value / abs(before_value) * 100.0


XGB_HIGH_D33_WEIGHT_CUTOFF = 20
XGB_HIGH_D33_WEIGHT_ALPHA = 0.0


def get_xgb_sample_weight(y_values):
    y_arr = np.asarray(y_values, dtype=float)
    scale = max(30.0 - XGB_HIGH_D33_WEIGHT_CUTOFF, 1e-12)
    high_d33_boost = np.clip((y_arr - XGB_HIGH_D33_WEIGHT_CUTOFF) / scale, 0.0, None)
    return 1.0 + XGB_HIGH_D33_WEIGHT_ALPHA * high_d33_boost


def fit_pipeline_with_optional_weights(model_name, pipeline, X, y):
    if model_name == 'gpr':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')
            pipeline.fit(X, y)
        return pipeline
    if model_name == 'xgboost':
        pipeline.fit(X, y, model__sample_weight=get_xgb_sample_weight(y))
        return pipeline
    pipeline.fit(X, y)
    return pipeline


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
# Configure the sans-serif font fallback list.
# SimHei supports Chinese glyphs. Use math text ($R^2$) to render R-squared independently of the font.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Render superscripts with math text.
plt.rcParams['mathtext.default'] = 'regular'
MODEL_RF = Pipeline([
    ('model', RandomForestRegressor(
        n_estimators=300,          # Increase to 300–500 to reduce variance
        max_depth=6,               # Increase the previous depth of 4 while keeping trees shallow
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',       # Consider 0.5–0.8 when many features are available
        random_state=42,
        n_jobs=-1
    ))
])
MODEL_XGBOOST = Pipeline([
    ('model', xgb.XGBRegressor(
        n_estimators=897,
        learning_rate=0.06625672133240959,
        max_depth=3,
        min_child_weight=5,
        subsample=0.9320837825359516,
        colsample_bytree=0.6965488623333802,
        reg_lambda=3.4611014245944305,
        reg_alpha=0.3805515087673986,
        gamma=0.8920465551771133,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ))
])


def build_pipeline(model_name, random_seed=42):
    if model_name == 'ridge':
        from sklearn.linear_model import Ridge
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
    if model_name == 'rf':
        return MODEL_RF
    if model_name == 'svr':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10, epsilon=0.1))
        ])
    if model_name == 'gpr':
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) \
                 + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=30, alpha=0.0,
                normalize_y=True, random_state=random_seed,
                optimizer='fmin_l_bfgs_b'
            ))
        ])
    if model_name == 'xgboost':
        return MODEL_XGBOOST
    if model_name == 'gbdt':
        return Pipeline([
            ('model', GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=random_seed
            ))
        ])
    if model_name == 'adaboost':
        from sklearn.tree import DecisionTreeRegressor
        base_estimator = DecisionTreeRegressor(max_depth=3, random_state=random_seed)
        return Pipeline([
            ('model', AdaBoostRegressor(
                estimator=base_estimator, n_estimators=100,
                learning_rate=0.1, random_state=random_seed
            ))
        ])
    if model_name == 'rfr':
        return MODEL_RF
    raise ValueError(f"Unknown model name: {model_name}")


def load_and_clean(json_file, max_threshold=30, min_threshold=3,  use_swap=True, selected_features=selected_features):
    """
    Load and clean the training data.
    
    Args:
        json_file: Path to the JSON file.
        max_threshold: Exclude samples with d33 above this value.
        use_swap: Include swap-augmented samples when True.
    
    Returns:
        DataFrame: Cleaned data.
        dict: Map from row indices to chemical compositions (structure names).
        list: GroupKFold group labels shared by original and swap-augmented samples.
    """
    logger.info(f"Loading and cleaning data from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    composition_map = {}  # Map row indices to chemical compositions.
    groups = []  # Group labels for GroupKFold.
    idx = 0
    
    for key, val in data.items():
        try:
            # Exclude entries with the _swap suffix when swap augmentation is disabled.
            if not use_swap and key.endswith('_swap'):
                continue
                
            d33_val = val["d33"]
            if d33_val is None or d33_val > max_threshold or d33_val < min_threshold:
                continue
            
            rows.append(gen_row(key,val))
            composition_map[idx] = key  # Store the chemical composition (structure name).
            
            # Assign each original sample and its swapped variant to the same GroupKFold group.
            if key.endswith('_swap'):
                # Use the original structure name without the _swap suffix.
                group_name = key.replace('_swap', '')
            else:
                # Use the structure name as the group label.
                group_name = key
            groups.append(group_name)
            
            idx += 1
        except Exception as e:
            logger.error(f"Error processing structure {key}: {e}")
            continue
    
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    required_columns = selected_features + ['target']
    missing_features = [col for col in required_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing selected feature columns: {missing_features}")

    # Only the active feature set should decide whether a row is usable.
    # Dropping on every gen_row column discards valid CV samples because some
    # unused exploratory descriptors are missing.
    df = df[required_columns]
    valid_indices = df.dropna().index.tolist()
    df = df.loc[valid_indices]

    # Remap composition_map and groups to the new indices.
    new_composition_map = {}
    new_groups = []
    for new_idx, old_idx in enumerate(valid_indices):
        if old_idx < len(composition_map):
            new_composition_map[new_idx] = composition_map[old_idx]
        if old_idx < len(groups):
            new_groups.append(groups[old_idx])
    composition_map = new_composition_map
    groups = new_groups
    
    # Reset the index to contiguous values from 0 to N-1.
    df = df.reset_index(drop=True)
    
    return df, composition_map, groups


def load_predefined_folds(folds_file):
    with open(folds_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    folds = payload.get("folds", [])
    if not folds:
        raise ValueError(f"No valid folds in {folds_file}")
    return payload.get("meta", {}), folds


def build_name_to_idx(composition_map):
    return {str(v): int(k) for k, v in composition_map.items()}


def resolve_fold_indices(fold_entry, name_to_idx):
    train_idx = [name_to_idx[n] for n in fold_entry.get("train_names", []) if n in name_to_idx]
    test_idx = [name_to_idx[n] for n in fold_entry.get("test_names", []) if n in name_to_idx]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Fold {fold_entry.get('fold', '?')} has an empty training or test set after mapping")
    return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)


def load_fold_feature_map(fold_feature_file):
    path = Path(fold_feature_file)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("fold_feature_lists", {})


def train_model_with_predefined_folds(model_name, X, y, folds, composition_map, fold_feature_map=None):
    """
    Train a model using predefined folds, which determine the fold count and test set size.
    Optionally extend the feature set with per-fold PySR formulas.
    """
    random_seed = 42
    name_to_idx = build_name_to_idx(composition_map)
    pipeline = build_pipeline(model_name, random_seed)

    r2_scores = []
    mae_scores = []
    all_y_train = []
    all_y_train_pred = []
    all_train_indices = []
    all_y_test = []
    all_y_pred = []
    all_test_indices = []
    splits_list = []
    fold_feature_map = fold_feature_map or {}

    pbar = tqdm(folds, desc=f"Predefined 5-fold CV: {model_name.upper()}", ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for fold_entry in pbar:
        fold_id = int(fold_entry.get("fold", 0))
        train_idx, test_idx = resolve_fold_indices(fold_entry, name_to_idx)
        splits_list.append((train_idx, test_idx))

        X_train_split = X.iloc[train_idx]
        X_test_split = X.iloc[test_idx]
        y_train_split = y.iloc[train_idx]
        y_test_split = y.iloc[test_idx]

        fold_features = fold_feature_map.get(str(fold_id), [])
        if fold_features:
            valid_cols = [c for c in fold_features if c in X.columns]
            if len(valid_cols) > 0:
                X_train_split = X_train_split[valid_cols]
                X_test_split = X_test_split[valid_cols]

        fit_pipeline_with_optional_weights(model_name, pipeline, X_train_split, y_train_split)

        y_train_pred_split = pipeline.predict(X_train_split)
        y_pred_split = pipeline.predict(X_test_split)
        r2 = r2_score(y_test_split, y_pred_split)
        mae = mean_absolute_error(y_test_split, y_pred_split)
        r2_scores.append(r2)
        mae_scores.append(mae)
        all_train_indices.extend(train_idx.tolist())
        all_y_train.extend(y_train_split.values.tolist())
        all_y_train_pred.extend(np.asarray(y_train_pred_split).tolist())
        all_test_indices.extend(test_idx.tolist())
        all_y_test.extend(y_test_split.values.tolist())
        all_y_pred.extend(np.asarray(y_pred_split).tolist())
        pbar.set_postfix({'Median R2': f'{np.median(r2_scores):.4f}', 'Median MAE': f'{np.median(mae_scores):.4f}'})

    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores, ddof=1) if len(r2_scores) > 1 else 0.0
    r2_median = np.median(r2_scores)
    r2_iqr = np.percentile(r2_scores, 75) - np.percentile(r2_scores, 25)
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores, ddof=1) if len(mae_scores) > 1 else 0.0
    mae_median = np.median(mae_scores)
    mae_iqr = np.percentile(mae_scores, 75) - np.percentile(mae_scores, 25)

    return (r2_mean, r2_std, r2_median, r2_iqr,
            mae_mean, mae_std, mae_median, mae_iqr,
            all_y_train, all_y_train_pred, all_train_indices,
            all_y_test, all_y_pred, all_test_indices, splits_list)


def train_model_with_groupkfold(model_name, X, y, groups, n_splits=5, n_repeats=3, use_groups=True):
    """
    Train a model with GroupKFold or KFold cross-validation.
    
    Args:
        model_name: Name of the model.
        X: Feature data.
        y: Target values.
        groups: Group labels for GroupKFold.
        n_splits: Number of cross-validation folds.
        n_repeats: Number of repetitions.
        use_groups: Use GroupKFold. Required when using swap-augmented data.
    
    Returns:
        r2_mean, r2_std, r2_median, r2_iqr,
        mae_mean, mae_std, mae_median, mae_iqr,
        all_y_test, all_y_pred, all_test_indices, splits_list
    """
    random_seed = 42
    
    
    # Build splits for repeated cross-validation (n_splits * n_repeats).
    splits_list = []
    for repeat_idx in range(n_repeats):
        if use_groups and groups is not None:
            splitter = GroupKFold(n_splits=n_splits)
            splits_list.extend(list(splitter.split(X, y, groups)))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat_idx)
            splits_list.extend(list(splitter.split(X, y)))
    
    pipeline = build_pipeline(model_name, random_seed)
    
    r2_scores = []
    mae_scores = []
    all_y_test = []
    all_y_pred = []
    all_test_indices = []
    
    # Display training progress with tqdm.
    pbar = tqdm(splits_list, desc=f"{n_splits} folds × {n_repeats} repeats: {model_name.upper()}", ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for split_idx, (train_idx, test_idx) in enumerate(pbar):
        X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
        y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
        
        fit_pipeline_with_optional_weights(model_name, pipeline, X_train_split, y_train_split)
        y_pred_split = pipeline.predict(X_test_split)
        
        r2 = r2_score(y_test_split, y_pred_split)
        mae = mean_absolute_error(y_test_split, y_pred_split)
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        all_test_indices.extend(test_idx)
        all_y_test.extend(y_test_split.values)
        all_y_pred.extend(y_pred_split)
        
        # Display the current median R² and MAE in the progress bar.
        if len(r2_scores) > 0:
            current_r2_median = np.median(r2_scores)
            current_mae_median = np.median(mae_scores)
            pbar.set_postfix({'Median R²': f'{current_r2_median:.4f}', 'Median MAE': f'{current_mae_median:.4f}'})
    
    # Calculate medians and interquartile ranges (IQR).
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores, ddof=1) if len(r2_scores) > 1 else 0.0
    r2_median = np.median(r2_scores)
    r2_q25 = np.percentile(r2_scores, 25)
    r2_q75 = np.percentile(r2_scores, 75)
    r2_iqr = r2_q75 - r2_q25
    
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores, ddof=1) if len(mae_scores) > 1 else 0.0
    mae_median = np.median(mae_scores)
    mae_q25 = np.percentile(mae_scores, 25)
    mae_q75 = np.percentile(mae_scores, 75)
    mae_iqr = mae_q75 - mae_q25
    
    return (r2_mean, r2_std, r2_median, r2_iqr,
            mae_mean, mae_std, mae_median, mae_iqr,
            all_y_test, all_y_pred, all_test_indices, splits_list)


def plot_all_models_comparison(all_predictions, output_dir, filename_prefix=""):
    """
    Plot predicted vs. actual values with one panel for each model.
    
    Args:
        all_predictions: Map from model names to (y_test, y_pred, r2, mae) or (y_test, y_pred, r2, mae, sigma).
        output_dir: Output directory.
        filename_prefix: Optional prefix for the output filename.
    """
    n_models = len(all_predictions)
    if n_models == 0:
        return

    # Arrange the panels in a nearly square grid.
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))

    train_color = '#065F46'
    test_color = '#EA580C'
    line_color = '#64748B'
    figure_bg = PAPER_FIGURE_BG
    axis_bg = '#FFFFFF'
    grid_color = '#D1FAE5'
    spine_color = '#CBD5E1'
    title_color = '#14213D'
    panel_color = '#000000'
    metric_color = '#14213D'

    with plt.rc_context(PAPER_PLOT_STYLE_CONTEXT):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 5.6 * n_rows))
        fig.patch.set_facecolor(figure_bg)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
            ax = axes[idx]
            ax.set_facecolor(axis_bg)
            panel_label = f"({chr(97 + idx)})" if idx < 26 else f"({idx + 1})"

            payload = extract_prediction_payload(pred_data)
            y_test = payload["test_actual"]
            y_pred = payload["test_pred"]
            y_train = payload["train_actual"]
            y_train_pred = payload["train_pred"]
            r2 = payload["r2"]
            mae = payload["mae"]
            r2_std = payload["r2_std"]
            train_fraction = payload["train_fraction"]
            test_fraction = payload["test_fraction"]

            if len(y_train) > 0 and len(y_train_pred) > 0:
                train_label = 'Train'
                if train_fraction is not None:
                    train_label = f'Train {train_fraction * 100:.0f}%'
                ax.scatter(
                    y_train,
                    y_train_pred,
                    alpha=0.58,
                    s=42,
                    color=train_color,
                    edgecolors='white',
                    linewidth=0.4,
                    label=train_label,
                    zorder=2,
                )
            test_label = 'Test'
            if test_fraction is not None:
                test_label = f'Test {test_fraction * 100:.0f}%'
            ax.scatter(
                y_test,
                y_pred,
                alpha=0.68,
                s=52,
                color=test_color,
                edgecolors='none',
                linewidth=0.0,
                label=test_label,
                zorder=3,
            )

            value_blocks = [y_test, y_pred]
            if len(y_train) > 0:
                value_blocks.extend([y_train, y_train_pred])
            local_min = min(float(np.min(arr)) for arr in value_blocks if len(arr) > 0)
            local_max = max(float(np.max(arr)) for arr in value_blocks if len(arr) > 0)
            local_span = local_max - local_min
            local_margin = local_span * 0.08 if local_span > 0 else 1.0
            local_axis_range = [local_min - local_margin, local_max + local_margin]

            ax.plot(local_axis_range, local_axis_range, color=line_color, linestyle='--', lw=1.9, label='Ideal', zorder=1)
            ax.set_xlim(local_axis_range)
            ax.set_ylim(local_axis_range)
            ax.set_xlabel(r'Actual $\mathbf{\mathit{d}}_{33}$ (pC/N)', **PAPER_AXIS_LABEL_STYLE)
            ax.set_ylabel(r'Predicted $\mathbf{\mathit{d}}_{33}$ (pC/N)', **PAPER_AXIS_LABEL_STYLE)
            title_style = PAPER_TITLE_STYLE.copy()
            title_style['color'] = title_color
            ax.set_title(f'{model_name}', pad=12, **title_style)
            panel_style = PAPER_PANEL_LABEL_STYLE.copy()
            panel_style['color'] = panel_color
            ax.text(
                -0.16,
                1.08,
                panel_label,
                transform=ax.transAxes,
                ha='left',
                va='top',
                clip_on=False,
                **panel_style,
            )
            ax.grid(True, color=grid_color, linewidth=1.0, alpha=0.95)
            ax.tick_params(axis='both', labelsize=PAPER_TICK_LABEL_STYLE['fontsize'], direction='in', length=5, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight(PAPER_TICK_LABEL_STYLE['fontweight'])
                label.set_fontfamily(PAPER_TICK_LABEL_STYLE['fontfamily'])
                label.set_color(PAPER_TICK_LABEL_STYLE['color'])
            for spine in ax.spines.values():
                spine.set_color(spine_color)
                spine.set_linewidth(1.8)
            ax.text(
                0.68,
                0.04,
                # f"R² = {r2:.2f}" + (f" ± {r2_std:.2f}" if r2_std is not None else "") + f"\nMAE = {mae:.2f}",
                f"R² = {r2:.2f}" + f"\nMAE = {mae:.2f}",
                transform=ax.transAxes,
                ha='left',
                va='bottom',
                fontsize=15,
                fontweight='bold',
                fontfamily='Times New Roman',
                multialignment='left',
                color=metric_color,
                bbox=dict(boxstyle='round,pad=0.28', facecolor='none', edgecolor='none'),
            )
            ax.legend(
                loc='upper left',
                frameon=False,
                markerscale=0.9,
                prop={'family': PAPER_FONT_FAMILY, 'size': 15, 'weight': 'bold'},
            )
            ax.set_aspect('equal', adjustable='box')

        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout(pad=1.4)
        center_last_row_axes(axes, n_models, n_cols)
        comparison_path = str(Path(output_dir) / prefixed_filename(filename_prefix, 'all_models_predictions_comparison.png'))
        save_figure_with_tif(plt.gcf(), comparison_path, dpi=400, bbox_inches='tight', facecolor=figure_bg)
        print(f"\nAll models prediction comparison saved to: {comparison_path}")
        plt.close()


def get_feature_importance(model_name, pipeline, X, y, feature_names, random_seed=42):
    """
    Get feature importances for a model.
    
    Args:
        model_name: Name of the model.
        pipeline: Fitted pipeline.
        X: Feature data.
        y: Target values.
        feature_names: List of feature names.
        random_seed: Random seed.
    
    Returns:
        feature_importance: Array of normalized feature importances.
    """
    # Extract the estimator from the pipeline.
    if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
    elif hasattr(pipeline, 'named_steps') and len(pipeline.named_steps) > 0:
        # Use the last step, which is usually the estimator.
        model = list(pipeline.named_steps.values())[-1]
    else:
        model = pipeline
    
    # Determine feature importance based on the model type.
    if model_name == 'ridge':
        # Linear models use absolute coefficient values.
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            # Return an array of zeros if the model has no coef_ attribute.
            importance = np.zeros(len(feature_names))
    elif model_name in ['rf', 'rfr', 'adaboost', 'xgboost', 'gbdt']:
        # Tree-based models use feature_importances_.
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Return an array of zeros if the model has not been fitted.
            importance = np.zeros(len(feature_names))
    elif model_name in ['svr', 'gpr']:
        # SVR and GPR use permutation importance.
        try:
            # Calculate permutation importance.
            perm_importance = permutation_importance(
                pipeline, X, y, n_repeats=10, random_state=random_seed, n_jobs=-1
            )
            importance = perm_importance.importances_mean
        except:
            # Return an array of zeros on failure.
            importance = np.zeros(len(feature_names))
    else:
        # Other models use permutation importance.
        try:
            perm_importance = permutation_importance(
                pipeline, X, y, n_repeats=10, random_state=random_seed, n_jobs=-1
            )
            importance = perm_importance.importances_mean
        except:
            importance = np.zeros(len(feature_names))
    
    # Normalize values to the range 0–1.
    if np.sum(importance) > 0:
        importance = importance / np.sum(importance)
    
    return importance


def plot_all_models_feature_importance(
    X,
    y,
    groups,
    models_to_train,
    output_dir,
    use_groups=True,
    random_seed=42,
    filename_prefix="",
):
    """
    Plot feature importances with one panel for each model.
    
    Args:
        X: Feature data.
        y: Target values.
        groups: Group labels.
        models_to_train: List of models to train.
        output_dir: Output directory.
        use_groups: Use GroupShuffleSplit.
        random_seed: Random seed.
    """
    from sklearn.model_selection import train_test_split
    
    # Use 80% of the data to fit models for feature importance.
    if use_groups and groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
        train_idx, _ = next(splitter.split(X, y, groups))
        X_train_imp = X.iloc[train_idx]
        y_train_imp = y.iloc[train_idx]
    else:
        X_train_imp, _, y_train_imp, _ = train_test_split(
            X, y, test_size=0.2, random_state=random_seed
        )
    
    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    top_n_features = min(8, n_features)
    
    # Store feature importances for every model.
    all_importances = {}
    
    # Fit each model and calculate its feature importances.
    for model_name in models_to_train:
        print(f"  Calculating feature importance for {model_name.upper()}...")
        
        try:
            # Create the pipeline.
            if model_name == 'ridge':
                from sklearn.linear_model import Ridge
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', Ridge(alpha=1.0, random_state=random_seed))
                ])
            elif model_name == 'rf':
                pipeline = MODEL_RF
            elif model_name == 'svr':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', SVR(kernel='rbf', C=10, epsilon=0.1))
                ])
            elif model_name == 'gpr':
                # Tune kernel parameters to reduce convergence warnings.
                kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) \
                         + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1))
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GaussianProcessRegressor(
                        kernel=kernel, n_restarts_optimizer=30, alpha=0.0,
                        normalize_y=True, random_state=random_seed,
                        optimizer='fmin_l_bfgs_b'
                    ))
                ])
            elif model_name == 'xgboost':
                pipeline = MODEL_XGBOOST
            elif model_name == 'adaboost':
                from sklearn.tree import DecisionTreeRegressor
                base_estimator = DecisionTreeRegressor(max_depth=3, random_state=random_seed)
                pipeline = Pipeline([
                    ('model', AdaBoostRegressor(
                        estimator=base_estimator, n_estimators=100,
                        learning_rate=0.1, random_state=random_seed
                    ))
                ])
            elif model_name == 'rfr':
                pipeline = MODEL_RF
            else:
                continue
            
            # Fit the model.
            fit_pipeline_with_optional_weights(model_name, pipeline, X_train_imp, y_train_imp)
            
            # Get feature importances.
            importance = get_feature_importance(model_name, pipeline, X_train_imp, y_train_imp, feature_names, random_seed)
            all_importances[model_name.upper()] = importance
            
        except Exception as e:
            print(f"    Error calculating feature importance for {model_name}: {e}")
            continue
    
    # Plot feature importances.
    n_models = len(all_importances)
    if n_models == 0:
        print("  No feature importance data available")
        return
    
    # Set up the subplot grid.
    n_cols = int(np.ceil(np.sqrt(n_models)))
    n_rows = int(np.ceil(n_models / n_cols))
    
    figure_bg = PAPER_FIGURE_BG
    axis_bg = '#FFFFFF'
    grid_color = '#D7E3E8'
    spine_color = '#D5E3E7'
    panel_color = '#000000'
    value_color = '#7C3A4B'
    importance_label_size = 15

    with plt.rc_context(PAPER_PLOT_STYLE_CONTEXT):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.9 * n_cols, 5.2 * n_rows), facecolor=figure_bg)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot one panel per model.
        for idx, (model_name, importance) in enumerate(all_importances.items()):
            ax = axes[idx]
            ax.set_facecolor(axis_bg)
            panel_label = f"({chr(97 + idx)})" if idx < 26 else f"({idx + 1})"

            # Sort features by importance.
            sorted_idx = np.argsort(importance)[::-1][:top_n_features]
            sorted_importance = importance[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]
            sorted_display_names = format_feature_labels(sorted_names)

            color_scale = np.clip(sorted_importance / sorted_importance.max(), 0, 1) if sorted_importance.max() > 0 else np.zeros_like(sorted_importance)
            light_rgb = np.array([248, 242, 244]) / 255.0
            dark_rgb = np.array([201, 110, 126]) / 255.0
            bar_colors = [tuple(light_rgb * (1.0 - val) + dark_rgb * val) for val in color_scale]
            bars = ax.barh(
                range(top_n_features),
                sorted_importance,
                color=bar_colors,
                edgecolor='white',
                linewidth=0.8,
                zorder=3,
            )

            # Set the y-axis labels.
            ax.set_yticks(range(top_n_features))
            ax.set_yticklabels(sorted_display_names, fontsize=importance_label_size)
            ax.invert_yaxis()  # Show the most important feature at the top.
            apply_feature_label_text_style(
                ax.get_yticklabels(),
                font_family=PAPER_FEATURE_LABEL_STYLE['fontfamily'],
                font_weight='bold',
                font_size=importance_label_size,
                font_style=PAPER_FEATURE_LABEL_STYLE['fontstyle'],
                color=PAPER_FEATURE_LABEL_STYLE['color'],
            )

            title_style = PAPER_TITLE_STYLE.copy()
            title_style['fontsize'] = 15
            ax.set_title(model_name, pad=10, **title_style)
            ax.set_xlabel('Feature importance', **PAPER_AXIS_LABEL_STYLE)
            ax.set_ylabel('')
            ax.grid(True, axis='x', color=grid_color, linewidth=0.9, alpha=0.95, zorder=1)
            ax.tick_params(
                axis='x',
                labelsize=PAPER_TICK_LABEL_STYLE['fontsize'],
                direction='in',
                length=4,
                width=1.0,
                colors=PAPER_TICK_LABEL_STYLE['color'],
            )
            ax.tick_params(axis='y', length=0)
            for label in ax.get_xticklabels():
                label.set_fontweight(PAPER_TICK_LABEL_STYLE['fontweight'])
                label.set_fontfamily(PAPER_TICK_LABEL_STYLE['fontfamily'])
                label.set_color(PAPER_TICK_LABEL_STYLE['color'])
            for spine in ax.spines.values():
                spine.set_color(spine_color)
                spine.set_linewidth(1.4)
            panel_style = PAPER_PANEL_LABEL_STYLE.copy()
            panel_style['color'] = panel_color
            ax.text(
                -0.14,
                1.10,
                panel_label,
                transform=ax.transAxes,
                ha='left',
                va='top',
                clip_on=False,
                **panel_style,
            )

            # Add value labels.
            x_max = float(sorted_importance.max()) if sorted_importance.size > 0 else 1.0
            ax.set_xlim(0, x_max * 1.12 if x_max > 0 else 1.0)
            for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
                if val > 0.01:  # Label only features with importance greater than 1%.
                    ax.text(
                        val + x_max * 0.012,
                        i,
                        f'{val:.3f}',
                        va='center',
                        ha='left',
                        fontsize=12,
                        fontweight='bold',
                        fontfamily=PAPER_FONT_FAMILY,
                        color=value_color,
                    )

        # Hide unused subplots.
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09, top=0.92, wspace=0.28, hspace=0.36)
        center_last_row_axes(axes, n_models, n_cols, y_offset=0.012)
        output_path = str(Path(output_dir) / prefixed_filename(filename_prefix, 'all_models_correlation.png'))
        save_figure_with_tif(plt.gcf(), output_path, dpi=300, bbox_inches='tight', facecolor=figure_bg)
        print(f"\nFeature importance comparison saved to: {output_path}")
        plt.close()

    # Plot the combined heatmap and per-model heatmaps.
    try:
        # heatmap_path = plot_all_models_heatmap(all_importances, feature_names, output_dir)
        # if heatmap_path:
        #     print(f"Combined feature importance heatmap saved to: {heatmap_path}")

        subplot_heatmap_path = plot_per_model_heatmaps(all_importances, X, feature_names, output_dir, n_cols=3, top_k=12)
        subplot_heatmap_path = prefix_saved_file(subplot_heatmap_path, filename_prefix)
        if subplot_heatmap_path:
            print(f"Per-model correlation matrices saved to: {subplot_heatmap_path}")
    except Exception as e:
        traceback.print_exc()
        print(f"  Error generating feature importance heatmaps: {e}")


def save_prediction_table(
    output_dir,
    model_name,
    test_indices,
    y_test,
    y_pred,
    composition_map,
    X=None,
    residual_top_k=5,
    filename_prefix="",
):
    """
    Save material names, predictions, and actual values, and print the materials with the largest residuals.
    """
    if test_indices is None or composition_map is None:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    materials = [composition_map.get(int(i), str(i)) for i in test_indices]
    table = pd.DataFrame({
        "material": materials,
        "predicted": y_pred,
        "actual": y_test
    })
    table["residual"] = table["actual"] - table["predicted"]
    table["abs_residual"] = table["residual"].abs()

    if X is not None:
        try:
            feature_rows = X.iloc[test_indices].reset_index(drop=True)
            table = pd.concat([table.reset_index(drop=True), feature_rows.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"  Could not add feature columns: {e}")

    top_k = max(1, int(residual_top_k))
    worst_cases = table.nlargest(top_k, "abs_residual")[[
        "material", "actual", "predicted", "residual", "abs_residual"
    ]]
    print(f"  {model_name}: {len(worst_cases)} materials with the largest residuals")
    for _, row in worst_cases.iterrows():
        print(
            "    - {material}: actual={actual:.4f}, predicted={predicted:.4f}, residual={residual:+.4f}, abs={abs_residual:.4f}".format(
                material=row["material"],
                actual=row["actual"],
                predicted=row["predicted"],
                residual=row["residual"],
                abs_residual=row["abs_residual"],
            )
        )

    table_path = output_dir / prefixed_filename(filename_prefix, f"{safe_name}_predictions_table.csv")
    table.to_csv(table_path, index=False)
    print(f"  Prediction table for {model_name} saved to: {table_path}")

def aggregate_oof_predictions(test_indices, y_test, y_pred):
    """
    Aggregate repeated out-of-fold predictions by sample index, averaging repeated entries.
    Return the unique indices and aggregated actual and predicted values.
    """
    oof_df = pd.DataFrame({
        'idx': np.asarray(test_indices, dtype=int),
        'actual': np.asarray(y_test, dtype=float),
        'predicted': np.asarray(y_pred, dtype=float)
    })
    agg_df = (
        oof_df.groupby('idx', as_index=False, sort=True)
        .agg(actual=('actual', 'mean'), predicted=('predicted', 'mean'))
    )
    return (
        agg_df['idx'].to_numpy(dtype=int),
        agg_df['actual'].to_numpy(dtype=float),
        agg_df['predicted'].to_numpy(dtype=float)
    )


def extract_prediction_payload(pred_data):
    """
    Accept prediction results in both the legacy tuple format and the current dictionary format.
    """
    if isinstance(pred_data, dict):
        return {
            "test_actual": np.asarray(pred_data.get("test_actual", []), dtype=float),
            "test_pred": np.asarray(pred_data.get("test_pred", []), dtype=float),
            "train_actual": np.asarray(pred_data.get("train_actual", []), dtype=float),
            "train_pred": np.asarray(pred_data.get("train_pred", []), dtype=float),
            "r2": float(pred_data.get("r2", np.nan)),
            "mae": float(pred_data.get("mae", np.nan)),
            "r2_std": pred_data.get("r2_std", None),
            "train_fraction": pred_data.get("train_fraction", None),
            "test_fraction": pred_data.get("test_fraction", None),
        }

    y_test, y_pred, r2, mae = pred_data[:4]
    return {
        "test_actual": np.asarray(y_test, dtype=float),
        "test_pred": np.asarray(y_pred, dtype=float),
        "train_actual": np.array([], dtype=float),
        "train_pred": np.array([], dtype=float),
        "r2": float(r2),
        "mae": float(mae),
        "r2_std": pred_data[4] if len(pred_data) >= 5 else None,
        "train_fraction": None,
        "test_fraction": None,
    }


def default_feature_config(feature_set):
    if feature_set == 'after':
        return selected_features_after, 'selected_features_after'
    return selected_features, 'selected_features'


def build_runtime_feature_config(feature_set, fold_feature_config):
    default_selected_features, config_name = default_feature_config(feature_set)
    fold_feature_map = load_fold_feature_map(fold_feature_config) if str(fold_feature_config).strip() else {}

    if len(fold_feature_map) > 0:
        selected_features_runtime = []
        for _, feats in fold_feature_map.items():
            if isinstance(feats, list):
                selected_features_runtime.extend(feats)
        selected_features_runtime = list(dict.fromkeys(selected_features_runtime))
        if len(selected_features_runtime) == 0:
            selected_features_runtime = default_selected_features
    else:
        selected_features_runtime = default_selected_features

    return selected_features_runtime, fold_feature_map, config_name


def compute_mapes_by_model(results, all_predictions):
    mapes = {}
    for result in results:
        name = result[0]
        if name not in all_predictions:
            mapes[name] = np.nan
            continue
        payload = extract_prediction_payload(all_predictions[name])
        mapes[name] = calculate_mape(payload["test_actual"], payload["test_pred"])
    return mapes


def print_detailed_summary(results, mapes_by_name):
    print("\n" + "="*80)
    print("Detailed model performance summary (including MAPE)")
    print("="*80)
    print(f"{'Model':<20} {'R2 (median, IQR)':<26} {'MAE (median, IQR)':<26} {'MAPE (%)':<15}")
    print("-"*80)
    for result in results:
        name, r2_median, mae_median, r2_iqr, mae_iqr = result
        mape = mapes_by_name.get(name, np.nan)
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
        print(f"{name:<20} {r2_median:.4f} (IQR:{r2_iqr:.4f})   {mae_median:.4f} (IQR:{mae_iqr:.4f})   {mape_str:<15}")
    print("="*80)


def format_change_value(value, digits=4, suffix=""):
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:+.{digits}f}{suffix}"


def average_metric(metrics_by_model, models, metric_key):
    values = [metrics_by_model.get(model, {}).get(metric_key, np.nan) for model in models]
    values = [float(v) for v in values if not pd.isna(v)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def print_before_after_changes(before_summary, after_summary, output_dir):
    before_metrics = before_summary.get("metrics", {})
    after_metrics = after_summary.get("metrics", {})
    common_models = [name for name in before_metrics if name in after_metrics]
    if not common_models:
        print("\nNo models are shared by the before and after runs. No change table was generated.")
        return

    metric_specs = [
        ("r2_mean", "R2_mean"),
        ("r2_std", "R2_std"),
        ("r2_median", "R2_median"),
        ("r2_iqr", "R2_IQR"),
        ("r2_oof", "R2_OOF"),
        ("mae_mean", "MAE_mean"),
        ("mae_std", "MAE_std"),
        ("mae_median", "MAE_median"),
        ("mae_iqr", "MAE_IQR"),
        ("mae_oof", "MAE_OOF"),
        ("mape", "MAPE"),
    ]

    rows = []
    for scope in common_models + ["ALL_AVERAGE"]:
        row = {"model": scope}
        for metric_key, label in metric_specs:
            if scope == "ALL_AVERAGE":
                before_value = average_metric(before_metrics, common_models, metric_key)
                after_value = average_metric(after_metrics, common_models, metric_key)
            else:
                before_value = before_metrics.get(scope, {}).get(metric_key, np.nan)
                after_value = after_metrics.get(scope, {}).get(metric_key, np.nan)
            delta = after_value - before_value if not (pd.isna(before_value) or pd.isna(after_value)) else np.nan
            pct_change = safe_metric_pct_change(before_value, delta)
            row[f"{label}_before"] = before_value
            row[f"{label}_after"] = after_value
            row[f"{label}_delta"] = delta
            row[f"{label}_delta_pct"] = pct_change
        rows.append(row)

    output_path = Path(output_dir) / "before_after_metric_changes.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print("\n" + "="*110)
    print("AFTER vs. BEFORE (Δ = after - before). Negative MAE or MAPE changes indicate lower error.")
    print("="*110)
    print(
        f"{'Model':<16} {'ΔR2_mean':>11} {'ΔR2%':>9} {'ΔMAE_mean':>12} {'ΔMAE%':>9} "
        f"{'ΔR2_median':>12} {'ΔMAE_median':>13} {'ΔR2_OOF':>10} {'ΔMAE_OOF':>11} {'ΔMAPE':>9}"
    )
    print("-"*110)
    for row in rows:
        print(
            f"{row['model']:<16} "
            f"{format_change_value(row['R2_mean_delta']):>11} "
            f"{format_change_value(row['R2_mean_delta_pct'], digits=2, suffix='%'):>9} "
            f"{format_change_value(row['MAE_mean_delta']):>12} "
            f"{format_change_value(row['MAE_mean_delta_pct'], digits=2, suffix='%'):>9} "
            f"{format_change_value(row['R2_median_delta']):>12} "
            f"{format_change_value(row['MAE_median_delta']):>13} "
            f"{format_change_value(row['R2_OOF_delta']):>10} "
            f"{format_change_value(row['MAE_OOF_delta']):>11} "
            f"{format_change_value(row['MAPE_delta'], digits=2):>9}"
        )
    print("="*110)
    print(f"Detailed before/after metric changes saved to: {output_path}")


def plot_model_comparison(results, all_predictions, mapes_by_name, feature_set, output_dir, filename_prefix):
    if len(results) <= 1:
        return

    names = [r[0] for r in results]
    r2_scores = [r[1] for r in results]
    maes = [r[2] for r in results]
    r2_iqrs = [r[3] for r in results]
    mae_iqrs = [r[4] for r in results]
    mapes = [mapes_by_name.get(name, np.nan) for name in names]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.bar(names, r2_scores, yerr=r2_iqrs, capsize=5, color='steelblue', alpha=0.7, edgecolor='black', label='Median ± IQR')
    ax1.set_ylabel('$R^2$ Score', fontsize=12)
    ax1.set_title(f'{feature_set.upper()} Model $R^2$ Comparison (with IQR)', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, min(1.0, max(r2_scores) + max(r2_iqrs)) * 1.1])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(r2_scores):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2.bar(names, maes, yerr=mae_iqrs, capsize=5, color='coral', alpha=0.7, edgecolor='black', label='Median ± IQR')
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title(f'{feature_set.upper()} Model MAE Comparison (with IQR)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(maes):
        ax2.text(i, v + 0.05, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    valid_mapes = [m for m in mapes if not np.isnan(m)]
    if len(valid_mapes) > 0:
        ax3.bar(names, mapes, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('MAPE (%)', fontsize=12)
        ax3.set_title(f'{feature_set.upper()} Model MAPE Comparison', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        for i, v in enumerate(mapes):
            if not np.isnan(v):
                ax3.text(i, v + max(valid_mapes) * 0.02, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'MAPE is undefined\n(all actual values are zero)', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'{feature_set.upper()} Model MAPE Comparison', fontsize=13, fontweight='bold')

    plt.tight_layout()
    comparison_path = Path(output_dir) / prefixed_filename(filename_prefix, 'model_comparison.png')
    save_figure_with_tif(plt.gcf(), comparison_path, dpi=300, bbox_inches='tight')
    print(f"\nModel comparison saved to: {comparison_path}")
    plt.close()


def make_extra_outputs(feature_set, args, models_to_train, X, y, groups, use_groups, output_dir, filename_prefix):
    if args.shap:
        print(f"\nGenerating {feature_set.upper()} SHAP summary plot...")
        try:
            shap_model = args.shap_model
            pipeline = build_pipeline(shap_model, random_seed=42)
            fit_pipeline_with_optional_weights(shap_model, pipeline, X, y)
            save_shap_summary_plot(
                output_dir,
                shap_model,
                pipeline,
                X,
                sample_size=args.shap_sample,
                max_display=args.shap_top,
                filename_prefix=filename_prefix
            )
        except Exception as e:
            print(f"  Could not generate the SHAP plot: {e}")

    if args.release_figure and feature_set == "after":
        print(f"\nGenerating {feature_set.upper()} XGBoost landscape–SHAP release figure...")
        try:
            prediction_table_path = Path(output_dir) / prefixed_filename(filename_prefix, "xgboost_predictions_table.csv")
            shap_summary_path = Path(output_dir) / prefixed_filename(filename_prefix, "xgboost_shap_summary.png")
            plot_landscape_shap_release_figure(
                prediction_table_path,
                shap_summary_path,
                output_dir,
                filename_prefix=filename_prefix,
                alias_filename="figure6_landscape_shap_release.png",
            )
        except Exception as e:
            print(f"  Could not generate the landscape–SHAP release figure: {e}")

    print(f"\nGenerating {feature_set.upper()} structure–property landscape plot...")
    try:
        plot_structure_property_landscape(
            X,
            y,
            output_dir,
            filename_prefix=filename_prefix,
            title_prefix=feature_set,
        )
    except Exception as e:
        print(f"  Could not generate the structure–property landscape plot: {e}")

    print("\n" + "="*80)
    print(f"Generating {feature_set.upper()} feature importance comparison across all models...")
    print("="*80)
    try:
        plot_all_models_feature_importance(
            X,
            y,
            groups,
            models_to_train,
            str(output_dir),
            use_groups,
            filename_prefix=filename_prefix
        )
    except Exception as e:
        print(f"  Could not generate the feature importance plot: {e}\n")

    if args.save_models:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        print("\n" + "="*80)
        print(f"Saving {feature_set.upper()} models...")
        for model_name in models_to_train:
            try:
                pipeline = build_pipeline(model_name, random_seed=42)
                fit_pipeline_with_optional_weights(model_name, pipeline, X, y)
                save_path = models_dir / f"{feature_set}_{model_name}.joblib"
                joblib.dump(pipeline, save_path)
                print(f"  {feature_set.upper()} / {model_name.upper()} saved to: {save_path}")
            except Exception as e:
                print(f"  Could not save {feature_set.upper()} / {model_name}: {e}")
        print("="*80)


def run_feature_set(feature_set, args, models_to_train, json_file, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = feature_set

    print("\n" + "="*80)
    print(f"Running feature set: {feature_set.upper()}")
    print("="*80)
    print("Loading data...")
    print(f"  Swap augmentation: {args.use_swap}")
    print(f"  d33 threshold: {args.threshold}")
    print(f"  Feature set: {feature_set}")

    selected_features_runtime, fold_feature_map, config_name = build_runtime_feature_config(
        feature_set,
        args.fold_feature_config
    )
    df, composition_map, groups = load_and_clean(
        json_file,
        max_threshold=args.threshold,
        min_threshold=3,
        use_swap=args.use_swap,
        selected_features=selected_features_runtime
    )
    print(f"  Loaded {len(df)} samples")
    print(f"  Number of features: {len(df.columns) - 1}")

    folds_meta, predefined_folds = load_predefined_folds(args.folds_file)
    print(f"  Using predefined folds: {args.folds_file}")
    print(f"  Fold metadata: n_splits={folds_meta.get('n_splits', len(predefined_folds))}, random_state={folds_meta.get('random_state', 'NA')}")
    if len(fold_feature_map) > 0:
        if str(args.fold_feature_config).strip():
            print(f"  Loaded per-fold feature lists from JSON: {args.fold_feature_config}")
    else:
        print(f"  Using the shared feature list from config.py: {config_name}")

    X = df.drop(columns=['target'])
    y = df['target']
    results = []
    all_predictions = {}
    stats_summary = {}
    use_groups = args.use_swap and groups is not None

    for model_name in models_to_train:
        print(f"Training {feature_set.upper()} / {model_name.upper()}...")
        try:
            cv_result = train_model_with_predefined_folds(
                model_name,
                X,
                y,
                predefined_folds,
                composition_map,
                fold_feature_map=fold_feature_map
            )
            (
                r2_mean, r2_std, r2_median, r2_iqr,
                mae_mean, mae_std, mae_median, mae_iqr,
                all_y_train, all_y_train_pred, train_indices_cv,
                all_y_test, all_y_pred, test_indices_cv, _
            ) = cv_result

            agg_idx, agg_y_test, agg_y_pred = aggregate_oof_predictions(test_indices_cv, all_y_test, all_y_pred)
            agg_train_idx, agg_y_train, agg_y_train_pred = aggregate_oof_predictions(train_indices_cv, all_y_train, all_y_train_pred)
            r2_oof = r2_score(agg_y_test, agg_y_pred)
            mae_oof = mean_absolute_error(agg_y_test, agg_y_pred)
            total_cv_points = len(all_y_train) + len(all_y_test)
            train_fraction = (len(all_y_train) / total_cv_points) if total_cv_points > 0 else None
            test_fraction = (len(all_y_test) / total_cv_points) if total_cv_points > 0 else None
            name = model_name.upper()

            results.append((name, r2_median, mae_median, r2_iqr, mae_iqr))
            all_predictions[name] = {
                "test_actual": agg_y_test,
                "test_pred": agg_y_pred,
                "train_actual": agg_y_train,
                "train_pred": agg_y_train_pred,
                "r2": r2_oof,
                "mae": mae_oof,
                "r2_std": r2_std,
                "mae_std": mae_std,
                "r2_oof": r2_oof,
                "mae_oof": mae_oof,
                "train_fraction": train_fraction,
                "test_fraction": test_fraction,
                "test_indices": agg_idx,
                "train_indices": agg_train_idx,
            }
            stats_summary[name] = {
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "r2_median": r2_median,
                "r2_iqr": r2_iqr,
                "r2_oof": r2_oof,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mae_median": mae_median,
                "mae_iqr": mae_iqr,
                "mae_oof": mae_oof,
            }

            save_prediction_table(
                output_dir,
                name,
                agg_idx,
                agg_y_test,
                agg_y_pred,
                composition_map,
                X=X,
                filename_prefix=filename_prefix
            )
            plot_predictions(
                agg_y_test,
                agg_y_pred,
                name,
                r2_oof,
                mae_oof,
                output_dir,
                r2_std=r2_std,
                filename_prefix=filename_prefix
            )
            print(
                f"  {feature_set.upper()} / {name} - R2: {r2_mean:.4f} ± {r2_std:.4f} (IQR: {r2_iqr:.4f}), "
                f"MAE: {mae_mean:.4f} ± {mae_std:.4f} (IQR: {mae_iqr:.4f}), "
                f"OOF R2: {r2_oof:.4f}, OOF MAE: {mae_oof:.4f}"
            )
            print("")
        except Exception as e:
            print(f"  Error training {feature_set.upper()} / {model_name}: {e}\n")

    print("\n" + "="*80)
    print(f"{feature_set.upper()} model performance summary (predefined 5-fold CV)")
    print("="*80)
    print(f"{'Model':<20} {'R2 (mean±std)':<22} {'R2 (median, IQR)':<24} {'MAE (mean±std)':<22} {'MAE (median, IQR)':<24}")
    print("-"*80)
    for result in results:
        name, r2_median, mae_median, r2_iqr, mae_iqr = result
        s = stats_summary.get(name, {})
        print(
            f"{name:<20} "
            f"{s.get('r2_mean', np.nan):.4f}±{s.get('r2_std', np.nan):.4f}    "
            f"{r2_median:.4f} (IQR:{r2_iqr:.4f})   "
            f"{s.get('mae_mean', np.nan):.4f}±{s.get('mae_std', np.nan):.4f}    "
            f"{mae_median:.4f} (IQR:{mae_iqr:.4f})"
        )
    print("="*80)

    if len(all_predictions) > 0:
        print(f"\nGenerating {feature_set.upper()} prediction comparisons for all models (predefined 5-fold out-of-fold results)...")
        plot_all_models_comparison(all_predictions, str(output_dir), filename_prefix=filename_prefix)

    mapes_by_name = compute_mapes_by_model(results, all_predictions)
    for name, mape in mapes_by_name.items():
        stats_summary.setdefault(name, {})["mape"] = mape

    plot_model_comparison(results, all_predictions, mapes_by_name, feature_set, output_dir, filename_prefix)
    make_extra_outputs(feature_set, args, models_to_train, X, y, groups, use_groups, output_dir, filename_prefix)
    print_detailed_summary(results, mapes_by_name)

    return {
        "feature_set": feature_set,
        "metrics": stats_summary,
        "results": results,
        "all_predictions": all_predictions,
        "mapes": mapes_by_name,
    }


def main(output_dir='data'):
    parser = argparse.ArgumentParser(description='Unified machine learning training script')
    parser.add_argument('--no-swap', dest='use_swap', action='store_false',
                       help='Disable swap augmentation')
    parser.add_argument('--threshold', type=float, default=30,
                       help='Upper d33 cutoff (default: 30, lower cutoff fixed at 3)')
    parser.add_argument('--models', nargs='+',
                       choices=['ridge', 'rf', 'svr', 'gpr', 'xgboost', 'gbdt', 'adaboost', 'rfr', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    parser.add_argument('--folds-file', type=str, default='data/cv5_folds.json',
                       help='Path to the predefined 5-fold split file (default: data/cv5_folds.json)')
    parser.add_argument('--fold-feature-config', type=str, default='',
                       help='Optional per-fold feature lists in JSON (defaults to the shared list in config.py)')
    parser.add_argument('--feature-set', type=str, default='both', choices=['before', 'after', 'both'],
                       help='Retained for compatibility. Both feature sets run every time.')
    parser.add_argument('--shap', action='store_true',
                       default=True,
                       help='Generate a SHAP summary plot (enabled by default)')
    parser.add_argument('--shap-model', type=str, default='xgboost',
                       choices=['rf', 'xgboost'],
                       help='Model for the SHAP plot (default: xgboost)')
    parser.add_argument('--shap-sample', type=int, default=200,
                       help='Number of samples for SHAP (default: 200)')
    parser.add_argument('--shap-top', type=int, default=8,
                       help='Number of features shown in the SHAP plot (default: 8)')
    parser.add_argument('--save-models', dest='save_models', action='store_true',
                       default=False,
                       help='Save trained models to models/ (disabled by default)')
    parser.add_argument('--no-release-figure', dest='release_figure', action='store_false',
                       default=True,
                       help='Skip the XGBoost landscape–SHAP release figure for the after feature set')

    args = parser.parse_args()

    # Select the models to train.
    if 'all' in args.models:
        models_to_train = [
            'ridge',
            'rf',
            'svr',
            # 'gbdt',
            'xgboost',
            'adaboost',
            # 'rfr',
            ]
    else:
        models_to_train = args.models

    json_file = output_dir + '/results_preprocessed.json'

    if not Path(json_file).exists():
        print(f"Error: File {json_file} does not exist")
        return

    if args.feature_set != 'both':
        print(f"--feature-set={args.feature_set} is retained for compatibility. Both the before and after feature sets will run.")

    summaries = {}
    for feature_set in ("before", "after"):
        summaries[feature_set] = run_feature_set(
            feature_set,
            args,
            models_to_train,
            json_file,
            output_dir
        )
    print_before_after_changes(summaries["before"], summaries["after"], output_dir)
    return


if __name__ == '__main__':
    main()

