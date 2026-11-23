"""
Definitions of candidate features for feature selection.

by flyingrtx 2026-03-03
"""
import json
import pandas as pd
import numpy as np
from loguru import logger

CA_DIFF_THRESHOLD = 1.24 # Threshold for a complete phase transition
PEAK2_RATIO_DIFF_THRESHOLD = 1.78 # Threshold for the ratio of the second peak position to the first

def get_structure_feature(val, name, default=0.0):
    if val.get(name) is not None:
        return val.get(name)
    structure_features = val.get("structure_features", {}) or {}
    value = structure_features.get(name, default)
    return default if value is None else value

def gen_row(key,val):
    rdf_analysis = val.get("rdf_analysis", {}) or {}
    a_amount_norm = val.get("A_amount") / 16
    ca_diff_sq_value = (val.get("c_over_a") - CA_DIFF_THRESHOLD) ** 2
    pysr_peak2_logA_logca2 = rdf_analysis.get("Peak2_Pos") * (
        np.log(max(a_amount_norm, 1e-6))
        + np.log(max(ca_diff_sq_value, 1e-6)) ** 2
    )
    pysr_ben18_ca = (val.get("B_electronegativity_diff", 0) - 1.8) * ca_diff_sq_value
    pysr_peak2_ben18_sq = (pysr_peak2_logA_logca2 * pysr_ben18_ca) ** 2
    row = {
                "name": key, # Structure name
                "volume": val.get("volume_ang3"),
                "A_amount":val.get("A_amount") / 16, # Number of atoms of element A divided by 16
                "B_amount":val.get("B_amount") / 16, # Number of atoms of element B divided by 16
                "u": val.get("rdf_analysis", {}).get("u"), # u
                "u_minus_0_375_abs": val.get("u_minus_0_375_abs"), # |u-0.375|
                "u_minus_0_19_abs": abs(0.19-val.get("rdf_analysis", {}).get("u")), # |u-0.19|
                "u_minus_05_abs": abs(0.5-val.get("rdf_analysis", {}).get("u")), # |u-0.5|
                "total_mass": val.get("total_mass"), # Total mass
                "density": val.get("density"), # Density
                "log_density": np.log1p(val.get("density")), # log(density)
                "log_density_diff_ploar_factor":(np.log1p(val.get("density")) + val.get("polar_factor") - 2.06) ** 4,
                "u_ca_overlap": val.get("rdf_analysis", {}).get("u") * ((val.get("c_over_a") - 1) ** 4),
                "peak1_pos": val.get("rdf_analysis", {}).get("Peak1_Pos"),
                "peak2_pos": val.get("rdf_analysis", {}).get("Peak2_Pos"),
                "double_peak2_pos_minus_89": 8.9 - (val.get("rdf_analysis", {}).get("Peak2_Pos") * 2),
                "peak2_ratio": val.get("rdf_analysis", {}).get("Peak2_Pos_Peak1_Pos"),
                "peak2_ratio_diff": val.get("rdf_analysis", {}).get("Peak2_Pos_Peak1_Pos") - PEAK2_RATIO_DIFF_THRESHOLD,
                "exp_peak2_ratio_diff_div_u":(np.exp(val.get("rdf_analysis", {}).get("Peak2_Pos_Peak1_Pos") - PEAK2_RATIO_DIFF_THRESHOLD) / val.get("rdf_analysis", {}).get("u")),
                "peak2_height": val.get("rdf_analysis", {}).get("Peak2_Height"),
                "peak2_fwhm": val.get("rdf_analysis", {}).get("Peak2_FWHM"),
                # "A_electronegativity": val.get("A_electronegativity"),
                # "B_electronegativity": val.get("B_electronegativity"),
                "A_radius_diff": val.get("A_radius_diff"),
                "B_radius_diff": val.get("B_radius_diff",0),
                "A_en_diff": val.get("A_electronegativity_diff",0),
                "B_en_diff": val.get("B_electronegativity_diff",0),
                "B_en_diff_minus_18":val.get("B_electronegativity_diff",0)-1.8,
                "A_valence_electrons_diff": val.get("A_valence_electrons_diff") or 0,
                "B_valence_electrons_diff": val.get("B_valence_electrons_diff") or 0,
                'A_ionic_radius_diff': np.abs(val.get("A_ionic_radius_diff",0)),
                'B_ionic_radius_diff': np.abs(val.get("B_ionic_radius_diff",0)),
                "B_en_diff_A_en_diff": val.get("B_electronegativity_diff") / val.get("A_electronegativity_diff"),
                "energy_per_16": val.get("energy_per_16"), # Energy from qvasp -e divided by 16
                "energy_per_16_sq": val.get("energy_per_16") ** 2, # (Energy from qvasp -e divided by 16) ** 2
                "desq": val.get("distortion") * (val.get("energy_per_16") ** 2), # distortion * (energy from qvasp -e divided by 16) ** 2
                "sharpness": val.get("sharpness"),  # Peak1_Height / Peak1_FWHM
                "Reciprocal_sharpness": 1 / val.get("sharpness"), # 1 / (Peak1_Height / Peak1_FWHM)
                "distortion": val.get("distortion"), # Tetrahedral distortion index: distortion = Peak1_Pos / Volume^(1/3)
                "goldschmidt_tolerance_factor": val.get("goldschmidt_tolerance_factor"), # Goldschmidt tolerance factor: t = (R_A + R_O) / (√2 * (R_B + R_O))
                "octahedral_factor": val.get("octahedral_factor") or 0, # Octahedral factor: R_B / R_O
                "c_over_a": val.get("c_over_a"),  # c/a
                "ca_diff": val.get("c_over_a") - CA_DIFF_THRESHOLD,  # c/a - 1.25
                "ca_diff_dis":val.get("distortion") / (val.get("c_over_a") - CA_DIFF_THRESHOLD),  # distortion / (c/a - 1.25)
                "ca_diff_sq": (val.get("c_over_a") - CA_DIFF_THRESHOLD) ** 2,  # (c/a - 1.25) ** 2
                "ca_diff_sq_minus_0124": (val.get("c_over_a") - CA_DIFF_THRESHOLD) ** 2 - 0.124,
                "ca_diff_sq4": (val.get("c_over_a") - CA_DIFF_THRESHOLD) ** 4,  # (c/a - 1.25) ** 4
                "log_ca_diff_sq4": np.log1p((val.get("c_over_a") - CA_DIFF_THRESHOLD) ** 4), # log( (c/a - 1.25) ** 4)
                "polar_factor": val.get("polar_factor"), # c_over_a * (A_electronegativity_diff + B_electronegativity_diff)
                "polar_factor2": (val.get("A_electronegativity_diff", "0") + val.get("B_electronegativity_diff", "0")) * (val.get("c_over_a") - 1),
                "polar_factor3": (val.get("A_electronegativity_diff", "0") + val.get("B_electronegativity_diff", "0")) * val.get("rdf_analysis", {}).get("u"),
                "polar_factor_sq": val.get("polar_factor") ** 2,  # polar_factor ** 2
                "polar_factor_sq_sq": val.get("polar_factor") ** 4, # polar_factor ** 4
                "distortion_per_vol": val.get("distortion_per_vol"), # distortion / volume_ang3
                "u_distort_inter": val.get("u_distort_inter"), # u * distortion
                # "pack_density": pack_density,
                # "log_pack_density":np.log1p(pack_density),
                # Repeated runs often found pack_density - 29 in the denominator, so define it as a feature.
                # "pack_density_diff":pack_density_diff,
                # "log_pack_density_diff": np.log1p(pack_density_diff),
                # "unknown_1": val.get("polar_factor") / pack_density,
                # Repeated runs consistently found the important expression pack_density_diff / (square(polar_factor) - ca_sq_2).
                # "unknown_2": unknown_2,
                # pack_density_diff + unknown2 may be useful.
                # "unknown_3": pack_density_diff + (pack_density_diff / ((val.get("polar_factor") ** 2) - (val.get("c_over_a") - 1) ** 4)),
                # ca_sq_2*log(log_pack_density_diff) + exp(unknown_2)
                # "unknown_4": (val.get("c_over_a") - 1) ** 2 * np.log1p(pack_density_diff) + exp(unknown_2),
                # unknown_2 / log(ca_sq_2 / pack_density_diff)
                # "unknown_5": unknown_2 / np.log1p((val.get("c_over_a") - 1) ** 4 / pack_density_diff),
                "feat1":val.get("polar_factor") / (val.get("c_over_a") + 1e-6),
                "feat2":np.log1p(val.get("density")) * (1/val.get("sharpness")),
                "feat11":val.get("density")*val.get("c_over_a"),
                "feat12":0.63-(1/val.get("c_over_a")),
                "rdf_n_peaks": get_structure_feature(val, "rdf_n_peaks"),
                "rdf_peak_separation": get_structure_feature(val, "rdf_peak_separation"),
                "rdf_height_ratio_21": get_structure_feature(val, "rdf_height_ratio_21"),
                "rdf_width_ratio_21": get_structure_feature(val, "rdf_width_ratio_21"),
                "rdf_area_ratio_21": get_structure_feature(val, "rdf_area_ratio_21"),
                "rdf_peak1_width_norm": get_structure_feature(val, "rdf_peak1_width_norm"),
                "rdf_peak2_width_norm": get_structure_feature(val, "rdf_peak2_width_norm"),
                "rdf_peak2_sharpness": get_structure_feature(val, "rdf_peak2_sharpness"),
                "rdf_sharpness_ratio_21": get_structure_feature(val, "rdf_sharpness_ratio_21"),
                "rdf_local_order": get_structure_feature(val, "rdf_local_order"),
                "rdf_disorder": get_structure_feature(val, "rdf_disorder"),
                "rdf_gap_over_width": get_structure_feature(val, "rdf_gap_over_width"),
                "volume_per_atom": get_structure_feature(val, "volume_per_atom"),
                "tetragonal_strain": get_structure_feature(val, "tetragonal_strain"),
                "cation_n_bond_mean": get_structure_feature(val, "cation_n_bond_mean"),
                "cation_n_bond_std": get_structure_feature(val, "cation_n_bond_std"),
                "cation_n_bond_range": get_structure_feature(val, "cation_n_bond_range"),
                "cation_n_bond_cv": get_structure_feature(val, "cation_n_bond_cv"),
                "cation_n_coord_mean": get_structure_feature(val, "cation_n_coord_mean"),
                "cation_n_coord_std": get_structure_feature(val, "cation_n_coord_std"),
                "tetra_angle_dev_mean": get_structure_feature(val, "tetra_angle_dev_mean"),
                "tetra_angle_dev_max": get_structure_feature(val, "tetra_angle_dev_max"),
                "tetra_angle_std": get_structure_feature(val, "tetra_angle_std"),
                "cation_offcenter_mean": get_structure_feature(val, "cation_offcenter_mean"),
                "cation_offcenter_max": get_structure_feature(val, "cation_offcenter_max"),
                "dopant_n_bond_delta": get_structure_feature(val, "dopant_n_bond_delta"),
                "dopant_n_bond_std": get_structure_feature(val, "dopant_n_bond_std"),
                "pysr_peak2_logA_logca2": pysr_peak2_logA_logca2,
                "pysr_Ben18_ca": pysr_ben18_ca,
                "pysr_peak2_ben18_sq": pysr_peak2_ben18_sq,
                "pysr_Aen_Ben_ca": val.get("A_electronegativity_diff", 0)
                * val.get("B_electronegativity_diff", 0)
                * val.get("c_over_a"),
                "pysr_density_A_radius": val.get("density") * val.get("A_radius_diff"),
                "pysr_B_radius_B": val.get("B_radius_diff", 0) * (val.get("B_amount") / 16),
                "pysr_u_peak2_over_sharp": (
                    val.get("rdf_analysis", {}).get("u")
                    * val.get("rdf_analysis", {}).get("Peak2_Pos")
                    / max(val.get("sharpness"), 1e-6)
                ),
                "pysr_density_Ben": val.get("density") * val.get("B_electronegativity_diff", 0),
                "pysr_density_AB": val.get("density") * (val.get("A_amount") / 16) * (val.get("B_amount") / 16),
                "pysr_peak1_minus_ca": val.get("rdf_analysis", {}).get("Peak1_Pos") - val.get("c_over_a"),
                "pysr_Ben_over_ca_minus_Ben": val.get("B_electronegativity_diff", 0)
                / max(abs(val.get("c_over_a") - val.get("B_electronegativity_diff", 0)), 1e-6),
                "pysr_log_peak1_minus_ca_abs": np.log1p(
                    abs(val.get("rdf_analysis", {}).get("Peak1_Pos") - val.get("c_over_a"))
                ),
                "pysr_peak2_over_sharp": val.get("rdf_analysis", {}).get("Peak2_Pos")
                / max(val.get("sharpness"), 1e-6),
                "target": val["d33"]
            }

    return row

def load_and_clean(json_file, max_threshold=30, min_threshold=3, use_swap=False):
    logger.info(f"Loading and cleaning data from {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    groups = []
    idx = 0
    skip_count = 0
    for key, val in data.items():
        try:
            # logger.info(f"Processing structure {key}")
            if not use_swap and key.endswith('_swap'): 
                # skip_count += 1
                # logger.warning(f"Skipping swap-augmented structure {key} (total skipped: {skip_count})")
                continue
            d33_val = val["d33"]
            if d33_val is None or d33_val > max_threshold or d33_val < min_threshold: 
                skip_count += 1
                logger.warning(f"Skipping structure {key} with d33 = {d33_val} (total skipped: {skip_count})")
                continue
            
            rows.append(gen_row(key,val))
            groups.append(key.replace('_swap', ''))
            idx += 1
        except Exception as e:
            logger.error(f"Error processing structure {key}: {e}")
            continue
    df = pd.DataFrame(rows)
    # Count missing values for each feature.
    nan_counts = df.isna().sum()
    logger.info(f"Missing values by feature:\n{nan_counts[nan_counts > 0]}")
    logger.info(f"Structures retained before dropping missing values: {len(rows)}")
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    logger.info(f"Data cleaning complete. Original entries: {len(data)}, retained entries: {len(df)}")
    return df, groups

selected_features = [
    # # Structural distortion features
    "c_over_a",
    # # Statistical features
    "density",
    "sharpness",
    "A_amount",
    "B_amount",
    "A_en_diff",
    "B_en_diff",
    # "B_en_diff_A_en_diff",
    "A_radius_diff",
    "B_radius_diff",
    # "A_ionic_radius_diff",
    # "B_ionic_radius_diff",
    "energy_per_16",
    "u",                # Key structural parameter

    
    "peak1_pos",
    "peak2_pos",
    "cation_n_bond_mean",
    "cation_n_bond_std",
    "cation_n_bond_range",
    "cation_n_bond_cv",
    "cation_n_coord_mean",
    "cation_n_coord_std",
    # "double_peak2_pos_minus_89",
    # "peak2_ratio_diff", # Long-range and short-range structural order
    # "goldschmidt_tolerance_factor", # Stability criterion
    # "Reciprocal_sharpness" # Reciprocal measure of local structural order
    # "rdf_peak1_width_norm",
    # "rdf_peak2_width_norm",
    # "rdf_peak2_sharpness",
    # "rdf_sharpness_ratio_21",
    # "rdf_local_order",
    # "rdf_disorder"
]


selected_features_after = [
    # # Structural distortion features
    # "c_over_a",
    # # Statistical features
    "density",
    "sharpness",
    "A_amount",
    "B_amount",
    "A_en_diff",
    "B_en_diff",
    # "B_en_diff_A_en_diff",
    "A_radius_diff",
    "B_radius_diff",
    # "A_ionic_radius_diff",
    # "B_ionic_radius_diff",
    "energy_per_16",
    "u",                # Key structural parameter

    
    "peak1_pos",
    "peak2_pos",

    # Derived replacements / compact nonlinear terms.
    # "ca_diff_sq",
    "ca_diff_sq4",
    "log_density",
    "B_en_diff_minus_18",
    "feat11",
    "polar_factor",
    "u_minus_0_375_abs",
    "distortion",
    "cation_n_bond_std",
    "tetra_angle_dev_mean",
    "dopant_n_bond_delta",
    "Reciprocal_sharpness",
    # PySR candidate feature. It slightly improved CV metrics but worsened external top-30 rankings, so it is excluded for now.
    # "pysr_Ben18_ca",
    # "pysr_peak2_ben18_sq",
    # "pysr_u_peak2_over_sharp",
    # "pysr_density_Ben",
]

if __name__ == "__main__":
    json_file = "data/results_preprocessed.json"
    df, groups = load_and_clean(json_file)
    # print(df)
    # print(groups)
