#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess the merged raw DFT data in JSON format.
Remove selected fields and add differences in atomic radius, electronegativity, valence electron count, and ionic radius.
Add the Goldschmidt tolerance factor (t) and the octahedral factor (μ).
- Goldschmidt tolerance factor: t = (R_A + R_O) / (√2 * (R_B + R_O))
- Octahedral factor: μ = R_B / R_O
Add other derived features:
- polar_factor: c_over_a * (A_electronegativity_diff + B_electronegativity_diff)
- distortion_per_vol: distortion / volume_ang3
- u_distort_inter: u * distortion
"""

import json
import traceback
import sys
import math
from pathlib import Path
from loguru import logger
from typing import Optional
from sklearn.model_selection import KFold, GroupKFold
# Try to import pymatgen
try:
    from pymatgen.core import Element
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    print("Warning: pymatgen is required to calculate atomic radii and electronegativities")
    sys.exit(1)


def get_atomic_radius(element_str):
    """
    Get the atomic radius of an element in Å.
    
    Args:
        element_str: Element symbol, such as 'Mg' or 'Al'.
    
    Returns:
        float: Atomic radius in Å, or None if unavailable.
    """
    if not element_str:
        return None
    
    try:
        element_str = str(element_str).strip().capitalize()
        elem = Element(element_str)
        
        # Try the tabulated atomic radius
        if hasattr(elem, 'atomic_radius'):
            radius = elem.atomic_radius
            if radius is not None:
                return float(radius)
        
        # Try the calculated atomic radius
        if hasattr(elem, 'atomic_radius_calculated'):
            radius = elem.atomic_radius_calculated
            if radius is not None:
                return float(radius)
        
        # Fall back to the van der Waals radius
        if hasattr(elem, 'van_der_waals_radius'):
            radius = elem.van_der_waals_radius
            if radius is not None:
                return float(radius)
        
        return None
    except Exception as e:
        print(f"Warning: Could not get the atomic radius for {element_str}: {e}")
        return None

def get_atmoic_mass(element_str):
    """
    Get the atomic mass of an element.
    
    Args:
        element_str: Element symbol.
    
    Returns:
        float: Atomic mass, or None if unavailable.
    """
    if not element_str:
        return 0
    
    try:
        element_str = str(element_str).strip().capitalize()
        elem = Element(element_str)
        return float(elem.atomic_mass)
    except Exception as e:
        logger.error(f"Warning: Could not get the atomic mass for {element_str}: {e}")
        return 0

def get_electronegativity(element_str):
    """
    Get the Pauling electronegativity of an element.
    
    Args:
        element_str: Element symbol.
    
    Returns:
        float: Pauling electronegativity, or None if unavailable.
    """
    if not element_str:
        return None
    
    try:
        element_str = str(element_str).strip().capitalize()
        elem = Element(element_str)
        if hasattr(elem, 'X'):
            return float(elem.X)
        return None
    except Exception as e:
        print(f"Warning: Could not get the electronegativity for {element_str}: {e}")
        return None


def get_valence_electrons(element_str):
    """
    Estimate the valence electron count for electron counting, charge compensation, and feature engineering.
    
    Apply these rules in order:
    1. For main-group elements, use the modern group number: groups 1–2 use the group number, and groups 13–18 subtract 10.
    2. For common transition metals and d-block elements, use the d + s electron count for the commonly used high oxidation state or a typical group valence.
    3. Fall back to the outermost s + p + d electron count, which may differ from the chemical valence electron count.
    
    Returns:
        int or None.
    """
    if not element_str:
        return None
    
    try:
        elem_str = str(element_str).strip().capitalize()
        elem = Element(elem_str)
        
        # -------------------------------
        # Method 1: Main-group elements in the s and p blocks
        # -------------------------------
        if hasattr(elem, 'group') and elem.group is not None:
            group = elem.group
            if 1 <= group <= 2:
                return group  # Li and Na: 1. Be, Mg, Ca, Sr, and Ba: 2.
            elif 13 <= group <= 18:
                return group - 10  # Al →3, Si→4, P→5, S→6, etc.
        
        # -------------------------------
        # Method 2: Common transition metals and d-block elements, often found on perovskite B sites
        # Use conventional valence electron counts by element
        # -------------------------------
        common_valence_electrons = {
            # Alkaline earth metals often found on A sites
            'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
            # Common B-site elements in perovskites and nitrides
            'Sc': 3, 'Y': 3,
            'Ti': 4, 'Zr': 4, 'Hf': 4,
            'V': 5, 'Nb': 5, 'Ta': 5,
            'Cr': 3,  # Chromium can be +6, but +3 is common in perovskites
            'Mn': 4,  # Common oxidation states include +2, +3, and +4
            'Fe': 3, 'Co': 3, 'Ni': 2,
            # Other possible elements
            'Zn': 2, 'Cu': 1, 'Ag': 1,
            'Al': 3,  # Keep aluminum as a fallback despite its main-group assignment
            'Ga': 3, 'In': 3,
        }
        
        if elem_str in common_valence_electrons:
            return common_valence_electrons[elem_str]
        
        # -------------------------------
        # Method 3: Fall back to the outermost electrons (s + p + d)
        # pymatgen's full_electronic_structure format: [(n, l, occupancy), ...]
        # -------------------------------
        if hasattr(elem, 'full_electronic_structure'):
            es = elem.full_electronic_structure
            if es:
                # Count all electrons at the highest n (s + p + d)
                max_n = max(n for n, l, occ in es)
                valence = sum(occ for n, l, occ in es if n == max_n)
                if valence > 0:
                    return int(valence)
        
        # No rule produced a reliable estimate
        print(f"Warning: Could not determine a reliable valence electron count for {elem_str}")
        return None
        
    except Exception as e:
        print(f"Warning: Could not get the valence electron count for {element_str}: {e}")
        return None

def get_ionic_radius(
    element_str: str,
    valence: Optional[int] = None,
    coordination: int = 6,
    prefer_high_spin: bool = False
) -> Optional[float]:
    """
    Get the Shannon ionic radius of an element in Å.

    Args:
        element_str: Element symbol, such as 'Mg', 'Nb', 'Al', or 'O'.
        valence: Oxidation state. Use positive values for cations and negative values for anions, such as +2, +3, +5, or -2.
                 If None, infer a common oxidation state for the element.
        coordination: Coordination number (commonly 4, 6, 8, or 12). Defaults to 6 for octahedral coordination.
        prefer_high_spin: Prefer high-spin radii for transition metals. Defaults to False (low spin).

    Returns:
        float or None: Ionic radius in Å, or None if unavailable.

    Features:
    - Prefer the Shannon ionic radii tabulated in pymatgen.
    - Fall back across common coordination numbers (6, then 4 or 8, then 12).
    - Use typical oxidation states for common perovskite A- and B-site elements.
    - Handle oxide ions separately, usually using 1.40 Å for six-coordinate O²⁻.
    """
    if not element_str:
        return None

    try:
        elem_str = str(element_str).strip().capitalize()
        elem = Element(elem_str)

        # Handle oxide ions separately, using 1.40 Å for the common six-coordinate O²⁻ ion
        if elem_str == 'O' and valence in (None, -2):
            if coordination in (4, 6):
                return 1.40
            elif coordination == 2:
                return 1.35  # Less common coordination
            else:
                return 1.40  # Default to the most common radius

        # Infer a common oxidation state when none is provided
        if valence is None:
            common_valences = {
                # Common A-site oxidation state (+2)
                'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2, 'Zn': 2,
                # Common B-site oxidation states (+3, +4, +5)
                'Al': 3, 'Sc': 3, 'Y': 3,
                'Ti': 4, 'Zr': 4, 'Hf': 4,
                'V': 5, 'Nb': 5, 'Ta': 5,
                'Cr': 3, 'Fe': 3, 'Mn': 4,
                # Other elements
                'La': 3, 'Ce': 3, 'Nd': 3,
            }
            valence = common_valences.get(elem_str)
            if valence is None:
                # Fall back to the first common oxidation state listed by pymatgen
                if hasattr(elem, 'common_oxidation_states') and elem.common_oxidation_states:
                    valence = elem.common_oxidation_states[0]

        if valence is None:
            print(f"Warning: Could not infer an oxidation state for {elem_str}")
            return None

        # Get the ionic radii by oxidation state
        if not hasattr(elem, 'ionic_radii'):
            print(f"Warning: No ionic_radii data available for {elem_str}")
            return None

        ionic_radii = elem.ionic_radii
        if valence in ionic_radii:
            return ionic_radii[valence]
        else:
            logger.error(f"Warning: No ionic radius found for {elem_str} at oxidation state {valence} and coordination {coordination}")
            return None

        print(f"Warning: No ionic radius found for {elem_str} at oxidation state {valence} and coordination {coordination}")
        return None

    except Exception as e:
        traceback.print_exc()
        print(f"Error: Could not get the ionic radius for {element_str} (oxidation state={valence}, CN={coordination}): {e}")
        return None

def safe_float(value):
    """Return a finite float or None."""
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def safe_div(numerator, denominator):
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def calculate_rdf_lattice_features(record):
    """
    Compute structure proxy descriptors from RDF/lattice fields already present
    in the result JSON. True POSCAR-level descriptors are produced by
    scripts/get_final_results.py on the server and are merged separately.
    """
    rdf = record.get('rdf_analysis', {}) or {}
    peak1_pos = safe_float(rdf.get('Peak1_Pos'))
    peak1_height = safe_float(rdf.get('Peak1_Height'))
    peak1_fwhm = safe_float(rdf.get('Peak1_FWHM'))
    peak2_pos = safe_float(rdf.get('Peak2_Pos'))
    peak2_height = safe_float(rdf.get('Peak2_Height'))
    peak2_fwhm = safe_float(rdf.get('Peak2_FWHM'))
    n_peaks = safe_float(rdf.get('n_peaks'))
    u = safe_float(rdf.get('u'))
    volume = safe_float(record.get('volume_ang3'))
    c_over_a = safe_float(record.get('c_over_a'))

    peak1_area = (
        peak1_height * peak1_fwhm
        if peak1_height is not None and peak1_fwhm is not None else None
    )
    peak2_area = (
        peak2_height * peak2_fwhm
        if peak2_height is not None and peak2_fwhm is not None else None
    )
    peak_separation = (
        peak2_pos - peak1_pos
        if peak1_pos is not None and peak2_pos is not None else None
    )
    total_fwhm = (
        peak1_fwhm + peak2_fwhm
        if peak1_fwhm is not None and peak2_fwhm is not None else None
    )
    total_height = (
        peak1_height + peak2_height
        if peak1_height is not None and peak2_height is not None else None
    )

    return {
        'rdf_n_peaks': n_peaks,
        'rdf_peak_separation': peak_separation,
        'rdf_height_ratio_21': safe_div(peak2_height, peak1_height),
        'rdf_width_ratio_21': safe_div(peak2_fwhm, peak1_fwhm),
        'rdf_area_ratio_21': safe_div(peak2_area, peak1_area),
        'rdf_peak1_width_norm': safe_div(peak1_fwhm, peak1_pos),
        'rdf_peak2_width_norm': safe_div(peak2_fwhm, peak2_pos),
        'rdf_peak2_sharpness': safe_div(peak2_height, peak2_fwhm),
        'rdf_sharpness_ratio_21': safe_div(
            safe_div(peak2_height, peak2_fwhm),
            safe_div(peak1_height, peak1_fwhm),
        ),
        'rdf_total_fwhm': total_fwhm,
        'rdf_local_order': safe_div(total_height, total_fwhm),
        'rdf_disorder': safe_div(total_fwhm, total_height),
        'rdf_peak_balance': safe_div(
            peak2_height - peak1_height
            if peak1_height is not None and peak2_height is not None else None,
            total_height,
        ),
        'rdf_width_balance': safe_div(
            peak2_fwhm - peak1_fwhm
            if peak1_fwhm is not None and peak2_fwhm is not None else None,
            total_fwhm,
        ),
        'rdf_gap_over_width': safe_div(peak_separation, total_fwhm),
        'rdf_gap_over_peak1': safe_div(peak_separation, peak1_pos),
        'rdf_peak1_area': peak1_area,
        'rdf_peak2_area': peak2_area,
        'volume_per_atom': volume / 32.0 if volume is not None else None,
        'volume_per_cation': volume / 16.0 if volume is not None else None,
        'tetragonal_strain': abs(c_over_a - 1.0) if c_over_a is not None else None,
        'ca_centered_abs': abs(c_over_a - 1.2) if c_over_a is not None else None,
        'u_peak1_pos': u * peak1_pos if u is not None and peak1_pos is not None else None,
        'u_peak2_pos': u * peak2_pos if u is not None and peak2_pos is not None else None,
        'u_rdf_width_ratio': (
            u * safe_div(peak2_fwhm, peak1_fwhm)
            if u is not None and safe_div(peak2_fwhm, peak1_fwhm) is not None else None
        ),
    }


def merge_structure_features(record):
    features = dict(record.get('structure_features') or {})
    for name, value in calculate_rdf_lattice_features(record).items():
        if value is not None:
            features[name] = value
    return features


def preprocess_data(data, filter_structures=None,enable_swap=True):
    """
    Remove selected fields, add atomic features, and augment the data by swapping A and B.
    
    Args:
        data: Raw data dictionary.
        filter_structures: Structure names to exclude, such as ['6Sc', '10Mg5V']. None uses the default exclusion list.
        enable_swap: Whether to augment the data by swapping A and B.
    Returns:
        The processed data dictionary, including the original and A/B-swapped entries.
    """
    # Use the properties of Al as a reference
    al_radius = get_atomic_radius('Al')
    al_electronegativity = get_electronegativity('Al')
    al_valence_electrons = get_valence_electrons('Al')
    al_ionic_radius = get_ionic_radius('Al', valence=3)  # Al typically has oxidation state +3
    
    # Get the ionic radius of O²⁻
    # Perovskites typically use the six-coordinate oxide ion radius of about 1.40 Å
    o_ionic_radius = get_ionic_radius('O', valence=-2)
    if o_ionic_radius is None:
        # Fall back to the standard radius for six-coordinate O²⁻
        o_ionic_radius = 1.40
        logger.error("Warning: Could not get the oxide ion radius. Using the default of 1.40 Å.")
    
    if al_radius is None or al_electronegativity is None:
        logger.error("Error: Could not get the atomic properties of Al")
        return None, 0
    
    # Exclude structures with known or suspected structural collapse by default
    if filter_structures is None:
        filter_structures = [
            '6Sc', '6Y', 
            "3Ca3Nb","2Ca2Nb", "1Ca1Nb","8Ca8Nb",
            "1Ca1Ta","2Ca2Ta","3Ca3Ta","6Ca3Ta",
            "1Ca1V","2Ca2V","3Ca3V","8Ca4V","10Ca5V",
            "4Mg4V","3Mg3V","2Mg2V","1Mg1V",
            "3Mg3Ta","2Mg2Ta","1Mg1Ta",
            "3Mg3Nb","2Mg2Nb","1Mg1Nb",
            
            "3Zn3Ta","2Zn2Ta","1Zn1Ta",
            "6Zn3V","3Zn3V","2Zn2V","1Zn1V",
            "3Zn3Nb","2Zn2Nb","1Zn1Nb","4Zn2Nb",
            '10Mg5V', "4Mg2Ta","4Mg2V","8Mg8Nb",
            "2Mg1Ta", "3Mg3Nb", "8Zn4Nb","6Mg6Nb", "6Ca3V","10Mg5Nb", "6Mg6V","8Mg8Ta", "8Mg4Ta","8Mg8V","10Ca5Ta","10Ca5Nb",
            
            # Structures confirmed to have collapsed
            "6Zn3Nb","POSCAR-K1Nb1Al14N16-2","POSCAR-K3Ta3Al10N16-3","POSCAR-K2Nb2Al12N16-1","POSCAR-K1Al14V1N16-1",
            "6Zn3Ta","POSCAR-K1Nb1Al14N16-3","POSCAR-Na2Al12V2N16-1","POSCAR-K1Ta1Al14N16-3","POSCAR-Hf3Be3Al10N16-2",
            "4Y","POSCAR-Na2Ta2Al12N16-2",
            
            # Structures to review later
            "POSCAR-Ca3Ti3Al10N16-2",
            "POSCAR-Hf2Mg2Al12N16-3",
            "POSCAR-K1Ta1Al14N16-1",
            
            "POSCAR-Ca1Ti1Al14N16-3",
            "POSCAR-Ba1Zr1Al14N16-3",
            
            "POSCAR-Na1Nb1Al14N16-3",
            "POSCAR-Ca2Ti2Al12N16-2",
            
            "POSCAR-K1Ta1Al14N16-1",
            "POSCAR-Na3Al10V3N16-2",
            "4Ca2Nb",
            "POSCAR-Na2Nb2Al12N16-1",
            "POSCAR-Na2Ta2Al12N16-2",
            "POSCAR-Li3Al10V3N16-3",
            "POSCAR-Na1Nb1Al14N16-1",
            "POSCAR-Na3Ta3Al10N16-1",
            "POSCAR-Ba1Hf1Al14N16-2",
            "POSCAR-Ca3Zr3Al10N16-1",
            
            "POSCAR-Hf1Mg1Al14N16-2",
            "POSCAR-Y5Al11N16-3",
            "POSCAR-Li3Al10V3N16-2",
            "POSCAR-Ca1Hf1Al14N16-1",
            "POSCAR-Ca1Ti1Al14N16-1",
            "POSCAR-Al13B3N16-3",
            "POSCAR-Al13B3N16-1",
            "POSCAR-Na3Nb3Al10N16-2",
            "POSCAR-Na2Al12V2N16-3",
            "POSCAR-Na1Al14V1N16-1",
            
            "POSCAR-Na1Al14V1N16-3",
            "POSCAR-K2Ta2Al12N16-2",
            "POSCAR-Ca2Zr2Al12N16-2",
            "POSCAR-Y3Al13N16-3",
            "POSCAR-Y5Al11N16-1",
            "POSCAR-Na1Ta1Al14N16-1",
            "POSCAR-K3Nb3Al10N16-2",
            "POSCAR-K3Al10V3N16-2",
            "4Ca2Ta",
            "POSCAR-Y3Al11B2N16-3",
            
            "POSCAR-K3Ta3Al10N16-1",
            "POSCAR-Li3Ta3Al10N16-4",
            "POSCAR-Ba1Zr1Al14N16-1",
            "POSCAR-Na1Ta1Al14N16-3",
            "POSCAR-Ca3Ti3Al10N16-3",
            
            "POSCAR-Li2Nb2Al12N16-2",
            "POSCAR-Sc5Al11N16-1",
            "POSCAR-Y1Sc2Al13N16-3",
            "POSCAR-Na2Al12V2N16-2",
            "POSCAR-Y2Al14N16-1",
            "POSCAR-K3Al10V3N16-1",
            "POSCAR-Li1Ta1Al14N16-1",
            "POSCAR-Li2Ta2Al12N16-3",
            "POSCAR-Mg2Ti2Al12N16-4",
            "POSCAR-Li1Al14V1N16-2",
            "2Y",
            "POSCAR-Sc1Al15N16-1",
            "POSCAR-Zr2Be2Al12N16-3",
            "POSCAR-Ba1Ti1Al14N16-1",
            
            "POSCAR-Ti2Be2Al12N16-3",
            "POSCAR-Ti1Al14Zn1N16-2",
            "POSCAR-K1Nb1Al14N16-1",
            "POSCAR-Ba1Ti1Al14N16-1",
            "POSCAR-Ti1Al14Zn1N16-2",
            "POSCAR-Ca3Ti3Al10N16-1",
            "POSCAR-Ca2Zr2Al12N16-1",
            "POSCAR-Ba1Ti1Al14N16-2",
            "POSCAR-K1Nb1Al14N16-1",
            "POSCAR-Ba3Ti3Al10N16-2",
            "POSCAR-Hf1Al14Zn1N16-1",
            "POSCAR-Zr1Be1Al14N16-1",
            "POSCAR-Hf3Al10Zn3N16-2",
            "POSCAR-Ba3Zr3Al10N16-1",
            "POSCAR-Hf1Al14Zn1N16-3",
            "POSCAR-Zr1Be1Al14N16-2",
            "POSCAR-Mg2Ti2Al12N16-1",
            "POSCAR-Ca3Zr3Al10N16-2",
            "POSCAR-Y1Al14B1N16-3",
            "POSCAR-Li2Ta2Al12N16-1",

            "POSCAR-Ba3Hf3Al10N16-3",
            "POSCAR-Ba3Hf3Al10N16-1",
            "POSCAR-Ba2Zr2Al12N16-3",
            "POSCAR-Li2Al12V2N16-5",
            "POSCAR-Zr3Be3Al10N16-3",

            "4Zn2Ta",
            "POSCAR-Y5Al11N16-2",

            "POSCAR-Li3Nb3Al10N16-2",
            "POSCAR-Sc3Al10B3N16-3",
            "POSCAR-Zr3Be3Al10N16-2",
            "POSCAR-Hf1Be1Al14N16-2",
            "POSCAR-Hf1Be1Al14N16-1",
            "POSCAR-Hf2Be2Al12N16-3",
            "POSCAR-Ca2Hf2Al12N16-3",
            "POSCAR-Al13Ga1B2N16-2",
            "POSCAR-Li2Al12V2N16-4",
            ]
        # filter_structures = []

    
    # Use a set for fast membership checks
    filter_set = set(filter_structures)
    
    processed_data = {}
    filtered_count = 0
    
    for key, value in data.items():
        # Check whether the structure is excluded
        if key in filter_set:
            logger.info(f"Excluding structure: {key}")
            filtered_count += 1
            continue
        
        # Exclude swapped entries if their original structures are excluded
        if key.endswith('_swap'):
            original_key = key.replace('_swap', '')
            if original_key in filter_set:
                logger.info(f"Excluding swapped structure: {key} (original: {original_key})")
                filtered_count += 1
                continue
        if 'd33' not in value:
            logger.error(f"Structure {key} has no d33 field")
            continue
        if value['d33'] is None:
            logger.warning(f"Skipping structure {key} because d33 is missing")
            continue
        processed_entry = value.copy()
        
        # Remove selected fields and retain e33
        fields_to_remove = ['c33', 'e33_matrix1', 'e33_matrix2']
        for field in fields_to_remove:
            processed_entry.pop(field, None)
        
        # Extract the A and B dopants
        dopants = value.get('dopants', {})
        a_element = dopants.get('A', {}).get('element')
        a_amount = dopants.get('A', {}).get('amount', 0)
        b_element = dopants.get('B', {}).get('element')
        b_amount = dopants.get('B', {}).get('amount', 0)
        
        if b_amount == None:
            b_amount = 0
        
        # Add A_amount and B_amount
        processed_entry['A_amount'] = a_amount
        processed_entry['B_amount'] = b_amount
        
        # Calculate the charge compensation parameter charge_delta
        # Formula: |(A_amount × 2 + B_amount × 5) - (A_amount + B_amount) × 3|
        # A dopants such as Mg are typically +2, and B dopants such as Nb are typically +5
        # The total charge before replacing Al (+3) is (A_amount + B_amount) × 3
        try:
            a_amount_val = float(a_amount) if a_amount is not None else 0.0
            b_amount_val = float(b_amount) if b_amount is not None else 0.0
            
            # Dopant charge: A_amount × 2 + B_amount × 5
            actual_charge = a_amount_val * 2 + b_amount_val * 5
            
            # Original charge of the Al ions (+3) being replaced
            original_charge = (a_amount_val + b_amount_val) * 3
            
            # Charge difference
            charge_delta = abs(actual_charge - original_charge)
            processed_entry['charge_delta'] = charge_delta
        except (ValueError, TypeError) as e:
            processed_entry['charge_delta'] = None
            print(f"Warning: Could not calculate charge_delta with A_amount={a_amount}, B_amount={b_amount}: {e}")
        
        a_ionic_radius = None
        b_ionic_radius = None

        # Calculate the differences in atomic radius, electronegativity, valence electron count, and ionic radius between A and Al
        if a_element:
            a_radius = get_atomic_radius(a_element)
            a_electronegativity = get_electronegativity(a_element)
            a_valence_electrons = get_valence_electrons(a_element)
            processed_entry['A_electronegativity'] = a_electronegativity
            
            a_ionic_radius = get_ionic_radius(a_element, valence=a_valence_electrons)  # A dopants typically have oxidation state +2
            logger.info(f"{a_element}: {a_valence_electrons},{a_ionic_radius}")
            if a_radius is not None:
                processed_entry['A_radius_diff'] = a_radius - al_radius
            else:
                processed_entry['A_radius_diff'] = None
                print(f"Warning: Could not get the atomic radius for {a_element}")
            
            if a_electronegativity is not None:
                processed_entry['A_electronegativity_diff'] = a_electronegativity - al_electronegativity
            else:
                processed_entry['A_electronegativity_diff'] = None
                print(f"Warning: Could not get the electronegativity for {a_element}")
            
            if a_valence_electrons is not None and al_valence_electrons is not None:
                processed_entry['A_valence_electrons_diff'] = a_valence_electrons - al_valence_electrons
            else:
                processed_entry['A_valence_electrons_diff'] = None
                if a_valence_electrons is None:
                    print(f"Warning: Could not get the valence electron count for {a_element}")
            
            if a_ionic_radius is not None and al_ionic_radius is not None:
                processed_entry['A_ionic_radius_diff'] = a_ionic_radius - al_ionic_radius
            else:
                processed_entry['A_ionic_radius_diff'] = None
                if a_ionic_radius is None:
                    print(f"Warning: Could not get the ionic radius for {a_element}")
        else:
            processed_entry['A_radius_diff'] = None
            processed_entry['A_electronegativity_diff'] = None
            processed_entry['A_valence_electrons_diff'] = None
            processed_entry['A_ionic_radius_diff'] = None
        
        # Calculate the differences in atomic radius, electronegativity, valence electron count, and ionic radius between B and Al
        # Use zero when B is absent or its amount is zero
        if b_element and b_amount > 0:
            b_radius = get_atomic_radius(b_element)
            b_electronegativity = get_electronegativity(b_element)
            b_valence_electrons = get_valence_electrons(b_element)
            processed_entry['B_electronegativity'] = b_electronegativity
            b_ionic_radius = get_ionic_radius(b_element, valence=b_valence_electrons)  # B dopants typically have oxidation state +5
            logger.info(f"{b_element}: {b_valence_electrons},{b_ionic_radius}")
            if b_radius is not None:
                processed_entry['B_radius_diff'] = b_radius - al_radius
            else:
                processed_entry['B_radius_diff'] = None
                print(f"Warning: Could not get the atomic radius for {b_element}")
            
            if b_electronegativity is not None:
                processed_entry['B_electronegativity_diff'] = b_electronegativity - al_electronegativity
            else:
                processed_entry['B_electronegativity_diff'] = None
                print(f"Warning: Could not get the electronegativity for {b_element}")
            
            if b_valence_electrons is not None and al_valence_electrons is not None:
                processed_entry['B_valence_electrons_diff'] = b_valence_electrons - al_valence_electrons
            else:
                processed_entry['B_valence_electrons_diff'] = None
                if b_valence_electrons is None:
                    print(f"Warning: Could not get the valence electron count for {b_element}")
            
            if b_ionic_radius is not None and al_ionic_radius is not None:
                processed_entry['B_ionic_radius_diff'] = b_ionic_radius - al_ionic_radius
            else:
                processed_entry['B_ionic_radius_diff'] = None
                if b_ionic_radius is None:
                    print(f"Warning: Could not get the ionic radius for {b_element}")
        else:
            # B is absent or its amount is zero, so the differences are zero
            processed_entry['B_radius_diff'] = 0.0
            processed_entry['B_electronegativity_diff'] = 0.0
            processed_entry['B_valence_electrons_diff'] = 0.0
            processed_entry['B_ionic_radius_diff'] = 0.0
            
        # Calculate the total mass
        if b_amount is None:
            b_amount = 0
        if a_amount is None:
            a_amount = 0
        al_amount = 16 - a_amount - b_amount
        # logger.info(f"{b_amount}, {a_amount}, {al_amount},{get_atmoic_mass(a_element)},{get_atmoic_mass(b_element)},{get_atmoic_mass('Al')}")
        processed_entry['total_mass'] = a_amount * get_atmoic_mass(a_element) + b_amount * get_atmoic_mass(b_element) + al_amount * get_atmoic_mass('Al')
        processed_entry['density'] = processed_entry['total_mass'] / value.get('volume_ang3')
        # Calculate the Goldschmidt tolerance factor (t) and the octahedral factor (μ)
        # Goldschmidt tolerance factor: t = (R_A + R_O) / (√2 * (R_B + R_O))
        # Octahedral factor: μ = R_B / R_O
        # R_A, R_B, and R_O are the ionic radii of the A-site, B-site, and oxide ions, respectively
        
        # Calculate the Goldschmidt tolerance factor
        if a_ionic_radius is not None and b_ionic_radius is not None and o_ionic_radius is not None:
            try:
                # t = (R_A + R_O) / (√2 * (R_B + R_O))
                numerator = a_ionic_radius + o_ionic_radius
                denominator = (2.0 ** 0.5) * (b_ionic_radius + o_ionic_radius)
                if denominator > 0:
                    processed_entry['goldschmidt_tolerance_factor'] = numerator / denominator
                else:
                    processed_entry['goldschmidt_tolerance_factor'] = None
                    print(f"Warning: The denominator of the Goldschmidt tolerance factor is zero")
            except (ValueError, TypeError) as e:
                processed_entry['goldschmidt_tolerance_factor'] = None
                print(f"Warning: Could not calculate the Goldschmidt tolerance factor: {e}")
        else:
            processed_entry['goldschmidt_tolerance_factor'] = None
            if a_ionic_radius is None:
                print(f"Warning: The Goldschmidt tolerance factor requires the ionic radius of A dopant {a_element}")
            if b_ionic_radius is None and b_element and b_amount > 0:
                print(f"Warning: The Goldschmidt tolerance factor requires the ionic radius of B dopant {b_element}")
        
        # Calculate the octahedral factor
        if b_ionic_radius is not None and o_ionic_radius is not None:
            try:
                # μ = R_B / R_O
                if o_ionic_radius > 0:
                    processed_entry['octahedral_factor'] = b_ionic_radius / o_ionic_radius
                else:
                    processed_entry['octahedral_factor'] = None
                    print(f"Warning: The oxide ion radius is zero, so the octahedral factor is undefined")
            except (ValueError, TypeError) as e:
                processed_entry['octahedral_factor'] = None
                print(f"Warning: Could not calculate the octahedral factor: {e}")
        else:
            processed_entry['octahedral_factor'] = None
            if b_ionic_radius is None and b_element and b_amount > 0:
                print(f"Warning: The octahedral factor requires the ionic radius of B dopant {b_element}")
        
        # Calculate the |u-0.375| parameter
        rdf_analysis = value.get('rdf_analysis', {})
        u = rdf_analysis.get('u')
        if u is not None:
            try:
                u_value = float(u)
                processed_entry['u_minus_0_375_abs'] = abs(u_value - 0.375)
            except (ValueError, TypeError):
                processed_entry['u_minus_0_375_abs'] = None
                print(f"Warning: Could not calculate |u-0.375| for u={u}")
        else:
            processed_entry['u_minus_0_375_abs'] = None
        
        # Calculate peak sharpness: sharpness = Peak1_Height / Peak1_FWHM
        peak1_height = rdf_analysis.get('Peak1_Height')
        peak1_fwhm = rdf_analysis.get('Peak1_FWHM')
        if peak1_height is not None and peak1_fwhm is not None:
            try:
                height_val = float(peak1_height)
                fwhm_val = float(peak1_fwhm)
                if fwhm_val > 0:
                    processed_entry['sharpness'] = height_val / fwhm_val
                else:
                    processed_entry['sharpness'] = None
                    print(f"Warning: Could not calculate sharpness because Peak1_FWHM is zero or negative")
            except (ValueError, TypeError) as e:
                processed_entry['sharpness'] = None
                print(f"Warning: Could not calculate sharpness with Peak1_Height={peak1_height}, Peak1_FWHM={peak1_fwhm}: {e}")
        else:
            processed_entry['sharpness'] = None
        
        # Calculate the tetrahedral distortion index: distortion = Peak1_Pos / Volume^(1/3)
        peak1_pos = rdf_analysis.get('Peak1_Pos')
        volume = value.get('volume_ang3')
        if peak1_pos is not None and volume is not None:
            try:
                pos_val = float(peak1_pos)
                volume_val = float(volume)
                if volume_val > 0:
                    # Calculate the cube root of the volume
                    volume_cube_root = volume_val ** (1.0 / 3.0)
                    processed_entry['distortion'] = pos_val / volume_cube_root
                else:
                    processed_entry['distortion'] = None
                    print(f"Warning: Could not calculate distortion because volume_ang3 is zero or negative")
            except (ValueError, TypeError) as e:
                processed_entry['distortion'] = None
                print(f"Warning: Could not calculate distortion with Peak1_Pos={peak1_pos}, volume_ang3={volume}: {e}")
        else:
            processed_entry['distortion'] = None
        
        # Calculate polar_factor: c_over_a * (A_electronegativity_diff + B_electronegativity_diff)
        c_over_a = value.get('c_over_a')
        a_electronegativity_diff = processed_entry.get('A_electronegativity_diff')
        b_electronegativity_diff = processed_entry.get('B_electronegativity_diff')
        if c_over_a is not None and a_electronegativity_diff is not None and b_electronegativity_diff is not None:
            try:
                processed_entry['polar_factor'] = c_over_a * (a_electronegativity_diff + b_electronegativity_diff)
            except (ValueError, TypeError) as e:
                processed_entry['polar_factor'] = None
                print(f"Warning: Could not calculate polar_factor: {e}")
        else:
            processed_entry['polar_factor'] = None
        
        # Calculate distortion_per_vol: distortion / volume_ang3
        distortion = processed_entry.get('distortion')
        volume = value.get('volume_ang3')
        if distortion is not None and volume is not None:
            try:
                volume_val = float(volume)
                if volume_val > 0:
                    processed_entry['distortion_per_vol'] = distortion / volume_val
                else:
                    processed_entry['distortion_per_vol'] = None
                    print(f"Warning: Could not calculate distortion_per_vol because volume_ang3 is zero or negative")
            except (ValueError, TypeError) as e:
                processed_entry['distortion_per_vol'] = None
                print(f"Warning: Could not calculate distortion_per_vol: {e}")
        else:
            processed_entry['distortion_per_vol'] = None
        
        # Calculate u_distort_inter: u * distortion
        u = rdf_analysis.get('u')
        if u is not None and distortion is not None:
            try:
                u_value = float(u)
                processed_entry['u_distort_inter'] = u_value * distortion
            except (ValueError, TypeError) as e:
                processed_entry['u_distort_inter'] = None
                print(f"Warning: Could not calculate u_distort_inter: {e}")
        else:
            processed_entry['u_distort_inter'] = None

        structure_features = merge_structure_features(processed_entry)
        processed_entry['structure_features'] = structure_features
        processed_entry.update(structure_features)
        
        processed_data[key] = processed_entry
        
        # Augment the data with a copy that swaps A and B
        # Swap only when both dopants are present and the amount of B is positive
        if enable_swap and a_element and b_element and b_amount > 0:
            # Create the A/B-swapped entry
            swapped_entry = processed_entry.copy()
            swapped_entry['structure_features'] = dict(processed_entry.get('structure_features') or {})
            
            # Swap A and B in dopants
            original_dopants = swapped_entry.get('dopants', {}).copy()
            swapped_entry['dopants'] = {
                'A': original_dopants.get('B', {}).copy(),
                'B': original_dopants.get('A', {}).copy()
            }
            
            # Swap A_amount and B_amount
            original_a_amount = swapped_entry.get('A_amount')
            original_b_amount = swapped_entry.get('B_amount')
            swapped_entry['A_amount'] = original_b_amount
            swapped_entry['B_amount'] = original_a_amount
            
            # Recalculate charge_delta.
            # Estimate oxidation states from the elements after swapping A and B.
            # Assigning fixed +5 and +2 states by site in this AlN system would produce
            # invalid Shannon radius lookups such as Zn5+ and Hf2+.
            try:
                new_a_amount_val = float(original_b_amount) if original_b_amount is not None else 0.0
                new_b_amount_val = float(original_a_amount) if original_a_amount is not None else 0.0
                new_a_element = original_dopants.get('B', {}).get('element')
                new_b_element = original_dopants.get('A', {}).get('element')
                new_a_valence = get_valence_electrons(new_a_element)
                new_b_valence = get_valence_electrons(new_b_element)
                
                if new_a_valence is not None and new_b_valence is not None:
                    actual_charge = new_a_amount_val * new_a_valence + new_b_amount_val * new_b_valence
                    original_charge = (new_a_amount_val + new_b_amount_val) * 3
                    charge_delta = abs(actual_charge - original_charge)
                    swapped_entry['charge_delta'] = charge_delta
                else:
                    swapped_entry['charge_delta'] = processed_entry.get('charge_delta')
                
            except (ValueError, TypeError) as e:
                swapped_entry['charge_delta'] = None
                print(f"Warning: Could not calculate charge_delta after swapping A and B: {e}")
            
            # Swap all feature differences associated with A and B
            # Atomic radius differences
            original_a_radius_diff = swapped_entry.get('A_radius_diff')
            original_b_radius_diff = swapped_entry.get('B_radius_diff')
            swapped_entry['A_radius_diff'] = original_b_radius_diff
            swapped_entry['B_radius_diff'] = original_a_radius_diff
            
            # Electronegativity differences
            original_a_electronegativity_diff = swapped_entry.get('A_electronegativity_diff')
            original_b_electronegativity_diff = swapped_entry.get('B_electronegativity_diff')
            swapped_entry['A_electronegativity_diff'] = original_b_electronegativity_diff
            swapped_entry['B_electronegativity_diff'] = original_a_electronegativity_diff
            
            # Valence electron count differences
            original_a_valence_electrons_diff = swapped_entry.get('A_valence_electrons_diff')
            original_b_valence_electrons_diff = swapped_entry.get('B_valence_electrons_diff')
            swapped_entry['A_valence_electrons_diff'] = original_b_valence_electrons_diff
            swapped_entry['B_valence_electrons_diff'] = original_a_valence_electrons_diff
            
            # Ionic radius differences
            original_a_ionic_radius_diff = swapped_entry.get('A_ionic_radius_diff')
            original_b_ionic_radius_diff = swapped_entry.get('B_ionic_radius_diff')
            swapped_entry['A_ionic_radius_diff'] = original_b_ionic_radius_diff
            swapped_entry['B_ionic_radius_diff'] = original_a_ionic_radius_diff
            
            # Recalculate the Goldschmidt tolerance factor and octahedral factor after swapping A and B.
            # Continue to use typical oxidation states for the elements at the swapped sites.
            new_a_element = original_dopants.get('B', {}).get('element')
            new_b_element = original_dopants.get('A', {}).get('element')
            new_a_valence = get_valence_electrons(new_a_element)
            new_b_valence = get_valence_electrons(new_b_element)
            new_a_ionic_radius = get_ionic_radius(new_a_element, valence=new_a_valence) if new_a_element else None
            new_b_ionic_radius = get_ionic_radius(new_b_element, valence=new_b_valence) if new_b_element else None
            
            # Recalculate the Goldschmidt tolerance factor: t = (R_A + R_O) / (√2 * (R_B + R_O))
            if new_a_ionic_radius is not None and new_b_ionic_radius is not None and o_ionic_radius is not None:
                try:
                    numerator = new_a_ionic_radius + o_ionic_radius
                    denominator = (2.0 ** 0.5) * (new_b_ionic_radius + o_ionic_radius)
                    if denominator > 0:
                        swapped_entry['goldschmidt_tolerance_factor'] = numerator / denominator
                    else:
                        swapped_entry['goldschmidt_tolerance_factor'] = None
                except (ValueError, TypeError):
                    swapped_entry['goldschmidt_tolerance_factor'] = None
            else:
                swapped_entry['goldschmidt_tolerance_factor'] = None
            
            # Recalculate the octahedral factor: μ = R_B / R_O
            if new_b_ionic_radius is not None and o_ionic_radius is not None:
                try:
                    if o_ionic_radius > 0:
                        swapped_entry['octahedral_factor'] = new_b_ionic_radius / o_ionic_radius
                    else:
                        swapped_entry['octahedral_factor'] = None
                except (ValueError, TypeError):
                    swapped_entry['octahedral_factor'] = None
            else:
                swapped_entry['octahedral_factor'] = None
            
            # Recalculate polar_factor: c_over_a * (A_electronegativity_diff + B_electronegativity_diff)
            # Use the swapped values of A_electronegativity_diff and B_electronegativity_diff
            swapped_c_over_a = swapped_entry.get('c_over_a')
            swapped_a_electronegativity_diff = swapped_entry.get('A_electronegativity_diff')
            swapped_b_electronegativity_diff = swapped_entry.get('B_electronegativity_diff')
            if swapped_c_over_a is not None and swapped_a_electronegativity_diff is not None and swapped_b_electronegativity_diff is not None:
                try:
                    swapped_entry['polar_factor'] = swapped_c_over_a * (swapped_a_electronegativity_diff + swapped_b_electronegativity_diff)
                except (ValueError, TypeError) as e:
                    swapped_entry['polar_factor'] = None
                    print(f"Warning: Could not calculate polar_factor after swapping A and B: {e}")
            else:
                swapped_entry['polar_factor'] = None
            
            # Copy the original distortion_per_vol and u_distort_inter values.
            # Both depend only on distortion, volume, and u,
            # which are independent of the A/B swap.
            swapped_entry['distortion_per_vol'] = processed_entry.get('distortion_per_vol')
            swapped_entry['u_distort_inter'] = processed_entry.get('u_distort_inter')
            for feature_name, feature_value in swapped_entry['structure_features'].items():
                swapped_entry[feature_name] = feature_value
            
            # Append the _swap suffix to the original key
            swapped_key = f"{key}_swap"
            processed_data[swapped_key] = swapped_entry
            logger.info(f"Created A/B-swapped entry: {key} -> {swapped_key}")
    
    return processed_data, filtered_count


def build_cv5_folds(
    processed_data: dict,
    output_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
    min_threshold: float = 3,
    max_threshold: float = 30,
    use_swap: bool = False,
):
    """
    Build reusable cross-validation folds from preprocessed data and store structure names.
    Training scripts map those names to each feature table to avoid index shifts during cleaning.
    """
    sample_names = []
    groups = []
    for name, item in processed_data.items():
        if (not use_swap) and str(name).endswith("_swap"):
            continue
        d33 = item.get("d33", None)
        if d33 is None or d33 < min_threshold or d33 > max_threshold:
            continue
        sample_names.append(name)
        groups.append(name.replace("_swap", "") if str(name).endswith("_swap") else name)

    if len(sample_names) < n_splits:
        raise ValueError(f"Too few samples for {n_splits} folds: samples={len(sample_names)}")

    if use_swap:
        n_groups = len(set(groups))
        if n_groups < n_splits:
            raise ValueError(f"Too few groups for {n_splits} folds: groups={n_groups}")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(sample_names, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(sample_names)

    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
        folds.append(
            {
                "fold": fold_id,
                "train_names": [sample_names[i] for i in train_idx.tolist()],
                "test_names": [sample_names[i] for i in test_idx.tolist()],
            }
        )

    payload = {
        "meta": {
            "n_splits": n_splits,
            "random_state": random_state,
            "min_threshold": min_threshold,
            "max_threshold": max_threshold,
            "use_swap": use_swap,
            "n_samples": len(sample_names),
        },
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    """Run preprocessing and export the cross-validation folds."""
    data_dir = "data"
    
    input_file_list = [
        f"{data_dir}/results_combined.json",
    ]
    data = {}
    for input_file in input_file_list:
        if not Path(input_file).exists():
            print(f"Error: Input file {input_file} does not exist")
            sys.exit(1)
        print(f"Reading {input_file}...")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data.update(json.load(f))
                logger.success(f"Read {input_file} successfully")
        except Exception as e:
            print(f"Error: Could not read the input file: {e}")
            sys.exit(1)
    output_file = f"{data_dir}/results_preprocessed.json"
    folds_file = Path(f"{data_dir}/cv5_folds.json")
    
    print(f"Loaded {len(data)} raw records")
    enable_swap = True
    result = preprocess_data(data, enable_swap=enable_swap)
    
    if result is None:
        print("Error: Preprocessing failed")
        sys.exit(1)
    
    processed_data, filtered_count = result
    
    # Count the original, excluded, and augmented records
    original_count = len(data)
    augmented_count = len(processed_data)
    swap_count = augmented_count - (original_count - filtered_count)  # Account for excluded records
    
    # Save the processed data
    print(f"Saving to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Could not save the processed data: {e}")
        sys.exit(1)
    
    print(f"Preprocessing complete. Results saved to {output_file}")
    print(f"Original records: {original_count}")
    print(f"Excluded structures: {filtered_count}")
    print(f"A/B-swapped records: {swap_count}")
    print(f"Total processed records: {augmented_count}")

    # Generate the five-fold split for train.py and train_pysr.py
    try:
        out_folds = build_cv5_folds(
            processed_data=processed_data,
            output_path=folds_file,
            n_splits=5,
            random_state=42,
            min_threshold=3,
            max_threshold=30,
            use_swap=enable_swap,
        )
        print(f"Saved the five-fold split to {out_folds}")
    except Exception as e:
        print(f"Warning: Could not generate the five-fold split: {e}")


if __name__ == '__main__':
    main()
