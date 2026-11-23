"""
Write the final results from the c33 and e33 calculations to JSON.
"""

import shutil
import os
import subprocess
import re
import csv
import json
from pathlib import Path
import traceback
import numpy as np
from loguru import logger
try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    SpacegroupAnalyzer = None
    logger.warning("pymatgen is not installed. Install it with pip install pymatgen to enable RDF calculations.")


def check_covered(output):
    """
    Check whether the output contains "Get COVERAGED:  YES !".
    """
    return "YES" in output or "Get COVERAGED: YES !" in output

def check_is_compute_completed(target_dir):
    """
    Return True if the calculation has already completed, or False otherwise.
    """
    if os.path.exists(target_dir / "OUTCAR"):
        with open(target_dir / "OUTCAR", "r", encoding="utf-8") as f:
            content = f.read()
        if "General timing and accounting informations for this job" in content:
            return True
    return False

def extract_energy_from_output(output):
    """
    Extract the energy from the output of qvasp -e.
    Example output: The Final Energy in This Folder is:   -238.24048093
    """
    try:
        test_output2 = output.split('\n')[1].split(' ')[-1]
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', test_output2)
        return float(clean_output)
    except:
        return None

def extract_structure_name(parent_name, child_name):
    """
    Extract the structure name from the parent and child directory names.
    
    Rules:
    1. Use the child directory name as is when it contains multiple elements (e.g., 8Ca4Nb).
    2. When the child name contains one element (e.g., 10Zn), get the other element from the parent name and calculate its count.
    
    Examples:
    - ZnV-w + 10Zn -> 10Zn5V (half as many V atoms as Zn atoms)
    - ZnV-w + 8Zn -> 8Zn4V (half as many V atoms as Zn atoms)
    - 8Ca4Nb -> 8Ca4Nb (unchanged)
    
    Args:
        parent_name: Parent directory name, such as "ZnV-w".
        child_name: Child directory name, such as "10Zn" or "8Ca4Nb".
    
    Returns: The structure name, such as "10Zn5V" or "8Ca4Nb".
    """
    try:
        # Check for multiple element counts in the child directory name.
        # Match a count followed by an element symbol with an optional lowercase letter.
        pattern = r'(\d+)([A-Z][a-z]?)'
        matches = re.findall(pattern, child_name)
        
        if len(matches) > 1:
            # Use the child directory name as is when it contains multiple elements.
            return child_name
        
        # Get the second element from the parent name when the child name contains only one.
        if len(matches) == 1:
            child_amount, child_element = matches[0]
            child_amount = int(child_amount)
            
            # Strip suffixes such as -w before extracting elements from the parent name.
            # Example: ZnV-w -> ZnV
            parent_clean = re.sub(r'-[a-zA-Z]+$', '', parent_name)
            
            # Extract all element symbols from the parent directory name.
            parent_pattern = r'([A-Z][a-z]?)'
            parent_elements = re.findall(parent_pattern, parent_clean)
            
            # Find the element that differs from the child directory's element.
            other_element = None
            for elem in parent_elements:
                if elem != child_element:
                    other_element = elem
                    break
            
            if other_element is None:
                # Use the child directory name if no second element is found.
                logger.warning(f"No second element found in {parent_name} alongside {child_element}. Using the child directory name.")
                return child_name
            
            # Set the second element count to half the first.
            other_amount = child_amount // 2
            
            # Build the structure name with the child directory's element first.
            structure_name = f"{child_amount}{child_element}{other_amount}{other_element}"
            return structure_name
        
        # Use the child directory name if parsing fails.
        logger.warning(f"Could not parse child directory name {child_name}. Using the original name")
        return child_name
        
    except Exception as e:
        logger.error(f"Could not extract a structure name from parent={parent_name}, child={child_name}: {e}")
        return child_name

def extract_piezoelectric_tensor(outcar_path, tensor_type='local_field'):
    """
    Extract the piezoelectric tensor from an OUTCAR file.
    
    Args:
        outcar_path: Path to the OUTCAR file.
        tensor_type: Either 'local_field' or 'ionic_contr'.
            - 'local_field': Look for "PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)".
            - 'ionic_contr': Look for "PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)".
    
    Read the matrix below the header and keep the first three columns (XX, YY, ZZ).
    
    Example matrix:
          XX          YY          ZZ          XY          YZ          ZX
    --------------------------------------------------------------------------------
    x    -0.00871    -0.01476     0.00240    -0.04797    -0.00297     0.32513
    y    -0.07502     0.10531     0.02172    -0.00106     0.32974    -0.00310
    z     0.27751     0.26536    -0.40496     0.00879     0.01468    -0.01409
    
    Returns: A 3x3 nested list in x, y, z row order, or None if parsing fails.
    """
    try:
        with open(outcar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Select the marker for the requested tensor type.
        start_idx = None
        for i, line in enumerate(lines):
            line_upper = line.upper()
            line_lower = line.lower()
            
            if tensor_type == 'local_field':
                # Look for "PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)".
                if ("PIEZOELECTRIC TENSOR" in line_upper and 
                    "(including local field effects)" in line_lower and 
                    "for field in x, y, z" in line_lower and 
                    "(c/m^2)" in line_lower):
                    start_idx = i
                    break
            elif tensor_type == 'ionic_contr':
                # Look for "PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)".
                if ("PIEZOELECTRIC TENSOR" in line_upper and 
                    "IONIC CONTR" in line_upper and 
                    "for field in x, y, z" in line_lower and 
                    "(c/m^2)" in line_lower):
                    start_idx = i
                    break
            else:
                logger.error(f"Unknown tensor type: {tensor_type}")
                return None
        
        if start_idx is None:
            logger.warning(f"No PIEZOELECTRIC TENSOR header found for {tensor_type} in {outcar_path}")
            return None
        
        logger.info(f"Found the {tensor_type} header: {lines[start_idx].strip()}")
        
        # Read the x, y, and z rows in that order.
        target_rows = ['x', 'y', 'z']
        matrix = [None, None, None]  # Reserve three row slots.
        
        for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
            line = lines[i].strip()
            
            # Skip blank lines and separators.
            if not line or line.startswith('-'):
                continue
            
            # Look for data rows that begin with x, y, or z.
            parts = line.split()
            if len(parts) < 4:  # Require a row label and at least three values.
                continue
            
            # Check whether the first field is a target row label (x, y, or z).
            if parts[0].lower() in target_rows:
                try:
                    # Keep the first three values after the label (XX, YY, ZZ).
                    values = [float(x) for x in parts[1:4]]
                    if len(values) == 3:
                        # Map row labels to indices: x -> 0, y -> 1, z -> 2.
                        idx = target_rows.index(parts[0].lower())
                        matrix[idx] = values
                except (ValueError, IndexError):
                    continue
        
        # Check that the matrix has three rows of three values.
        if all(row is not None for row in matrix) and all(len(row) == 3 for row in matrix):
            return matrix
        else:
            logger.warning(f"Found {len([r for r in matrix if r is not None])} rows in {outcar_path}. Expected a 3x3 matrix.")
            return None
            
    except FileNotFoundError:
        logger.error(f"File not found: {outcar_path}")
        return None
    except Exception as e:
        logger.error(f"Could not parse the piezoelectric tensor in {outcar_path}: {e}")
        return None

def extract_elastic_moduli(output):
    """
    Extract the elastic moduli matrix from the output of qvasp -elasticd.
    
    Example output:
    TOTAL ELASTIC MODULI (kBar)
    
     Direction    XX          YY          ZZ          XY          YZ          ZX
     --------------------------------------------------------------------------------
     XX        3268.0317   1245.2482   1006.7377     21.2409     19.7083     -2.0414
     YY        1245.2482   3294.0697   1080.5299     -6.7877     10.2641     -2.4441
     ZZ        1006.7377   1080.5299   2802.3471      3.9547    -15.1674     -1.9348
     XY          21.2409     -6.7877      3.9547    987.7900      0.3899     17.0815
     YZ          19.7083     10.2641    -15.1674      0.3899    980.8160     -6.3401
     ZX          -2.0414     -2.4441     -1.9348     17.0815     -6.3401    996.8744
     --------------------------------------------------------------------------------
    
    Returns: A 3x3 nested list with the first three values of the XX, YY, and ZZ rows, or None if parsing fails.
    """
    try:
        lines = output.split('\n')
        
        # Find the "TOTAL ELASTIC MODULI" header.
        start_idx = None
        for i, line in enumerate(lines):
            if "TOTAL ELASTIC MODULI" in line.upper():
                start_idx = i
                break
        
        if start_idx is None:
            logger.warning("No TOTAL ELASTIC MODULI header found")
            return None
        
        # Keep the first three values of the XX, YY, and ZZ rows in that order.
        target_directions = ['XX', 'YY', 'ZZ']
        matrix = [None, None, None]  # Reserve three row slots.
        
        for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
            line = lines[i].strip()
            
            # Skip blank lines and separators.
            if not line or line.startswith('-'):
                continue
            
            # Look for data rows that begin with a direction label.
            parts = line.split()
            if len(parts) < 4:  # Require a direction label and at least three values.
                continue
            
            # Check whether the first field is a target direction label (XX, YY, or ZZ).
            if parts[0] in target_directions:
                try:
                    # Keep the first three values after the direction label.
                    values = [float(x) for x in parts[1:4]]
                    if len(values) == 3:
                        # Map direction labels to indices: XX -> 0, YY -> 1, ZZ -> 2.
                        idx = target_directions.index(parts[0])
                        matrix[idx] = values
                except (ValueError, IndexError):
                    continue
        
        # Check that the matrix has three rows of three values.
        if all(row is not None for row in matrix) and all(len(row) == 3 for row in matrix):
            return matrix
        else:
            logger.warning(f"Expected a 3x3 elastic moduli matrix, parsed {len(matrix)} row slots")
            return None
            
    except Exception as e:
        logger.error(f"Could not parse the elastic moduli matrix: {e}")
        return None

def calculate_e33_matrix(e33_matrix1, e33_matrix2):
    """
    Calculate the e33 matrix by adding e33_matrix1 and e33_matrix2.
    
    Args:
        e33_matrix1: First piezoelectric tensor matrix (3x3).
        e33_matrix2: Second piezoelectric tensor matrix (3x3).
    
    Returns: The e33 matrix (3x3), or None if either input is None.
    """
    if e33_matrix1 is None or e33_matrix2 is None:
        return None
    
    try:
        # Convert the inputs to NumPy arrays for matrix addition.
        e33_1 = np.array(e33_matrix1)
        e33_2 = np.array(e33_matrix2)
        
        # Add the matrices.
        e33 = e33_1 + e33_2
        
        # Convert the result back to a nested list.
        return e33.tolist()
    except Exception as e:
        logger.error(f"Could not calculate the e33 matrix: {e}")
        return None

def calculate_d33(c33_matrix, e33_matrix):
    """
    Calculate the d33 piezoelectric coefficient.
    
    Formula: d_{33} = [e_{33}(c_{11} + c_{12}) - 2e_{31}c_{13}] / [(c_{11} + c_{12})c_{33} - 2c_{13}^2]
    
    Terms:
        - c33_matrix: Elastic moduli matrix (3x3) in kbar. Divide by 10 to convert to GPa.
        - e33_matrix: Piezoelectric coefficient matrix (3x3) in C/m^2.
        - c11 = c33_matrix[0][0] (XX component of the XX row)
        - c12 = c33_matrix[0][1] (YY component of the XX row)
        - c13 = c33_matrix[0][2] (ZZ component of the XX row)
        - c33 = c33_matrix[2][2] (ZZ component of the ZZ row)
        - e33 = e33_matrix[2][2] (ZZ component of the z row)
        - e31 = e33_matrix[2][0] (XX component of the z row)
    
    Args:
        c33_matrix: Elastic moduli matrix (3x3) in kbar.
        e33_matrix: Piezoelectric coefficient matrix (3x3) in C/m^2.
    
    Returns: The scalar d33, or None if an input is None or the calculation fails.
    """
    if c33_matrix is None or e33_matrix is None:
        return None
    
    try:
        # Convert the inputs to NumPy arrays.
        C = np.array(c33_matrix)
        e = np.array(e33_matrix)
        
        # Divide all elements of C by 10 to convert kbar to GPa.
        C = C / 10.0
        
        # Select the required matrix elements.
        c11 = C[0][0]  # XX component of the XX row.
        c12 = C[0][1]  # YY component of the XX row.
        c13 = C[0][2]  # ZZ component of the XX row.
        c33 = C[2][2]  # ZZ component of the ZZ row.
        
        e33 = e[2][2]  # ZZ component of the z row.
        e31 = e[2][0]  # XX component of the z row.
        
        # Numerator: e_{33}(c_{11} + c_{12}) - 2e_{31}c_{13}
        numerator = e33 * (c11 + c12) - 2 * e31 * c13
        logger.debug(f"Numerator: {numerator}")
        
        # Denominator: (c_{11} + c_{12})c_{33} - 2c_{13}^2
        denominator = (c11 + c12) * c33 - 2 * (c13 ** 2)
        logger.debug(f"Denominator: {denominator}")

        # Check for a zero denominator.
        if abs(denominator) < 1e-10:
            logger.error(f"Could not calculate d33 because the denominator is zero")
            return None
        
        # Calculate d33.
        d33 = numerator / denominator
        
        return float(d33 * 1000)
    except Exception as e:
        logger.error(f"Could not calculate d33: {e}")
        return None

def extract_dopants(structure_name):
    """
    Extract dopants A and B and their atom counts from the structure name.

    Example inputs: Ba2Ti2, Y2Sc4, Y3

    - Ba2Ti2 => dopant A: Ba, count: 2. Dopant B: Ti, count: 2.
    - Y2Sc4  => dopant A: Y, count: 2. Dopant B: Sc, count: 4.
    - Y3     => dopant A: Y, count: 3. Dopant B: None, count: None.

    Returns: (dopant_A, amount_A, dopant_B, amount_B)
    """
    try:
        # Match an uppercase letter, an optional lowercase letter, and a count.
        # e.g. Ba2, Ti2, Y2, Sc4
        pattern = r'([A-Z][a-z]*)(\d+)'
        matches = re.findall(pattern, structure_name)
        logger.debug(f"matches: {matches}")

        # Keep elements other than Al and N.
        base_elements = {'Al', 'N'}

        dopants_with_amounts = []
        for element, amount_str in matches:
            if element not in base_elements:
                try:
                    amount = int(amount_str)
                    dopants_with_amounts.append((element, amount))
                except ValueError:
                    continue

        # Remove duplicates while preserving order.
        seen = set()
        unique_dopants = []
        for dopant, amount in dopants_with_amounts:
            if dopant not in seen:
                seen.add(dopant)
                unique_dopants.append((dopant, amount))

        dopant_A, amount_A = unique_dopants[0] if len(unique_dopants) > 0 else (None, None)
        dopant_B, amount_B = unique_dopants[1] if len(unique_dopants) > 1 else (None, None)
        return (dopant_A, amount_A, dopant_B, amount_B)

    except Exception as e:
        logger.error(f"Could not extract dopants from {structure_name}: {e}")
        return (None, None, None, None)

def process_scf_results(scf_folder_path="../scf", limit=5):
    """
    Process structures under ../scf.
    
    Example entries in the scf directory:
    POSCAR-Y3Al13N16-1
    POSCAR-Y3Al13N16-2
    POSCAR-Mg2Ti2Al12N16-3
    
    1. Extract dopants A and B from the structure name.
    2. Get the energy from qvasp -e and divide it by 16.
    3. Calculate unit cell volume from POSCAR.
    4. Calculate the RDF and extract its peak information.
    Save all results to results.json.
    
    Example results:
    {
        "POSCAR-Mg2Ti2Al12N16-3": {
            'dopants': {
                    'A': {
                        'element': dopant_A if dopant_A else None,
                        'amount': int(amount_A) if amount_A is not None else None
                    },
                    'B': {
                        'element': dopant_B if dopant_B else None,
                        'amount': int(amount_B) if amount_B is not None else None
                    }
                },
            "energy_per_16": 10.0,
            "volume": 10.0,
            "rdf_analysis": {
                "n_peaks": 1,
                "first_peak": {
                    "position": 1.0,
                    "height": 1.0
                },
                "second_peak": {
                    "position": 2.0,
                    "height": 2.0
                },
                "peak_ratio": 1.0,
                "u": 1.0
            },
            "lattice_a": 1.0,
            "lattice_c": 1.0,
            "c_over_a": 1.0
        },
        ...
    }
    """
    # Define paths.
    script_dir = Path(__file__).parent
    
    # Resolve the absolute path.
    script_dir = script_dir.resolve()
    
    logger.info("Scanning the scf directory for structures such as POSCAR-Mg2Ti2Al12N16-3")
    results = {}  # Store the results by structure.
    
    for scf_child_folder in os.listdir(scf_folder_path)[:limit]:
        logger.info(f"Processing {scf_child_folder}...")
        
        structure_name = scf_child_folder.split("-")[1].split("-")[0]
        logger.info(f"Structure name: {structure_name}")
        
        # Extract the dopant elements and their atom counts.
        dopant_A, amount_A, dopant_B, amount_B = extract_dopants(structure_name)
        logger.info(f"Dopant A: {dopant_A}, count: {amount_A}. Dopant B: {dopant_B}, count: {amount_B}")
        
        # Check that all required files are present.
        required_files = ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]
        missing_files = [f for f in required_files if not (Path(f"{scf_folder_path}/{scf_child_folder}/{f}").exists())]
        
        if missing_files:
            logger.error(f"Skipping {structure_name} because files are missing: {', '.join(missing_files)}")
            continue
        
        # Initialize result variables.
        energy_per_16 = None
        volume = None
        rdf_analysis = None
        lattice_a = None
        lattice_c = None
        c_over_a = None
        structure_features = {}
        
        # 1. Run qvasp -e to obtain the energy.
        logger.info(f"Running qvasp -e for {scf_child_folder}...")
        try:
            result = subprocess.run(
                ["qvasp", "-e"],
                cwd=str(f"{scf_folder_path}/{scf_child_folder}"),
                text=True,
                capture_output=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Check for the convergence marker in the qvasp output.
            is_covered = check_covered(output)
            
            if not is_covered:
                logger.warning(f"Skipping {structure_name} because it failed the COVERAGED check")
                continue
            
            # Extract the energy from the output.
            energy = extract_energy_from_output(output)
            if energy is None:
                logger.warning(f"Skipping {structure_name} because its energy could not be extracted")
                continue
            energy_per_16 = energy / 16.0
            logger.success(f"  {structure_name} passed the COVERAGED check. Energy/16 = {energy_per_16:.10f}")
            
        except Exception as e:
            logger.error(f"  Skipping {structure_name} after qvasp -e failed: {e}")
            continue
        
        POSCAR_PATH = Path(f"{scf_folder_path}/{scf_child_folder}/POSCAR")
        
        # 2. Calculate the unit cell volume from POSCAR.
        logger.info(f"  Calculating the unit cell volume for {structure_name}...")
        volume = calculate_volume_from_poscar(POSCAR_PATH)
        
        if volume is None:
            logger.warning(f"Skipping {structure_name} because its volume could not be calculated")
            continue
        
        logger.success(f"  {structure_name} volume = {volume:.10f} Å³")

        lattice_params = extract_lattice_constants(POSCAR_PATH)
        lattice_a = lattice_params['a']
        lattice_c = lattice_params['c']
        c_over_a = lattice_params['c_over_a']

        logger.info(f"  Calculating local structure descriptors for {structure_name}...")
        structure_features = calculate_structure_descriptors(POSCAR_PATH)
        
        # # 3. Calculate and analyze the RDF.
        logger.info(f"  Calculating the RDF for {structure_name}...")
        rdf_analysis = calculate_rdf_analysis(POSCAR_PATH)
        
        if rdf_analysis is None:
            logger.warning(f"Could not calculate the RDF for {structure_name}. Using default values")
            rdf_analysis = {
                'n_peaks': 0,
                'first_peak': None,
                'second_peak': None,
                'peak_ratio': None,
                'u': None
            }
        else:
            # Get the lattice parameters a, c, and c/a and calculate the internal coordinate u.
            u = None
            if lattice_c is not None and rdf_analysis.get('first_peak') and rdf_analysis['first_peak'].get('position'):
                peak1_pos = rdf_analysis['first_peak']['position']
                if peak1_pos > 0:
                    u = peak1_pos / lattice_c
                    if lattice_a is not None and c_over_a is not None:
                        logger.info(f"  {structure_name} lattice parameters: a = {lattice_a:.6f} Å, c = {lattice_c:.6f} Å, c/a = {c_over_a:.6f}, internal coordinate u = {u:.6f}")
                    else:
                        logger.info(f"  {structure_name} lattice parameter c = {lattice_c:.6f} Å, internal coordinate u = {u:.6f}")
                else:
                    logger.warning(f"  Could not calculate u for {structure_name} because Peak1_Pos is invalid")
            else:
                if lattice_c is None:
                    logger.warning(f"  Could not calculate u for {structure_name} because lattice parameter c could not be extracted")
                elif not rdf_analysis.get('first_peak'):
                    logger.warning(f"  Could not calculate u for {structure_name} because no first peak was found")
            
            rdf_analysis['u'] = u

        structure_features.update(
            {
                name: value
                for name, value in calculate_rdf_lattice_features(
                    rdf_analysis,
                    volume=volume,
                    c_over_a=c_over_a,
                ).items()
                if value is not None
            }
        )
            
        results[scf_child_folder] = {
            'dopants': {
                'A': {
                    'element': dopant_A if dopant_A else None,
                    'amount': int(amount_A) if amount_A is not None else None
                },
                'B': {
                    'element': dopant_B if dopant_B else None,
                    'amount': int(amount_B) if amount_B is not None else None
                }
            },
            'energy_per_16': energy_per_16,
            'volume_ang3': volume,
            'rdf_analysis': {
                    'n_peaks': rdf_analysis.get('n_peaks', 0),
                    'Peak1_Pos': rdf_analysis['first_peak']['position'] if rdf_analysis.get('first_peak') else None,
                    'Peak1_Height': rdf_analysis['first_peak']['height'] if rdf_analysis.get('first_peak') else None,
                    'Peak1_FWHM': rdf_analysis['first_peak']['fwhm'] if rdf_analysis.get('first_peak') else None,
                    'Peak2_Pos': rdf_analysis['second_peak']['position'] if rdf_analysis.get('second_peak') else None,
                    'Peak2_Height': rdf_analysis['second_peak']['height'] if rdf_analysis.get('second_peak') else None,
                    'Peak2_FWHM': rdf_analysis['second_peak']['fwhm'] if rdf_analysis.get('second_peak') else None,
                    'Peak2_Pos_Peak1_Pos': rdf_analysis.get('peak_ratio'),
                    'u': rdf_analysis.get('u')
                },
            'lattice_a': lattice_a,
            'lattice_c': lattice_c,
            'c_over_a': c_over_a,
            'structure_features': structure_features
        }
        
        
    return results
        
    
    
    
    # # Iterate over parent directories such as Cav-w and CaTa-w.
    # for parent_folder in script_dir.iterdir():
    #     if not parent_folder.is_dir():
    #         continue
        
    #     parent_name = parent_folder.name
        
    #     # Skip the cohp and bulk parent directories.
    #     if parent_name.lower() in ['cohp', 'bulk']:
    #         logger.info(f"Skipping parent directory: {parent_name}")
    #         continue
        
    #     # Iterate over child directories such as 10Ca and 8Ca.
    #     for child_folder in parent_folder.iterdir():
    #         if not child_folder.is_dir():
    #             continue
            
    #         child_name = child_folder.name
            
    #         # Path to the scf directory.
    #         scf_folder_path = child_folder / "scf"
            
    #         # Check whether the scf directory exists.
    #         if not scf_folder_path.exists() or not scf_folder_path.is_dir():
    #             logger.warning(f"Skipping {parent_name}/{child_name} because the scf directory is missing")
    #             continue
            
    #         # Derive the structure name from the parent and child directory names.
    #         structure_name = extract_structure_name(parent_name, child_name)
    #         poscar_path = scf_folder_path / "POSCAR"
            
    #         # Extract the dopant elements and their atom counts.
    #         dopant_A, amount_A, dopant_B, amount_B = extract_dopants(structure_name)
            
    #         # Check that all required files are present.
    #         required_files = ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]
    #         missing_files = [f for f in required_files if not (scf_folder_path / f).exists()]
            
    #         if missing_files:
    #             logger.error(f"Skipping {structure_name} because files are missing: {', '.join(missing_files)}")
    #             continue
            
    #         # Initialize result variables.
    #         energy_per_16 = None
    #         volume = None
    #         rdf_analysis = None
    #         lattice_a = None
    #         lattice_c = None
    #         c_over_a = None
    #         c33_matrix = None
    #         e33_matrix1 = None
    #         e33_matrix2 = None
    #         e33_matrix = None
    #         d33 = None
            
    #         # 1. Run qvasp -e to obtain the energy.
    #         logger.info(f"Running qvasp -e for {structure_name}...")
    #         try:
    #             result = subprocess.run(
    #                 ["qvasp", "-e"],
    #                 cwd=str(scf_folder_path),
    #                 text=True,
    #                 capture_output=True,
    #                 timeout=60
    #             )
                
    #             output = result.stdout + result.stderr
                
    #             # Check for the convergence marker in the qvasp output.
    #             is_covered = check_covered(output)
                
    #             if not is_covered:
    #                 logger.warning(f"Skipping {structure_name} because it failed the COVERAGED check")
    #                 continue
                
    #             # Extract the energy from the output.
    #             energy = extract_energy_from_output(output)
    #             if energy is None:
    #                 logger.warning(f"Skipping {structure_name} because its energy could not be extracted")
    #                 continue
    #             energy_per_16 = energy / 16.0
    #             logger.success(f"  {structure_name} passed the COVERAGED check. Energy/16 = {energy_per_16:.10f}")
                
    #         except Exception as e:
    #             logger.error(f"  Skipping {structure_name} after qvasp -e failed: {e}")
    #             continue
            
    #         # 2. Calculate the unit cell volume from POSCAR.
    #         logger.info(f"  Calculating the unit cell volume for {structure_name}...")
    #         volume = calculate_volume_from_poscar(poscar_path)
            
    #         if volume is None:
    #             logger.warning(f"Skipping {structure_name} because its volume could not be calculated")
    #             continue
            
    #         logger.success(f"  {structure_name} volume = {volume:.10f} Å³")
            
    #         # 3. Calculate and analyze the RDF.
    #         logger.info(f"  Calculating the RDF for {structure_name}...")
    #         rdf_analysis = calculate_rdf_analysis(poscar_path)
            
    #         if rdf_analysis is None:
    #             logger.warning(f"Could not calculate the RDF for {structure_name}. Using default values")
    #             rdf_analysis = {
    #                 'n_peaks': 0,
    #                 'first_peak': None,
    #                 'second_peak': None,
    #                 'peak_ratio': None,
    #                 'u': None
    #             }
    #         else:
    #             # Get the lattice parameters a, c, and c/a and calculate the internal coordinate u.
    #             lattice_params = extract_lattice_constants(poscar_path)
    #             lattice_a = lattice_params['a']
    #             lattice_c = lattice_params['c']
    #             c_over_a = lattice_params['c_over_a']
                
    #             u = None
    #             if lattice_c is not None and rdf_analysis.get('first_peak') and rdf_analysis['first_peak'].get('position'):
    #                 peak1_pos = rdf_analysis['first_peak']['position']
    #                 if peak1_pos > 0:
    #                     u = peak1_pos / lattice_c
    #                     if lattice_a is not None and c_over_a is not None:
    #                         logger.info(f"  {structure_name} lattice parameters: a = {lattice_a:.6f} Å, c = {lattice_c:.6f} Å, c/a = {c_over_a:.6f}, internal coordinate u = {u:.6f}")
    #                     else:
    #                         logger.info(f"  {structure_name} lattice parameter c = {lattice_c:.6f} Å, internal coordinate u = {u:.6f}")
    #                 else:
    #                     logger.warning(f"  Could not calculate u for {structure_name} because Peak1_Pos is invalid")
    #             else:
    #                 if lattice_c is None:
    #                     logger.warning(f"  Could not calculate u for {structure_name} because lattice parameter c could not be extracted")
    #                 elif not rdf_analysis.get('first_peak'):
    #                     logger.warning(f"  Could not calculate u for {structure_name} because no first peak was found")
                
    #             rdf_analysis['u'] = u
            
    #         # Extract lattice parameters separately if the RDF analysis did not provide them.
    #         if lattice_a is None or lattice_c is None:
    #             lattice_params = extract_lattice_constants(poscar_path)
    #             lattice_a = lattice_params['a']
    #             lattice_c = lattice_params['c']
    #             c_over_a = lattice_params['c_over_a']
                
    #             if rdf_analysis['n_peaks'] > 0:
    #                 peak_info = f"Found {rdf_analysis['n_peaks']} peaks"
    #                 if rdf_analysis.get('first_peak'):
    #                     peak_info += f", first peak position: {rdf_analysis['first_peak']['position']:.3f} Å"
    #                 if rdf_analysis.get('second_peak'):
    #                     peak_info += f", second peak position: {rdf_analysis['second_peak']['position']:.3f} Å"
    #                 if rdf_analysis.get('peak_ratio'):
    #                     peak_info += f", peak ratio: {rdf_analysis['peak_ratio']:.3f}"
    #                 if u is not None:
    #                     peak_info += f", internal coordinate u: {u:.6f}"
    #                 logger.success(f"  Completed RDF analysis for {structure_name}. {peak_info}")
    #             else:
    #                 logger.warning(f"  RDF analysis found no peaks for {structure_name}")
            
    #         # 4. Run qvasp -elasticd in the pie directory to obtain the elastic moduli.
    #         pie_folder_path = child_folder / "pie"
    #         if pie_folder_path.exists() and pie_folder_path.is_dir():
    #             logger.info(f"  Extracting elastic moduli for {structure_name} from the pie directory...")
    #             try:
    #                 result = subprocess.run(
    #                     ["qvasp", "-elasticd"],
    #                     cwd=str(pie_folder_path),
    #                     text=True,
    #                     capture_output=True,
    #                     timeout=120
    #                 )
                    
    #                 output = result.stdout + result.stderr
                    
    #                 # Extract the elastic moduli matrix from the output.
    #                 c33_matrix = extract_elastic_moduli(output)
                    
    #                 if c33_matrix is not None:
    #                     logger.success(f"  Extracted the elastic moduli matrix for {structure_name}")
    #                 else:
    #                     logger.warning(f"  Could not extract the elastic moduli matrix for {structure_name}")
                        
    #             except Exception as e:
    #                 logger.error(f"  qvasp -elasticd failed for {structure_name}: {e}")
    #         else:
    #             logger.info(f"  Skipping elastic moduli for {structure_name} because the pie directory is missing")
            
    #         # 5. Extract the piezoelectric tensor from OUTCAR in the eij-born directory.
    #         eij_born_folder_path = child_folder / "eij-born"
    #         if eij_born_folder_path.exists() and eij_born_folder_path.is_dir():
    #             outcar_path = eij_born_folder_path / "OUTCAR"
    #             if outcar_path.exists():
    #                 logger.info(f"  Extracting the piezoelectric tensor for {structure_name} from the eij-born directory...")
    #                 # Extract the piezoelectric tensor including local field effects.
    #                 e33_matrix1 = extract_piezoelectric_tensor(outcar_path, tensor_type='local_field')
    #                 # Extract the ionic contribution to the piezoelectric tensor.
    #                 e33_matrix2 = extract_piezoelectric_tensor(outcar_path, tensor_type='ionic_contr')
                    
    #                 if e33_matrix1 is not None:
    #                     logger.success(f"  Extracted the piezoelectric tensor including local field effects for {structure_name}")
    #                 else:
    #                     logger.warning(f"  Could not extract the piezoelectric tensor including local field effects for {structure_name}")
                    
    #                 if e33_matrix2 is not None:
    #                     logger.success(f"  Extracted the ionic contribution to the piezoelectric tensor for {structure_name}")
    #                 else:
    #                     logger.warning(f"  Could not extract the ionic contribution to the piezoelectric tensor for {structure_name}")
    #             else:
    #                 logger.warning(f"  OUTCAR is missing from the eij-born directory for {structure_name}")
    #         else:
    #             logger.info(f"  Skipping the piezoelectric tensor for {structure_name} because the eij-born directory is missing")
            
    #         # 6. Calculate the e33 matrix and d33.
    #         if e33_matrix1 is not None and e33_matrix2 is not None:
    #             logger.info(f"  Calculating the e33 matrix and d33 for {structure_name}...")
    #             e33_matrix = calculate_e33_matrix(e33_matrix1, e33_matrix2)
    #             if e33_matrix is not None:
    #                 logger.success(f"  Calculated the e33 matrix for {structure_name}")
    #             else:
    #                 logger.warning(f"  Could not calculate the e33 matrix for {structure_name}")
                
    #             # Calculate d33.
    #             if c33_matrix is not None and e33_matrix is not None:
    #                 d33 = calculate_d33(c33_matrix, e33_matrix)
    #                 if d33 is not None:
    #                     logger.success(f"  {structure_name} d33 = {d33:.6f}")
    #                 else:
    #                     logger.warning(f"  Could not calculate d33 for {structure_name}")
    #             else:
    #                 if c33_matrix is None:
    #                     logger.warning(f"  Could not calculate d33 for {structure_name} because the c33 matrix is missing")
    #                 if e33_matrix is None:
    #                     logger.warning(f"  Could not calculate d33 for {structure_name} because the e33 matrix is missing")
    #         else:
    #             if e33_matrix1 is None or e33_matrix2 is None:
    #                 logger.info(f"  Skipping e33 and d33 for {structure_name} because the piezoelectric tensor matrices are missing")
            
    #         # Build the results dictionary with all RDF features.
    #         structure_result = {
    #             'dopants': {
    #                 'A': {
    #                     'element': dopant_A if dopant_A else None,
    #                     'amount': int(amount_A) if amount_A is not None else None
    #                 },
    #                 'B': {
    #                     'element': dopant_B if dopant_B else None,
    #                     'amount': int(amount_B) if amount_B is not None else None
    #                 }
    #             },
    #             'energy_per_16': float(energy_per_16),
    #             'volume_ang3': float(volume),
    #             'lattice_a': float(lattice_a) if lattice_a is not None else None,
    #             'lattice_c': float(lattice_c) if lattice_c is not None else None,
    #             'c_over_a': float(c_over_a) if c_over_a is not None else None,
    #             'rdf_analysis': {
    #                 'n_peaks': rdf_analysis.get('n_peaks', 0),
    #                 'Peak1_Pos': rdf_analysis['first_peak']['position'] if rdf_analysis.get('first_peak') else None,
    #                 'Peak1_Height': rdf_analysis['first_peak']['height'] if rdf_analysis.get('first_peak') else None,
    #                 'Peak1_FWHM': rdf_analysis['first_peak']['fwhm'] if rdf_analysis.get('first_peak') else None,
    #                 'Peak2_Pos': rdf_analysis['second_peak']['position'] if rdf_analysis.get('second_peak') else None,
    #                 'Peak2_Height': rdf_analysis['second_peak']['height'] if rdf_analysis.get('second_peak') else None,
    #                 'Peak2_FWHM': rdf_analysis['second_peak']['fwhm'] if rdf_analysis.get('second_peak') else None,
    #                 'Peak2_Pos_Peak1_Pos': rdf_analysis.get('peak_ratio'),
    #                 'u': rdf_analysis.get('u')
    #             },
    #             'c33': c33_matrix,
    #             'e33_matrix1': e33_matrix1,
    #             'e33_matrix2': e33_matrix2,
    #             'e33': e33_matrix,
    #             'd33': d33
    #         }
            
    #         results[structure_name] = structure_result
    
    # # Save results to JSON.
    # json_path = script_dir / "results.json"
    # try:
    #     with open(json_path, 'w', encoding='utf-8') as f:
    #         json.dump(results, f, indent=4, ensure_ascii=False)
        
    #     logger.info(f"\nSaved results for {len(results)} structures to {json_path}")
    # except Exception as e:
    #     logger.error(f"Could not save the JSON file: {e}")


def process_c33_results(c33_folder_path="../c33", limit=5):
    """
    Run qvasp -elasticd for each structure under ../c33 and save the elastic moduli matrices to results_c33.json.
    The first column contains the structure name and the second contains the elastic moduli matrix.
    """
    results_c33 = {}
    for c33_child_folder in os.listdir(c33_folder_path)[:limit]:
        logger.info(f"Processing C33: {c33_child_folder}...")
        structure_name = c33_child_folder.split("-")[1].split("-")[0].split("Al")[0]
        logger.info(f"Structure name: {structure_name}")
        try:
            result = subprocess.run(
                ["qvasp", "-elasticd"],
                cwd=str(f"{c33_folder_path}/{c33_child_folder}"),
                text=True,
                capture_output=True,
                timeout=120
            )
            
            output = result.stdout + result.stderr
            
            # Extract the elastic moduli matrix from the output.
            c33_matrix = extract_elastic_moduli(output)
            
            if c33_matrix is not None:
                logger.success(f"Extracted the elastic moduli matrix for {structure_name}: {c33_matrix}")
                results_c33[c33_child_folder] = c33_matrix
            else:
                logger.warning(f"Could not extract the elastic moduli matrix for {structure_name}")
            
        except Exception as e:
            logger.error(f"  qvasp -elasticd failed for {structure_name}: {e}")
        # c33_matrix = extract_c33_matrix(Path(f"{c33_folder_path}/{c33_child_folder}/c33.txt"))
        # logger.info(f"Elastic moduli matrix: {c33_matrix}")
        # results_c33[structure_name] = c33_matrix
    return results_c33

def process_e33_results(e33_folder_path="../e33", limit=5):
    """
    Run qvasp -e33 for each structure under ../e33 and save the piezoelectric tensors to results_e33.json.
    The first column contains the structure name and the second contains the piezoelectric tensor.
    """
    results_e33 = {}
    for e33_child_folder in os.listdir(e33_folder_path)[:limit]:
        logger.info(f"Processing E33: {e33_child_folder}...")
        structure_name = e33_child_folder.split("-")[1].split("-")[0].split("Al")[0]
        logger.info(f"Structure name: {structure_name}")
        outcar_path = Path(f"{e33_folder_path}/{e33_child_folder}/OUTCAR")
        try:
            logger.info(f"Processing the piezoelectric tensor for {structure_name}")
            # Extract the piezoelectric tensor including local field effects.
            e33_matrix1 = extract_piezoelectric_tensor(outcar_path, tensor_type='local_field')
            # Extract the ionic contribution to the piezoelectric tensor.
            e33_matrix2 = extract_piezoelectric_tensor(outcar_path, tensor_type='ionic_contr')
            
            if e33_matrix1 is not None:
                logger.success(f"Extracted the piezoelectric tensor including local field effects for {structure_name}: {e33_matrix1}")
            else:
                logger.warning(f"  Could not extract the piezoelectric tensor including local field effects for {structure_name}")
            
            if e33_matrix2 is not None:
                logger.success(f"Extracted the ionic contribution to the piezoelectric tensor for {structure_name}: {e33_matrix2}")
            else:
                logger.warning(f"  Could not extract the ionic contribution to the piezoelectric tensor for {structure_name}")
                
            e33_matrix = calculate_e33_matrix(e33_matrix1, e33_matrix2)
            if e33_matrix is not None:
                logger.success(f"Calculated the e33 matrix for {structure_name}: {e33_matrix}")
                results_e33[e33_child_folder] = e33_matrix
            else:
                logger.warning(f"  Could not calculate the e33 matrix for {structure_name}")
                
            
        except Exception as e:
            logger.error(f"  qvasp -e33 failed for {structure_name}: {e}")
    return results_e33

def process_results(
    scf_folder_path="../scf", 
    c33_folder_path="../c33", 
    e33_folder_path="../e33",
    limit=9999
    ):
    final_results = {}
    results_scf = process_scf_results(scf_folder_path, limit)
    results_c33 = process_c33_results(c33_folder_path, limit)
    results_e33 = process_e33_results(e33_folder_path, limit)
    for structure_name in results_scf.keys():
        
        if structure_name in results_c33.keys() and structure_name in results_e33.keys():
            c33_matrix = results_c33[structure_name]
            e33_matrix = results_e33[structure_name]
            try:
                final_results[structure_name] = results_scf[structure_name].copy()
                final_results[structure_name]['c33'] = c33_matrix
                final_results[structure_name]['e33'] = e33_matrix
                d33 = calculate_d33(c33_matrix, e33_matrix)
                final_results[structure_name]['d33'] = d33
                logger.success(f"Calculated d33 for {structure_name}: {d33}")
            except Exception as e:
                logger.error(f"Could not calculate d33 for {structure_name}: {e}")
        else:
            logger.error(f"Skipping d33 for {structure_name} because the c33 or e33 matrix is missing")
    with open('final_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    logger.success(f"Saved the final results to final_results.json")
    return final_results
    

def process_energy():
    """
    Run qvasp -e for each structure under ../scf and save each structure name and its energy divided by 16 to results_scf_energy.csv.
    The first column contains the structure name and the second contains the energy divided by 16.
    Deprecated. Use process_results() instead.
    """
    logger.warning("process_energy() is deprecated. Use process_results() instead.")
    process_results()

def calculate_volume_from_poscar(poscar_path):
    """
    Read the lattice vectors from POSCAR and calculate the unit cell volume.
    POSCAR format:
    Line 1: Description
    Line 2: Scaling factor
    Lines 3–5: Lattice vectors (one vector of three numbers per line)
    
    Volume = |det(lattice_vectors)| = |a · (b × c)|
    """
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip line 1 (description).
        # Read the scaling factor from line 2.
        scale_factor = float(lines[1].strip())
        
        # Read the lattice vectors from lines 3–5.
        lattice_vectors = []
        for i in range(2, 5):  # Lines 3–5 (indices 2–4).
            vector = [float(x) for x in lines[i].split()]
            if len(vector) != 3:
                raise ValueError(f"Expected 3 numbers in the lattice vector, found {len(vector)}")
            lattice_vectors.append(vector)
        
        # Convert the vectors to a NumPy array.
        lattice_matrix = np.array(lattice_vectors)
        
        # Calculate the volume as |det(lattice_vectors)| * scale_factor^3.
        volume = abs(np.linalg.det(lattice_matrix)) * (scale_factor ** 3)
        
        return volume
    except Exception as e:
        logger.error(f"Could not read POSCAR file {poscar_path}: {e}")
        return None

def extract_lattice_constant_c(poscar_path):
    """
    Extract the lattice parameter c from POSCAR.
    
    POSCAR format:
    Line 1: Description
    Line 2: Scaling factor
    Lines 3–5: Lattice vectors (one vector of three numbers per line)
    
    The lattice parameter c is the magnitude of the third lattice vector (line 5, index 4) multiplied by the scaling factor.
    
    Args:
        poscar_path: Path to the POSCAR file.
    
    Returns:
        float: The lattice parameter c in Å, or None if extraction fails.
    """
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Read the scaling factor from line 2.
        scale_factor = float(lines[1].strip())
        
        # Read the third lattice vector from line 5 (index 4), typically along c.
        c_vector = [float(x) for x in lines[4].split()]
        if len(c_vector) != 3:
            raise ValueError(f"Expected 3 numbers in the lattice vector, found {len(c_vector)}")
        
        # Calculate the magnitude of the c vector.
        c_vector_np = np.array(c_vector)
        c_length = np.linalg.norm(c_vector_np)
        
        # Multiply by the scaling factor to get the lattice parameter.
        c = c_length * scale_factor
        
        return float(c)
    except Exception as e:
        logger.error(f"Could not extract lattice parameter c from {poscar_path}: {e}")
        return None

def extract_lattice_constant_a(poscar_path):
    """
    Extract the lattice parameter a from POSCAR.
    
    POSCAR format:
    Line 1: Description
    Line 2: Scaling factor
    Lines 3–5: Lattice vectors (one vector of three numbers per line)
    
    The lattice parameter a is the magnitude of the first lattice vector (line 3, index 2) multiplied by the scaling factor.
    
    Args:
        poscar_path: Path to the POSCAR file.
    
    Returns:
        float: The lattice parameter a in Å, or None if extraction fails.
    """
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Read the scaling factor from line 2.
        scale_factor = float(lines[1].strip())
        
        # Read the first lattice vector from line 3 (index 2), typically along a.
        a_vector = [float(x) for x in lines[2].split()]
        if len(a_vector) != 3:
            raise ValueError(f"Expected 3 numbers in the lattice vector, found {len(a_vector)}")
        
        # Calculate the magnitude of the a vector.
        a_vector_np = np.array(a_vector)
        a_length = np.linalg.norm(a_vector_np)
        
        # Multiply by the scaling factor to get the lattice parameter.
        a = a_length * scale_factor
        
        return float(a)
    except Exception as e:
        logger.error(f"Could not extract lattice parameter a from {poscar_path}: {e}")
        return None

def extract_lattice_constants(poscar_path):
    """
    Extract lattice parameters a and c from POSCAR and calculate c/a.
    
    Args:
        poscar_path: Path to the POSCAR file.
    
    Returns:
        dict: Values for 'a', 'c', and 'c_over_a'. Missing values are None.
    """
    a = extract_lattice_constant_a(poscar_path)
    c = extract_lattice_constant_c(poscar_path)
    
    c_over_a = None
    if a is not None and c is not None and a > 0:
        c_over_a = c / a
    
    return {
        'a': a,
        'c': c,
        'c_over_a': c_over_a
    }

def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def safe_div(numerator, denominator):
    numerator = safe_float(numerator)
    denominator = safe_float(denominator)
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def finite_stats(values, prefix):
    clean = [float(v) for v in values if safe_float(v) is not None]
    if not clean:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_range": None,
        }
    arr = np.asarray(clean, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
    }


def calculate_rdf_lattice_features(rdf_analysis, volume=None, c_over_a=None):
    rdf_analysis = rdf_analysis or {}
    first_peak = rdf_analysis.get('first_peak') or {}
    second_peak = rdf_analysis.get('second_peak') or {}
    peak1_pos = safe_float(first_peak.get('position', rdf_analysis.get('Peak1_Pos')))
    peak1_height = safe_float(first_peak.get('height', rdf_analysis.get('Peak1_Height')))
    peak1_fwhm = safe_float(first_peak.get('fwhm', rdf_analysis.get('Peak1_FWHM')))
    peak2_pos = safe_float(second_peak.get('position', rdf_analysis.get('Peak2_Pos')))
    peak2_height = safe_float(second_peak.get('height', rdf_analysis.get('Peak2_Height')))
    peak2_fwhm = safe_float(second_peak.get('fwhm', rdf_analysis.get('Peak2_FWHM')))
    n_peaks = safe_float(rdf_analysis.get('n_peaks'))
    u = safe_float(rdf_analysis.get('u'))
    volume = safe_float(volume)
    c_over_a = safe_float(c_over_a)

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


def _neighbor_distance(neighbor):
    return safe_float(getattr(neighbor, 'nn_distance', getattr(neighbor, 'distance', None)))


def get_species_neighbors(structure, center_index, target_symbol='N', radius=3.0, max_neighbors=None):
    center_site = structure[center_index]
    neighbors = []
    for neighbor in structure.get_neighbors(center_site, radius):
        symbol = getattr(neighbor.specie, 'symbol', str(neighbor.specie))
        distance = _neighbor_distance(neighbor)
        if symbol == target_symbol and distance is not None and distance > 0.01:
            neighbors.append((neighbor, distance))
    neighbors.sort(key=lambda item: item[1])
    if max_neighbors is not None:
        neighbors = neighbors[:max_neighbors]
    return neighbors


def angle_degrees(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return None
    cos_value = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(np.degrees(np.arccos(cos_value)))


def calculate_structure_descriptors(poscar_path, neighbor_radius=3.0):
    """
    POSCAR-level descriptors for local cation-N geometry. These are intended
    to distinguish same-composition variants that RDF/composition features
    cannot separate well.
    """
    if not PYMATGEN_AVAILABLE:
        return {}
    try:
        structure = Structure.from_file(str(poscar_path))
    except Exception as e:
        logger.warning(f"Could not read POSCAR file {poscar_path} for structure descriptors: {e}")
        return {}

    species = [getattr(site.specie, 'symbol', str(site.specie)) for site in structure]
    cation_indices = [i for i, symbol in enumerate(species) if symbol != 'N']
    ideal_tetra_angle = 109.471220634

    cation_bonds = []
    al_bonds = []
    dopant_bonds = []
    coord_counts = []
    angle_values = []
    angle_devs = []
    offcenters = []

    for center_index in cation_indices:
        neighbors = get_species_neighbors(
            structure,
            center_index,
            target_symbol='N',
            radius=neighbor_radius,
            max_neighbors=4,
        )
        if not neighbors:
            continue
        distances = [distance for _, distance in neighbors]
        coord_counts.append(len(distances))
        cation_bonds.extend(distances)
        if species[center_index] == 'Al':
            al_bonds.extend(distances)
        else:
            dopant_bonds.extend(distances)

        center_coords = np.asarray(structure[center_index].coords, dtype=float)
        vectors = [np.asarray(neighbor.coords, dtype=float) - center_coords for neighbor, _ in neighbors]
        if len(vectors) >= 2:
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    angle = angle_degrees(vectors[i], vectors[j])
                    if angle is not None:
                        angle_values.append(angle)
                        angle_devs.append(abs(angle - ideal_tetra_angle))
        if len(vectors) >= 3:
            neighbor_coords = np.asarray([neighbor.coords for neighbor, _ in neighbors], dtype=float)
            centroid = np.mean(neighbor_coords, axis=0)
            mean_distance = float(np.mean(distances))
            if mean_distance > 1e-12:
                offcenters.append(float(np.linalg.norm(center_coords - centroid) / mean_distance))

    features = {}
    features.update(finite_stats(cation_bonds, 'cation_n_bond'))
    features.update(finite_stats(al_bonds, 'al_n_bond'))
    features.update(finite_stats(dopant_bonds, 'dopant_n_bond'))
    features.update(finite_stats(coord_counts, 'cation_n_coord'))
    features.update(finite_stats(angle_values, 'tetra_angle'))
    features.update(finite_stats(angle_devs, 'tetra_angle_dev'))
    features.update(finite_stats(offcenters, 'cation_offcenter'))

    features['cation_n_bond_cv'] = safe_div(
        features.get('cation_n_bond_std'),
        features.get('cation_n_bond_mean'),
    )
    features['dopant_n_bond_delta'] = (
        features.get('dopant_n_bond_mean') - features.get('al_n_bond_mean')
        if features.get('dopant_n_bond_mean') is not None and features.get('al_n_bond_mean') is not None
        else None
    )

    try:
        lengths = structure.lattice.abc
        angles = structure.lattice.angles
        features['lattice_a_len'] = float(lengths[0])
        features['lattice_b_len'] = float(lengths[1])
        features['lattice_c_len'] = float(lengths[2])
        features['lattice_alpha'] = float(angles[0])
        features['lattice_beta'] = float(angles[1])
        features['lattice_gamma'] = float(angles[2])
        features['lattice_angle_deviation'] = float(np.mean(np.abs(np.asarray(angles) - 90.0)))
    except Exception:
        pass

    if SpacegroupAnalyzer is not None:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            features['sg_number'] = int(analyzer.get_space_group_number())
        except Exception:
            features['sg_number'] = None

    return features

def calculate_rdf_analysis(poscar_path, r_max=6.0, dr=0.02):
    """
    Calculate the radial distribution function (RDF) with pymatgen and extract peak information.
    
    Args:
        poscar_path: Path to the POSCAR file.
        r_max: Maximum distance in Å (default: 6.0).
        dr: Distance resolution in Å (default: 0.02).
    
    Returns a dictionary with these keys:
        - n_peaks: Number of peaks.
        - first_peak: Position, height, and FWHM of the first peak.
        - second_peak: Position, height, and FWHM of the second peak.
        - peak_ratio: Structural ratio Peak2_Pos / Peak1_Pos.
        Returns None if calculation fails or no peaks are found.
    """
    if not PYMATGEN_AVAILABLE:
        logger.error("pymatgen is required for RDF calculations")
        return None
    
    try:
        # Read the structure with pymatgen.
        structure = Structure.from_file(str(poscar_path))
        
        # Get the Cartesian coordinates of all atoms.
        coords = np.array([site.coords for site in structure])
        n_atoms = len(coords)
        
        # Get the lattice matrix.
        lattice_matrix = np.array(structure.lattice.matrix)
        
        # Calculate the volume.
        volume = structure.volume
        
        # Calculate pair distances using periodic boundary conditions.
        # Work in fractional coordinates.
        coords_frac = np.array([site.frac_coords for site in structure])
        
        # Calculate fractional coordinate differences for all atom pairs.
        diff_frac = coords_frac[:, np.newaxis, :] - coords_frac[np.newaxis, :, :]  # (n, n, 3)
        
        # Apply periodic boundaries by wrapping fractional differences into [-0.5, 0.5].
        diff_frac = diff_frac - np.round(diff_frac)
        
        # Convert fractional differences to Cartesian differences.
        diff_cartesian = diff_frac @ lattice_matrix  # (n, n, 3)
        
        # Calculate the distance matrix.
        distances_matrix = np.linalg.norm(diff_cartesian, axis=2)  # (n, n)
        
        # Keep the upper triangle above the diagonal to exclude self-pairs and duplicates.
        triu_indices = np.triu_indices(n_atoms, k=1)
        distances = distances_matrix[triu_indices]
        
        # Filter the distances.
        valid_mask = (distances > 0.01) & (distances < r_max)
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            logger.warning(f"No valid atom pair distances found for {poscar_path}")
            return None
        
        # Create distance bins.
        r = np.arange(0, r_max + dr, dr)
        n_bins = len(r)
        
        # Bin the distances with numpy.histogram.
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max + dr))
        
        # Use the bin centers as r values.
        r = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        g_r = hist.astype(float)
        
        # Normalize by spherical shell volume and atomic density.
        rho = n_atoms / volume  # Atomic density.
        shell_volumes = 4.0 * np.pi * r ** 2 * dr
        # Avoid division by zero.
        valid_shell = shell_volumes > 0
        g_r[valid_shell] = g_r[valid_shell] / (rho * shell_volumes[valid_shell] * n_atoms)
        g_r[~valid_shell] = 0.0
        
        # Extract the peaks.
        peak_info = extract_rdf_peaks(r, g_r)
        
        if peak_info is None or peak_info['n_peaks'] == 0:
            logger.warning(f"No RDF peaks found for {poscar_path}")
            return {
                'n_peaks': 0,
                'first_peak': None,
                'second_peak': None,
                'peak_ratio': None
            }
        
        # Build the result dictionary.
        result = {
            'n_peaks': peak_info['n_peaks'],
            'first_peak': None,
            'second_peak': None,
            'peak_ratio': None
        }
        
        # Record the first peak.
        if len(peak_info['peak_positions']) > 0:
            first_peak = {
                'position': float(peak_info['peak_positions'][0]),
                'height': float(peak_info['peak_heights'][0]),
                'fwhm': float(peak_info['fwhm'][0]) if len(peak_info['fwhm']) > 0 else None
            }
            result['first_peak'] = first_peak
        
        # Record the second peak.
        if len(peak_info['peak_positions']) > 1:
            second_peak = {
                'position': float(peak_info['peak_positions'][1]),
                'height': float(peak_info['peak_heights'][1]),
                'fwhm': float(peak_info['fwhm'][1]) if len(peak_info['fwhm']) > 1 else None
            }
            result['second_peak'] = second_peak
            
            # Calculate the structural ratio Peak2_Pos / Peak1_Pos.
            if result['first_peak'] and result['first_peak']['position'] > 0:
                result['peak_ratio'] = float(second_peak['position'] / result['first_peak']['position'])
        
        return result
        
    except Exception as e:
        logger.error(f"Could not calculate the RDF for {poscar_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def extract_rdf_peaks(r, g_r, min_height=None, min_distance=0.5):
    """
    Extract peak positions, heights, and full widths at half maximum (FWHM) from an RDF curve.
    
    Args:
        r: Array of distances.
        g_r: Array of RDF values.
        min_height: Minimum peak height. Defaults to 0.1 times the maximum RDF value.
        min_distance: Minimum separation between peaks in Å (default: 0.5).
    
    Returns a dictionary with these keys:
        - peak_positions: Array of peak positions in Å.
        - peak_heights: Array of peak heights.
        - fwhm: Array of full widths at half maximum in Å.
        - n_peaks: Number of peaks.
    """
    try:
        if r is None or g_r is None or len(r) == 0 or len(g_r) == 0:
            return None
        
        # Set default parameters.
        if min_height is None:
            min_height = np.max(g_r) * 0.1  # Use 10% of the maximum RDF value.
        
        # Calculate the distance resolution.
        dr = r[1] - r[0] if len(r) > 1 else 0.01
        
        # Detect peaks as local maxima.
        peaks = []
        peak_heights = []
        
        for i in range(1, len(g_r) - 1):
            # Check whether this point is a local maximum.
            if g_r[i] > g_r[i-1] and g_r[i] > g_r[i+1] and g_r[i] > min_height:
                # Check the distance to peaks already found.
                is_valid = True
                for peak_idx in peaks:
                    if abs(r[i] - r[peak_idx]) < min_distance:
                        # Keep the higher peak if two are too close.
                        if g_r[i] > g_r[peak_idx]:
                            peaks.remove(peak_idx)
                            peak_heights.remove(g_r[peak_idx])
                        else:
                            is_valid = False
                        break
                
                if is_valid:
                    peaks.append(i)
                    peak_heights.append(g_r[i])
        
        if len(peaks) == 0:
            return {
                'peak_positions': np.array([]),
                'peak_heights': np.array([]),
                'fwhm': np.array([]),
                'n_peaks': 0
            }
        
        # Sort peaks by position.
        sorted_indices = sorted(range(len(peaks)), key=lambda i: r[peaks[i]])
        peaks = [peaks[i] for i in sorted_indices]
        peak_heights = [peak_heights[i] for i in sorted_indices]
        
        peak_positions = r[peaks]
        peak_heights = np.array(peak_heights)
        
        # Calculate the full width at half maximum (FWHM).
        fwhm = []
        for peak_idx in peaks:
            peak_height = g_r[peak_idx]
            half_max = peak_height / 2.0
            
            # Find where the curve crosses half maximum on each side.
            # Left side.
            left_idx = peak_idx
            while left_idx > 0 and g_r[left_idx] > half_max:
                left_idx -= 1
            
            # Right side.
            right_idx = peak_idx
            while right_idx < len(g_r) - 1 and g_r[right_idx] > half_max:
                right_idx += 1
            
            # Calculate FWHM.
            if left_idx < peak_idx and right_idx > peak_idx:
                # Use linear interpolation to estimate each crossing.
                # Interpolate on the left.
                if left_idx < peak_idx - 1:
                    y1, y2 = g_r[left_idx], g_r[left_idx + 1]
                    if y2 > y1:
                        t = (half_max - y1) / (y2 - y1) if (y2 - y1) > 1e-10 else 0
                        left_pos = r[left_idx] + t * (r[left_idx + 1] - r[left_idx])
                    else:
                        left_pos = r[left_idx]
                else:
                    left_pos = r[left_idx]
                
                # Interpolate on the right.
                if right_idx > peak_idx + 1:
                    y1, y2 = g_r[right_idx - 1], g_r[right_idx]
                    if y1 > y2:
                        t = (half_max - y2) / (y1 - y2) if (y1 - y2) > 1e-10 else 0
                        right_pos = r[right_idx - 1] + t * (r[right_idx] - r[right_idx - 1])
                    else:
                        right_pos = r[right_idx]
                else:
                    right_pos = r[right_idx]
                
                width = right_pos - left_pos
            else:
                # Use a default width if either crossing is missing.
                width = dr * 2.0
            
            fwhm.append(width)
        
        fwhm = np.array(fwhm)
        
        return {
            'peak_positions': peak_positions,
            'peak_heights': peak_heights,
            'fwhm': fwhm,
            'n_peaks': len(peaks)
        }
        
    except Exception as e:
        logger.error(f"Could not extract RDF peaks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    process_results()

