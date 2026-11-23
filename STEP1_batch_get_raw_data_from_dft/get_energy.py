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
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    logger.warning("pymatgen is required for RDF calculations. Install it with pip install pymatgen.")


def check_covered(output):
    """
    Check for the qvasp success marker "Get COVERAGED: YES !".
    """
    return "YES" in output or "Get COVERAGED: YES !" in output

def check_is_compute_completed(target_dir):
    """
    Return True if the calculation has already completed.
    """
    if os.path.exists(target_dir / "OUTCAR"):
        with open(target_dir / "OUTCAR", "r", encoding="utf-8") as f:
            content = f.read()
        if "General timing and accounting informations for this job" in content:
            return True
    return False

def extract_energy_from_output(output):
    """
    Extract the final energy from the qvasp -e output.
    Example: The Final Energy in This Folder is:   -238.24048093
    """
    try:
        test_output2 = output.split('\n')[1].split(' ')[-1]
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', test_output2)
        return float(clean_output)
    except:
        return None

def extract_dopants(structure_name):
    """
    Extract up to two dopant species and their counts from the structure name.
    Examples:
    - POSCAR-K1Al14V1N16-1 -> ('K', 1, 'V', 1)
    - POSCAR-Sc4Al12N16-2 -> ('Sc', 4, None, None)
    
    Al and N are the host elements. Treat other elements as dopants.
    
    Return a (dopant_A, amount_A, dopant_B, amount_B) tuple.
    """
    try:
        # Remove the POSCAR- prefix and the trailing numeric suffix
        # Example: POSCAR-K1Al14V1N16-1 -> K1Al14V1N16
        name_without_prefix = structure_name.replace("POSCAR-", "")
        # Remove the trailing numeric suffix
        name_parts = name_without_prefix.rsplit("-", 1)
        if len(name_parts) > 1 and name_parts[1].isdigit():
            name_without_suffix = name_parts[0]
        else:
            name_without_suffix = name_without_prefix
        
        # Extract each element symbol and its count
        # Match an uppercase letter, an optional lowercase letter, and digits
        # Examples: K1, Al14, V1, N16
        pattern = r'([A-Z][a-z]?)(\d+)'
        matches = re.findall(pattern, name_without_suffix)
        
        # Host elements
        base_elements = {'Al', 'N'}
        
        # Collect non-host elements and their counts
        dopants_with_amounts = []
        for element, amount_str in matches:
            if element not in base_elements:
                try:
                    amount = int(amount_str)
                    dopants_with_amounts.append((element, amount))
                except ValueError:
                    # Skip non-integer counts
                    continue
        
        # Keep the first occurrence of each dopant in formula order
        seen = set()
        unique_dopants = []
        for dopant, amount in dopants_with_amounts:
            if dopant not in seen:
                seen.add(dopant)
                unique_dopants.append((dopant, amount))
        
        # Return the two dopant species and their counts
        dopant_A, amount_A = unique_dopants[0] if len(unique_dopants) > 0 else (None, None)
        dopant_B, amount_B = unique_dopants[1] if len(unique_dopants) > 1 else (None, None)
        
        return (dopant_A, amount_A, dopant_B, amount_B)
    except Exception as e:
        logger.error(f"Failed to extract dopants from {structure_name}: {e}")
        return (None, None, None, None)

def process_results():
    """
    For each structure directory in ../scf:
    1. Parse the dopant species from the structure name.
    2. Run qvasp -e and divide the reported energy by 16.
    3. Calculate the cell volume from POSCAR.
    4. Compute the RDF and extract peak features.
    Save the combined results to results.json.
    """
    # Define input and output paths
    script_dir = Path(__file__).parent
    scf_dir = script_dir / ".." / "scf"
    
    # Resolve the absolute path
    scf_dir = scf_dir.resolve()
    
    # Check that the scf directory exists
    if not scf_dir.exists():
        logger.error(f"Directory does not exist: {scf_dir}")
        return
    
    if not PYMATGEN_AVAILABLE:
        logger.error("pymatgen is required for RDF calculations. Install it with pip install pymatgen.")
        return
    
    logger.info("Scanning the scf directory to calculate energy, volume, and RDF...")
    results = {}  # Results by structure name
    
    # Iterate over structure directories
    for folder_path in scf_dir.iterdir():
        if not folder_path.is_dir():
            continue
        
        structure_name = folder_path.name
        poscar_path = folder_path / "POSCAR"
        
        # Extract the dopant species and their counts
        dopant_A, amount_A, dopant_B, amount_B = extract_dopants(structure_name)
        
        # Check for required files
        required_files = ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]
        missing_files = [f for f in required_files if not (folder_path / f).exists()]
        
        if missing_files:
            logger.error(f"Skipping {structure_name}: missing files {', '.join(missing_files)}")
            continue
        
        # Initialize per-structure results
        energy_per_16 = None
        volume = None
        rdf_analysis = None
        
        # 1. Run qvasp -e to obtain the energy
        logger.info(f"Running qvasp -e for {structure_name}...")
        try:
            result = subprocess.run(
                ["qvasp", "-e"],
                cwd=str(folder_path),
                text=True,
                capture_output=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # Check the qvasp result
            is_covered = check_covered(output)
            
            if not is_covered:
                logger.warning(f"Skipping {structure_name}: qvasp convergence check failed")
                continue
            
            # Extract the energy from the output
            energy = extract_energy_from_output(output)
            if energy is None:
                logger.warning(f"Skipping {structure_name}: could not extract the energy")
                continue
            energy_per_16 = energy / 16.0
            logger.success(f"  {structure_name} passed the qvasp convergence check. Energy/16 = {energy_per_16:.10f}")
            
        except Exception as e:
            logger.error(f"  Skipping {structure_name}: qvasp -e failed: {e}")
            continue
        
        # 2. Calculate the cell volume from POSCAR
        logger.info(f"  Calculating the cell volume for {structure_name}...")
        volume = calculate_volume_from_poscar(poscar_path)
        
        if volume is None:
            logger.warning(f"Skipping {structure_name}: could not calculate the cell volume")
            continue
        
        logger.success(f"  Cell volume for {structure_name}: {volume:.10f} Å³")
        
        # 3. Compute the RDF and extract peak features
        logger.info(f"  Calculating the RDF for {structure_name}...")
        rdf_analysis = calculate_rdf_analysis(poscar_path)
        
        if rdf_analysis is None:
            logger.warning(f"Could not calculate the RDF for {structure_name}. Using default values.")
            rdf_analysis = {
                'n_peaks': 0,
                'first_peak': None,
                'second_peak': None,
                'peak_ratio': None
            }
        else:
            if rdf_analysis['n_peaks'] > 0:
                peak_info = f"peaks found: {rdf_analysis['n_peaks']}"
                if rdf_analysis.get('first_peak'):
                    peak_info += f", first peak position: {rdf_analysis['first_peak']['position']:.3f} Å"
                if rdf_analysis.get('second_peak'):
                    peak_info += f", second peak position: {rdf_analysis['second_peak']['position']:.3f} Å"
                if rdf_analysis.get('peak_ratio'):
                    peak_info += f", peak position ratio: {rdf_analysis['peak_ratio']:.3f}"
                logger.success(f"  RDF analysis for {structure_name}: {peak_info}")
            else:
                logger.warning(f"  RDF analysis found no peaks for {structure_name}")
        
        # Build the result with all RDF features
        structure_result = {
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
            'energy_per_16': float(energy_per_16),
            'volume_ang3': float(volume),
            'rdf_analysis': {
                'n_peaks': rdf_analysis.get('n_peaks', 0),
                'Peak1_Pos': rdf_analysis['first_peak']['position'] if rdf_analysis.get('first_peak') else None,
                'Peak1_Height': rdf_analysis['first_peak']['height'] if rdf_analysis.get('first_peak') else None,
                'Peak1_FWHM': rdf_analysis['first_peak']['fwhm'] if rdf_analysis.get('first_peak') else None,
                'Peak2_Pos': rdf_analysis['second_peak']['position'] if rdf_analysis.get('second_peak') else None,
                'Peak2_Height': rdf_analysis['second_peak']['height'] if rdf_analysis.get('second_peak') else None,
                'Peak2_FWHM': rdf_analysis['second_peak']['fwhm'] if rdf_analysis.get('second_peak') else None,
                'Peak2_Pos_Peak1_Pos': rdf_analysis.get('peak_ratio')
            }
        }
        
        results[structure_name] = structure_result
    
    # Save the results to JSON
    json_path = script_dir / "results.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"\nSaved results for {len(results)} structures to {json_path}.")
    except Exception as e:
        logger.error(f"Failed to save the JSON file: {e}")

def process_energy():
    """
    The original workflow used qvasp -e on the structures in ../scf.
    It wrote the structure name and energy divided by 16 to results_scf_energy.csv.
    This function is deprecated. Use process_results() instead.
    """
    logger.warning("process_energy() is deprecated. Use process_results() instead.")
    process_results()

def calculate_volume_from_poscar(poscar_path):
    """
    Calculate the cell volume from the lattice vectors in a POSCAR file.
    POSCAR layout:
    Line 1: Comment
    Line 2: Scaling factor
    Lines 3-5: Lattice vectors (three numbers per line)
    
    Volume = |det(lattice_vectors)| = |a · (b × c)|
    """
    try:
        with open(poscar_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip the comment on line 1
        # Read the scaling factor on line 2
        scale_factor = float(lines[1].strip())
        
        # Read the three lattice vectors on lines 3-5
        lattice_vectors = []
        for i in range(2, 5):  # Lines 3-5 (indices 2-4)
            vector = [float(x) for x in lines[i].split()]
            if len(vector) != 3:
                raise ValueError(f"Expected three numbers in a lattice vector, found {len(vector)}")
            lattice_vectors.append(vector)
        
        # Convert the lattice vectors to a NumPy array
        lattice_matrix = np.array(lattice_vectors)
        
        # Compute the volume as |det(lattice_vectors)| * scale_factor^3
        volume = abs(np.linalg.det(lattice_matrix)) * (scale_factor ** 3)
        
        return volume
    except Exception as e:
        logger.error(f"Failed to read POSCAR file {poscar_path}: {e}")
        return None

def calculate_rdf_analysis(poscar_path, r_max=6.0, dr=0.02):
    """
    Calculate the radial distribution function (RDF) with pymatgen and extract peak features.
    
    Args:
        poscar_path: Path to the POSCAR file.
        r_max: Maximum distance in Å (default: 6.0).
        dr: Distance bin width in Å (default: 0.02).
    
    Returns a dictionary with the following keys:
        - n_peaks: Number of peaks.
        - first_peak: First peak position, height, and FWHM.
        - second_peak: Second peak position, height, and FWHM.
        - peak_ratio: Second peak position divided by the first peak position.
        Returns None if the calculation fails or no peaks are found.
    """
    if not PYMATGEN_AVAILABLE:
        logger.error("pymatgen is required for RDF calculations")
        return None
    
    try:
        # Read the structure with pymatgen
        structure = Structure.from_file(str(poscar_path))
        
        # Get the Cartesian coordinates of all atoms
        coords = np.array([site.coords for site in structure])
        n_atoms = len(coords)
        
        # Get the lattice matrix
        lattice_matrix = np.array(structure.lattice.matrix)
        
        # Get the cell volume
        volume = structure.volume
        
        # Compute pairwise distances with periodic boundary conditions
        # Start from fractional coordinates
        coords_frac = np.array([site.frac_coords for site in structure])
        
        # Compute the fractional coordinate differences for all pairs
        diff_frac = coords_frac[:, np.newaxis, :] - coords_frac[np.newaxis, :, :]  # (n, n, 3)
        
        # Apply periodic boundary conditions by wrapping differences into [-0.5, 0.5]
        diff_frac = diff_frac - np.round(diff_frac)
        
        # Convert the differences to Cartesian coordinates
        diff_cartesian = diff_frac @ lattice_matrix  # (n, n, 3)
        
        # Compute the distance matrix
        distances_matrix = np.linalg.norm(diff_cartesian, axis=2)  # (n, n)
        
        # Use the upper triangle to exclude self-pairs and count each pair once
        triu_indices = np.triu_indices(n_atoms, k=1)
        distances = distances_matrix[triu_indices]
        
        # Filter the distances
        valid_mask = (distances > 0.01) & (distances < r_max)
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            logger.warning(f"No valid interatomic distances found in {poscar_path}")
            return None
        
        # Define the distance bin edges
        r = np.arange(0, r_max + dr, dr)
        n_bins = len(r)
        
        # Bin distances with NumPy's histogram function
        hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max + dr))
        
        # Use the bin centers as the radial distances
        r = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        g_r = hist.astype(float)
        
        # Normalize by spherical shell volume and atomic number density
        rho = n_atoms / volume  # Atomic number density
        shell_volumes = 4.0 * np.pi * r ** 2 * dr
        # Avoid division by zero
        valid_shell = shell_volumes > 0
        g_r[valid_shell] = g_r[valid_shell] / (rho * shell_volumes[valid_shell] * n_atoms)
        g_r[~valid_shell] = 0.0
        
        # Detect the peaks
        peak_info = extract_rdf_peaks(r, g_r)
        
        if peak_info is None or peak_info['n_peaks'] == 0:
            logger.warning(f"No RDF peaks found in {poscar_path}")
            return {
                'n_peaks': 0,
                'first_peak': None,
                'second_peak': None,
                'peak_ratio': None
            }
        
        # Build the result dictionary
        result = {
            'n_peaks': peak_info['n_peaks'],
            'first_peak': None,
            'second_peak': None,
            'peak_ratio': None
        }
        
        # Record the first peak
        if len(peak_info['peak_positions']) > 0:
            first_peak = {
                'position': float(peak_info['peak_positions'][0]),
                'height': float(peak_info['peak_heights'][0]),
                'fwhm': float(peak_info['fwhm'][0]) if len(peak_info['fwhm']) > 0 else None
            }
            result['first_peak'] = first_peak
        
        # Record the second peak
        if len(peak_info['peak_positions']) > 1:
            second_peak = {
                'position': float(peak_info['peak_positions'][1]),
                'height': float(peak_info['peak_heights'][1]),
                'fwhm': float(peak_info['fwhm'][1]) if len(peak_info['fwhm']) > 1 else None
            }
            result['second_peak'] = second_peak
            
            # Divide the second peak position by the first
            if result['first_peak'] and result['first_peak']['position'] > 0:
                result['peak_ratio'] = float(second_peak['position'] / result['first_peak']['position'])
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate the RDF for {poscar_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def extract_rdf_peaks(r, g_r, min_height=None, min_distance=0.5):
    """
    Extract peak positions, heights, and full widths at half maximum (FWHM) from the RDF.
    
    Args:
        r: Array of radial distances.
        g_r: Array of RDF values.
        min_height: Minimum peak height (defaults to 10% of the maximum RDF value).
        min_distance: Minimum peak separation in Å (default: 0.5).
    
    Returns a dictionary with the following keys:
        - peak_positions: Array of peak positions in Å.
        - peak_heights: Array of peak heights.
        - fwhm: Array of full widths at half maximum in Å.
        - n_peaks: Number of peaks.
    """
    try:
        if r is None or g_r is None or len(r) == 0 or len(g_r) == 0:
            return None
        
        # Set default parameters
        if min_height is None:
            min_height = np.max(g_r) * 0.1  # 10% of the maximum RDF value
        
        # Calculate the radial grid spacing
        dr = r[1] - r[0] if len(r) > 1 else 0.01
        
        # Detect peaks as local maxima
        peaks = []
        peak_heights = []
        
        for i in range(1, len(g_r) - 1):
            # Check for a local maximum
            if g_r[i] > g_r[i-1] and g_r[i] > g_r[i+1] and g_r[i] > min_height:
                # Check the distance to existing peaks
                is_valid = True
                for peak_idx in peaks:
                    if abs(r[i] - r[peak_idx]) < min_distance:
                        # Keep the higher peak if the two are too close
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
        
        # Sort by peak position
        sorted_indices = sorted(range(len(peaks)), key=lambda i: r[peaks[i]])
        peaks = [peaks[i] for i in sorted_indices]
        peak_heights = [peak_heights[i] for i in sorted_indices]
        
        peak_positions = r[peaks]
        peak_heights = np.array(peak_heights)
        
        # Compute the full width at half maximum (FWHM)
        fwhm = []
        for peak_idx in peaks:
            peak_height = g_r[peak_idx]
            half_max = peak_height / 2.0
            
            # Find the half-maximum crossings on both sides of the peak
            # Left side
            left_idx = peak_idx
            while left_idx > 0 and g_r[left_idx] > half_max:
                left_idx -= 1
            
            # Right side
            right_idx = peak_idx
            while right_idx < len(g_r) - 1 and g_r[right_idx] > half_max:
                right_idx += 1
            
            # Calculate the FWHM
            if left_idx < peak_idx and right_idx > peak_idx:
                # Refine the crossing positions by linear interpolation
                # Left crossing
                if left_idx < peak_idx - 1:
                    y1, y2 = g_r[left_idx], g_r[left_idx + 1]
                    if y2 > y1:
                        t = (half_max - y1) / (y2 - y1) if (y2 - y1) > 1e-10 else 0
                        left_pos = r[left_idx] + t * (r[left_idx + 1] - r[left_idx])
                    else:
                        left_pos = r[left_idx]
                else:
                    left_pos = r[left_idx]
                
                # Right crossing
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
                # Use the fallback width if a crossing cannot be found
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
        logger.error(f"Failed to extract RDF peaks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    process_results()

