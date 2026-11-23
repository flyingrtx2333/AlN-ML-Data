import shutil
import os
import subprocess
from pathlib import Path
from loguru import logger


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

def process_c33():
    """
    Prepare c33 jobs for the structures in ../scf.
    Copy POSCAR, POTCAR, and KPOINTS into the matching directories under ../c33.
    Copy the job templates and set the job name in run.sh to c33-<structure-name>.
    """
    # Define input and output paths
    script_dir = Path(__file__).parent
    scf_dir = script_dir / ".." / "scf"
    c33_dir = script_dir / ".." / "c33"
    
    # Resolve absolute paths
    scf_dir = scf_dir.resolve()
    c33_dir = c33_dir.resolve()
    
    # Ensure the c33 directory exists
    c33_dir.mkdir(parents=True, exist_ok=True)
    
    # Check that the scf directory exists
    if not scf_dir.exists():
        logger.error(f"Directory does not exist: {scf_dir}")
        return
    
    logger.info("Scanning the scf directory...")
    copied_count = 0
    
    # Iterate over structure directories
    for folder_path in scf_dir.iterdir():
        if not folder_path.is_dir():
            continue
        
        structure_name = folder_path.name
        
        # Set the destination for this structure
        target_structure_dir = c33_dir / structure_name
        
        # Check for required files
        required_files = ["POSCAR", "POTCAR", "KPOINTS", "run.sh"]
        missing_files = [f for f in required_files if not (folder_path / f).exists()]
        
        if missing_files:
            logger.warning(f"Skipping {structure_name}: missing files {', '.join(missing_files)}")
            continue
        
        # Run qvasp -e to check for a YES result
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
            
            logger.info(f"  {structure_name} passed the qvasp convergence check")
            
        except subprocess.TimeoutExpired:
            logger.error(f"  Skipping {structure_name}: qvasp -e timed out")
            continue
        except FileNotFoundError:
            logger.error(f"  qvasp was not found. Install qvasp and add it to PATH.")
            break
        except Exception as e:
            logger.error(f"  Skipping {structure_name}: qvasp -e failed: {e}")
            continue
        
        # Create the destination directory
        target_structure_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy the structure files and job templates
            shutil.copy2(folder_path / "POSCAR", target_structure_dir / "POSCAR")
            shutil.copy2(folder_path / "POTCAR", target_structure_dir / "POTCAR")
            shutil.copy2(folder_path / "KPOINTS", target_structure_dir / "KPOINTS")
            shutil.copy2(script_dir / "run.sh", target_structure_dir / "run.sh")
            shutil.copy2(script_dir / "INCAR-c33", target_structure_dir / "INCAR")
            
            logger.info(f"  ✓ Copied files for {structure_name}")
            
            # Update the job name on the second line of run.sh
            target_runsh = target_structure_dir / "run.sh"
            if target_runsh.exists():
                with open(target_runsh, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Replace the name in the job directive
                if len(lines) > 1:
                    # Change #SBATCH -J xxx to #SBATCH -J c33-<structure-name>
                    job_name = f"c33-{structure_name}"
                    lines[1] = f"#SBATCH -J {job_name}\n"
                    
                    with open(target_runsh, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    logger.info(f"  Set the run.sh job name to {job_name}")
            
            
            if not check_is_compute_completed(target_structure_dir):
                logger.warning(f"Submitting a C33 calculation for {structure_name}")
                subprocess.run(
                    ["sbatch", "run.sh"],
                    cwd=str(target_structure_dir),
                    text=True,
                    capture_output=True,
                    timeout=10
                )
            else:
                logger.success(f"Skipping {structure_name}: C33 calculation already completed")
            copied_count += 1
            
        except Exception as e:
            logger.error(f"  ✗ Failed to copy files for {structure_name}: {e}")
            # Remove the destination directory after a failed copy
            if target_structure_dir.exists():
                try:
                    shutil.rmtree(target_structure_dir)
                except:
                    pass
    
    logger.info(f"\nCopied {copied_count} structures to the c33 directory.")


if __name__ == "__main__":
    process_c33()

