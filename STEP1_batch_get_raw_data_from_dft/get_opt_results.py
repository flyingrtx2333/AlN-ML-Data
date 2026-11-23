
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

def process_poscar_files():
    """
    Scan ../POSCAR-base for input files other than POSCAR.
    Create a matching directory under ../OPT_results and copy each input there as POSCAR.
    """
    # Define input and output paths
    script_dir = Path(__file__).parent
    poscar_base_dir = script_dir / ".." / "POSCAR-base"
    opt_results_dir = script_dir / ".." / "OPT_results"
    
    # Resolve absolute paths
    poscar_base_dir = poscar_base_dir.resolve()
    opt_results_dir = opt_results_dir.resolve()
    
    # Ensure the OPT_results directory exists
    opt_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan the POSCAR-base directory
    if not poscar_base_dir.exists():
        print(f"Directory does not exist: {poscar_base_dir}")
        return
    
    processed_count = 0
    for file_path in poscar_base_dir.iterdir():
        # Process files only
        if not file_path.is_file():
            continue
        
        # Skip the file named POSCAR
        if file_path.name == "POSCAR":
            continue
        
        # Use the filename without its extension
        file_name = file_path.name
        
        # Check for an existing optimization directory
        if (opt_results_dir / file_name).exists():
            logger.info(f"Checking whether the existing calculation for {file_name} completed")
            # result = subprocess.run(
            #     ["qvasp", "-e"],
            #     cwd=str(opt_results_dir / file_name),
            #     text=True,
            #     capture_output=True,
            #     timeout=60
            # )
            
            # output = result.stdout + result.stderr
            
            # # Check the qvasp result
            # is_covered = check_covered(output)
            
            if check_is_compute_completed(opt_results_dir / file_name):
                logger.success(f"Skipping {file_name}: calculation already completed")
                continue
        
        # Create the matching directory under OPT_results
        target_dir = opt_results_dir / file_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the input to the destination as POSCAR
        target_file = target_dir / "POSCAR"
        shutil.copy2(file_path, target_file)
        
        incar_path = script_dir / "INCAR-opt"
        shutil.copy2(incar_path, target_dir / "INCAR")
        
        # Copy and update run.sh
        runsh_path = script_dir / "run.sh"
        target_runsh = target_dir / "run.sh"
        shutil.copy2(runsh_path, target_runsh)
        
        # Update the job name on the second line of run.sh
        if target_runsh.exists():
            with open(target_runsh, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Replace the name in the job directive
            if len(lines) > 1:
                # Change #SBATCH -J vasp to #SBATCH -J vasp-<directory-name>
                job_name = f"{file_name.replace('POSCAR-', '')}"
                lines[1] = f"#SBATCH -J {job_name}\n"
                
                with open(target_runsh, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                logger.info(f"  Set the run.sh job name to {job_name}")
        
        # Run vaspkit to generate POTCAR
        print(f"Generating POTCAR for {file_name}...")
        potcar_generated = False
        
        try:
            result = subprocess.run(
                ["vaspkit", "-task", "103"],
                cwd=str(target_dir),
                text=True,
                capture_output=True,
                timeout=60  # Time out after 60 seconds
            )
            
            # Check whether POTCAR was generated
            potcar_file = target_dir / "POTCAR"
            if potcar_file.exists():
                potcar_generated = True
                logger.info(f"  ✓ Generated POTCAR")
                
            else:
                logger.error(f"  ⚠ POTCAR was not generated. Check the job directory.")
                
            # Run qvasp -k to generate KPOINTS
            try:
                qvasp_result = subprocess.run(
                    ["qvasp", "-k"],
                    cwd=str(target_dir),
                    text=True,
                    capture_output=True,
                    timeout=60  # Time out after 60 seconds
                )
                
                if os.path.exists(target_dir / "KPOINTS"):
                    logger.info(f"  ✓ Generated KPOINTS")
                else:
                    logger.error(f"  ⚠ KPOINTS was not generated. Check the job directory.")
            except subprocess.TimeoutExpired:
                logger.error(f"  ⚠ qvasp timed out")
            except FileNotFoundError:
                logger.error(f"  ⚠ qvasp was not found")
            except Exception as e:
                logger.error(f"  ⚠ qvasp failed: {e}")
                
            
            # Submit run.sh after running qvasp -k
            logger.info(f"  Submitting the job: sbatch run.sh")
            try:
                sbatch_result = subprocess.run(
                    ["sbatch", "run.sh"],
                    cwd=str(target_dir),
                    text=True,
                    capture_output=True,
                    timeout=10
                )
                if sbatch_result.returncode == 0:
                    logger.success(f"  ✓ Job submitted")
                    if sbatch_result.stdout:
                        logger.info(f"    {sbatch_result.stdout.strip()}")
                else:
                    logger.error(f"  ⚠ Job submission failed with exit code {sbatch_result.returncode}")
                    if sbatch_result.stderr:
                        logger.error(f"    Error: {sbatch_result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.error(f"  ⚠ sbatch timed out")
            except FileNotFoundError:
                logger.error(f"  ⚠ sbatch was not found. Make sure Slurm is installed.")
            except Exception as e:
                logger.error(f"  ⚠ sbatch failed: {e}")
        except subprocess.TimeoutExpired:
            logger.error(f"  ⚠ vaspkit timed out")
        except FileNotFoundError:
            logger.error(f"  ⚠ vaspkit was not found. Install vaspkit and add it to PATH.")
        except Exception as e:
            logger.error(f"  ⚠ vaspkit failed: {e}")
        
        if not potcar_generated:
            logger.error(f"  ⚠ POTCAR was not generated. Check the job directory.")
        
        logger.info(f"Processed: {file_name} -> {target_dir / 'POSCAR'}")
        processed_count += 1
    
    logger.info(f"\nProcessed {processed_count} files.")


if __name__ == "__main__":
    process_poscar_files()

