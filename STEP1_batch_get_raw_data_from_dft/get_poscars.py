import os
import random
from pymatgen.core import Structure
from loguru import logger

def generate_multi_doping_data(doping_dict: dict, num_variants: int = 5, input_file: str = "../POSCAR-base/POSCAR"):
    """
    Generate structures with substitutions by one or more dopant species.
    :param doping_dict: Dopant counts, e.g. {"Mg": 2, "Si": 1} substitutes two Mg atoms and one Si atom.
    :param num_variants: Number of random variants to generate.
    :param input_file: Path to the original POSCAR.
    """
    # 1. Read the original structure
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return
        
    structure = Structure.from_file(input_file)
    
    # 2. Find the indices of the atoms to replace (Al in this case)
    target_base_element = "Al"
    base_indices = [i for i, site in enumerate(structure) if site.species_string == target_base_element]
    
    total_to_replace = sum(doping_dict.values())
    if total_to_replace > len(base_indices):
        logger.error(f"Requested {total_to_replace} substitutions, but the number of available {target_base_element} atoms is {len(base_indices)}.")
        return

    # 3. Generate the variants
    for v in range(num_variants):
        temp_structure = structure.copy()
        
        # Randomly select the sites to replace
        all_target_indices = random.sample(base_indices, total_to_replace)
        
        # Assign the selected sites to the dopant species
        current_start = 0
        for element, count in doping_dict.items():
            end_idx = current_start + count
            subset_indices = all_target_indices[current_start:end_idx]
            
            for idx in subset_indices:
                temp_structure.replace(idx, element)
            current_start = end_idx
            
        # Group atoms of the same element in the order required by VASP
        temp_structure.sort()
        
        # Use the chemical formula to name the variant
        comp_dict = temp_structure.composition.as_dict()
        comp_str = "".join([f"{el}{int(count)}" for el, count in comp_dict.items()])
        
        # Set the output directory and filename
        output_dir = "../POSCAR-base"
        os.makedirs(output_dir, exist_ok=True)
        poscar_path = os.path.join(output_dir, f"POSCAR-{comp_str}-{v+1}")
        
        # Save the POSCAR
        temp_structure.to(fmt="poscar", filename=poscar_path)
        logger.info(f"Generated variant {v+1}: {comp_str}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate POSCAR structures with one or more dopant species")
    # Accept multiple pairs, e.g. --doping Mg 2 --doping Si 1
    parser.add_argument("-dope", nargs=2, action='append', metavar=('ELEMENT', 'COUNT'), 
                        help="Dopant element and count (repeatable)")
    parser.add_argument("-num", type=int, default=5, help="Number of variants to generate")
    parser.add_argument("-input_file", type=str, default="../POSCAR-base/POSCAR", help="Path to the input POSCAR")

    args = parser.parse_args()

    # Convert the parsed pairs to a dictionary: [['Mg', '2'], ['Si', '1']] -> {'Mg': 2, 'Si': 1}
    if args.dope:
        doping_config = {el: int(cnt) for el, cnt in args.dope}
    else:
        logger.warning("No dopant specified. Using the default example: Mg:1")
        doping_config = {"Mg": 1}

    logger.info(f"Dopants: {doping_config} | Variants: {args.num}")
    generate_multi_doping_data(doping_config, args.num, args.input_file)
