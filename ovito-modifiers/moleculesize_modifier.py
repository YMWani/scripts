from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
import numpy as np


class CustomModifier(ModifierInterface):

    def modify(self, data: DataCollection, *, frame: int, **kwargs):
        # This user-defined modifier function selects molecules with a specific number of atoms
        
        if not data.particles:
            return

        # Configuration parameters
        TARGET_MOLECULE_SIZE = 140  # Change this to select molecules with desired number of atoms
        
        print(f"Starting with {data.particles.count} particles")
        
        # Check if we have molecule ID information
        molecule_id_property = None
        possible_names = ['Molecule', 'MoleculeID', 'Molecule Identifier', 'mol_id', 'molecule_id']
        
        for prop_name in possible_names:
            if prop_name in data.particles:
                molecule_id_property = prop_name
                break
        
        if molecule_id_property is None:
            print("WARNING: No molecule ID property found!")
            print("Looking for properties with names like:", possible_names)
            print("Available particle properties:")
            for prop_name in data.particles.keys():
                print(f"  '{prop_name}'")
            print("Please ensure your data has a molecule ID property assigned to particles.")
            return
            
        print(f"Using molecule ID property: '{molecule_id_property}'")
        
        # Get molecule IDs and calculate molecule sizes
        molecule_ids = data.particles[molecule_id_property][...]
        unique_molecules, molecule_sizes = np.unique(molecule_ids, return_counts=True)
        
        print(f"Found {len(unique_molecules)} molecules")
        print(f"Molecule size distribution:")
        size_counts = np.bincount(molecule_sizes)
        for size, count in enumerate(size_counts):
            if count > 0:
                print(f"  {count} molecules with {size} atoms")
        
        # Find molecules with the target size
        target_molecules = unique_molecules[molecule_sizes == TARGET_MOLECULE_SIZE]
        
        print(f"\nFound {len(target_molecules)} molecules with exactly {TARGET_MOLECULE_SIZE} atoms")
        
        # Create a property to store molecule sizes for each particle
        mol_sizes = np.zeros(data.particles.count, dtype=int)
        for i, mol_id in enumerate(molecule_ids):
            if mol_id >= 0:  # Valid molecule ID
                mol_index = np.where(unique_molecules == mol_id)[0]
                if len(mol_index) > 0:
                    mol_sizes[i] = molecule_sizes[mol_index[0]]
        
        data.particles_.create_property('MoleculeSize', data=mol_sizes)
        
        print(f"\nCreated property:")
        print(f"  'MoleculeSize' - Number of atoms in each molecule")
        
        # Summary statistics
        if len(target_molecules) > 0:
            print(f"\nTarget molecule statistics:")
            print(f"  Molecule IDs: {target_molecules}")
            print(f"  Total atoms in target molecules: {len(target_molecules) * TARGET_MOLECULE_SIZE}")
            print(f"  Percentage of atoms in target molecules: {100*len(target_molecules)*TARGET_MOLECULE_SIZE/data.particles.count:.1f}%")
