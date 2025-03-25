import os
import numpy as np
import pickle
import zipfile
from rdkit import Chem
from rdkit.Chem import rdmolops

class QM9Preprocessor:
    def __init__(self, config):
        self.max_atoms = config.MAX_NODES  # Max atoms per molecule (e.g., 9 for QM9)
        self.num_bond_types = config.NUM_EDGE_TYPES  # 4 bond types (single, double, triple, no bond)
        self.atom_types = [6, 8, 7, 9]  # Carbon (C), Oxygen (O), Nitrogen (N), Fluorine (F)
        self.dataset_path = config.DATASET_PATH  # Path to QM9 dataset (can be .xyz files or .zip)
        self.extracted_path = os.path.join(config.DATASET_PATH, "qm9_extracted")  # Extracted folder
        self.preprocessed_path = config.PREPROCESSED_DATA_PATH  # Save processed graphs
        self.smiles_path = config.SMILES_PATH  # Save SMILES strings

        # Ensure extracted directory exists
        os.makedirs(self.extracted_path, exist_ok=True)

        # Extract dataset if it's zipped
        self.extract_zip_if_needed()

    def extract_zip_if_needed(self):
        """Checks if dataset is a ZIP file and extracts it."""
        if os.path.isfile(self.dataset_path) and zipfile.is_zipfile(self.dataset_path):
            print(f"Extracting QM9 dataset from {self.dataset_path}...")
            with zipfile.ZipFile(self.dataset_path, "r") as zip_ref:
                zip_ref.extractall(self.extracted_path)
            print("Extraction complete!")
            self.dataset_path = self.extracted_path  # Update path to extracted directory

    def xyz_to_mol(self, xyz_file):
        """Reads an XYZ file and converts it into an RDKit molecule."""
    
        # Open normally or with gzip based on file type
        if xyz_file.endswith(".gz"):
            with gzip.open(xyz_file, "rt", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            with open(xyz_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        # Ensure file is not empty
        if len(lines) < 3:
            print(f"Skipping {xyz_file}: File too short!")
            return None  

        try:
            num_atoms = int(lines[0].strip())  # First line gives atom count
        except ValueError:
            print(f"Skipping {xyz_file}: Invalid atom count!")
            return None  

        if num_atoms > self.max_atoms:
            return None  # Skip molecules with too many atoms

        mol = Chem.RWMol()
        atom_positions = []
        atom_indices = {}

        # Read atoms and positions
        for i in range(2, 2 + num_atoms):  # Skip first two lines
            parts = lines[i].split()
            if len(parts) < 4:
                print(f"Skipping {xyz_file}: Invalid atom format!")
                return None  
            
            atom_symbol, x, y, z = parts[0], parts[1], parts[2], parts[3]
            
            try:
                atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom_symbol)
            except:
                print(f"Skipping {xyz_file}: Unknown atom '{atom_symbol}'!")
                return None

            if atomic_num in self.atom_types:
                atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
                atom_positions.append([float(x), float(y), float(z)])
                atom_indices[i - 2] = atom_idx
            else:
                print(f"Skipping {xyz_file}: Atom type {atomic_num} not allowed!")
                return None  # Skip unknown atoms

        # Convert positions to NumPy array
        atom_positions = np.array(atom_positions)

        # Compute bonds based on distances
        bond_distances = {
            (1, 1): 1.5,  # Single bond distance
            (1, 2): 1.3,  # Double bond distance
            (1, 3): 1.2   # Triple bond distance
        }
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(atom_positions[i] - atom_positions[j])
                if dist < bond_distances[(1, 1)]:  # Assume single bond threshold
                    mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)

        return mol

    def mol_to_graph(self, mol):
        """Converts RDKit Mol to adjacency matrix & node features."""
        if mol is None:
            return None, None

        mol = Chem.RemoveHs(mol)
        num_atoms = mol.GetNumAtoms()

        if num_atoms > self.max_atoms:
            return None, None

        adjacency_matrix = np.zeros((self.max_atoms, self.max_atoms, self.num_bond_types), dtype=np.float32)
        node_features = np.zeros((self.max_atoms, len(self.atom_types) + 1), dtype=np.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = atom.GetAtomicNum()
            if atom_type in self.atom_types:
                node_features[i, self.atom_types.index(atom_type)] = 1
            else:
                return None, None

        bond_type_to_index = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2
        }
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            if bond_type in bond_type_to_index:
                adjacency_matrix[i, j, bond_type_to_index[bond_type]] = 1
                adjacency_matrix[j, i, bond_type_to_index[bond_type]] = 1

        adjacency_matrix[:, :, 3] = 1 - np.sum(adjacency_matrix[:, :, :3], axis=-1)

        return adjacency_matrix, node_features

    def load_qm9_data(self):
        """Process .xyz files and save graph representations."""
        if os.path.exists(self.preprocessed_path) and os.path.exists(self.smiles_path):
            print("Loading preprocessed dataset...")
            with open(self.preprocessed_path, "rb") as f:
                data = pickle.load(f)
            with open(self.smiles_path, "r") as f:
                smiles_list = [s.strip() for s in f.readlines()]
            return data["adjacency_tensor"], data["node_tensor"], smiles_list

        print("Processing QM9 dataset from .xyz files...")

        # Ensure the dataset directory exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path '{self.dataset_path}' does not exist!")

        adjacency_list, node_feature_list, smiles_list = [], [], []

        xyz_files = [f for f in os.listdir(self.dataset_path) if f.endswith(".xyz") and os.path.isfile(os.path.join(self.dataset_path, f))]

        if not xyz_files:
            raise FileNotFoundError("No .xyz files found in the dataset directory. Check extraction!")

        for file_name in xyz_files:
            xyz_path = os.path.join(self.dataset_path, file_name)

            print(f"Processing file: {xyz_path}")

            try:
                mol = self.xyz_to_mol(xyz_path)
                adjacency_matrix, node_features = self.mol_to_graph(mol)

                if adjacency_matrix is not None and node_features is not None:
                    adjacency_list.append(adjacency_matrix)
                    node_feature_list.append(node_features)
                    smiles_list.append(Chem.MolToSmiles(mol))

            except Exception as e:
                print(f"Error processing {xyz_path}: {e}")

        print(f"Total molecules attempted: {len(os.listdir(self.dataset_path))}")
        print(f"Valid adjacency matrices: {len(adjacency_list)}")
        print(f"Valid node features: {len(node_feature_list)}")
        print(f"Valid SMILES: {len(smiles_list)}")


        if not adjacency_list or not node_feature_list:
            raise ValueError("No valid molecules were processed. Check your dataset!")

        adjacency_tensor = np.array(adjacency_list, dtype=np.float32)
        node_tensor = np.array(node_feature_list, dtype=np.float32)

        with open(self.preprocessed_path, "wb") as f:
            pickle.dump({"adjacency_tensor": adjacency_tensor, "node_tensor": node_tensor}, f)

        with open(self.smiles_path, "w") as f:
            for smiles in smiles_list:
                f.write(smiles + "\n")

        return adjacency_tensor, node_tensor, smiles_list
