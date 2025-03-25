import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from data.preprocess import QM9Preprocessor

class MolGANMetrics:
    def __init__(self, config, generator):
        self.config = config
        self.generator = generator
        self.preprocessor = QM9Preprocessor(config)

    def generate_molecules(self, num_samples):
        z = tf.random.normal((num_samples, self.config.LATENT_DIM))
        adj_pred, node_pred = self.generator(z, training=False)  # Ignore entropy_reg

        # Convert tensors to numpy for debugging
        adj_pred = adj_pred.numpy()
        node_pred = node_pred.numpy()

        # Adaptive thresholding for adjacency matrix
        adj_pred = np.where(adj_pred > 0.6, 1, 0)

        return adj_pred, node_pred

    def sanitize_molecule(self, mol):
        """Sanitize the molecule and fix valence issues."""
        try:
            Chem.SanitizeMol(mol)
            return mol  # Return valid molecule
        except:
            return None  # Skip invalid molecules

    def adjacency_to_molecule(self, adj, node_features):
        """Convert adjacency & node feature matrices to RDKit molecule, ensuring valid valencies."""
        mol = Chem.RWMol()
        node_indices = []
        atom_valences = []
        
        # Add atoms with valency check
        for i, features in enumerate(node_features):
            atom_type = np.argmax(features[:-1])  # Ignore padding dimension

            if atom_type >= len(self.preprocessor.atom_types):
                continue  # Skip invalid atom

            atom_symbol = self.preprocessor.atom_types[atom_type]

            try:
                atom = Chem.Atom(atom_symbol)
                max_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
                if max_valence is None:
                    continue
                
                node_idx = mol.AddAtom(atom)
                node_indices.append(node_idx)
                atom_valences.append(max_valence)
            except:
                continue

        if len(node_indices) == 0:
            return None  # No valid atoms

        # Add bonds with strict valency check
        for i in range(len(node_indices)):
            for j in range(i + 1, len(node_indices)):
                bond_type_idx = np.argmax(adj[i, j, :-1])

                if bond_type_idx in [0, 1, 2]:  
                    bond_type = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE][bond_type_idx]
                    
                    # Ensure both atoms have remaining valency
                    if atom_valences[i] > 0 and atom_valences[j] > 0:
                        mol.AddBond(node_indices[i], node_indices[j], bond_type)
                        atom_valences[i] -= (bond_type_idx + 1)  # Reduce valency properly
                        atom_valences[j] -= (bond_type_idx + 1)
                    else:
                        continue  # Skip if valency exceeded
        
        return self.check_valency(mol)

    def check_valency(self, mol):
        """Ensure atoms obey valency rules before returning a molecule."""
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol  # Valid molecule
        except:
            return None  # Invalid molecule, discard it

    def compute_validity(self, num_samples=1000):
        """Compute the validity of generated molecules."""
        adj_matrices, node_matrices = self.generate_molecules(num_samples)
        valid_count = 0

        for i in range(num_samples):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is not None:
                valid_count += 1  # Count valid molecules

        return valid_count / num_samples

    def compute_novelty(self, num_samples=1000):
        _, _, known_smiles = self.preprocessor.load_qm9_data()
        known_smiles_set = set(known_smiles)
        adj_matrices, node_matrices = self.generate_molecules(num_samples)
        novel_count = 0

        for i in range(num_samples):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in known_smiles_set:
                    novel_count += 1

        return novel_count / num_samples

    def compute_uniqueness(self, num_samples=1000):
        adj_matrices, node_matrices = self.generate_molecules(num_samples)
        generated_smiles = set()

        for i in range(num_samples):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is not None:
                generated_smiles.add(Chem.MolToSmiles(mol))

        return len(generated_smiles) / num_samples

    def compute_qed(self, num_samples=1000):
        adj_matrices, node_matrices = self.generate_molecules(num_samples)
        qed_scores = []

        for i in range(num_samples):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is not None:
                qed_scores.append(QED.qed(mol))

        return np.mean(qed_scores) if qed_scores else 0.0

    def compute_diversity(self, num_samples=1000):
        adj_matrices, node_matrices = self.generate_molecules(num_samples)
        fps = []

        for i in range(num_samples):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is None:
                continue

            try:
                fps.append(GetMorganGenerator(radius=2).GetFingerprint(mol))
            except:
                continue

        if len(fps) < 2:
            return 0.0

        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                similarities.append(sim)

        return 1.0 - np.mean(similarities)

    def evaluate(self, num_samples=1000):
        validity = self.compute_validity(num_samples)
        novelty = self.compute_novelty(num_samples)
        uniqueness = self.compute_uniqueness(num_samples)
        qed = self.compute_qed(num_samples)
        diversity = self.compute_diversity(num_samples)

        metrics = {
            "Validity": validity,
            "Novelty": novelty,
            "Uniqueness": uniqueness,
            "QED": qed,
            "Diversity": diversity
        }

        return metrics
