import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from evaluation.metrics import MolGANMetrics

class RewardNetwork(tf.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.metrics = MolGANMetrics(config, None)

    def compute_reward(self, adj_matrices, node_matrices):
        """Compute RL rewards: QED, Diversity, Validity, and Uniqueness."""
        qed_scores, diversity_scores, validity_scores, uniqueness_scores, synthesizability_scores = [], [], [], [], []
        generated_smiles, fps = set(), []

        for i in range(len(adj_matrices)):
            mol = self.adjacency_to_molecule(adj_matrices[i], node_matrices[i])
            if mol is None:
                qed_scores.append(0.0)
                diversity_scores.append(0.0)
                validity_scores.append(0.0)
                uniqueness_scores.append(0.0)
                synthesizability_scores.append(0.0)
                continue  

            # Compute QED Score
            qed_scores.append(QED.qed(mol))

            # Compute Validity Score
            try:
                Chem.SanitizeMol(mol)
                validity_scores.append(1.0)  # Valid molecule
            except:
                validity_scores.append(0.0)

            # Compute Uniqueness
            smiles = Chem.MolToSmiles(mol)
            if smiles in generated_smiles:
                uniqueness_scores.append(0.0)  # Penalize duplicates
            else:
                uniqueness_scores.append(1.0)
                generated_smiles.add(smiles)

            # Compute Diversity (Tanimoto Similarity)
            try:
                fp = GetMorganGenerator(radius=2).GetFingerprint(mol)
                fps.append(fp)  # Store fingerprints
            except:
                continue

            # Compute Synthesizability Score
            synthesis_score = Descriptors.NumRotatableBonds(mol) / max(Descriptors.HeavyAtomCount(mol), 1)
            synthesizability_scores.append(synthesis_score)

        # Compute pairwise diversity from fingerprints
        if len(fps) > 1:
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                    diversity_scores.append(1.0 - sim)  # Diversity = 1 - similarity

        # Ensure diversity isn't zero due to no comparisons
        if len(diversity_scores) == 0:
            diversity_scores = [0.0]

        # Convert lists to NumPy arrays
        qed_scores = np.array(qed_scores)
        diversity_scores = np.array(diversity_scores)
        validity_scores = np.array(validity_scores)
        uniqueness_scores = np.array(uniqueness_scores)
        synthesizability_scores = np.array(synthesizability_scores)

        return qed_scores, diversity_scores, validity_scores, uniqueness_scores, synthesizability_scores
