import tensorflow as tf
import numpy as np

# Maximum valencies for common elements (extend as needed)
MAX_VALENCIES = {6: 4, 7: 3, 8: 2, 9: 1}  # C:4, N:3, O:2, F:1

def threshold_with_valency(adjacency, nodes, threshold=0.5):
    """
    Binarizes adjacency matrix while enforcing valency constraints.
    
    Args:
        adjacency: Tensor (batch, num_atoms, num_atoms) with edge probabilities.
        nodes: Tensor (batch, num_atoms, atom_types) one-hot encoded atom types.
        threshold: Float value for binarization.
    
    Returns:
        Adjusted adjacency tensor (binarized and valency-corrected).
    """
    batch_size, num_atoms, _ = adjacency.shape
    
    # Step 1: Apply threshold to get initial bonds
    adj_bin = tf.cast(adjacency > threshold, tf.float32)
    
    # Step 2: Enforce valency constraints
    for i in range(batch_size):
        for atom_idx in range(num_atoms):
            # Determine atom type
            atom_type = tf.argmax(nodes[i, atom_idx]).numpy()
            max_valency = MAX_VALENCIES.get(atom_type, 4)  # Default to 4 if unknown
            
            # Count current bonds
            bond_count = tf.reduce_sum(adj_bin[i, atom_idx]).numpy()
            
            # If too many bonds, remove weakest ones
            if bond_count > max_valency:
                # Get indices of bonds
                bond_indices = np.argsort(adjacency[i, atom_idx].numpy())[::-1]  # Sort by strength
                excess_bonds = int(bond_count - max_valency)
                
                # Remove weakest bonds
                for bond in bond_indices[-excess_bonds:]:
                    adj_bin = tf.tensor_scatter_nd_update(adj_bin, [[i, atom_idx, bond]], [0])
                    adj_bin = tf.tensor_scatter_nd_update(adj_bin, [[i, bond, atom_idx]], [0])  # Keep symmetric

    return adj_bin
