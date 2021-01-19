import tensorflow as tf

"""
Calculates the LJ potential force on particles
    pos: Tensor containing the position of particles in the shape [[x,y,z]]
    edges_half: Edges of the simulation box as a tensor in shape [x,y,z], divided by two
    neg_edges_half: Negation of Tensor `edges_half`
    edges: Edges of the simulation box as a tensor in shape [x,y,z]
    forces_zeroes_tf: A Tensor of all zeroes that matches shape with `pos`
    ljatom_diameter_tf: Tensor containing diameters of the particles, must match positions with `pos`
"""
def lj_force(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf, ljatom_diameter_tf):
    with tf.name_scope("lj_force"):
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: atom_pos - pos, elems=pos)
        distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances, name="where_neg_edges_half")
        m = tf.norm(distances,ord='euclidean', axis=2, keepdims=True)
        
        magnitude = m #common.magnitude(distances)
        twelve = tf.math.pow(ljatom_diameter_tf, 12.0, name="diam_12_pow") / tf.math.pow(magnitude, 12.0, name="mag_12_pow")    #()
        six = tf.math.pow(ljatom_diameter_tf, 6.0, name="diam_6_pow") / tf.math.pow(magnitude, 6.0, name="mag_6_pow")    #(8, 8, 3)
        mag_squared = 1.0 / tf.math.pow(magnitude,2, name="mag_sqr_pow")
      
        slice_forces = distances * (48.0 * 1.0 * (((twelve - 0.5 * six) * mag_squared)))
        # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
        # can't see a way to remove that case in all above computations, easier to do it all at once at the end
        filter = tf.math.logical_or(tf.math.is_nan(slice_forces), magnitude >= (ljatom_diameter_tf*2.5), name="or")
        filtered = tf.compat.v1.where_v2(filter, forces_zeroes_tf, slice_forces, name="where_or")
        forces =  tf.math.reduce_sum(filtered, axis=1)
        return forces