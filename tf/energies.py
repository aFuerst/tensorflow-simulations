import tensorflow as tf

"""
Compute the kinetic energy of particles as described by their velocity and mass
Parameters:
    vel: Tensor containing the velocity of particles in the shape [[x,y,z]]
    ljatom_diameter_tf: Tensor containing diameters of the particles, must match positions with `vel`
"""
def kinetic_energy(vel, ljatom_diameter_tf):
    with tf.name_scope("kinetic_energy"):
        m = tf.norm(vel,ord='euclidean', axis=1, keepdims=True)
        magnitude = m #common.magnitude(distances)
        #out = tf.Print(ljatom_diameter_tf,[ljatom_diameter_tf])
        ke = tf.math.reduce_sum((0.5 * ljatom_diameter_tf * magnitude * magnitude),name="ke_energy_reduce_sum")
        return ke

"""
Compute the kinetic energy of particles as described by their velocity and mass
Parameters:
    pos: Tensor containing the position of particles in the shape [[x,y,z]]
    edges_half: Edges of the simulation box as a tensor in shape [x,y,z], divided by two
    neg_edges_half: Negation of Tensor `edges_half`
    edges: Edges of the simulation box as a tensor in shape [x,y,z]
    ljatom_diameter_tf: Tensor containing diameters of the particles, must match positions with `pos`
"""
def potential_energy(pos, edges_half, neg_edges_half, edges, ljatom_diameter_tf):
    with tf.name_scope("potential_energy"):
        dcut = tf.constant(2.5, dtype=pos.dtype, name="dcut")
        energy_zeros = tf.constant(0.0, dtype=pos.dtype, name="zeros")
        elj = 1.0
        four = 4.0
        one = 1.0
        two = 2.0
        dcut_6 = tf.pow(dcut, 6, name="dcut_6")
        dcut_12 = tf.pow(dcut, 12, name="dcut_12")
        energy_shift = four*elj*(one/dcut_12 - one/dcut_6)
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: atom_pos - pos, elems=pos)
        distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="edges_half_where")
        distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances, name="neg_edges_half_where")
        m = tf.norm(distances,ord='euclidean', axis=2, keepdims=True)
        magnitude = m #common.magnitude(distances)
        d_6 = tf.math.pow(ljatom_diameter_tf, 6, name="energy_d_6")
        r_6 = tf.math.pow(magnitude, 6, name="energy_r_6")
        ljpair = four * elj * (d_6 / r_6) * ( ( d_6 / r_6 ) - one ) - energy_shift
        filter = tf.math.logical_or(tf.math.is_inf(ljpair), magnitude >= (ljatom_diameter_tf*2.5), name="or")       
        filtered = tf.compat.v1.where_v2(filter, energy_zeros, ljpair, name="where_or")
        energies =  tf.math.reduce_sum(filtered, axis=1)
        ret = tf.math.reduce_sum(energies, name="energy_reduce_sum") / two
        return ret