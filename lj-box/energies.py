import tensorflow as tf
import common

"""
Compute the kinetic energy of particles as described by their velocity and mass
Parameters:
    vel: Tensor containing the velocity of particles in the shape [[x,y,z]]
    ljatom_diameter_tf: Tensor containing diameters of the particles, must match positions with `vel`
"""
def kinetic_energy(vel, ljatom_diameter_tf):
    with tf.name_scope("kinetic_energy"):
        half = tf.constant(0.5, dtype=tf.float64)
        magnitude = common.magnitude(vel)
        return tf.reduce_sum(half * ljatom_diameter_tf * magnitude * magnitude)

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
        elj = tf.constant(1.0, dtype=pos.dtype, name="elj")
        four = tf.constant(4.0, dtype=tf.float64)
        two = tf.constant(2.0, dtype=tf.float64)
        one = tf.constant(1.0, dtype=tf.float64)
        dcut_6 = tf.pow(dcut, 6, name="dcut_6")
        dcut_12 = tf.pow(dcut, 12, name="dcut_12")
        energy_shift = four*elj*(one/dcut_12 - one/dcut_6)
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: pos - atom_pos, elems=pos)
        distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="edges_half_where")
        distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances, name="neg_edges_half_where")
        magnitude = common.magnitude(distances)
        d_6 = tf.pow(ljatom_diameter_tf, 6, name="energy_d_6")
        r_6 = tf.pow(magnitude, 6, name="energy_r_6")
        ljpair = four * elj * (d_6 / r_6) * ( ( d_6 / r_6 ) - one ) - energy_shift
        ret = tf.reduce_sum(ljpair, name="energy_reduce_sum") / two
        return ret