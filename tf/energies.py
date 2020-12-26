import tensorflow as tf
import common
tf.compat.v1.logging.set_verbosity('INFO')
"""
Compute the kinetic energy of particles as described by their velocity and mass
Parameters:
    vel: Tensor containing the velocity of particles in the shape [[x,y,z]]
    ljatom_diameter_tf: Tensor containing diameters of the particles, must match positions with `vel`
"""
def kinetic_energy(vel, ljatom_diameter_tf):
    with tf.name_scope("kinetic_energy"):
        # half = tf.constant(0.5, dtype=tf.float64)
        #out_m = tf.Print(vel,[vel[0],vel[1],vel[2],vel[3]])
        m = tf.norm(vel,ord='euclidean', axis=1, keepdims=True)
        #out_m = tf.Print(m,[m[0],m[1],m[2],m[3]])
        #tf.print('out_m:'+str(out_m))
        magnitude = m #common.magnitude(distances)
        ke = tf.math.reduce_sum((0.5 * ljatom_diameter_tf * magnitude * magnitude),name="ke_energy_reduce_sum")
        #out_ke = tf.Print(ke,[ke[0],ke[1],ke[2],ke[3]])
        #tf.print('out_ke:'+str(out_ke))
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
        #tf.print('energy_shift'+str(energy_shift))
        #out_energyShift = tf.Print(energy_shift,[energy_shift])
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: atom_pos - pos, elems=pos)
        distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="edges_half_where")
        distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances, name="neg_edges_half_where")
        #out_dist = tf.Print(distances,[distances[0][1][0],distances[0][1][1],distances[0][1][2],distances[0][2][0],distances[0][2][1],distances[0][2][2],distances[0][3][0],distances[0][3][1], distances[0][3][2]])
        
        m = tf.norm(distances,ord='euclidean', axis=2, keepdims=True)
        magnitude = m #common.magnitude(distances)
        #tf.compat.v1.logging.info('magnitude:'+str(m))

        d_6 = tf.math.pow(ljatom_diameter_tf, 6, name="energy_d_6")
        r_6 = tf.math.pow(magnitude, 6, name="energy_r_6")
        ljpair = four * elj * (d_6 / r_6) * ( ( d_6 / r_6 ) - one ) - energy_shift
        #out_ljpair = tf.Print(ljpair,[ljpair[0][0],ljpair[0][1],ljpair[0][2],ljpair[0][3],ljpair[0][4],ljpair[0][5],ljpair[0][6],ljpair[0][7]])
        #tf.print('filter:'+str(out_ljpair))
        #tf.compat.v1.logging.info('ljpair:'+str(ljpair))

        filter = tf.math.logical_or(tf.math.is_inf(ljpair), magnitude >= (ljatom_diameter_tf*2.5), name="or")
        #out_filter = tf.Print(filter,[filter[0][0],filter[0][1],filter[0][2],filter[0][3],filter[0][4],filter[0][5],filter[0][6],filter[0][7]])
        #tf.print('filter:'+str(out_filter))
        
        filtered = tf.compat.v1.where_v2(filter, energy_zeros, ljpair, name="where_or")
        #out_filtered = tf.Print(filtered,[filtered[0][0],filtered[0][1],filtered[0][2],filtered[0][3],filtered[0][4],filtered[0][5],filtered[0][6],filtered[0][7]])
        #tf.print('filtered:'+str(out_filtered))

        energies =  tf.math.reduce_sum(filtered, axis=1)
        #out_energies = tf.Print(energies,[energies[0],energies[1],energies[2]])#,forces[1][0],forces[1][1],forces[1][2],forces[2][0],forces[2][1], forces[2][2]])
        #tf.print('out_energies:'+str(out_energies))



        ret = tf.math.reduce_sum(energies, name="energy_reduce_sum") / two
        return ret