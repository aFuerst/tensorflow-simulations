import tensorflow as tf
import common
tf.compat.v1.logging.set_verbosity('INFO')
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
        #out_pos = tf.Print(pos,[pos[0][0],pos[0][1],pos[0][2],pos[1][0],pos[1][1],pos[1][2]])
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: atom_pos - pos, elems=pos)
        #out_dist = tf.Print(distances,[distances[0][1][0],distances[0][1][1],distances[0][1][2],distances[0][2][0],distances[0][2][1],distances[0][2][2],distances[0][3][0],distances[0][3][1], distances[0][3][2]])
        distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances, name="where_neg_edges_half")
        m = tf.norm(distances,ord='euclidean', axis=2, keepdims=True)
        
        magnitude = m #common.magnitude(distances)
        #out_mag = tf.Print(magnitude,[m[0][0],m[0][1],m[0][2],m[0][3], m[0][4],m[0][5],m[0][6],m[0][7]])
        twelve = tf.math.pow(ljatom_diameter_tf, 12.0, name="diam_12_pow") / tf.math.pow(magnitude, 12.0, name="mag_12_pow")    #()
        six = tf.math.pow(ljatom_diameter_tf, 6.0, name="diam_6_pow") / tf.math.pow(magnitude, 6.0, name="mag_6_pow")    #(8, 8, 3)
        mag_squared = 1.0 / tf.math.pow(magnitude,2, name="mag_sqr_pow")
        #out_twelve = tf.Print(twelve,[twelve[0][1],twelve[0][2],twelve[0][3],twelve[0][4],twelve[0][5],twelve[0][6],twelve[0][7]])
        #out_six = tf.Print(six,[six[0][1],six[0][2],six[0][3],six[0][4],six[0][5],six[0][6],six[0][7]])
        #out_mag_squared = tf.Print(mag_squared,[mag_squared[0][1],mag_squared[0][2],mag_squared[0][3],mag_squared[0][4],mag_squared[0][5],mag_squared[0][6],mag_squared[0][7]])
        
        slice_forces = distances * (48.0 * 1.0 * (((twelve - 0.5 * six) * mag_squared)))
        #out_slice_forces = tf.Print(slice_forces,[slice_forces[0][0][0],slice_forces[0][0][1],slice_forces[0][0][2],slice_forces[0][1][0],slice_forces[0][1][1],slice_forces[0][1][2],slice_forces[0][2][0],slice_forces[0][2][1],slice_forces[0][2][2],slice_forces[0][3][0],slice_forces[0][3][1], slice_forces[0][3][2]])
        #tf.print('slice_forces:'+str(slice_forces))
        # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
        # can't see a way to remove that case in all above computations, easier to do it all at once at the end
        filter = tf.math.logical_or(tf.math.is_nan(slice_forces), magnitude >= (ljatom_diameter_tf*2.5), name="or")
        #tf.print('filter:'+str(filter))
        #out_filter = tf.Print(filter,[filter[0][0][0],filter[0][0][1],filter[0][0][2],filter[0][1][0],filter[0][1][1],filter[0][1][2],filter[0][2][0],filter[0][2][1],filter[0][2][2],filter[0][3][0],filter[0][3][1], filter[0][3][2]])
        
        filtered = tf.compat.v1.where_v2(filter, forces_zeroes_tf, slice_forces, name="where_or")
        #out_filtered = tf.Print(filtered,[filtered[0][0][0],filtered[0][0][1],filtered[0][0][2],filtered[0][1][0],filtered[0][1][1],filtered[0][1][2],filtered[0][2][0],filtered[0][2][1],filtered[0][2][2],filtered[0][3][0],filtered[0][3][1], filtered[0][3][2],
        #filtered[0][4][0],filtered[0][4][1],filtered[0][4][2],filtered[0][5][0],filtered[0][5][1],filtered[0][5][2],filtered[0][6][0],filtered[0][6][1],filtered[0][6][2],filtered[0][7][0],filtered[0][7][1],filtered[0][7][2],])
        #tf.print('filtered:'+str(filtered))

        forces =  tf.math.reduce_sum(filtered, axis=1)
        #out_forces = tf.Print(forces,[forces[0][0],forces[0][1],forces[0][2]])#,forces[1][0],forces[1][1],forces[1][2],forces[2][0],forces[2][1], forces[2][2]])
        #tf.print('out_forces:'+str(out_forces))
        return forces