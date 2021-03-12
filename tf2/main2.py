import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from box import Box
import energies, forces
import tensorflow_manip
import time, argparse, shutil, math, sys, os
import statistics



pi = 3.141593
kB = 1.38e-23
mol = 6.0e-23
unit_len = 0.4505e-9
unit_energy = 119.8 * kB # Joules; unit of energy is these many Joules (typical interaction strength)
unit_mass = 0.03994 / mol # kg; unit of mass (mass of a typical atom)
unit_time = math.sqrt(unit_mass * unit_len * unit_len / unit_energy) # Unit of time
unit_temp = unit_energy/kB # unit of temperature
df_type=np.float64

temperature = 1.0
dcut = 1 # cutoff distance for potential in reduced units

def update_pos(pos, vel, edges_half, neg_edges_half, edges, delta_t):
    with tf.name_scope("update_pos"):
        pos_graph = pos + (vel * delta_t)
        pos_graph = tf.compat.v1.where_v2(pos_graph > edges_half, pos_graph - edges, pos_graph, name="where_edges_half")
        return tf.compat.v1.where_v2(pos_graph < neg_edges_half, pos_graph + edges, pos_graph, name="where_neg_edges_half")

def update_vel(vel, force, delta_t, ljatom_mass_tf):
    with tf.name_scope("update_vel"):
        temp_vel = vel + (force * (0.5*delta_t/ljatom_mass_tf))
        return temp_vel

def velocity_verlet(curr_iter, vel, pos, force, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf, pe_graph, ke_graph):
    with tf.name_scope("run_one_iter"):
        if(curr_iter==0):
            force = forces.lj_force(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf, ljatom_diameter_tf)
            pe_graph = energies.potential_energy(pos, edges_half, neg_edges_half, edges, ljatom_diameter_tf)
            ke_graph = energies.kinetic_energy(vel, ljatom_mass_tf)
        vel_graph = update_vel(vel, force, delta_t, ljatom_mass_tf)
        pos_graph = update_pos(pos, vel_graph, edges_half, neg_edges_half, edges, delta_t)
        force_graph = forces.lj_force(pos_graph, edges_half, neg_edges_half, edges, forces_zeroes_tf, ljatom_diameter_tf)
        pe_graph = energies.potential_energy(pos_graph, edges_half, neg_edges_half, edges, ljatom_diameter_tf)
        vel_graph = update_vel(vel_graph, force_graph, delta_t, ljatom_mass_tf)
        ke_graph = energies.kinetic_energy(vel_graph, ljatom_mass_tf)
        return vel_graph, pos_graph, force_graph, pe_graph, ke_graph

def save_energies(filename, kinetics, potentials, ctr):
    with open(filename, "a+") as text_file:
        print_str = str(ctr)+"  "+str(kinetics)+"  "+str(potentials)+"    "+str(potentials+kinetics)
        text_file.write(f'\n{print_str}')

def save(path, id, velocities, positions, forces):
    np.savetxt(os.path.join(path, "{}-forces".format(id)), forces)
    np.savetxt(os.path.join(path, "{}-velocities".format(id)), velocities)
    np.savetxt(os.path.join(path, "{}-positions".format(id)), positions)

# def toggle_xla(xla):
#     if xla:
#         tf.config.optimizer.set_jit(xla)

# def toggle_cpu(cpu, thread_count):
#     if cpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         tf.config.threading.set_inter_op_parallelism_threads(thread_count)
#         tf.config.threading.set_intra_op_parallelism_threads(thread_count)
#         return tf.compat.v1.ConfigProto(intra_op_parallelism_threads=thread_count, inter_op_parallelism_threads=thread_count)
#     return None

# def manual_optimizer(optimizer):
#     if optimizer:
#         # , "pin_to_host_optimization":True
#         tf.config.optimizer.set_experimental_options({'constant_folding': True, "layout_optimizer": True, "shape_optimization":True, 
#                         "remapping":True, "arithmetic_optimization":True, "dependency_optimization":True, "loop_optimization":True, 
#                         "function_optimization":True, "debug_stripper":True, "scoped_allocator_optimization":True, 
#                         "implementation_selector":True, "auto_mixed_precision":True, "debug_stripper": True})

def run_simulation(subfolder, totaltime=10, steps=10000, number_ljatom=108, ljatom_density=0.8442, sess=None, profile=False, xla=True, force_cpu=False, optimizer=False, thread_count=os.cpu_count()):
    log_freq = 100
    tensorflow_manip.toggle_xla(xla)
    tensorflow_manip.manual_optimizer(optimizer)
    config = tensorflow_manip.toggle_cpu(force_cpu, thread_count)
    num_iterations = steps // log_freq
    delta_t = totaltime / steps
    tot_pe = 0.0
    tot_ke = 0.0
    avg_ke_arr = []
    avg_pe_arr = []
    curr_iter = 0
    samples = 1
    hit_eqm = 5000
    bx = by = bz = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
    box = Box(bx, by, bz, df_type)
    positions, number_ljatom, masses, diameters = box.fill(number_ljatom, atom_diam=1)
    ljatom_mass_tf = tf.constant(masses, dtype=df_type)
    ljatom_diameter_tf = tf.constant(diameters, dtype=df_type)
    forces = np.zeros(positions.shape, dtype=df_type)
    velocities = np.zeros(positions.shape, dtype=df_type)
    potentials = np.zeros((1), dtype=df_type)
    kinetics = np.zeros((1), dtype=df_type)
    edges = box.get_edges_as_tf()
    edges_half = edges / 2.0
    neg_edges_half = tf.negative(edges_half)
    forces_zeroes_tf = tf.constant(np.zeros((number_ljatom, 3)), dtype=df_type)
    position_p = tf.compat.v1.placeholder(dtype=df_type, shape=positions.shape, name="position_placeholder_n")
    forces_p = tf.compat.v1.placeholder(dtype=df_type, shape=forces.shape, name="forces_placeholder_n")
    velocities_p = tf.compat.v1.placeholder(dtype=df_type, shape=velocities.shape, name="velocities_placeholder_n")
    potentials_p = tf.compat.v1.placeholder(dtype=df_type, shape=potentials.shape, name="potentials_placeholder_n")
    kinetics_p = tf.compat.v1.placeholder(dtype=df_type, shape=kinetics.shape, name="potentials_placeholder_n")

    if sess is None:
        sess = tf.compat.v1.Session(config=config)
        sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    v_g, p_g, f_g, pe_g, ke_g = velocity_verlet(curr_iter, velocities_p, position_p, forces_p, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, 
        ljatom_diameter_tf, potentials_p, kinetics_p)
    if profile:
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    # writer = tf.compat.v1.summary.FileWriter(subfolder)
    # writer.add_graph(sess.graph)
    #comp_start = time.time()
    for curr_iter in range(steps):
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities, potentials_p:potentials, kinetics_p:kinetics}
        velocities, positions, forces, potentials, kinetics = sess.run([v_g, p_g, f_g, pe_g, ke_g], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        avg_pe = potentials/number_ljatom 
        avg_ke = kinetics/number_ljatom
        #save_energies(os.path.join(subfolder, "energies-{}-{}.out".format(ljatom_density, number_ljatom)),round(avg_ke,3), round(avg_pe,3), round(curr_iter,3))
        if(curr_iter>hit_eqm and curr_iter%log_freq==0):
            avg_ke_arr.append(avg_ke)
            avg_pe_arr.append(avg_pe)
            tot_ke+=kinetics
            tot_pe+=potentials
            samples += 1

        # if profile:
        #     from tensorflow.python.client import timeline
        #     tl = timeline.Timeline(run_metadata.step_stats)
        #     ctf = tl.generate_chrome_trace_format()
        #     writer.add_run_metadata(run_metadata, 'step%d' % curr_iter)
        #     with open(os.path.join(subfolder, "{}-timeline.json".format((1+curr_iter)*log_freq)), 'w') as f:
        #         f.write(ctf)
        potentials = np.zeros((1), dtype=df_type)
        kinetics = np.zeros((1), dtype=df_type)
    
    pe_stdev = statistics.stdev(avg_ke_arr)
    ke_stdev = statistics.stdev(avg_pe_arr)
    temp_stdev = statistics.stdev(map(lambda x:2*x/3,avg_ke_arr))
    cumm_avg_pe = tot_pe/(samples*number_ljatom)
    cumm_avg_ke = tot_ke/(samples*number_ljatom)
    avg_temp = (2*cumm_avg_ke)/3

    print(ljatom_density, " ",round(cumm_avg_pe,3), " ", round(avg_temp,3)," ",round(pe_stdev,3)," ", round(temp_stdev,3))
    #meta_graph_def = tf.compat.v1.train.export_meta_graph(filename='./tf/outputs/my-model.meta')
    #exec_time = time.time() - comp_start
    return (cumm_avg_pe, avg_temp, pe_stdev, temp_stdev)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--cpu", action="store_true")
    parser.add_argument('-x', "--xla", action="store_true")
    parser.add_argument('-r', "--prof", action="store_true")
    parser.add_argument('-o', "--opt", action="store_true")
    parser.add_argument('-g', "--generate", action="store", default=0, type=int)
    parser.add_argument('-p', "--parts", action="store", default=108, type=int)
    parser.add_argument('-s', "--steps", action="store", default=10000, type=int)
    parser.add_argument('-t', "--time", action="store", default=10, type=int)
    parser.add_argument('-d', "--density", action="store", default=0.1, type=float)
    parser.add_argument("--threads", action="store", default=os.cpu_count(), type=int)
    args = parser.parse_args()
    
    # main_folder = "./outputs"
    # if not os.path.exists(main_folder):
    #     os.mkdir(main_folder)
    # subfolder = os.path.join(main_folder, "output-{}-{}-{}-{}".format(args.threads, args.time, args.steps, args.parts))
    # if os.path.exists(subfolder):
    #     shutil.rmtree(subfolder)
    # os.mkdir(subfolder)
    subfolder = ''
    if args.generate:
        densities = np.arange(args.density, 0.95, 0.01)
        for rho in densities: 
            rho = round(rho,3)
            (potential_energy, temp, pe_stdev, temp_stdev) = run_simulation(subfolder, profile=args.prof, xla=args.xla, force_cpu=args.cpu, number_ljatom=args.parts, steps=args.steps, ljatom_density=rho, totaltime=args.time, optimizer=args.opt, thread_count=args.threads) 
            # with open(os.path.join(subfolder, "data_dump-{}-{}.out".format(args.parts,args.steps)), 'a') as f:
            #     f.write("\n"+str(rho)+"\t"+str(potential_energy)+"\t"+str(temp)+"\t"+str(pe_stdev)+"\t"+str(temp_stdev))
            
    else:
        (potential_energy, temp, pe_stdev, temp_stdev) = run_simulation(subfolder, profile=args.prof, xla=args.xla, force_cpu=args.cpu, number_ljatom=args.parts, steps=args.steps, ljatom_density=args.density, totaltime=args.time, optimizer=args.opt, thread_count=args.threads)
    
    
