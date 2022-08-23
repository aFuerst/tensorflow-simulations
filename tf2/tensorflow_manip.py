import tensorflow as tf
import os

def toggle_xla(xla):
    if xla:
        tf.config.optimizer.set_jit(xla)

def toggle_cpu(cpu, thread_count=os.cpu_count()):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.threading.set_inter_op_parallelism_threads(thread_count)
        tf.config.threading.set_intra_op_parallelism_threads(thread_count)
        return tf.compat.v1.ConfigProto(intra_op_parallelism_threads=thread_count, inter_op_parallelism_threads=thread_count)
    return None

def manual_optimizer(optimizer):
    if optimizer:
        tf.compat.v1.enable_control_flow_v2()
        # , "pin_to_host_optimization":True # TODO: test for better peformance on high num atom runs?
        tf.config.optimizer.set_experimental_options({'constant_folding': True, "layout_optimizer": True, "shape_optimization":True, 
                        "remapping":True, "arithmetic_optimization":True, "dependency_optimization":True, "loop_optimization":True, 
                        "function_optimization":True, "debug_stripper":True, "scoped_allocator_optimization":True, 
                        "implementation_selector":True, "auto_mixed_precision":True, "debug_stripper": True})

def silence(log=False):
    if not log:
        # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 