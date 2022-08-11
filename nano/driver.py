import os
import cluster

def main():
    # Create a Master in a new process
    # master_pid = os.fork()
    # if master_pid == 0:
    master = cluster.Cluster()
    # master.config_cluster()
    master.run_simulation()
    # master.run_tfmd()
    # master.start_surrogate()
    master.destroy_cluster()

if __name__ == "__main__":
    main()
    # os.wait()