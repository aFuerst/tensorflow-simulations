import main

for num_atoms in range(100, 2000, 100):
    print(main.run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms))