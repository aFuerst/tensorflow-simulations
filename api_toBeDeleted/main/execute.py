# from ..simulation import nano_init

import api.surrogate.surrogate_md as surrogate


class Execute:
    def __init__(self):
        self.surrogate = surrogate.Surrogate()
        # Maybe read the config file here if any
        return

    def run_simulation(self):
        # nano_init.start()
        return

    def run_surrogate(self):
        self.surrogate.load_data()
        self.surrogate.initialize_model()
        self.surrogate.train_model()
        return

    # def generate_bulk_input(self):
    #     for Z_val in [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]:
    #         for p_val in [1, 2, 3]:
    #             for n_val in [-1, - 2]:
    #                 for c_val in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    #                     for d_val in [0.5, 0.55, 0.553, 0.6, 0.65, 0.7, 0.714, 0.75]:
    #                         pass