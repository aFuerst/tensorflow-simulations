import nano
import nano.surrogate_md


class Execute:
    def __init__(self):
        self.surrogate = nano.surrogate_md.Surrogate()
        # Maybe read the config file here if any
        return

    def run_simulation(self):
        nano.nano_init()
        return

    def run_surrogate(self):
        self.surrogate.load_data()
        self.surrogate.initialize_model()
        self.surrogate.train_model()
        return




