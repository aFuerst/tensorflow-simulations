class Control:
    def __init__(self, args):
        self.verbose=args.verbose
        self.timestep=args.delta_t	# timestep used in molecular dynamics
        self.steps=args.steps			# number of steps in molecular dynamics
        self.hiteqm=100000		# wait till this step
        self.freq=100 			# frequency of sampling
        self.extra_compute=10000		# energy computed after these many steps
        self.verify=None			# verification with correct answer for ind. density done after these many steps # not relevant for this code
        self.writeverify=None		# write the verification files # not relevant for this code
        self.writedensity=100000 		# write the density files
        self.moviefreq=10000		# frequency of making movie files