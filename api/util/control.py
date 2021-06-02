class Control:
    def __init__(self, args):
        # self.freq=args.freq 			# frequency of sampling
        self.hiteqm=args.hiteqm		# wait till this step
        self.extra_compute=args.extracompute		# energy computed after these many steps
        self.writedensity=args.writedensity 		# write the density files
        self.moviefreq=args.moviefreq		# frequency of making movie files        self.verbose=args.verbose
        self.timestep=args.delta_t	# timestep used in molecular dynamics
        self.steps=args.steps 
        self.freq=args.freq			# number of steps in molecular dynamics
        self.validate=args.validate
        self.ein=args.ein
        self.eout=args.eout