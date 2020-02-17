import utility, common
import math
import numpy as np

class Interface:
    def __init__(self, salt_conc_in: float, salt_conc_out: float, salt_valency_in: int, salt_valency_out: int, bx: float, by: float, bz: float, initial_ein: float=1, initial_eout: float=1):
        self.ein = initial_ein
        self.eout = initial_eout
        #   useful combinations of different dielectric constants (inside and outside)
        self.em = 0.5 * (self.ein + self.eout)
        self.ed = (self.eout - self.ein) / (4 * math.pi)

        # // useful length scales signifying competition between electrostatics and entropy
        self.lB_in = (utility.lB_water * utility.epsilon_water / self.ein) / utility.unitlength
        self.lB_out = (utility.lB_water * utility.epsilon_water / self.eout) / utility.unitlength
        if (salt_conc_in != 0):
            self.inv_kappa_in = (0.257 / (salt_valency_in * math.sqrt(self. lB_in * utility.unitlength * salt_conc_in))) / utility.unitlength
            self.mean_sep_in = pow(1.2 * salt_conc_in, -1.0/3.0) / utility.unitlength
        else:
            self.inv_kappa_in = 0
            self.mean_sep_in = 0
        if (salt_conc_out != 0):
            self.inv_kappa_out = (0.257 / (salt_valency_out * math.sqrt(self.lB_out * utility.unitlength * salt_conc_out))) / utility.unitlength
            self.mean_sep_out = pow(1.2 * salt_conc_out, -1.0/3.0) / utility.unitlength
        else:
            self.inv_kappa_out = 0
            self.mean_sep_out = 0

        # simulation box size (in reduced units)
        self.lx = bx
        self.ly = by
        self.lz = bz

    def put_saltions_inside(self, pz: int, nz: int, concentration: float, positive_diameter_in: float, negative_diameter_in: float, counterions: int, valency_counterion: int, counterion_diameter_in: float, bigger_ion_diameter: float):
        # establish the number of inside salt ions first
        # Note: salt concentration is the concentration of one kind of ions.
        # also the factor of 0.6 is there in order to be consistent with units.

        volume_box = self.lx*self.ly*self.lz

        total_nions_inside = int((concentration * 0.6022) * (volume_box * utility.unitlength * utility.unitlength * utility.unitlength))
        if (total_nions_inside % pz !=0):
            total_nions_inside = total_nions_inside - (total_nions_inside % pz) + pz

        total_pions_inside = abs(nz) * total_nions_inside / pz
        total_saltions_inside = total_nions_inside + total_pions_inside + counterions

        # express diameter in consistent units
        bigger_ion_diameter = bigger_ion_diameter / utility.unitlength # the bigger_ion_diameter can be cation or anion depending on their sizes
        positive_diameter_in = positive_diameter_in / utility.unitlength
        negative_diameter_in = negative_diameter_in / utility.unitlength
        counterion_diameter_in = counterion_diameter_in / utility.unitlength

        # distance of closest approach between the ion and the interface
        # choosing bigger_ion_diameter to define distance of closest approach helps us to avoid overlapping the ions when we generate salt ions inside
        r0_x = 0.5 * self.lx - 0.5 * bigger_ion_diameter
        r0_y = 0.5 * self.ly - 0.5 * bigger_ion_diameter
        r0_z = 0.5 * self.lz - 0.5 * bigger_ion_diameter

        np.random.seed(0) # be consistent

        # generate salt ions inside
        saltion_in_pos = []
        ion_pos = []
        ion_diameter = []
        ion_valency = []
        ion_charges = []
        ion_masses = []
        ion_discont = []
        while (len(saltion_in_pos) != total_saltions_inside):
            x = np.random.random()
            x = (1 - x) * (-r0_x) + x * (r0_x)
            
            y = np.random.random()
            y = (1 - y) * (-r0_y) + y * (r0_y)
            
            z = np.random.random()
            z = (1 - z) * (-r0_z) + z * (r0_z)
            
            posvec = np.asarray([x,y,z])
            continuewhile = False
            i = 0
            while (i < len(ion_pos) and continuewhile == False): # ensure ions are far enough apart
                if (common.magnitude_np(posvec - ion_pos[i]) <= (0.5*bigger_ion_diameter+0.5*ion_diameter[i])):
                    continuewhile = True
            if (continuewhile == True):
                continue
            if (len(saltion_in_pos) < counterions):
                ion_diameter.append(counterion_diameter_in)
                ion_valency.append(valency_counterion)
                ion_charges.append(valency_counterion*1.0)
                ion_masses.append(1.0)
                ion_discont.append(self.ein)
            elif (len(saltion_in_pos) >= counterions and len(saltion_in_pos) < (total_pions_inside + counterions)):
                ion_diameter.append(positive_diameter_in)
                ion_valency.append(pz)
                ion_charges.append(pz*1.0)
                ion_masses.append(1.0)
                ion_discont.append(self.ein)
            else:
                ion_diameter.append(negative_diameter_in)
                ion_valency.append(nz)
                ion_charges.append(nz*1.0)
                ion_masses.append(1.0)
                ion_discont.append(self.ein)
            saltion_in_pos.append(posvec)		# create a salt ion
            ion_pos.append(posvec)			# copy the salt ion to the stack of all ions
        return np.array(saltion_in_pos), np.array(ion_pos), np.array(ion_charges), np.array(ion_masses), np.array(ion_diameter), np.array(ion_discont)