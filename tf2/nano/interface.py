import utility, common
import math
import numpy as np
import tensorflow as tf
from common import py_array_to_np as conv

ion_for_str = "ion_forces"
ion_pos_str = "ion_pos"
ion_charges_str = "ion_charges"
ion_diameters_str = "ion_diameters"
ion_masses_str = "ion_masses"
ion_epsilon_str = "ion_epsilon"
ion_valency_str = "ion_valency"

class Interface:
    #def __init__(self, salt_conc_in: float, salt_conc_out: float, salt_valency_in: int, salt_valency_out: int, bx: float, by: float, bz: float, initial_ein: float=80, initial_eout: float=80):
    def __init__(self, salt_conc_in, salt_conc_out: float, salt_valency_in: int, salt_valency_out: int, bx: float, by: float, bz: float, initial_ein: float=80, initial_eout: float=80):
        self.salt_conc_in = salt_conc_in
        self.salt_conc_out = salt_conc_out
        self.salt_valency_in = salt_valency_in
        self.salt_valency_out = salt_valency_out
        self.ein = initial_ein
        self.eout = initial_eout
        #   useful combinations of different dielectric constants (inside and outside)
        self.em = 0.5 * (self.ein + self.eout)
        self.ed = (self.eout - self.ein) / (4 * math.pi)

        # useful length scales signifying competition between electrostatics and entropy
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
        print("box_size:", " bx:", bx, " by:", by, " bz:", bz)
        self.lx = bx
        self.ly = by
        self.lz = bz

    def put_saltions_inside(self, pz: int, nz: int, concentration: float, positive_diameter_in: float, negative_diameter_in: float, counterions: int, valency_counterion: int, counterion_diameter_in: float, bigger_ion_diameter: float, crystal_pack: bool):
        # establish the number of inside salt ions first
        # Note: salt concentration is the concentration of one kind of ions, also the factor of 0.6 is there in order to be consistent with units.

        volume_box = self.lx*self.ly*self.lz

        total_nions_inside = int((concentration * 0.6022) * (volume_box * utility.unitlength * utility.unitlength * utility.unitlength))
        total_nions_inside = 4
        if (total_nions_inside % pz !=0):
            total_nions_inside = total_nions_inside - (total_nions_inside % pz) + pz

        total_pions_inside = abs(nz) * total_nions_inside / pz
        total_pions_inside = 4
        total_saltions_inside = total_nions_inside + total_pions_inside + counterions
        print("total_saltions_inside", total_saltions_inside)

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

        # generate salt ions inside
        ion_pos = []
        ion_diameter = []
        ion_valency = []
        ion_charges = []
        ion_masses = []
        ion_epsilon = []
        if not crystal_pack:
            while (len(ion_pos) != total_saltions_inside):
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
                    if (common.magnitude_np(posvec - ion_pos[i], axis=0) <= (0.5*bigger_ion_diameter+0.5*ion_diameter[i])):
                        continuewhile = True
                    i+=1
                if (continuewhile == True):
                    continue
                if (len(ion_pos) < counterions):
                    ion_diameter.append(counterion_diameter_in)
                    ion_valency.append(valency_counterion)
                    ion_charges.append(valency_counterion*1.0)
                    ion_masses.append(1.0)
                    ion_epsilon.append(self.ein)
                elif (len(ion_pos) >= counterions and len(ion_pos) < (total_pions_inside + counterions)):
                    ion_diameter.append(positive_diameter_in)
                    ion_valency.append(pz)
                    ion_charges.append(pz*1.0)
                    ion_masses.append(1.0)
                    ion_epsilon.append(self.ein)
                else:
                    ion_diameter.append(negative_diameter_in)
                    ion_valency.append(nz)
                    ion_charges.append(nz*1.0)
                    ion_masses.append(1.0)
                    ion_epsilon.append(self.ein)
                ion_pos.append(posvec)			# copy the salt ion to the stack of all ions
        else:
            num_ions_in_lx = int(self.lx/ bigger_ion_diameter)
            num_ions_in_ly = int(self.ly/ bigger_ion_diameter)
            num_ions_in_lz = int(self.lz/ bigger_ion_diameter)

            for i in range(num_ions_in_lx):
                x = (-self.lx/2 + (0.5 * bigger_ion_diameter)) + i * bigger_ion_diameter
                for j in range(num_ions_in_ly):
                    y = (-self.ly/2 + (0.5 * bigger_ion_diameter)) + j * bigger_ion_diameter
                    for k in range(num_ions_in_lz):
                        if len(ion_pos) < total_saltions_inside:
                            z = (-self.lz/2 + (0.5 * bigger_ion_diameter)) + k * bigger_ion_diameter
                            posvec = np.array([x,y,z])
                            if (x > ((self.lx/2)-(0.5 * bigger_ion_diameter)) or y > ((self.ly/2)-(0.5 * bigger_ion_diameter)) or z > ((self.lz/2)-(0.5 * bigger_ion_diameter))):
                                continue
                            if (len(ion_pos) < counterions):
                                ion_diameter.append(counterion_diameter_in)
                                ion_valency.append(valency_counterion)
                                ion_charges.append(valency_counterion*1.0)
                                ion_masses.append(1.0)
                                ion_epsilon.append(self.ein)
                            elif (len(ion_pos) >= counterions and len(ion_pos) < (total_pions_inside + counterions)):
                                ion_diameter.append(positive_diameter_in)
                                ion_valency.append(pz)
                                ion_charges.append(pz*1.0)
                                ion_masses.append(1.0)
                                ion_epsilon.append(self.ein)
                            else:
                                ion_diameter.append(negative_diameter_in)
                                ion_valency.append(nz)
                                ion_charges.append(nz*1.0)
                                ion_masses.append(1.0)
                                ion_epsilon.append(self.ein)
                            ion_pos.append(posvec)
        # print("\n Positions: put_salt_ions_inside:",ion_pos)   values verified with C++ code
        return {ion_pos_str:conv(ion_pos), ion_charges_str:conv(ion_charges),\
                 ion_masses_str:conv(ion_masses), ion_diameters_str:conv(ion_diameter), ion_epsilon_str:conv(ion_epsilon), ion_valency_str:conv(ion_valency)}  #, ion_valency_str:conv(ion_valency)}
        
    def discretize(self, smaller_ion_diameter: float, f: float, charge_meshpoint: float):
        print("charge_meshpoint", charge_meshpoint)
        self.width = f * self.lx
        nx = int(self.lx / self.width)
        ny = int(self.ly / self.width)
        left_plane = {"posvec":[], "q":[], "epsilon":[], "a":[], "normalvec":[]}
        right_plane = {"posvec":[], "q":[], "epsilon":[], "a":[], "normalvec":[]}
        area = self.width * self.width
                
        # creating a discretized hard wall interface at z = - l/2
        for j in range(ny):
            for i in range(nx):
                position = conv([-0.5*self.lx+0.5*smaller_ion_diameter+i*self.width, -0.5*self.ly+0.5*smaller_ion_diameter+j*self.width, -0.5*self.lz])
                normal = conv([0,0,-1])
                left_plane["posvec"].append(position)
                left_plane["q"].append(charge_meshpoint)
                left_plane["epsilon"].append(self.eout)
                left_plane["a"].append(area)
                left_plane["normalvec"].append(normal)

        # creating a discretized hard wall interface at z = l/2
        for j in range(ny):
            for i in range(nx):
                position = conv([-0.5*self.lx+0.5*smaller_ion_diameter+i*self.width, -0.5*self.ly+0.5*smaller_ion_diameter+j*self.width, 0.5*self.lz])
                normal = conv([0,0,1])
                right_plane["posvec"].append(position)
                right_plane["q"].append(charge_meshpoint)
                right_plane["epsilon"].append(self.eout)
                right_plane["a"].append(area)
                right_plane["normalvec"].append(normal)
        for key in left_plane.keys():
            left_plane[key] = conv(left_plane[key])
        for key in right_plane.keys():
            right_plane[key] = conv(right_plane[key])
        self.left_plane = left_plane
        self.right_plane = right_plane
        self.tf_left_plane, self.tf_place_left_plane = common.make_tf_versions_of_dict(left_plane)
        self.tf_right_plane, self.tf_place_right_plane = common.make_tf_versions_of_dict(right_plane)


    def electrostatics_between_walls(self):
        r_vec_walls = self.right_plane["posvec"] - self.left_plane["posvec"]
        # print("\n r_vec_walls:", type(r_vec_walls))
        r_vec_walls_magnitude = common.magnitude_np(r_vec_walls, axis=1)
        fqq_walls = 0.5 * self.right_plane["q"] * self.left_plane["q"] * 0.5 * (1.0/self.right_plane["epsilon"]+1.0/self.left_plane["epsilon"]) / r_vec_walls_magnitude
        # print("\n fqq_walls:", len(fqq_walls))
        # fqq_walls = tf.math.reduce_sum(fqq_walls, axis=1)
        electrostatics_between_walls = tf.math.reduce_sum(input_tensor=fqq_walls, axis=0)
        return electrostatics_between_walls*utility.scalefactor
