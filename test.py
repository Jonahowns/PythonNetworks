import nmodels as n
import numpy as np
import math
import json
from sklearn.cluster import KMeans
from copy import copy


def kmeans(X, K):
    km = KMeans(K).fit(X)
    return km.cluster_centers_

def lennard_jones_energy(rij, parameters):
    # Assumes you checked it is within distance cutoff
    excl_eps = 4.0
    c_dis = np.linalg.norm(rij)
    if(c_dis > parameters['rstar']): # rstar
        rrc = c_dis-parameters['rc']
        energy = excl_eps*parameters['b']*rrc**2
    else:
        tmp = parameters['sigma']**2 / c_dis**2
        lj_part = tmp**3
        energy = 4*excl_eps*(lj_part**2- lj_part)

    return energy

# class lj:
#     def __init__(self):
#
#     def energy(self, rij, sigma):
#         excl_eps = 4.0
#         c_dis = np.linalg.norm(rij)
#         if c_dis ** 2 >= sigma ** 2:
#             energy = 0.
#             return energy
#         else:
#             tmp = sigma ** 2 / c_dis ** 2
#             lj_part = tmp ** 3
#             energy = 4 * excl_eps * (lj_part ** 2 - lj_part)
#             return energy
#
#     def force(self, rij, sigma):
#         excl_eps = 4.0
#         c_dis = np.linalg.norm(rij)
#         if c_dis**2 >= sigma**2:
#             force = np.asarray([0., 0., 0.])
#             return force
#         else:
#             tmp = sigma ** 2 / c_dis ** 2
#             lj_part = tmp ** 3
#             # energy = 4 * excl_eps * (lj_part ** 2 - lj_part)
#             force = -rij * (24 * excl_eps * (lj_part - 2 * lj_part ** 2) / c_dis ** 2)
#             return force
#
#     def laplacian(self, rij, sigma):
#         excl_eps = 4.0
#         c_dis = np.linalg.norm(rij)
#         if c_dis ** 2 >= sigma ** 2:
#             l = np.asarray([0., 0., 0.])
#             return l
#         else:
#             tmp = sigma ** 2 / c_dis ** 2
#             lj_part = tmp ** 3
#             l = rij*24*excl_eps*(-7/c_dis**3 * lj_part + 26/c_dis**3 * lj_part**2)
#             return l


def lennard_jones_forces(rij, parameters):
    # Assumes you checked it is within distance cutoff
    excl_eps = 4.0
    c_dis = np.linalg.norm(rij)
    if (c_dis > parameters['rstar']):  # rstar
        rrc = c_dis - parameters['rc']
        # energy = excl_eps * parameters[3] * rrc ** 2
        force = -rij * (2 *excl_eps * parameters['b']*rrc)
    else:
        tmp = parameters[0] ** 2 / c_dis ** 2
        lj_part = tmp ** 3
        # energy = 4 * excl_eps * (lj_part ** 2 - lj_part)
        force = -rij * (24*excl_eps*(lj_part - 2*lj_part**2)/ c_dis**2)
    return force

# Coarse Grained Particle Excluded Volume Parameters
# parameters: sigma, rstar, rc, b (this order)


class HMC:
    def __init__(self, positions, masses):
        # Assumes positions is 2d numpy array shaped as [N, 3] <- output of get_pdb_info
        self.N = positions.shape[0]  # Number of particles
        self.positions = positions
        self.momenta = np.random.normal(0., 1., 3*self.N).reshape(self.N, 3)  # Randomly generate momenta by sampling from normal distribution
        self.masses = masses

        # CG parameters
        self.excluded_volume_cutoff = 1.5
        self.excluded_volume_parameters = {'rstar': 0.087, 'b': 671492, 'rc':0.100161, 'sigma': 0.117}


    def step(self, positions, momenta, timestep=0.003):
        start_momenta = momenta
        start_positions = positions
        momenta_half = start_momenta - timestep/2 * self.forces(start_positions)
        new_positions = start_positions + timestep*momenta_half
        new_momenta = momenta_half - timestep/2 * self.forces(new_positions)
        return new_positions, new_momenta

    def potential_energy(self, positions):
        # Apply potential energy functions appropriately here
        potential_energy = 0

        # Pairwise Lennard Jones for all particles
        for i in range(self.N):
            for j in range(self.N):
                if i >= j:
                    continue
                else:
                    rij = positions[j] - positions[i]
                    dis = np.linalg.norm(rij)
                    if(dis < self.excluded_volume_cutoff):
                        Eij = lennard_jones_energy(rij, self.excluded_volume_parameters)
                        potential_energy += Eij

        return potential_energy

    def forces(self, positions):
        forces = np.full((self.N, 3), 0.)

        # Pairwise Lennard Jones for all particles
        for i in range(self.N):
            for j in range(self.N):
                if i >= j:
                    continue
                else:
                    rij = positions[j] - positions[i]
                    dis = np.linalg.norm(rij)
                    if (dis < self.excluded_volume_cutoff):
                        Fij = lennard_jones_forces(rij, self.excluded_volume_parameters)
                        forces[i] -= Fij
                        forces[j] += Fij

        return forces

    def energy(self, positions, momenta):
        velocity_mag = np.true_divide(np.linalg.norm(momenta, axis=1), self.masses) # v = p/m
        kinetic_energy = np.sum(1/2*self.masses*velocity_mag**2)
        potential_energy = self.potential_energy(positions)
        return kinetic_energy + potential_energy

    def kinetic_energy(self, momenta):
        velocity_mag = np.true_divide(np.linalg.norm(momenta, axis=1), self.masses)  # v = p/m
        kinetic_energy = 1 / 2 * self.masses * velocity_mag ** 2
        return kinetic_energy

    def sample(self, steps=10, timestep=0.002):
        positions = self.positions
        momenta = self.momenta
        energy = self.energy(positions, momenta)

        for x in range(steps):
            # Move forward a step
            new_positions, new_momenta = self.step(positions, momenta, timestep=timestep)
            # Compute Acceptance Ratio
            acc = math.exp(energy - self.energy(new_positions, new_momenta))  # something like this, add temp factor later maybe
            rand = np.random.rand(1)
            # accept move
            if(acc > rand):
                positions = new_positions
                momenta = new_momenta
            else:
                continue

        self.positions = positions
        self.momenta = momenta

        return positions

class Coarse_Grainer:
    def __init__(self, target_coords, particle_number, kstart=False, distance_cutoff=100, ignore=None, votes=1, fix_handles=None,
                 handle_weight=10, max_radius_decimal=0.9):
        self.K = particle_number  # number of cg particles

        # ignored particles will ignore any particle index you give it by deleting it from the coordinates
        # and then adjusting the index mapping accordingly
        if ignore is None:
            self.ignored_particles = []

            self.N = target_coords.shape[0]
            self.target_coords = target_coords - np.sum(target_coords) / self.N

            self.index_mapping = np.arange(0, self.N)
        else:
            old_target_coords = target_coords.copy()

            ignored_particles = ignore
            ignored_particles.sort(reverse=True)  # indices listed in descending order

            self.target_coords = target_coords
            for ip in ignored_particles:
                np.delete(self.target_coords, ip)

            self.N = self.target_coords.shape[0]

            index_mapping = []
            for n in range(self.N):
                index_mapping.append(np.argwhere(old_target_coords == self.target_coords[n])[0][0])

            self.index_mapping = index_mapping
            self.target_coords = self.target_coords - np.sum(self.target_coords) / self.N  # Set COM to 0, 0, 0

        # target particles (DNA etc.) at a distance greater than the distance_cutoff away from a coarse grain particle
        # are given 0. probability of being assigned in the assignment matrix Z
        self.distance_cutoff = distance_cutoff  # in Angstroms
        self.votes = votes  # Number of samples to pull from each multinomial distribution in the assignment matrix

        # self.totalmass = self.N
        # Z = np.full((target_coords.shape[0], particle_number), 0)  # NxK assignment matrix
        self.s = 1.0  # initial value
        cg_start_index = list(np.random.choice(len(self.target_coords), replace=False, size=particle_number))
        if kstart:
            handle_coords = copy(self.target_coords)
            if fix_handles is not None:
                for ind in fix_handles:
                    for i in range(handle_weight):
                        handle_coords = np.append(handle_coords, [self.target_coords[ind]], axis=0)
            cg_start_positions = kmeans(handle_coords, self.K)
        else:
            cg_start_positions = np.asarray([self.target_coords[x] for x in cg_start_index])

        # if shift_toward_com > 0:
        #     pos_values = cg_start_positions > 0
        #     cg_start_positions[pos_values] -= shift_toward_com
        #     cg_start_positions[~pos_values] += shift_toward_com

        self.cg_coords = cg_start_positions
        self.max_radius_decimal = max_radius_decimal
        self.masses = []
        # masses updated in the assignment matrix
        self.Z = self.update_assignment_matrix(self.target_coords, self.cg_coords, self.s)  # NxK assignment matrix
        self.s = self.update_s(self.target_coords, self.cg_coords, self.Z)
        self.hmc = HMC(cg_start_positions, self.masses)

    def assign_radii(self, decimal_of_max_radius, cg_coords, masses):
        try:
            from sklearn.metrics import pairwise_distances
        except ImportError:
            print("Need sklearn for pairwise distance calculation")
            exit(1)


        dist_mat = pairwise_distances(cg_coords, cg_coords)

        # In Angstroms
        min_dist = np.min(dist_mat[dist_mat > 0])

        max_radius = min_dist / 2.

        max_radius_adj = decimal_of_max_radius * max_radius

        normed_masses = masses/np.max(masses)

        self.radii = normed_masses*max_radius_adj


    def update_assignment_matrix(self, target_coords, cg_coords, s):
        # stores exp{-1/2s^2 || xn - Xk ||^2}
        tmp = np.full((self.N, self.K), 0., dtype=float)

        # convert coords from angstrom to blank for numerical stability
        t_coords = target_coords/100
        c_coords = cg_coords/100
        distance_cut = self.distance_cutoff/100  # distance_cutoff in Angstroms
        p_cutoff = math.exp(-1. / (2 * s**2) * np.linalg.norm(distance_cut) ** 2)
        for n in range(self.N):
            for k in range(self.K):
                # if n == 50 and k == 0:
                #     # val = math.exp(-1. / (2 * s**2) * np.linalg.norm(t_coords[n] - c_coords[k]) ** 2) + 1e-64
                #     val = math.exp(-1. / (2 * s**2) * np.linalg.norm(t_coords[n] - c_coords[k]) ** 2) + 1e-64
                #     # tmp[n][k] = math.exp(-1. / (2 * s2) * np.linalg.norm(target_coords[n] - cg_coords[k]) ** 2)
                #     tmp[n][k] = val
                # else:

                val = math.exp(-1. / (2 * s**2) * np.linalg.norm(t_coords[n] - c_coords[k]) ** 2)
                # val = math.exp(-1. / (2 * s**2) * np.linalg.norm(t_coords[n] - c_coords[k]) ** 2)
                if val > p_cutoff:
                    tmp[n][k] = val
                else:
                    tmp[n][k] = 0.
                    # if val == 0. or math.isnan(val)

        # Using tmp to make assignment probability matrix
        probs = np.full((self.N, self.K), 0., dtype=float)
        for n in range(self.N):
            total = np.sum(tmp[n])
            for k in range(self.K):
                # if n == 50 and k == 0:
                #     probs[n][k] = tmp[n][k] / total
                # else:
                probs[n][k] = tmp[n][k] / total  # sum over k'

        # rewrite tmp with assignments
        zero = np.full(self.K, 0., dtype=float)
        for n in range(self.N):
            total = np.sum(probs[n])
            # if total != 1.:
            probvec = np.asarray([x/total for x in list(probs[n])])
            # psum = np.sum(probvec)
            # else:
            #     probvec = probs[n]

            if self.votes == 0:
                tmp[n] = zero.copy()
                index = np.argmax(probvec)
                tmp[n][index] = 1.
            else:
                voting = np.random.multinomial(self.votes, pvals=probvec)
                tmp[n] = zero.copy()
                index = np.argmax(voting)
                tmp[n][index] = 1.

        # update masses based off assignments
        new_masses = tmp.sum(axis=0)
        self.masses = new_masses

        # update radii based off positions of cg particles
        self.assign_radii(self.max_radius_decimal, cg_coords, self.masses)

        return tmp

    def likelihood(self):
        target_coords = self.target_coords
        cg_coords = self.cg_coords
        Z = self.Z
        return - 0.5 * (self.chi_squared(target_coords, cg_coords, Z) / self.s ** 2 + \
                        3*self.N * np.log(self.s ** 2))

    def update_s(self, target_coords, cg_coords, Z):
        # b = 1e-1 + 0.5 * self.chi_squared(target_coords, cg_coords, Z)
        b = 0.5 * self.chi_squared(target_coords, cg_coords, Z)
        # a = 1e-1 + 0.5 * 3 * self.N
        a = 0.5 * 3 * self.N

        s = (b / np.random.gamma(a)) ** 0.5

        return s

        # x2 = self.chi_squared(target_coords, cg_coords, Z)
        # s2 = 1./np.random.gamma(3*self.N/2, x2/2, 1)
        # return s2

    def chi_squared(self, target_coords, cg_coords, Z):
        # each target_coord assigned to 1 cg particle
        sum = 0.
        for n in range(self.N):
            k = np.argmax(Z[n])  # assigned particle index
            sum += np.linalg.norm(target_coords[n] - cg_coords[k]) ** 2

        return sum

    def coarse_grainer(self, target_coords, steps=10, timestep=0.002, hmc_steps=10):
        # # Initial starting place of our coarse-grained particles
        # cg_start_positions = np.random.choice(len(target_coords), replace=False, size=particle_number)
        # # Z = np.full((target_coords.shape[0], particle_number), 0)  # NxK assignment matrix
        # s2 = 2.0  # initial value
        # cg_coords = cg_start_positions
        # Z = update_assignment_matrix(target_coords, cg_coords, s2) # NxK assignment matrix
        # s2 = update_s2(target_coords, cg_coords, Z)

        # # Initialize Hamiltonian monte carlo sampler
        # hmc_sampler = HMC(cg_start_positions)
        cg_coords = self.cg_coords
        s = self.s

        for i in range(steps):
            # Sampled Postions
            cg_coords = self.hmc.sample(steps=hmc_steps, timestep=timestep)
            Z = self.update_assignment_matrix(target_coords, cg_coords, s)
            s = self.update_s(target_coords, cg_coords, Z)

        self.s = s
        self.Z = Z
        self.cg_coords = cg_coords

        final_positions = cg_coords
        x2 = self.chi_squared(target_coords, self.cg_coords, self.Z)
        print('Chi2: ', x2)
        return final_positions

    def get_indexing(self):
        indices = []
        masses = []
        assigned_particles = []
        for k in range(self.K): # for each coarse grained particle
            particle_indices = []
            for n in range(self.N):
                if self.Z[n][k] == 1.:  # find real particles assigned to coarse grained
                    particle_indices.append(self.index_mapping[n])  # add
                    assigned_particles.append(n)
                else:
                    continue  # do nothing
            indices.append(particle_indices)
            masses.append(len(particle_indices))
        unassigned_particles = np.full(self.N, True)
        unassigned_particles[assigned_particles] = False
        if True in unassigned_particles:
            print(f"Unassigned Particles in Given System!")
        return indices, masses

    def network_export(self, name):
        indices, masses = self.get_indexing()

        # coordinates and radii in Angstroms
        tmp = {'simMasses':masses, 'coordinates':self.cg_coords.tolist(), "radii":self.radii.tolist()}

        out = open(name+'.json', "w")
        json.dump(tmp, out)
        out.close()
        out = open(name + '_index.txt', "w")
        fullstring = ""
        for k in range(self.K):
            fullstring += ' '.join([str(x) for x in indices[k]]) # flip the numbering?
            fullstring += "\n"

        print(','.join([str(x) for x in indices[0]]))
        print(','.join([str(x) for x in indices[1]]))

        out.write(fullstring)
        out.close()




if __name__=='__main__':
    # Coarse Grain the handle version as that is what we have a trajectory of, this is for original icosahedron Micha sent
    # handle1 = np.arange(5657, 5676)
    # handle2 = np.arange(4222, 4241)
    # handle3 = np.arange(4073, 4092)
    # handle4 = np.arange(5215, 5234)
    # handle5 = np.arange(6071, 6084)
    # handle6 = np.arange(6251, 6270)
    # handles = list(np.concatenate([handle1, handle2, handle3, handle4, handle5, handle6]))



    #### This is for the ico_system from Hao (3p_patches_dimer_910)
    handle_connection_positions_p1 = [4069, 6405, 6032]
    handle_start_positions_p1 = [4088, 6424, 6051]

    handle_connection_positions_p2 = [6097, 6512, 5257]
    handle_start_positions_p2 = [6116, 6531, 5276]

    handle_connection_positions_p3 = [6138, 3817, 6465]
    handle_start_positions_p3 = [6157, 3836, 6484]

    handle_connection_positions_p4 = [6644, 6339, 5733]
    handle_start_positions_p4 = [6663, 6358, 5752]

    handle_connection_positions_p5 = [6567, 4925, 6204]
    handle_start_positions_p5 = [6598, 4944, 6223]

    handle_connection_positions_p6 = [6710, 5558, 6272]
    handle_start_positions_p6 = [6729, 5577, 5559]

    connection_positions = [*handle_connection_positions_p1, *handle_connection_positions_p2, *handle_connection_positions_p3,
                            *handle_connection_positions_p4, *handle_connection_positions_p5, *handle_connection_positions_p6]

    start_positions = [*handle_start_positions_p1, *handle_start_positions_p2, *handle_start_positions_p3,
                       *handle_start_positions_p4, *handle_start_positions_p5, *handle_start_positions_p6]

    handles = []
    for i in range(len(connection_positions)):
        handle_indices = np.arange(connection_positions[i] + 1, start_positions)
        handles.append(handle_indices)




    # target_coords = n.get_pdb_info('1bu4.pdb', returntype='c')
    # IcosahedronRT_mean stored as coordinates using the network export of oxView

    k = 90  # number of particles
    target_coords, masses = n.get_json_info('./system_jsons/IcoDNA.json')
    target_coords = np.asarray(target_coords)
    cg = Coarse_Grainer(target_coords, k, kstart=True, distance_cutoff=200, votes=0, ignore=handles, fix_handles=[5656, 4221, 4072, 5214, 6070, 6250], max_radius_decimal=0.8)
    adjusted_coords = cg.target_coords
    new_positions = cg.coarse_grainer(adjusted_coords, steps=1, timestep=0.002, hmc_steps=2)
    # print(new_positions.shape)
    # print(target_coords)
    cg.network_export('cg' + str(k) + '_nh')




