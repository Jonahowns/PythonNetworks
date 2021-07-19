import nmodels as n
import numpy as np
import math

coords = n.get_pdb_info('1bu4.pdb', returntype='c')


class HMC():
    def __init__(self, positions):
        # Assumes positions is 2d numpy array shaped as [N, 3] <- output of get_pdb_info
        self.N = positions.ndim[0] # Number of particles
        self.positions = positions
        self.momenta = np.random.normal(0., 1., 3*self.N).reshape(self.N, 3) # Randomly generate momenta by sampling from normal distribution
        self.masses = np.asarray([1. for x in np.arange(len(positions))])

        # CG parameters
        self.excluded_volume_cutoff = 1.5
        self.excluded_volume_parameters = {'rstar': 0.087, 'b': 671492, 'rc':0.100161, 'sigma': 0.117}


    def step(self, timestep):
        start_momenta = self.momenta
        start_positions = self.positions
        momenta_half = start_momenta - timestep/2 * self.forces(start_positions)
        new_positions = start_positions + timestep*momenta_half
        new_momenta = momenta_half - timestep/2* self.forces(new_positions)
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
        kinetic_energy = 1/2*self.masses*velocity_mag**2
        potential_energy = self.potential_energy(positions)
        return kinetic_energy + potential_energy

    def kinetic_energy(self, momenta):
        velocity_mag = np.true_divide(np.linalg.norm(momenta, axis=1), self.masses)  # v = p/m
        kinetic_energy = 1 / 2 * self.masses * velocity_mag ** 2
        return kinetic_energy


    def sample(self, steps=10):
        positions = self.positions
        momenta = self.momenta
        energy = self.energy(positions, momenta)

        for x in range(steps):
            # Move forward a step
            new_positions, new_momenta = self.step(.001)
            # Compute Acceptance Ratio
            acc = math.exp(self.energy(new_positions, new_momenta) - energy) # something like this










# Coarse Grained Particle Excluded Volume Parameters
# parameters: sigma, rstar, rc, b (this order)


# parameters: sigma, rstar, rc, b
def lennard_jones_energy(rij, parameters):
    # Assumes you checked it is within distance cutoff
    excl_eps = 4.0
    c_dis = np.linalg.norm(rij)
    if(c_dis > parameters['rstar']): # rstar
        rrc = c_dis-parameters['rc']
        energy = excl_eps*parameters[3]*rrc**2
    else:
        tmp = parameters['sigma']**2 / c_dis**2
        lj_part = tmp**3
        energy = 4*excl_eps*(lj_part**2- lj_part)
    return energy

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






