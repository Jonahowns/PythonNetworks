import nmodels as n
import numpy as np

coords = n.get_pdb_info('1bu4.pdb', returntype='c')


class HMC():
    def __init__(self, positions):
        # Assumes positions is 2d numpy array shaped as [N, 3] <- output of get_pdb_info
        self.N = positions.ndim[0] # Number of particles
        self.positions = positions
        self.momenta = np.random.normal(0., 1., 3*self.N).reshape(self.N, 3) # Randomly generate momenta by sampling from normal distribution
        self.masses = np.asarray([1. for x in np.arange(len(positions))])

    def step(self, timestep):
        start_momenta = self.momenta
        start_positions = self.positions
        momenta_half = start_momenta - timestep/2 * self.forces(start_positions)
        new_positions = start_positions + timestep*momenta_half
        new_momenta = momenta_half - timestep/2* self.forces(new_positions)
        return new_positions, new_momenta

    def potential_energy(self, positions):
        # Define Potential Functions, and apply appropiately here


    def forces(self, positions):
        # CG force field from potential functions


    def energy(self, positions):
        velocity_mag = np.true_divide(np.linalg.norm(self.momenta, axis=1), self.masses) # v = p/m
        kinetic_energy = 1/2*self.masses*velocity_mag**2
        potential_energy = self.potential_energy(positions)
        return kinetic_energy + potential_energy

    def kinetic_energy(self, positions):
        velocity_mag = np.true_divide(np.linalg.norm(self.momenta, axis=1), self.masses)  # v = p/m
        kinetic_energy = 1 / 2 * self.masses * velocity_mag ** 2







