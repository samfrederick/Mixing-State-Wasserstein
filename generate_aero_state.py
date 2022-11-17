"""Generate an idealized aerosol population with specified 
number of particles and species. The aerosol mixing state and associated 
entropy and diversity metrics are computed.

Mixing state metrics follow the form presented in
    Riemer, N. and West, M.: Quantifying aerosol mixing state with entropy and 
    diversity measures, Atmos. Chem. Phys., 13, 11423–11439, 
    https://doi.org/10.5194/acp-13-11423-2013, 2013.

"""
import numpy as np

class aero_state:

    def __init__(self, 
                 n_particles, 
                 particle_radius=1, # monodisperse population
                 species={'species_1': {'density': 1},
                          'species_2': {'density': 1}},
                 **kwargs):
        
        self.kwargs = kwargs
        self.__dict__.update(**kwargs)
        self.verbose = self.kwargs.get('verbose', True)
        self.round_mass_frac_base = self.kwargs.get('round_mass_frac_base', None)

        self.n_particles = n_particles
        self.particle_radius = particle_radius
        self.particle_volume = (4/3)*(np.pi*particle_radius**3)
        self.species = species
        self.n_species = len(self.species)
        self._generate_aero_state_frac()
        self._generate_aero_state_mass()
        
        # mu
        self.pop_mass = np.sum(self.aero_state_mass)
        # mu_i
        self.particle_mass = np.sum(self.aero_state_mass, axis=1)
        # p_i
        self.pop_particle_mass_frac = self.particle_mass / self.pop_mass
        # (mu)^a
        self.pop_species_mass = np.sum(self.aero_state_mass, axis=0)
        # (p)^a
        self.pop_species_mass_frac = self.pop_species_mass / self.pop_mass

        self._generate_mixing_entropy()
        self._generate_diversity_metrics()
        self._generate_mixing_state()

        if self.verbose:
            print(f'Generating {self.n_particles} particles with {self.n_species} species')
            print('\nDescription of aerosol species:')
            print(self.species)

            # print the total mass fraction for first particle, Should be 1
            print('\nSpecies fraction per particle:\n-----------------------------')
            print(self.aero_state_frac)
            #print(np.sum(self.aero_state_frac, axis=1))
            print('')
            print('Mass of species per particle:\n-----------------------------')
            print(self.aero_state_mass)
            #print(np.sum(self.aero_state_mass, axis=1))

            print('\n')
            #print('mass of each particle:', self.particle_mass)
            print('total population mass:', self.pop_mass)
            #print('mass fraction of each particle in pop', self.pop_particle_mass_frac)
            print('total mass of species in population', self.pop_species_mass)

            print(f'D_i: {self.particle_diversity}')
            print(f'D_alpha: {self.avg_particle_species_diversity:3.2f}')
            print(f'D_gamma: {self.pop_bulk_species_diversity:3.2f}')
            print(f'D_beta: {self.inter_particle_diversity:3.2f}')
            print(f'Chi: {self.mixing_state_index:3.2f}')


    def _generate_aero_state_frac(self):
        # Randomly generate n particles with m species
        # each row is a particle, each column is species fraction
        # (p_i)^a
        self.aero_state_frac = np.random.rand(self.n_particles, self.n_species)

        # round species mass fractions for each particle to a specified base and precision
        if self.round_mass_frac_base:
            base = self.round_mass_frac_base
            prec = len(str(base).split('.')[-1])
            self.aero_state_frac = (base * (np.array(self.aero_state_frac) / base).round()).round(prec)

        for part_i in range(self.n_particles):
            for species_i in range(self.n_species):
                
                prev_species_sum = self.aero_state_frac[part_i, :species_i].sum()

                if species_i + 1 == self.n_species:
                    self.aero_state_frac[part_i, species_i] = 1 - prev_species_sum
                elif species_i != 0:
                    self.aero_state_frac[part_i, species_i] = self.aero_state_frac[part_i, species_i]*(1- prev_species_sum)

    def _generate_aero_state_mass(self):
        # (mu_i)^a
        self.aero_state_mass = self.aero_state_frac.copy()
        for s_idx, s_name in enumerate(self.species):
            s_density = self.species[s_name]['density']
            s_mass = self.particle_volume*s_density
            self.aero_state_mass[:, s_idx] = self.aero_state_frac[:, s_idx]*s_mass

    def _generate_mixing_entropy(self):
        #H_i
        self.particle_mixing_entropy = np.zeros(self.n_particles)
        for species_i in range(self.n_species):
             species_entrop = np.nan_to_num(-1*self.aero_state_frac[:, species_i]*np.log(self.aero_state_frac[:, species_i]))
             self.particle_mixing_entropy += species_entrop

        # H_alpha
        self.avg_particle_mixing_entropy = np.sum(self.pop_particle_mass_frac*self.particle_mixing_entropy)

        # H_gamma
        self.pop_bulk_mixing_entropy = np.sum(-1*self.pop_species_mass_frac*np.log(self.pop_species_mass_frac))

    def _generate_diversity_metrics(self):
        # D_i
        self.particle_diversity = np.exp(self.particle_mixing_entropy)
        # D_alpha
        self.avg_particle_species_diversity = np.exp(self.avg_particle_mixing_entropy)
        # D_gamma
        self.pop_bulk_species_diversity = np.exp(self.pop_bulk_mixing_entropy)
        # D_beta
        self.inter_particle_diversity = self.pop_bulk_species_diversity / self.avg_particle_species_diversity

    def _generate_mixing_state(self):
        # Chi
        self.mixing_state_index = (self.avg_particle_species_diversity - 1) / (self.pop_bulk_species_diversity - 1)

if __name__ == '__main__':
    #aero = aero_state(n_particles=10)


    aero = aero_state(n_particles=10,
                      round_mass_frac_base=0.05)