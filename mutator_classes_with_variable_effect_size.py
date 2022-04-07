import os
import pickle
import sys
import numpy as np
import collections
from scipy import stats
import gzip
from scipy.integrate import quad
from copy import deepcopy

## for this file, I only made sure to comment on things that different from the default mutator classes
class Parameters:
    def __init__(self,
                 N,
                 h,
                 s,
                 loci,
                 mutation_rate,
                 phi,
                 mutator_mutation_rate,
                 M):
        self.N = N
        self.h = h
        self.s = s
        self.loci = loci
        self.mutation_rate = mutation_rate
        self.mutator_mutation_rate = mutator_mutation_rate
        self.phi = phi
        self.M = M

sys.path.append('/home/ec2-user/Mutator')
import stationary_distribution_aug as sd


class Population:
    def __init__(self,
                 N,
                 h,
                 s,
                 loci,
                 mutation_rate,
                 phi,
                 mutator_mutation_rate,
                 M,
                 invariable_mutator_mutation_rate,
                 outpath,
                 initialize = False,
                 variable_mutator_effect = False,
                 mutator_effect_size_distribution = None,
                 variable_selective_effect = False,
                 selected_effect_distribution = None):

        self.N = N
        self.h = h
        self.s = s
        self.loci = loci
        self.mutation_rate = mutation_rate
        self.mutator_mutation_rate = mutator_mutation_rate
        self.phi = phi
        self.M = M
        self.invariable_mutator_mutation_rate = invariable_mutator_mutation_rate

        self.fixed = []
        self.fixed_mutation_rate = 0
        self.mutations_at_fixed = collections.defaultdict(list)
        self.outpath = outpath
        self.mutator_counts_outpath = os.path.join(self.outpath,'mutator_counts.pickle')
        self.storage = []

        self.calculate_effective_population_size = True
        self.total_heterozygosity = 0
        self.NE_algorithm_active = False
        self.wait_time_NE_algorithm = int(2 * self.N)

        self.variable_mutator_effect = variable_mutator_effect
        assert not self.variable_mutator_effect

        # if selected sites have variable selection effects, assign a distribution
        self.variable_selective_effect = variable_selective_effect
        if self.variable_selective_effect:

            if selected_effect_distribution:
                self.selected_effect_distribution = selected_effect_distribution
            else:
                a = (10/15)**2
                b = (10/(2000)/a)
                gamma = stats.gamma(a = a,scale=b,loc=0.01)
                self.selected_effect_distribution = gamma
            self.s = self.selected_effect_distribution.mean()/self.h
        else:
            print(f'Using constant selected effect size of {self.h*self.s}')

        # initialization - is a bit different
        self.people = [Individual(self.selected_effect_distribution) for _ in range(self.N)]
        if initialize:
            self.init_from_params()

    # not changed
    def update_outpath(self,new_outpath):
        self.outpath = new_outpath
        self.mutator_counts_outpath = os.path.join(self.outpath, 'mutator_counts')

    # initialize the population
    def init_from_params(self):

        SD = sd.get_SD(p_input=self,N=470)

        mutator_freqs = np.random.choice(list(SD.keys()),
                                         p=list(SD.values()),
                                         size = self.M)

        # Determine the mean number of deleterious mutations carried by individuals
        # calculate E(1/s)
        poisson_param = 2 * self.mutation_rate * self.loci * quad(lambda s: 1/s*self.selected_effect_distribution.pdf(s),self.selected_effect_distribution.ppf(0),self.selected_effect_distribution.ppf(1))[0]

        for i in self.people:
            i._init_from_params(poisson_param = poisson_param, mutator_freqs = mutator_freqs)

    # fitness is different - here it's a function of the individual object since its more difficult to calculate
    def calc_relative_fitness(self):

        fit = np.array([i.fitness() for i in self.people])

        # return relative fitness
        return fit/sum(fit)

    # not changed
    def choose_parents(self, N):

        prob_fit = self.calc_relative_fitness()

        parents = np.random.choice(a=self.people, size=2*N, p=prob_fit, replace=True)
        
        return parents

    # not changed
    def deal_with_mutations_at_fixed_loci(self):

        # iterate over all mutated fixed loci
        for k in self.mutations_at_fixed.keys():
            # iterate over all people
            for p in range(len(self.people)):
                # if this person had a mutation, the k locus is now heterozygous
                if p in self.mutations_at_fixed[k]:
                    self.people[p].mutator_loci_het.append(k)
                # otherwise, it is homozygous
                else:
                    self.people[p].mutator_loci_homo.append(k)
            # this locus is no longer fixed
            self.fixed.remove(k)

    # not changed
    def update_fixed_loci(self,new_fixed):

        for k in new_fixed:
            # remove fixed loci from list of homozygous loci in each person
            for p in self.people:
                p.mutator_loci_homo.remove(k)
            # add them to list of fixed mutations
            self.fixed.append(k)

        # make sure no duplicate loci
        assert len(self.fixed) == len(np.unique(self.fixed))

        if self.variable_mutator_effect:
            for i in self.fixed:
                self.fixed_mutation_rate += 2*self.mutator_effects[i]

    # not changed
    # returns mutator allele frequencies at each locus and tells us which loci are (newly) fixed
    def get_mutator_counts(self):

        mutator_counts = np.zeros(self.M)

        # update mutator counts
        for p in self.people:
            mutator_counts[p.mutator_loci_het] += 1
            mutator_counts[p.mutator_loci_homo] += 2

        # figure out which loci are now fixed
        new_fixed = list(np.where(mutator_counts == 2 * self.N)[0])

        # make sure to include previously fixed loci
        mutator_counts[self.fixed] = 2*self.N

        # convert to frequencies
        mutator_count = mutator_counts / (2 * self.N)

        return mutator_count, new_fixed

    # not changed
    def next_gen(self, N):

        # get the indiviudals that will produce gametes
        chosen_parents = self.choose_parents(N=N)

        # store mutations that occur at fixed loci
        self.mutations_at_fixed = collections.defaultdict(list)

        # make N new individuals
        new_people = []

        for i in range(N):

            # create gametes from each parent
            gameteA = chosen_parents[i].make_gamete(population = self,
                                                    mutations_at_fixed = self.mutations_at_fixed,
                                                    index = i)

            gameteB = chosen_parents[i + N].make_gamete(population = self,
                                                        mutations_at_fixed = self.mutations_at_fixed,
                                                        index = i)

            # combine gametes to make people
            new_person = deepcopy(self.reproduce(gameteA, gameteB))

            new_people.append(new_person)

        self.people = new_people

        # update population size
        self.N = N

        # this must be done every generation
        self.deal_with_mutations_at_fixed_loci()
    
    # Changed
    def reproduce(self, gameteA, gameteB):

        # initialize an individual
        child = Individual(selected_effect_distribution=self.selected_effect_distribution)

        # now we store selected mutations as a list of selection coefficients
        child.selected_mutations = np.append(gameteA.selected_mutations,gameteB.selected_mutations)

        # figure out which mutator loci the child is homo or heterozygous for
        for k in gameteB.mutator_loci:
            if k not in gameteA.mutator_loci:
                child.mutator_loci_het.append(k)
            else:
                child.mutator_loci_homo.append(k)
                gameteA.mutator_loci.remove(k)

        for k in gameteA.mutator_loci:
            child.mutator_loci_het.append(k)

        child.hetero = [gameteA.hetero, gameteB.hetero]

        return child

    # not changed
    def store(self,a):
        self.storage.append(a)

    # not changed
    def write_out(self):
        #print('mean selected mutations: ',np.mean([len(i.selected_mutations) for i in self.people]))
        with gzip.open(self.mutator_counts_outpath,'ab+') as fout:
            for a in self.storage:
                pickle.dump(a, fout)
        self.storage=[]

# not changed - but implementation, selected mutations becomes a list not an int
class Gamete():
    def __init__(self,
                   selected_mutations = 0,
                   mutator_loci = [],
                   hetero = 0):
        self.selected_mutations = selected_mutations
        self.mutator_loci = mutator_loci
        self.hetero = hetero

# selected mutations is a list not an int
class Individual:
    def __init__(self,selected_effect_distribution = False):
        self.mutator_loci_het = []
        self.mutator_loci_homo = []
        self.selected_effect_distribution = selected_effect_distribution
        self.selected_mutations = []
        self.hetero = [0,0]

    # we only use this for the case with variable selected effects
    # calculate fitness multiplicatively assuming all mutations are heterozygous
    def fitness(self):
        if len(self.selected_mutations) == 0: 
            return 1
        fit = np.product(1-self.selected_mutations)
        return fit

    # difference is that we do not initialize the selected sites
    # It's faster to just allow them to burn in as they are strongly selected.
    def _init_from_params(self, mutator_freqs, poisson_param = 0):

        # same as before
        for i in range(len(mutator_freqs)):
            g = np.random.binomial(2,mutator_freqs[i])
            if g == 2:
               self.mutator_loci_homo.append(i)
            elif g == 1:
               self.mutator_loci_het.append(i)

    # changed how selected mutations are handled
    def make_gamete(self, population, mutations_at_fixed, index):

        # mendelian segregation in mutator loci
        one_chosen_bits = []
        if self.mutator_loci_het:
            one_chosen_bits = np.random.binomial(1, 1 / 2, len(self.mutator_loci_het))
            one_chosen_bits = [self.mutator_loci_het[i] for i in np.where(one_chosen_bits)[0]]
        mutator_loci = self.mutator_loci_homo + one_chosen_bits

        # mutation_rate at modifier loci
        if population.invariable_mutator_mutation_rate:
            mutator_mutation_rate = population.mutator_mutation_rate
        else:
            n_mutators = len(self.mutator_loci_het) + 2 * len(self.mutator_loci_homo) + 2 * len(population.fixed)
            mutator_mutation_rate = population.mutator_mutation_rate + population.phi * n_mutators

        # mutations at modifier loci
        # Realize the number of mutations at modifier loci and where they occur
        num_mutations = np.random.poisson(population.M * mutator_mutation_rate)
        mutated_mutator_loci = np.random.randint(0, population.M, num_mutations)
        # determine what actions need to be taken (i.e., reverse mutation, reverse mutation at fixed locus, new mutator)
        for k in mutated_mutator_loci:
            if k in mutator_loci:
                mutator_loci.remove(k)
            elif k in population.fixed:
                mutations_at_fixed[k].append(index)
            else:
                mutator_loci.append(k)

        # update mutation rate for selected loci
        if population.variable_mutator_effect:
            mutation_rate = population.mutation_rate + population.fixed_mutation_rate
            for i in self.mutator_loci_het:
                mutation_rate += population.mutator_effects[i]
            for i in self.mutator_loci_homo:
                mutation_rate += 2*population.mutator_effects[i]
        else:
            mutation_rate = population.mutation_rate + population.phi * (len(self.mutator_loci_het) + 2 * len(self.mutator_loci_homo) + 2 * len(population.fixed))

        # mendelian segregation at selected loci
        # can use this assumption so long as 2Nhs >> 1 and infinite sites applies
        # randomly pass on current selected mutations with probability 1/2
        n_inherited_mutations = np.random.binomial(len(self.selected_mutations), 1 / 2)
        inherited_mutations = np.random.choice(a = self.selected_mutations, size = n_inherited_mutations, replace = False)

        # new mutations at selected loci
        n_new_mutations = np.random.poisson(population.loci * mutation_rate)
        new_mutations = self.selected_effect_distribution.rvs(n_new_mutations)

        selected_mutations = np.append(new_mutations, inherited_mutations)
        assert len(selected_mutations) == n_inherited_mutations + n_new_mutations

        # update locus used to measure heterozygosity
        hetero = self.hetero[np.random.rand() > 0.5]

        return Gamete(selected_mutations = selected_mutations,
                   mutator_loci = mutator_loci,
                   hetero = hetero)
