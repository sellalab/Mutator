import os
import pickle
import sys
import numpy as np
import collections
from scipy import stats
import gzip

# simple object for when you just want the basic parameters
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

# load special custom packages
sys.path.append('/home/ec2-user/Mutator')
import stationary_distribution_aug as sd

# main object
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
                 invariable_mutator_mutation_rate = True,
                 outpath = '',
                 initialize = False,
                 variable_mutator_effect = False,
                 mutator_effect_size_distribution = None,
                 variable_selective_effect = False,
                 selective_effect_distribution = None,
                 somatic = False):

        # the basic parameters
        self.N = N
        self.h = h
        self.s = s
        self.loci = loci
        self.mutation_rate = mutation_rate
        self.mutator_mutation_rate = mutator_mutation_rate
        self.phi = phi
        self.M = M

        # bool to determine if mutator mutation rate is variable
        self.invariable_mutator_mutation_rate = invariable_mutator_mutation_rate

        # list of individual objects
        self.people = [Individual() for _ in range(self.N)]

        # keep track of fixed mutators, mutations at loci fixed for mutators, and the mutation rate contribution from fixed mutators
        self.fixed = []
        self.fixed_mutation_rate = 0
        self.mutations_at_fixed = collections.defaultdict(list)

        # storage stuff
        self.outpath = outpath
        self.mutator_counts_outpath = os.path.join(self.outpath,'mutator_counts.pickle.gz')
        self.storage = []

        # how much somatic selection
        self.somatic_selection = 2*self.h*self.s*self.loci*self.phi*int(somatic)

        # if we are simulating modifiers with a distribtuion of mutator effects
        self.variable_mutator_effect = variable_mutator_effect
        if self.variable_mutator_effect:

            # determine the effect size distribution
            if mutator_effect_size_distribution:
                self.mutator_effect_size_distribution = mutator_effect_size_distribution
            else:

                print(f'Using default mutator effect size distribution that is loguniform distributed over 0.01 < 2Ns < 1000')
                max_phi = 1000/(4*self.N*self.s*self.h*self.loci)
                min_phi = 0.01 / (4 * self.N * self.s * self.h * self.loci)
                self.mutator_effect_size_distribution = stats.loguniform(a=min_phi,b=max_phi)

            # randomly draw M effect sizes
            self.mutator_effects = self.mutator_effect_size_distribution.rvs(self.M)
            print(f'Mean mutator effect size is {np.mean(self.mutator_effects)} with variance {np.var(self.mutator_effects)}')

        else:
            print(f'Using constant mutator effect size of {self.phi}')

        # if selected sites have a distribution of selection effects use a different simulator
        self.variable_selective_effect = variable_selective_effect
        self.selective_effect_distribution = selective_effect_distribution
        if self.variable_selective_effect:
            raise ValueError

        # initialize the population
        if initialize:
            self.init_from_params()

    # change the outpath
    def update_outpath(self,new_outpath):
        self.outpath = new_outpath
        self.mutator_counts_outpath = os.path.join(self.outpath, 'mutator_counts')

    # initialization
    def init_from_params(self):

        # randomly draw mutator freqs from stationary distribution
        if self.variable_mutator_effect:
            mutator_freqs = []

            for phi in self.mutator_effects:
                SD = sd.get_SD(p=self,phi=phi)
                mutator_freqs.append(np.random.choice(list(SD.keys()),
                                                 p=list(SD.values()),
                                                 size=1)[0])
        else:

            # determine the stationary distribution of mutator allele frequencies
            SD = sd.get_SD(self,somatic=self.somatic_selection)

            # use the stationary distribution to randomly draw M starting mutator frequencies
            mutator_freqs = np.random.choice(list(SD.keys()),
                                             p=list(SD.values()),
                                             size = self.M)

        # Determine the mean number of deleterious mutations carried by individuals
        poisson_param = 2 * self.mutation_rate * self.loci / (self.h * self.s)

        # create N individual objects in the population based on:
        # mean number of deleterious mutations and starting mutator frequencies
        for i in self.people:
            i._init_from_params(poisson_param, mutator_freqs)

    # calculate fitness of individuals
    def calc_relative_fitness(self):

        # calculate the mean number of mutations carried by people
        mean_mutations = np.mean([i.selected_mutations for i in self.people])

        # calculate fitness by first scaling by 1/(1-hs)^(mean # of mutations)
        # this helps avoids floating point errors
        fit = np.array([(1 - self.h * self.s) ** (i.selected_mutations - mean_mutations) * (1 - self.somatic_selection) ** (len(i.mutator_loci_het)+2*len(i.mutator_loci_homo)) for i in self.people])
        
        relative_fit = fit/sum(fit)

        if any(np.isnan(relative_fit)):
            print(mean_mutations, fit, relative_fit)
            raise ValueError

        # return relative fitness
        return fit/sum(fit)

    # choose parents for the next generation based on fitness
    def choose_parents(self, N):

        # get relative fitness of individuals
        prob_fit = self.calc_relative_fitness()

        # choose N parents where the probability of being chosen is proportional to their fitness
        parents = np.random.choice(self.people, N, p=prob_fit)

        return parents

    # handle what happens when a mutation happens at a fixed site
    # everyone must now be homozygous at that site except for individuals that had mutations
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

    # determine which modifiers are fixed for mutators
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
                self.fixed_mutation_rate = 2*self.mutator_effects[i]

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

    # order of events for each generation
    def next_gen(self, N):

        # determine the indiviudals that will produce gametes
        chosen_parents = self.choose_parents(2 * N)

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
            new_people.append(self.reproduce(gameteA, gameteB))

        self.people = new_people

        # update population size
        self.N = N

        # this must be done every generation
        self.deal_with_mutations_at_fixed_loci()

    # turns 2 gametes into an individual
    def reproduce(self, gameteA, gameteB):

        # initialize an individual
        child = Individual()

        child.selected_mutations = gameteA.selected_mutations + gameteB.selected_mutations

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

    # store mutator counts
    def store(self,a):
        self.storage.append(a)

    # dump mutator counts
    def write_out(self):

        with gzip.open(self.mutator_counts_outpath,'ab+') as fout:
            for a in self.storage:
                pickle.dump(a, fout)
        self.storage = []

# object that represents a haploid individual
class Gamete():
    def __init__(self,
                   selected_mutations = 0,
                   mutator_loci = [],
                   hetero = 0):
        self.selected_mutations = selected_mutations
        self.mutator_loci = mutator_loci
        self.hetero = hetero

# object that represents a diploid individual
class Individual:

    # we only keep track of segregating modifier sites
    def __init__(self):
        self.mutator_loci_het = []
        self.mutator_loci_homo = []

        # we only keep track of the number of strongly selected derived alleles at selected sites
        self.selected_mutations = 0

        # used to measure effective population size
        self.hetero = [0,0]

    # initialize individuals
    def _init_from_params(self, poisson_param, mutator_freqs):

        # randomly draw the number of selected mutations assuming it is Poisson distributed around mean
        self.selected_mutations = np.random.poisson(poisson_param)

        # randomly assign mutator genotypes
        for i in range(len(mutator_freqs)):
            g = np.random.binomial(2,mutator_freqs[i])
            if g == 2:
               self.mutator_loci_homo.append(i)
            elif g == 1:
               self.mutator_loci_het.append(i)

    # create a haploid individual from a diploid individual
    def make_gamete(self, population, mutations_at_fixed, index):

        # mendelian segregation in mutator loci
        one_chosen_bits = []
        if self.mutator_loci_het:
            one_chosen_bits = np.random.binomial(1, 1 / 2, len(self.mutator_loci_het))
            one_chosen_bits = [self.mutator_loci_het[i] for i in np.where(one_chosen_bits)[0]]
        mutator_loci = self.mutator_loci_homo + one_chosen_bits

        # determine the mutation rate at modifier loci
        if population.invariable_mutator_mutation_rate:
            mutator_mutation_rate = population.mutator_mutation_rate
        else:
            print('whatthefuck')
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

        # determine the mutation rate for selected loci
        if population.variable_mutator_effect:
            mutation_rate = population.mutation_rate + population.fixed_mutation_rate
            for i in self.mutator_loci_het:
                mutation_rate += population.mutator_effects[i]
            for i in self.mutator_loci_homo:
                mutation_rate += 2*population.mutator_effects[i]
        else:
            mutation_rate = population.mutation_rate + population.phi * (len(self.mutator_loci_het) + 2 * len(self.mutator_loci_homo) + 2 * len(population.fixed))

        # mendelian segregation at selected loci
        # can use this assumption so long as 2Nhs >> 1 and infinite sites approximation applies
        inherited_mutations = np.random.binomial(self.selected_mutations, 1 / 2)

        # new mutations at selected loci
        new_mutations = np.random.poisson(population.loci * mutation_rate)

        # total number of derived alleles at selected sites
        selected_mutations = new_mutations + inherited_mutations

        # update locus used to measure heterozygosity
        hetero = self.hetero[np.random.rand() > 0.5]

        return Gamete(selected_mutations = selected_mutations,
                   mutator_loci = mutator_loci,
                   hetero = hetero)



