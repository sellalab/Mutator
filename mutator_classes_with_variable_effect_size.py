import os
import pickle
import sys
import numpy as np
import collections
from scipy import stats
import gzip
from scipy.integrate import quad
from copy import deepcopy

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

        self.calculate_effective_population_size = False
        self.total_heterozygosity = 0
        self.NE_algorithm_active = False
        self.wait_time_NE_algorithm = int(2 * self.N)

        self.variable_mutator_effect = variable_mutator_effect
        if self.variable_mutator_effect:

            if mutator_effect_size_distribution:
                self.mutator_effect_size_distribution = mutator_effect_size_distribution
            else:
                # print(f'Using default mutator effect size distribution that is exponentially distributed with mean {self.phi}')
                # self.mutator_effect_size_distribution = stats.expon(scale=self.phi)

                print(f'Using default mutator effect size distribution that is loguniform distributed over 0.01 < 2Ns < 1000')
                max_phi = 1000/(4*self.N*self.s*self.h*self.loci)
                min_phi = 0.01 / (4 * self.N * self.s * self.h * self.loci)
                self.mutator_effect_size_distribution = stats.loguniform(a=min_phi,b=max_phi)


            self.mutator_effects = self.mutator_effect_size_distribution.rvs(self.M)
            print(f'Mean mutator effect size is {np.mean(self.mutator_effects)} with variance {np.var(self.mutator_effects)}')
        else:
            print(f'Using constant mutator effect size of {self.phi}')

        # Define the distribution of effect sizes at the selected sites
        self.variable_selective_effect = variable_selective_effect
        if self.variable_selective_effect:

            if selected_effect_distribution:
                self.selected_effect_distribution = selected_effect_distribution
            else:

                min_hs = 20 /(2*self.N)
                max_hs = 20.1/(2*self.N)
                print(min_hs,max_hs)
                print(f'Using default selected effect size distribution that is loguniform distributed over {min_hs} < hs < {max_hs}')
                self.selected_effect_distribution = stats.loguniform(b=max_hs, a=min_hs)
                
            self.s = self.selected_effect_distribution.mean()*2 # because h = 0.5
        else:
            print(f'Using constant selected effect size of {self.h*self.s}')

        self.people = [Individual(self.selected_effect_distribution) for _ in range(self.N)]
        if initialize:
            self.init_from_params()

    def update_outpath(self,new_outpath):
        self.outpath = new_outpath
        self.mutator_counts_outpath = os.path.join(self.outpath, 'mutator_counts')

    def init_from_params(self):

        if self.variable_mutator_effect:
            mutator_freqs = []
            for phi in self.mutator_effects:
                SD = sd.get_SD(p=self,phi=phi)
                mutator_freqs.append(np.random.choice(list(SD.keys()),
                                                 p=list(SD.values()),
                                                 size=1)[0])
        else:

            # determine the stationary distribution of mutator allele frequencies
            SD = sd.get_SD(p=self)

            # use the stationary distribution to randomly draw M starting mutator frequencies
            mutator_freqs = np.random.choice(list(SD.keys()),
                                             p=list(SD.values()),
                                             size = self.M)

        # Determine the mean number of deleterious mutations carried by individuals
        if self.variable_selective_effect:
            poisson_param = 2 * self.mutation_rate * self.loci * quad(lambda s: 1/s*self.selected_effect_distribution.pdf(s),self.selected_effect_distribution.ppf(0),self.selected_effect_distribution.ppf(1))[0]
        else:
            poisson_param = 2 * self.mutation_rate * self.loci / (self.h * self.s)
        
        print(f'Poisson param {poisson_param}')
        
        # create N individual objects in the population based on:
        # mean number of deleterious mutations and starting mutator frequencies
        for i in self.people:
            i._init_from_params(poisson_param = poisson_param, mutator_freqs = mutator_freqs)

    def calc_relative_fitness(self):

        if self.selected_effect_distribution:
            fit = np.array([i.fitness() for i in self.people])
    
        else:

            # calculate the mean number of mutations carried by people
            mean_mutations = np.mean([i.selected_mutations for i in self.people])

            # calculate fitness by first scaling by 1/(1-hs)^(mean # of mutations)
            # this helps avoids floating point errors
            fit = np.array([(1 - self.h * self.s) ** (i.selected_mutations - mean_mutations) for i in self.people])

        # return relative fitness
        return fit/sum(fit)

    # choose parents for the next generation based on fitness
    def choose_parents(self, N):

        # get relative fitness of individuals
        prob_fit = self.calc_relative_fitness()
        
        # choose 2N parents where the probability of being chosen is proportional to their fitness
        parents = np.random.choice(a=self.people, size=2*N, p=prob_fit, replace=True)
        
        return parents

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
            try:
                new_person = deepcopy(self.reproduce(gameteA, gameteB))
            except MemoryError:
                print(len(gameteA.selected_mutations),len(gameteB.selected_mutations))
            new_people.append(new_person)

        self.people = new_people
        sys.stdout.flush()
        
        # update population size
        self.N = N

        # this must be done every generation
        self.deal_with_mutations_at_fixed_loci()
        
        if self.calculate_effective_population_size:
            if self.NE_algorithm_active:
                self.update_NE_algorithm()
            else:
                self.wait_time_NE_algorithm += -1
                if self.wait_time_NE_algorithm <= 0:
                    self.initiate_NE_algorithm()
    
    # turns 2 gametes into an individual
    def reproduce(self, gameteA, gameteB):

        # initialize an individual
        child = Individual(selected_effect_distribution=self.selected_effect_distribution)

        # if using variable selected effects, the selected mutations require a different implementation
        if self.selected_effect_distribution:
            child.selected_mutations = np.append(gameteA.selected_mutations,gameteB.selected_mutations)
        else:
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

    def store(self,a):
        self.storage.append(a)

    def write_out(self):
        #print('mean selected mutations: ',np.mean([len(i.selected_mutations) for i in self.people]))
        with gzip.open(self.mutator_counts_outpath,'ab+') as fout:
            for a in self.storage:
                pickle.dump(a, fout)
        self.storage=[]

    # initiates the algorithm to measure NE
    def initiate_NE_algorithm(self):

        # assigns each individual two unique alleles at a given locus
        for index,i in enumerate(self.people):
            i.hetero = [index*2,index*2+1]
        # markers to keep track of the total heterozygosity and if the algorithm is active or not
        self.total_heterozygosity = 0
        self.NE_algorithm_active = True

    def update_NE_algorithm(self):

        # gets the frequency of all currently segregating alleles
        allele_counts = {}
        for i in self.people:
            for index in i.hetero:
                if index not in allele_counts.keys():
                    allele_counts[index] = 0
                allele_counts[index] += 1
        allele_freqs = {k:v/(2*self.N) for k,v in allele_counts.items()}

        # calculates the current heterozygosity
        current_heterozygosity = 1-sum([i**2 for i in allele_freqs.values()])
        # add the current heterozygosity to the total
        self.total_heterozygosity += current_heterozygosity

        # if an allele has fixed, stop the algorithm
        if current_heterozygosity == 0:
            self.write_out_NE_algorithm()
            self.NE_algorithm_active = False
            self.wait_time_NE_algorithm = int(2*self.N)

    # store the results
    def write_out_NE_algorithm(self):
        NE_algorithm_outpath = os.path.join(os.path.dirname(self.mutator_counts_outpath),'NE_algorithm.pickle')
        with open(NE_algorithm_outpath,'ab+') as fout:
            pickle.dump(self.total_heterozygosity,fout)


class Gamete():
    def __init__(self,
                   selected_mutations = 0,
                   mutator_loci = [],
                   hetero = 0):
        self.selected_mutations = selected_mutations
        self.mutator_loci = mutator_loci
        self.hetero = hetero

class Individual:
    def __init__(self,selected_effect_distribution = False):
        self.mutator_loci_het = []
        self.mutator_loci_homo = []
        self.selected_effect_distribution = selected_effect_distribution

        if selected_effect_distribution:
            self.selected_mutations = []
        else:
            self.selected_mutations = 0

        self.hetero = [0,0]

    # we only use this for the case with variable selected effects
    def fitness(self):
        if len(self.selected_mutations) == 0: 
            return 1

        #assert np.all(self.selected_mutations <= 0.5)
        #assert np.all(self.selected_mutations > 0)
        fit = np.product(1-self.selected_mutations)
        # self.selected_mutations is an array of hs values
        #assert fit > 0
        return fit

    def _init_from_params(self, mutator_freqs, poisson_param = 0):

        # if using variable selected effects, we do not initialize the selected sites.
        # It's faster to just allow them to burn in as they are strongly selected.
        if not self.selected_effect_distribution:
            self.selected_mutations = np.random.poisson(poisson_param)
        elif False:
            
            L = np.random.poisson(20*poisson_param)
            hs_values = self.selected_effect_distribution.ppf(np.linspace(0,1,100))
            probs = 1/hs_values
            probs = probs/sum(probs)
            outcome = np.random.multinomial(L,probs)
            
            for hs, i  in zip(hs_values,outcome):
                self.selected_mutations += [hs]*i
            self.selected_mutations = np.array(self.selected_mutations)

        for i in range(len(mutator_freqs)):
            g = np.random.binomial(2,mutator_freqs[i])
            if g == 2:
               self.mutator_loci_homo.append(i)
            elif g == 1:
               self.mutator_loci_het.append(i)

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
        # print(mutation_rate)
        if self.selected_effect_distribution:
            # mendelian segregation at selected loci
            # can use this assumption so long as 2Nhs >> 1 and infinite sites applies
            # randomly pass on current selected mutations with probability 1/2
            n_inherited_mutations = np.random.binomial(len(self.selected_mutations), 1 / 2)
            inherited_mutations = np.random.choice(a = self.selected_mutations, size = n_inherited_mutations, replace = False)
           
            #chosen_mutations = np.random.binomial(1, 1 / 2, len(self.selected_mutations))
            #inherited_mutations = [self.selected_mutations[i] for i in np.where(chosen_mutations)[0]]
            #n_inherited_mutations = sum(chosen_mutations)

            # new mutations at selected loci
            n_new_mutations = np.random.poisson(population.loci * mutation_rate)
            new_mutations = self.selected_effect_distribution.rvs(n_new_mutations)
            
            selected_mutations = np.append(new_mutations, inherited_mutations)
            assert len(selected_mutations) == n_inherited_mutations + n_new_mutations

        else:
            # mendelian segregation at selected loci
            # can use this assumption so long as 2Nhs >> 1 and infinite sites applies
            inherited_mutations = np.random.binomial(self.selected_mutations, 1 / 2)

            # new mutations at selected loci
            new_mutations = np.random.poisson(population.loci * mutation_rate)

            selected_mutations = new_mutations + inherited_mutations
            print('wrong area')

        # update locus used to measure heterozygosity
        hetero = self.hetero[np.random.rand() > 0.5]

        return Gamete(selected_mutations = selected_mutations,
                   mutator_loci = mutator_loci,
                   hetero = hetero)
