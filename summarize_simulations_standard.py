import numpy as np
import pickle
import argparse
import mutator_classes
import os
import stationary_distribution_aug as sd
import gzip

def main():
    # a set of default params to use

    default_params = {'N': 2000,  # population size
                      'M': 1000,  # number of modifier loci, M
                      'h': 0.5,  # h
                      's': 0.01,  # s - together hs are the average fitness effects of mutations at selected loci
                      'phi': 1E-12,  # effect size of mutator alleles
                      'mutator_mutation_rate': 1.25E-7,  # Mutation rate at modifier sites
                      'mutation_rate': 1.25E-7,  # baseline mutation rate at selected sites, u0
                      'loci': 3E8 * 0.08,  # number of selected loci
                      'constant': True,  # is the population size constant
                      'split_gen': 0,
                      # the generation at which the ancestral population is split into europeans and africans
                      'backup_gen': 100,  # backup the population every 100 generations
                      'ignore_gen': 70,  # stop simulations this many generations from the present
                      'total_gen': 10000,  # how many total generations to simulate
                      'outpath': 'blah3',  # where do we store results
                      'NE_path': '/Users/will_milligan/PycharmProjects/MUTATOR_FINAL/MSMC_NE_dict.pickle',  # where do we get population size estimates
                      'invariable_mutator_mutation_rate': True,
                      'variable_mutator_effect': False,
                      'sampling_interval': 10}

    print(default_params)
    # # get and parse input string
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="population_size", type=int, default=default_params['N'])
    parser.add_argument("--M", help="number of modifier loci, M", type=int, default=default_params['M'])
    parser.add_argument("--h", help="h", type=float, default=default_params['h'])
    parser.add_argument("--s", help="s", type=float, default=default_params['s'])
    parser.add_argument("--phi", help="Mutator effect size", type=float, default=default_params['phi'])
    parser.add_argument("--mutator_mutation_rate", help="Mutation rate at modifier sites", type=float, default=default_params['mutator_mutation_rate'])
    parser.add_argument("--mutation_rate", help="baseline mutation rate at selected sites, u0", type=float, default=default_params['mutation_rate'])
    parser.add_argument("--loci", help="number of selected loci", type=float, default=default_params['loci'])
    parser.add_argument("--constant", help="Is pop. size constant?", type=bool, default=default_params['constant'])
    parser.add_argument("--invariable_mutator_mutation_rate", help="Is the mutator mutation rate invariable?", type=bool,
                        default=default_params['invariable_mutator_mutation_rate'])
    parser.add_argument("--split_gen", help="What generation do we split at, None if not split", type=int, default=default_params['split_gen'])
    parser.add_argument("--total_gen", help="Total num. of generations to simulate", type=int, default=default_params['total_gen'])
    parser.add_argument("--backup_gen", help="How many generations between backing up populations ", type=int, default=default_params['backup_gen'])
    parser.add_argument("--ignore_gen", help="Stop simulations at this generations", type=int, default=default_params['ignore_gen'])
    parser.add_argument("--outpath", help="Where to store populations, should be directory (i.e., end in /)", type=str, default=default_params['outpath'])
    parser.add_argument("--NE_path", help="Where are pop. sizes stored", type=str, default=default_params['NE_path'])
    parser.add_argument("--variable_mutator_effect", help="False is mutator effect size is constant", type=bool, default=default_params['variable_mutator_effect'])
    parser.add_argument("--store_trajectories", help="Should we consolidate and store all mutator trajectories", type=bool, default=False)
    parser.add_argument("--sampling_interval", help="How often to sample mutator frequencies in units of N ",
                        type=float, default=default_params['sampling_interval'])

    args = parser.parse_args()
    print(args)
    # check that results directory exists
    assert os.path.exists(args.outpath)

    for pop_dir in os.listdir(args.outpath):
        print(pop_dir)
        process_population_results(args = args, pop_dir = pop_dir)


def process_population_results(args,pop_dir):

    # determine if this is a population that predates or postdates a population split
    if pop_dir == 'ancestral':
        split = False
    else:
        split = True

    # directory where results are stored for this population
    population_directory = os.path.join(args.outpath, pop_dir)

    # load a copy of the population
    # useful for getting some specific parameters defined after initialization (e.g., distribution of fitness effects)
    population = load_final_population(population_directory)

    # load a matrix representing the mutator allele frequency at each
    mutator_counts = load_mutator_counts(args = args, population = population,split=split)

    # sample the mutator frequencies
    mutator_frequencies = mutator_counts[::int(args.sampling_interval*args.N), :]

    # calculate mutator freq. moments
    mean = np.nanmean(mutator_frequencies)
    variance = np.nanvar(mutator_frequencies)
    within = np.nanmean(mutator_frequencies*(1-mutator_frequencies))

    # calculate stationary distributions
    sD = calculate_stationary_distributions(population = population,args = args)

    write_out(args=args,
              population=population,
              mutator_frequencies = mutator_frequencies,
              mean = mean,
              variance = variance,
              within = within,
              sD = sD,
              population_directory = population_directory)

    if args.store_trajectories:
        write_out_mutator_trajectories(args = args, population = population, population_directory = population_directory, mutator_counts = mutator_counts)


def write_out(args,
              population,
              mutator_frequencies,
              mean,
              variance,
              within,
              sD,
              population_directory):

    with open(os.path.join(population_directory,'summarized_results.pickle'),'wb+') as fout:
        pickle.dump((args,population,mutator_frequencies,mean,variance,within,sD),fout)

def write_out_mutator_trajectories(args, population, population_directory, mutator_counts):

    mutator_count_storage_outpath = os.path.join(population_directory,'mutator_count_storage/')
    os.makedirs(mutator_count_storage_outpath, exist_ok=True)

    for i in args.M:
        with gzip.open(os.path.join(mutator_count_storage_outpath,f'trajectory_{i}.gz'),'wb+') as fout:
            pickle.dump(mutator_counts[:,i],fout)

def calculate_stationary_distributions(population, args):
    return sd.get_SD(p=population, phi=args.phi)

# load and return final population
def load_final_population(population_directory):

    with open(os.path.join(population_directory,'final_population.pickle'), 'rb') as fin:
        final_population = pickle.load(fin)
    return final_population

def load_mutator_counts(args, population,split):

    # determine what size the matrix should be and how many generations were simulated
    if split:
        total_gen = args.split_gen
        simulated_gen = args.split_gen - args.ignore_gen
    else:
        total_gen = args.total_gen - args.split_gen
        simulated_gen = total_gen - args.split_gen

    # make a blank matrix
    mutator_counts = np.zeros([total_gen,args.M],dtype=float)

    # load realized mutator allele frequencies
    with gzip.open(population.mutator_counts_outpath,'rb') as fin:
        for i in range(simulated_gen):
            mutator_counts[i,:] = pickle.load(fin)

    # set ignored generations to nan
    if split:
        mutator_counts[simulated_gen:,:] = np.nan

    return mutator_counts

if __name__ == '__main__':
    main()
