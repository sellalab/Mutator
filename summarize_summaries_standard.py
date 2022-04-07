import numpy as np
import pickle
import argparse
import mutator_classes
from collections import defaultdict as ddict
import os
import gzip
import stationary_distribution_aug as sd

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

    # check that results directory exists
    assert os.path.exists(args.outpath)

    # get the directory where all replicates are stored
    replicate_directory = os.path.dirname(args.outpath)

    # load all summarized results
    all_results = load_all_summaries(replicate_directory=replicate_directory)

    # summary_functions
    # calculate the mean, variance between and within populations.
    summary_functions = {'mean'   : (lambda a: np.nanmean(a[2]),
                                     lambda a: np.sqrt(np.nanvar(a[3])/len(a[3]))*1.96,
                                     lambda sd: sum([q*p for q,p in sd.items()])),

                         'var'    : (lambda a: np.nanvar(a[2]),
                                     lambda a: np.sqrt(np.nanvar(a[4])/len(a[4]))*1.96,
                                     lambda sd: sum([q**2*p for q,p in sd.items()])-sum([q*p for q,p in sd.items()])**2),

                         'within' : (lambda a: np.nanmean(a[2]*(1-a[2])),
                                     lambda a: np.sqrt(np.nanvar(a[5])/len(a[5]))*1.96,
                                     lambda sd: sum([q*(1-q)*p for q,p in sd.items()]))}

    results_summarized = summarize_all_results(results = all_results,
                                               summary_functions = summary_functions,
                                               args = args)

    write_out(replicate_directory=replicate_directory,all_results_summarized = results_summarized)


def write_out(replicate_directory,all_results_summarized):

    with open(os.path.join(replicate_directory,'all_results_summarized'),'wb+') as fout:
        pickle.dump((all_results_summarized),fout)

# use the provided functions to summarize summaries
def summarize_all_results(results,summary_functions,args):

    summarized_results = {}

    for summary, (sim_function, error_function, sd_function) in summary_functions.items():

        summarized_results[summary] = (sim_function(results), error_function(results), sd_function(results[-1]))

        print(summary,summarized_results[summary])
        
    return summarized_results

# load the summaries from simulations
def load_all_summaries(replicate_directory):

    all_freqs = None
    all_means = []
    all_vars = []
    all_within = []

    for v in os.listdir(replicate_directory):
       
        v_directory = os.path.join(replicate_directory,v)
        if not os.path.isdir(v_directory): continue

        # we only do this for simple (panmictic & constant N) simulations which are by default 'ancestral'
        with open(os.path.join(os.path.join(v_directory,'ancestral'), 'summarized_results.pickle'), 'rb') as fin:
            (args, population, mutator_frequencies, mean, variance, within, sD) = pickle.load(fin)

        if type(all_freqs)==type(None):
            all_freqs = mutator_frequencies
        else:
            all_freqs = np.append(mutator_frequencies,all_freqs,axis=1)

        all_means.append(mean)
        all_vars.append(variance)
        all_within.append(within)

    return (args,population,all_freqs,all_means,all_vars, all_within, sD)

if __name__ == '__main__':
    main()
