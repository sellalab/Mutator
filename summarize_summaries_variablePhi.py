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
                      'NE_path': '/Users/will_milligan/PycharmProjects/MUTATOR_FINAL/MSMC_NE_dict.pickle',
                      # where do we get population size estimates
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
    parser.add_argument("--mutator_mutation_rate", help="Mutation rate at modifier sites", type=float,
                        default=default_params['mutator_mutation_rate'])
    parser.add_argument("--mutation_rate", help="baseline mutation rate at selected sites, u0", type=float,
                        default=default_params['mutation_rate'])
    parser.add_argument("--loci", help="number of selected loci", type=float, default=default_params['loci'])
    parser.add_argument("--constant", help="Is pop. size constant?", type=bool, default=default_params['constant'])
    parser.add_argument("--invariable_mutator_mutation_rate", help="Is the mutator mutation rate invariable?",
                        type=bool,
                        default=default_params['invariable_mutator_mutation_rate'])
    parser.add_argument("--split_gen", help="What generation do we split at, None if not split", type=int,
                        default=default_params['split_gen'])
    parser.add_argument("--total_gen", help="Total num. of generations to simulate", type=int,
                        default=default_params['total_gen'])
    parser.add_argument("--backup_gen", help="How many generations between backing up populations ", type=int,
                        default=default_params['backup_gen'])
    parser.add_argument("--ignore_gen", help="Stop simulations at this generations", type=int,
                        default=default_params['ignore_gen'])
    parser.add_argument("--outpath", help="Where to store populations, should be directory (i.e., end in /)", type=str,
                        default=default_params['outpath'])
    parser.add_argument("--NE_path", help="Where are pop. sizes stored", type=str, default=default_params['NE_path'])
    parser.add_argument("--variable_mutator_effect", help="False is mutator effect size is constant", type=bool,
                        default=default_params['variable_mutator_effect'])
    parser.add_argument("--store_trajectories", help="Should we consolidate and store all mutator trajectories",
                        type=bool, default=False)
    parser.add_argument("--sampling_interval", help="How often to sample mutator frequencies in units of N ",
                        type=float, default=default_params['sampling_interval'])

    args = parser.parse_args()

    # check that results directory exists
    assert os.path.exists(args.outpath)

    # get the directory where all replicates are stored
    replicate_directory = os.path.dirname(args.outpath)

    # summary_functions
    summary_functions = {'mean': (lambda a: np.nanmean(a),
                                  lambda a: np.sqrt(np.nanvar(a) / np.shape(a)[1]) * 1.96,
                                  lambda j: sum([q * p for q, p in j.items()])),

                         'var': (lambda a: np.nanvar(a),
                                 lambda a: np.sqrt(
                                     (np.mean(a ** 4 - np.mean(a)) - np.mean(a ** 2 - np.mean(a))) / np.shape(a)[
                                         1] ** 2) * 1.96,
                                 lambda j: sum([q ** 2 * p for q, p in j.items()]) - sum(
                                     [q * p for q, p in j.items()]) ** 2),

                         'within': (lambda a, m: np.nanmean(m * a * (1 - a)),
                                    lambda a, m: np.sqrt(np.nanvar(m * a * (1 - a)) / np.shape(a)[1]) * 1.96,
                                    lambda j: sum([q * (1 - q) * p for q, p in j.items()]))}

    # results are seperated into bins of 10
    for bin_r in range(10):
        # load all summarized results
        all_results = load_all_summaries(replicate_directory=replicate_directory,
                                         bin_r=bin_r,
                                         summary_functions=summary_functions)

        reformatted_values = reformat_values(results_dict=all_results)

        results_summarized = summarize_all_results(results=reformatted_values)

        write_out(replicate_directory=replicate_directory,
                  all_results_summarized=results_summarized,
                  bin_r=bin_r)

# summarize all results from a given bin
def summarize_all_results(results):
    summarized_results = {}
    for summary, sdict in results.items():
        mean = np.nanmean(sdict['mean'])
        error = np.sqrt(sum([i ** 2 for i in sdict['error']]))
        error2 = np.sqrt(np.nanvar(sdict['mean']) / len(sdict['mean'])) * 2
        analytic = np.nanmean(sdict['analytic'])

        summarized_results[summary] = (mean, error, error2, analytic)
        print('summarize_all_results', summary, summarized_results[summary])
    return summarized_results

def write_out(replicate_directory, all_results_summarized, bin_r=np.nan):
    filename = 'all_results_summarized'
    if not np.isnan(bin_r):
        filename = filename + f'_bin{bin_r}'

    with open(os.path.join(replicate_directory, filename + '.pickle'), 'wb+') as fout:
        pickle.dump(all_results_summarized, fout)

# calculate summaries of interest from sampled mutator frequencies
def summarize_results(results, summary_functions):

    (args, population, mutator_frequencies, mean, variance, within, all_sD, phi_values) = results

    mutator_effects = 2 * np.array(population.mutator_effects[phi_values])

    summarized_results = {}

    n = np.shape(mutator_frequencies)[1]

    summarized_results['mean'] = (
        np.nanmean(mutator_frequencies),
        np.sqrt(np.nanvar(mutator_frequencies) / n) * 1.96,
        np.mean([sum([q * p for q, p in sd.items()]) for sd in all_sD.values()])
    )

    mean_rate_object = mutator_frequencies * mutator_effects
    summarized_results['mean_rate'] = (
        np.nanmean(mean_rate_object),
        np.sqrt(np.nanvar(mean_rate_object) / n) * 1.96,
        np.mean([sum([q * p for q, p in sd.items()]) * mutator_effects[i] for i, sd in all_sD.items()])
    )

    summarized_results['var'] = (
        np.nanvar(mutator_frequencies),
        np.sqrt(np.nanvar(mutator_frequencies) / n) * 1.96,
        np.mean([sum([q ** 2 * p for q, p in sd.items()]) for i, sd in enumerate(all_sD.values())]) - summarized_results['mean'][-1]**2
    )

    var_rate_object = mutator_frequencies * mutator_effects
    summarized_results['var_rate'] = (
        np.nanvar(var_rate_object),
        np.sqrt(np.nanvar(var_rate_object) / n) * 1.96,
        np.mean([sum([q ** 2 * p for q, p in sd.items()]) *
                 mutator_effects[i] ** 2 for i, sd in enumerate(all_sD.values())]) - summarized_results['mean_rate'][-1]**2
    )

    within_object = 2 * mutator_frequencies * (1 - mutator_frequencies)
    summarized_results['within'] = (
        np.nanmean(within_object),
        np.sqrt(np.nanvar(within_object) / n) * 1.96,
        np.mean([sum([2 * q * (1 - q) * p for q, p in sd.items()]) for sd in all_sD.values()])
    )

    within_rate_object = 2 * mutator_frequencies * (1 - mutator_frequencies) * (mutator_effects / 2) ** 2
    summarized_results['within_rate'] = (
        np.nanmean(within_rate_object),
        np.sqrt(np.nanvar(within_rate_object) / n) * 1.96,
        np.mean([sum([2 * q * (1 - q) * p for q, p in sd.items()]) *
                 (mutator_effects[i]/2) ** 2 for i, sd in enumerate(all_sD.values())])
    )

    for summary in summarized_results.keys():
        print('summarize_results', summary, summarized_results[summary])

    return summarized_results

# change how results are formatted from {version: {summary: values}} to {summary:[list of values]}
def reformat_values(results_dict):
    reformatted_values = {}

    for v in results_dict.values():
        for summary, (mean, error, analytic) in v.items():
            if summary not in reformatted_values.keys():
                reformatted_values[summary] = ddict(list)
            reformatted_values[summary]['mean'].append(mean)
            reformatted_values[summary]['error'].append(error)
            reformatted_values[summary]['analytic'].append(analytic)

    for summary in reformatted_values.keys():
        print('reformatted_values', summary, reformatted_values[summary]['mean'])
    return reformatted_values


# load sampled mutator frequencies and return summaries of interest
def load_all_summaries(replicate_directory, summary_functions, bin_r=np.nan):
    if not np.isnan(bin_r):
        filename = f'summarized_results_bin{bin_r}.pickle'
    else:
        filename = 'summarized_results.pickle'
    all_results = {}

    for v in os.listdir(replicate_directory):
        v_directory = os.path.join(replicate_directory, v)
        if not os.path.isdir(v_directory): continue

        with open(os.path.join(os.path.join(v_directory, 'ancestral'), filename), 'rb') as fin:
            (args, population, mutator_frequencies, mean, variance, within, all_sD, phi_values) = pickle.load(fin)

        results = (args, population, mutator_frequencies, mean, variance, within, all_sD, phi_values)

        all_results[v] = summarize_results(results=results,
                                           summary_functions=summary_functions)

    return all_results


if __name__ == '__main__':
    main()
