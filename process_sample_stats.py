#!/usr/bin/env python
# coding: utf-8

# import required packages
import argparse
from collections import OrderedDict
import numpy as np
import pickle
import os
import gzip

# import local python files
import sys
sys.path.append('/ifs/scratch/c2b2/gs_lab/wm2377/Mutator_Project/change_NE')
import stationary_distribution_aug as sd
import mutator_classes
from collections import Counter

# this function assigns default argument values and parses provided arguments
def get_args():
    # a set of default params to use

    default_params = {'N': 2000,  # population size
                      'M': 1000,  # number of modifier loci, M
                      'h': 0.5,  # h
                      's': 0.01,  # s - together hs are the average fitness effects of mutations at selected loci
                      'phi': 1E-12,  # effect size of mutator alleles
                      'mutator_mutation_rate': 1.25E-7,  # Mutation rate at modifier sites
                      'mutation_rate': 1.25E-7,  # baseline mutation rate at selected sites, u0
                      'loci': 3E8 * 0.08,  # number of selected loci
                      'constant': int(True),  # is the population size constant
                      'split_gen': 0,
                      # the generation at which the ancestral population is split into europeans and africans
                      'backup_gen': 100,  # backup the population every 100 generations
                      'ignore_gen': 70,  # stop simulations this many generations from the present
                      'total_gen': 10000,  # how many total generations to simulate
                      'outpath': 'blah3',  # where do we store results
                      'NE_path': '/Users/will_milligan/PycharmProjects/MUTATOR_FINAL/MSMC_NE_dict.pickle',
                      # where do we get population size estimates
                      'invariable_mutator_mutation_rate': int(True),
                      'variable_mutator_effect': int(False),
                      'variable_selective_effect': int(False),
                      'which_population': 'ancestral',
                      'mode': 'default',
                      'M_sample': 1}

    ## get and parse input string
    ## argparse can not handle bools so read as int and then convert to bool
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",
                        help="population_size",
                        type=int,
                        default=default_params['N'])
    parser.add_argument("--M",
                        help="number of modifier loci, M",
                        type=int,
                        default=default_params['M'])
    parser.add_argument("--h",
                        help="h",
                        type=float,
                        default=default_params['h'])
    parser.add_argument("--s",
                        help="s",
                        type=float,
                        default=default_params['s'])
    parser.add_argument("--phi",
                        help="Mutator effect size",
                        type=float,
                        default=default_params['phi'])
    parser.add_argument("--mutator_mutation_rate",
                        help="Mutation rate at modifier sites",
                        type=float,
                        default=default_params['mutator_mutation_rate'])
    parser.add_argument("--mutation_rate",
                        help="baseline mutation rate at selected sites, u0",
                        type=float,
                        default=default_params['mutation_rate'])
    parser.add_argument("--loci",
                        help="number of selected loci",
                        type=float,
                        default=default_params['loci'])
    parser.add_argument("--constant",
                        help="Is pop. size constant?",
                        type=int,
                        default=default_params['constant'])
    parser.add_argument("--split_gen",
                        help="What generation do we split at, None if not split",
                        type=int,
                        default=default_params['split_gen'])
    parser.add_argument("--backup_gen",
                        help="How many generations between backing up populations ",
                        type=int,
                        default=default_params['backup_gen'])
    parser.add_argument("--ignore_gen",
                        help="Stop simulations at this generations",
                        type=int,
                        default=default_params['ignore_gen'])
    parser.add_argument("--total_gen",
                        help="Total num. of generations to simulate",
                        type=int,
                        default=default_params['total_gen'])
    parser.add_argument("--outpath",
                        help="Where to store populations, should be directory (i.e., end in /)",
                        type=str,
                        default=default_params['outpath'])
    parser.add_argument("--NE_path",
                        help="Where are pop. sizes stored",
                        type=str,
                        default=default_params['NE_path'])
    parser.add_argument("--invariable_mutator_mutation_rate",
                        help="Is the mutator mutation rate invariable?",
                        type=int,
                        default=default_params['invariable_mutator_mutation_rate'])
    parser.add_argument("--variable_mutator_effect",
                        help="False is mutator effect size is constant",
                        type=int,
                        default=default_params['variable_mutator_effect'])
    parser.add_argument("--variable_selective_effect",
                        help="False is selected effect size is constant",
                        type=int,
                        default=default_params['variable_selective_effect'])
    parser.add_argument("--which_population",
                        help="which population are we simulating - default is ancestral",
                        type=str,
                        default=default_params['which_population'])
    parser.add_argument("--mode",
                        help="how to create mutator trajectories",
                        type=str,
                        default=default_params['mode'])
    parser.add_argument("--M_sample",
                        help="how many modifiers to sample",
                        type=str,
                        default=default_params['M_sample'])

    args = parser.parse_args()
    args.variable_selective_effect = bool(args.variable_selective_effect)
    args.variable_mutator_effect = bool(args.variable_mutator_effect)
    args.invariable_mutator_mutation_rate = bool(args.invariable_mutator_mutation_rate)
    args.constant = bool(args.constant)
    args.ignore_gen = max(60, args.ignore_gen)

    return args


def main():

    # get arguments
    args = get_args()
    mode = args.mode
    if mode == 'fewer':
        M = int(args.M_sample)*6
        M_write = int(args.M_sample)
    else:
        M = int(args.M_sample)
        M_write = int(args.M_sample)
    n_trials = int(1e4)
    
    outpath = os.path.join(os.path.dirname(args.outpath), f'sample_statistics_{M}_{mode}.pickle.gz')
    print(outpath)

    outcomes_all = []
    with gzip.open(outpath,'rb') as fin:
        for i in range(n_trials):
            outcomes_all.append(pickle.load(fin))
    print('loaded outcomes',len(outcomes_all),outcomes_all[0])

    for index,outcomes in enumerate([outcomes_all[:10000]]):
        results = {}

        HP_stats = np.array([i['HP'] for i in outcomes])
        HP_median = sum(HP_stats>1.1)/len(HP_stats)
        HP_quantile = np.quantile(HP_stats,0.95)

        results['HP'] = (HP_median,HP_quantile)
        print('finished HP',results['HP'])

        for modeSM in ['restricted','relaxed']:

            SM_set_outcomes = np.array([i[modeSM][0] for i in outcomes])
            SM_set_prob_peak = sum(SM_set_outcomes > 0) / len(SM_set_outcomes)
            SM_set_prob_over = sum(SM_set_outcomes == 0) / len(SM_set_outcomes)
            SM_set_prob_under = sum(SM_set_outcomes < 0) / len(SM_set_outcomes)
            SM_set_outer_color = (SM_set_prob_over > SM_set_prob_under)
            try:
                SM_set_inner_color = np.mean(SM_set_outcomes[SM_set_outcomes>0])
            except:
                SM_set_inner_color = np.nan
            SM_set = [SM_set_prob_peak, SM_set_inner_color, SM_set_outer_color]
            print('finished set',SM_set_prob_peak)

            SM_traj_outcomes = np.resize([i[modeSM][1] for i in outcomes],[len(outcomes),3])
            print(SM_traj_outcomes[0,:])
            SM_traj_prob_under,SM_traj_prob_peak,SM_traj_prob_over = np.sum(SM_traj_outcomes,axis=0)/sum(np.sum(SM_traj_outcomes,axis=0))
            print(SM_traj_prob_over,np.sum(SM_traj_outcomes,axis=0))
            SM_traj_outer_color = SM_traj_prob_over > SM_traj_prob_under
            SM_traj = [SM_traj_prob_peak, SM_traj_prob_over, SM_traj_outer_color]
            print('finished traj')

            SM_peak_locations_list = [i[modeSM][2] for i in outcomes]
            SM_peak_locations_all = []
            for i in SM_peak_locations_list:
                SM_peak_locations_all.extend(i)
            SM_peak_locations_summarized = dict(Counter(SM_peak_locations_all))
            print('finished peaks')

            results[modeSM] = (SM_set,SM_traj,SM_peak_locations_summarized)

        outpath = os.path.join(os.path.dirname(args.outpath), f'summarized_sample_statistics_{M_write}_{mode}.pickle')
        with open(outpath,'wb+') as fout:
            pickle.dump(results,fout)


if __name__ == '__main__':
    main()





