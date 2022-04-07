import pickle
import gzip
import numpy as np
import os
import argparse

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
                      'which_population': 'ancestral'}

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

    args = parser.parse_args()
    args.variable_selective_effect = bool(args.variable_selective_effect)
    args.variable_mutator_effect = bool(args.variable_mutator_effect)
    args.invariable_mutator_mutation_rate = bool(args.invariable_mutator_mutation_rate)
    args.constant = bool(args.constant)
    args.ignore_gen = max(60, args.ignore_gen)

    return args

def main():

    # figure out where replicate directories are
    args = get_args()
    in_path = os.path.dirname(args.outpath)

    # create a storage object
    all_mutator_tables = {}

    # for each replicate directory, load the corresponding mutator traj tables
    for vdir in os.listdir(in_path):
        if 'V' not in vdir: continue
        print(in_path)
        print(vdir)
        v = eval(vdir.split('V')[1])
        new_path = os.path.join(os.path.join(in_path,vdir),'summarized_trajectory_data.pickle.gz')
        with gzip.open(new_path,'rb') as fin:
            all_mutator_tables[v] = pickle.load(fin)

    # dump a dictionary with the structure {v:(EUR,AFR,ANC)}
    with gzip.open(os.path.join(in_path,'consolidated_mutator_tables.pickle.gz'),'wb+') as fout:
        pickle.dump(all_mutator_tables,fout)

if __name__ == '__main__':
    main()

