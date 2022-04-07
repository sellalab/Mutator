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


# loads data from a given path
def load_data(args,min_length, in_path):
    storage = []

    with gzip.open(in_path, 'rb') as fin:
        while True:
            try:
                storage.append(pickle.load(fin))
            except EOFError:
                break

    # check that the data is long enough
    if len(storage) < min_length:
        print(len(storage),min_length,in_path)
        raise ValueError

    # reformat the data from a list of lists to an array with cooridnates [locus,generation]
    storage = np.reshape(storage, [len(storage), args.M])
    storage = np.transpose(storage)

    return storage

def get_full_data(args, in_path):

    # information needed to load the data and ensure that they are the correct length
    pop_names = ['ancestral', 'CEU', 'YRI']
    minimum_lengths = {p:i for p,i in zip(pop_names,[60000,10000-args.ignore_gen,10000-args.ignore_gen])}
    data_lengths = {p: i for p, i in zip(pop_names, [50000, 10000, 10000])}
    # accidentally had the ancestral saved differently than the other populations
    filenames = {p:i for p,i in zip(pop_names,['mutator_counts.pickle.gz', 'mutator_counts', 'mutator_counts'])}

    # create storage object
    storage = {}

    # for each population
    for pop_name in pop_names:

        # get the relevant name and minimum length
        min_length = minimum_lengths[pop_name]
        filename = filenames[pop_name]

        # find, load, and store the data
        full_pdir = os.path.join(in_path,pop_name)
        mutator_counts_file = os.path.join(full_pdir, filename)
        new_data = load_data(args = args, in_path = mutator_counts_file,min_length = min_length)
        storage[pop_name] = new_data

    # trim burn-in generations and unsimulated generations
    trimmed_data,mean_data = trim_data(args = args, storage = storage,data_lengths=data_lengths)

    #
    collapsed_data = collapse_trajectories(trimmed_data, args)

    # dump the summarized data
    with gzip.open(os.path.join(in_path, 'summarized_trajectory_data.pickle.gz'), 'wb+') as fout:
        pickle.dump(collapsed_data, fout)

    with gzip.open(os.path.join(in_path, 'summarized_mean_freqs.pickle.gz'), 'wb+') as fout:
        pickle.dump(mean_data, fout)

def trim_data(args,storage,data_lengths):

    trimmed_data = {}
    mean_data = {}

    # for each population
    for pop_name, data in storage.items():

        # if the ancestral population, we chop of the burn in generations
        if pop_name == 'ancestral':
            data_new = data[:,-data_lengths[pop_name]:]
        # otherwise, we add 0 values for the last few generations that we didn't simulated
        else:
            data_new = np.zeros([args.M,data_lengths[pop_name]])+np.nan
            data_new[:, :np.shape(data)[1]] = data
            data_new[:,-args.ignore_gen:] = 0
            if np.any(np.isnan(data_new)):
               print(np.where(np.isnan(data_new)))
               raise ValueError

        # store trimmed data and the mean mutator frequencies
        trimmed_data[pop_name] = data_new
        mean_data[pop_name] = np.nanmean(data_new,axis=1)

    return trimmed_data,mean_data

# load the ARG summaries
def get_ARG_summaries(arg_path = '/home/ec2-user/Mutator/coalescent_summaries.pickle'):
    with open(arg_path,'rb') as fin:
        aCEU, aYRI, aANC = pickle.load(fin)
    return {'CEU':np.flip(aCEU),'YRI':np.flip(aYRI), 'ancestral':np.flip(aANC)}

# determine where the window boundaries are for the Speidel-Myers test
def get_window_boundaries():
    delimiters = np.logspace(np.log10(60), 4, 8)
    window_boundaries = [(int(-a),int(-b)) for a,b in zip(delimiters[1:],delimiters[:-1])]
    return window_boundaries

# for each locus, we create 3 trajectories that we use to measure enrichments
def get_locus_data(trimmed_data,locus):

    # load data for this locus
    locus_data = {}
    for pop_name,data in trimmed_data.items():
        locus_data[pop_name] = data[locus,:]

    # create mutator allele trajectories

    # the ancestral one needs to have the same shape as the other trajectories,
    # so we append the CEU-specific trajectory but it is not used
    aTrajectory = np.append(locus_data['ancestral'],locus_data['CEU'])
    cTrajectory = np.append(locus_data['ancestral'],locus_data['CEU'])
    yTrajectory = np.append(locus_data['ancestral'], locus_data['YRI'])
    
    return {pop_name:t for pop_name,t in zip(['ancestral','CEU','YRI'],[aTrajectory,cTrajectory,yTrajectory])}

# summarize each locus according to the two tests that we do
def summarize_locus(locus_data,ARG_summaries, window_boundaries):

    # calculate a proxy for the expected number of mutations this modifier locus would cuase in each population
    # specifically its sum(E(q(t))*o(t)) where o(t) is the number of mutational opportunities
    # later we multiply by 2*phi to get the true expectation
    HP_summaries = {}
    for pop_name in locus_data.keys():
        traj = locus_data[pop_name]
        arg = ARG_summaries[pop_name]
        mean_traj = np.mean(traj)
        past_contribution = sum(mean_traj*arg[:-len(traj)])
        HP_summaries[pop_name] = sum(traj*arg[-len(traj):]) + past_contribution

    # calculate the mean mutator frequency in each window for both populations
    SM_summaries = {}
    for pop_name in ['CEU','YRI']:
        SM_summaries[pop_name] = []
        for (t1,t2) in window_boundaries:
            SM_summaries[pop_name].append(np.mean(locus_data[pop_name][t1:t2]))
        SM_summaries[pop_name]  = np.array(SM_summaries[pop_name])

    return (HP_summaries,SM_summaries)

# we reduce the trajectories into only the information we need to save resources
def collapse_trajectories(trimmed_data,args):

    # create a storage object
    summarized_trajectories = {}

    # get ARG summaries and window boundaries for the two enrichment tests
    ARG_summaries = get_ARG_summaries()
    window_boundaries = get_window_boundaries()


    for i in range(args.M):

        # load the information for this locus and then summarize and store it
        locus_data = get_locus_data(trimmed_data = trimmed_data,locus = i)

        (HP_summaries, SM_summaries) = summarize_locus(locus_data = locus_data,
                                                       ARG_summaries = ARG_summaries,
                                                       window_boundaries = window_boundaries)
        
        summarized_trajectories[i] = (HP_summaries, SM_summaries)

    return summarized_trajectories


def main():

    args = get_args()
    get_full_data(args = args, in_path = args.outpath)

if __name__ == '__main__':
    main()
