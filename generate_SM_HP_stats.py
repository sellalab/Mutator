# import required packages
import argparse
import sys
from collections import OrderedDict
import numpy as np
import pickle
import os
import gzip

# import local python files

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
                        type=int,
                        default=default_params['M_sample'])

    args = parser.parse_args()
    args.variable_selective_effect = bool(args.variable_selective_effect)
    args.variable_mutator_effect = bool(args.variable_mutator_effect)
    args.invariable_mutator_mutation_rate = bool(args.invariable_mutator_mutation_rate)
    args.constant = bool(args.constant)
    args.ignore_gen = max(60, args.ignore_gen)

    return args

# load info from ARG. For this step, we just need to sum
def get_ARG_summaries(arg_path = '/home/ec2-user/Mutator/coalescent_summaries.pickle'):
    with open(arg_path,'rb') as fin:
        aCEU, aYRI, aANC = pickle.load(fin)
    arg_sums = np.array([sum(aANC), sum(aCEU), sum(aYRI)])
    arg_sums = np.resize(arg_sums,[1,3])
    return arg_sums

# get the consolidated tables
def load_data(in_path):

    with gzip.open(os.path.join(in_path,'consolidated_mutator_tables.pickle.gz'),'rb') as fin:
        data = pickle.load(fin)
    return data

# determine 2Ns
def get_s(params):

    return params.phi*params.h*params.s*params.loci*params.pop_size*4

# chooses which loci to sample
def choose_loci(data,M_simulated,M):
    chosen_replicates = np.random.multinomial(M,[1/len(list(data.keys()))]*len(list(data.keys())))

    chosen_loci = {}
    for replicate,n in zip(list(data.keys()),chosen_replicates):
        chosen_loci[replicate] = np.random.choice(M_simulated,n)
    return chosen_loci

# calculate the statistics for the given mutational process
# sum stats across modifier loci
def get_chosen_traj(data,chosen_loci,phi):

    HP = {'ancestral':0,
          'CEU':0,
          'YRI':0}

    SM = {'CEU':np.zeros(7),
          'YRI':np.zeros(7)}

    # sum across all trajectories chosen to make up a process
    for replicate in chosen_loci.keys():
        for locus in chosen_loci[replicate]:
            HP_summaries, SM_summaries = data[replicate][locus]
            for k,v in HP_summaries.items():
                HP[k] += v*2*phi

            for k,v in SM_summaries.items():
                SM[k] += v*2*phi

    # returns two dicts that correspond to the contribution of mutator alleles to this process's statistics
    return (HP,SM)

# convert from a set of mutational processes into mutation types
def turn_trajectories_into_96_mutations(data,
                                        u0_96,
                                        u_mean_96,
                                        process_to_type_mapping,
                                        arg_sums,
                                        identity_matrix = np.ones([1,14])):

    # create empty matrices to store results
    HP_all = np.zeros([96,3])
    SM_all = np.zeros([96,14])

    # we need to convert from mutational process to the 96 mutation types
    for process,(HP,SM) in data.items():

        # need to scale down ARG
        scaling = 1/32

        # get the mapping of this process to mutation types
        # most of the time, a mutational process contributes to only one mutation type
        # but we also consider a model where a mutation process impacts multiple mutation types
        # each mapping should be a vector of length 96 that sums to 1
        mapping = process_to_type_mapping[process]

        # map the contribution of process to the HP stats for mutation types
        HP_all[:, 0] += HP['ancestral']*mapping*scaling
        HP_all[:, 1] += HP['CEU']*mapping*scaling
        HP_all[:, 2] += HP['YRI']*mapping*scaling

        # Need to resize things for matrix math to work
        mapping_resized = np.resize(mapping,[96, 1])
        CEU_data_resized = np.resize(SM['CEU'],[1, 7])
        YRI_data_resized = np.resize(SM['YRI'], [1, 7])

        # map the contribution of a process to the SM stats for mutation types
        SM_all[:,:7] += np.matmul(mapping_resized,CEU_data_resized)
        SM_all[:, 7:] += np.matmul(mapping_resized,YRI_data_resized)

    # add the contribution from u0 which should be a matrix of size [96,1]
    # arg_sums should be a matrix of size [1,3]
    # identity_matrix should be a matrix of size [1,14]
    HP_all += np.matmul(u0_96,arg_sums*scaling)
    SM_all += np.matmul(u0_96,identity_matrix)

    # some final adjustments
    # realize the number of segregating variants (assuming poisson distributed around the mean we've calculated)
    HP_all = np.random.poisson(HP_all)
    # make SM relative to the mean rate for each mutation type
    # u_mean_96 should be a matrix of size [96,14] but all values in a given row are identical
    SM_all = SM_all/u_mean_96

    return HP_all,SM_all

# assign trajectories to be peak-like, under-elevated, or multi-elevated for each definition of peak-like trajectories.
# also determine where the peaks occur
def determine_SM_traj_outcomes(SM_all,threshold_dict = {'restricted': (1.1, 1.5),'relaxed':(1.5,1.5)}):

    # create storage
    SM_outcomes = {}

    # iterate over the two modes
    for mode,(elevated_threshold,peak_threshold) in threshold_dict.items():

        # create storage
        where_peaks = np.zeros(96) + np.nan

        assert elevated_threshold <= peak_threshold

        # count the number of elevated and peaked intervals for each trajectory
        elevated = np.sum(SM_all > elevated_threshold,axis=1)
        peak = np.sum(SM_all > peak_threshold, axis=1)

        # assign under_elevated if one or fewer intervals are elevated and none are peaked
        under_elevated = (elevated <= 1)*(peak == 0)

        # assign peaked if one interval is peaked and no other interval is elevated
        peak_like = (elevated == 1)*(peak == 1)

        # count the number of each one
        under = sum(under_elevated)
        peak = sum(peak_like)
        over = 96 - under - peak
        traj_outcomes = [under,peak,over]

        # for trajectories that are peak-like determine where peaks are
        peaks_only = SM_all[peak_like,:] > peak_threshold
        # x is rows, y is intervals that are peaked
        x, peaked_intervals = np.where(peaks_only)

        # store results
        SM_outcomes[mode] = (traj_outcomes,peaked_intervals)

    return SM_outcomes


def calculate_final_statistics(HP_all,SM_outcomes):

    outcomes = {}

    # calculate maximum enrichment value
    r_CEU = (HP_all[:,0]+HP_all[:,1])/sum(HP_all[:,0]+HP_all[:,1])
    r_YRI = (HP_all[:,0]+HP_all[:,2])/sum(HP_all[:,0]+HP_all[:,2])
    r     = r_CEU/r_YRI
    r_max = max(max(r),max(1/r))

    outcomes['HP'] = r_max

    # determine the set outcome for the Speidel-Myers test for each definition of peak-like
    for mode,(traj_outcomes,peaked_intervals) in SM_outcomes.items():

        # if any traj is multi-elevated, then the set is multi-elevated
        if traj_outcomes[-1] > 0:
            set_outcome = -1
        # otherwise determine the number of peak like traj (could be 0)
        else:
            set_outcome = traj_outcomes[1]

        # store all of this
        outcomes[mode] = (set_outcome,traj_outcomes,peaked_intervals)

    return outcomes

# determine if this M is too large to be used,
# calculate u0 and u_mean
# also calculate the mapping between mutation processes and mutation types
def calc_M(args,mode,statdist,M):

    mean_contribution = sum([p*q for p,q in statdist.items()])*2*args.phi

    u_mean = 1.25E-8/3

    # for default and somatic, it's straightforward
    if mode == 'default' or 'somatic':
        # u mean is the default value
        u_mean_matrix = np.zeros(shape=[96,14]) + u_mean
        # u0 is the difference between umean and the contribution from modifier sites
        u_0_temp = u_mean - M*mean_contribution
        u_0_matrix    = np.zeros(shape=[96,1]) + u_0_temp

        # one to one mapping between processes and mutation types
        n_process = 96
        process_to_type_mapping = {}
        for process in range(n_process):
            mapping = np.zeros(96)
            mapping[process] = 1
            process_to_type_mapping[process] = mapping

    # here, we allow mutation types to have different mean rates
    elif mode == 'different':

        # specifically, 4 have a 2-fold lower rate and 4 have a five-fold higher rate than the remaining 88
        temp = [0.5]*4+[5]*4+[1]*(96-8)
        temp = np.array(temp)
        u_mean_vector = temp*u_mean/np.mean(temp)
        assert np.isclose(np.mean(u_mean_vector),u_mean,rtol=0.01)

        u_mean_matrix = np.matmul(np.resize(u_mean_vector,[96,1]),np.ones(shape=[1,14]))
        u_0_matrix = np.resize(u_mean_vector,[96,1]) - M * mean_contribution
        n_process = 96

        process_to_type_mapping = {}
        for process in range(n_process):
            mapping = np.zeros(96)
            mapping[process] = 1
            process_to_type_mapping[process] = mapping

    # here there is not a one to one mapping
    # instead one process affects 6 mutation rates unevenly
    elif mode == 'fewer':
        n_process = 16
        n_motifs = int(96/n_process)

        u_mean_matrix = np.zeros(shape=[96, 14]) + u_mean
        u_0_matrix = np.zeros(shape=[96, 1]) + (u_mean - M * mean_contribution * 1/n_motifs)

        process_to_type_mapping = {}
        for process in range(n_process):
            effect_dist = np.random.random(n_motifs)
            effect_dist = effect_dist/sum(effect_dist)

            mapping = np.zeros(96)
            mapping[process*n_motifs:(process+1)*n_motifs] = effect_dist
            process_to_type_mapping[process] = mapping

    # if any u0 is less than 0, then this M exceeds the maximum number of modifier sites
    if np.any(u_0_matrix < 0):
        print('Too many modifier sites')
        quit()

    return u_0_matrix,u_mean_matrix,process_to_type_mapping,n_process,mean_contribution

def main():

    # get arguments
    args = get_args()
    args.loci = args.loci/32
    args.phi = args.phi*32
    print(args.phi)
    somatic = False

    mode = args.mode
    if mode == 'fewer':
        mod_count = 10
        # because each modifier site is spread over 6 mutation types, we allow for 6 times as many modifier sites
        # thus, the effective number of modifier sites per mutation type remains constant
        M = args.M_sample*6
    else:
        mod_count = 1000
        M = args.M_sample

    # calculate statationary distribution
    statdist = sd.get_SD(args,somatic=2*args.h*args.s*args.loci*args.phi*int(somatic))
    
    # load the mutator traj data
    data = load_data(in_path=os.path.dirname(args.outpath))

    # load ARG data
    arg_sums = get_ARG_summaries()
    file_name = f"sample_statistics_{M}_{mode}.pickle.gz"
    outpath = os.path.join(os.path.dirname(args.outpath),file_name)

    # generate 1e4 mutation rate sets
    for i in range(int(1e4)):

        if i % mod_count == 0:
            print(i)
            sys.stdout.flush()

        # create storage
        all_process_data = {}

        # determine the u0 & umean for each mutation type, number of processes, and mappings from process to mutation types
        u0_96, u_mean_96, process_to_type_mapping, n_process,mean_contribution = calc_M(args=args,
                                                                      mode=mode,
                                                                      statdist=statdist,
                                                                      M = M)

        # get data for each process
        for p in range(n_process):
            chosen_loci = choose_loci(data = data,M = M, M_simulated=args.M)
            chosen_data = get_chosen_traj(data = data,chosen_loci=chosen_loci,phi=args.phi)
            all_process_data[p] = chosen_data

        # convert from processes to motifs
        HP_all, SM_all = turn_trajectories_into_96_mutations(data=all_process_data,
                                                             u0_96 = u0_96,
                                                             u_mean_96 = u_mean_96,
                                                             process_to_type_mapping = process_to_type_mapping,
                                                             arg_sums = arg_sums)

        # calculate outcomes for motif trajectories
        SM_outcomes = determine_SM_traj_outcomes(SM_all)

        # calculate outcomes for the set of motifs
        final_outcomes = calculate_final_statistics(HP_all = HP_all, SM_outcomes = SM_outcomes)
        
        # save the results
        with gzip.open(outpath,'ab+') as fout:
            pickle.dump(final_outcomes,fout)


if __name__ == '__main__':
    main()

