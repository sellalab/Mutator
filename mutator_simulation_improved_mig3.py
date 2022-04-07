import numpy as np
import pickle
import argparse
from copy import deepcopy
import mutator_classes
import os
import sys
from scipy.stats import loguniform

# check that parameters make sense
def check_basic_assumptions(args):

    assert args.N*args.h*args.s*2 > 10
    assert args.phi > 0
    assert args.mutation_rate > 0
    assert args.mutator_mutation_rate > 0

    print(f'N: parameter value {args.N} | expected value 1000 or 14000')
    print(f'2Nhs: parameter value {args.N*args.h*args.s*2} | expected value {20}')
    print(f'Lhs: parameter value {args.loci*args.h*args.s} | expected value {3e9*0.08*20/(2*2e4)}')
    print(f'2Nu_0: parameter value {args.N*args.mutation_rate*2} | expected value {2e4*1.25E-8*2}')
    print(f'2N\mu: parameter value {args.N*args.mutator_mutation_rate*2} | expected value {2e4*1.25E-8*2}')

    f = 'Simulating a '
    if args.constant:
        f += 'constant size population '
    else:
        f += 'variable size population '
    if not args.invariable_mutator_mutation_rate:
        f += 'where mutators increase their own mutation rate '
    else:
        f += 'where mutators experience a constant mutation rate '
    if args.variable_mutator_effect:
        f += 'and mutators have a distribution of effect sizes'
    else:
        f += 'and mutators have a constant effect size'
    print(f)
    print(args.which_population)
    sys.stdout.flush()

# parse parameter values from arguments. Bools are passed as ints.
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
    print(args)
    return args


def main():

    # get parameters
    args = get_args()
    check_basic_assumptions(args)

    # create a directory to store results
    # optimal usage is a parent directory for all runs with same parameters and a subdirectory for each specific run.
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath,exist_ok=True)
    
    # create the ancestral population
    ancestral_population = mutator_classes.Population(N=args.N,
                                                    h=args.h,
                                                    s=args.s,
                                                    loci=args.loci,
                                                    mutation_rate=args.mutation_rate,
                                                    phi=args.phi,
                                                    mutator_mutation_rate=args.mutator_mutation_rate,
                                                    M=args.M,
                                                    invariable_mutator_mutation_rate=args.invariable_mutator_mutation_rate,
                                                    outpath= os.path.join(args.outpath,'ancestral'),
                                                    initialize=True,
                                                    variable_mutator_effect = args.variable_mutator_effect,
                                                    somatic = True)

    print('initial population created')
    # If a split time is provided then
    if args.split_gen:
        print('Using a population split')
        args.total_gen = int(7e4)
        # If pop size is not constant
        if not args.constant:
            print('with variable population sizes')
            # Find where population sizes are stored
            with open(args.NE_path, 'rb') as fin:
                NE_times = pickle.load(fin)

            # and create the file to determine what the pop size is each generation
            CEU_N_function = lambda t: get_variable_N(NE_times=NE_times, index=0, t=t)
            YRI_N_function = lambda t: get_variable_N(NE_times=NE_times, index=1, t=t)

        # otherwise it's just constant
        else:
            print('with costant population sizes')
            CEU_N_function = lambda t: args.N
            YRI_N_function = lambda t: args.N


        # if final population for this population does not exist, then finish simulations for it
        # For the ancestral population, we follow the CEU population sizes
        if not os.path.exists(os.path.join(ancestral_population.outpath,'final_population.pickle')):
            if args.which_population == 'ancestral':
                # we simulate the ancestral population for TOTAL_GEN - SPLIT_GEN generations
                ancestral_population = simulate(parent_population=ancestral_population,
                                              start_gen=args.total_gen,
                                              stop_gen = args.split_gen,
                                              N_function=CEU_N_function,
                                              backup_gen=args.backup_gen)
                print('Finished ancestral population')
                quit()
            else:
                print("Can't run job as ancestral population is not finished")
                quit()
        else:
            if args.which_population == 'ancestral':
                print('Ancestral population finished already')
                quit()
            else:
                with open(os.path.join(ancestral_population.outpath, 'final_population.pickle'),'rb') as fin:
                    ancestral_population = pickle.load(fin)

        if args.which_population == 'YRI':
            # make a copy of the ancestral population and run the YRI branch
            YRI_population = deepcopy(ancestral_population)

            try:
                YRI_population.update_outpath(os.path.join(args.outpath, 'YRI'))
                YRI_population = simulate(parent_population=YRI_population,
                                          start_gen=args.split_gen,
                                          stop_gen = args.ignore_gen,
                                          N_function=YRI_N_function,
                                          backup_gen=args.backup_gen,
                                          post_split = True)
            except MemoryError:
                print('Got memoryerror for YRI population')

            print('Finished YRI population')
            quit()

        elif args.which_population == 'CEU':
            # make a copy of the ancestral population and run the CEU branch
            CEU_population = deepcopy(ancestral_population)

            try:
                CEU_population.update_outpath(os.path.join(args.outpath,'CEU'))
                CEU_population = simulate(parent_population=CEU_population,
                                          start_gen=args.split_gen,
                                          stop_gen = args.ignore_gen,
                                          N_function=CEU_N_function,
                                          backup_gen=args.backup_gen,
                                          post_split=True)

            except MemoryError:
                print('Got memory error for CEU population')
            print('Finished CEU population')
            quit()

    # otherwise we just run a simple simulation
    else:
        print('Simple simulation') 
        if not os.path.exists(os.path.join(args.outpath,'final_population.pickle')):

            N_function = lambda t: args.N
            ancestral_population = simulate(parent_population = ancestral_population,
                                            start_gen = args.total_gen,
                                            stop_gen = 0,
                                            backup_gen = args.backup_gen,
                                            N_function = N_function)

    print('done')

# determine what the current population size is when using a variable population size
def get_variable_N(NE_times, index, t, burn_time = 5e4):
    if t > 6e4:
        N = 14000
    else:
        NE_keys = np.array(list(NE_times.keys()))
        current_step = max(NE_keys[NE_keys < t])
        N = NE_times[current_step][index]
    return int(N)

# check if there is a current version of this simulation already started and restart from there.
# otherwise, just start from begining
def check_for_backups(parent_population, gen):

    if os.path.exists(os.path.join(parent_population.outpath,'backup.pickle')):
        with open(os.path.join(parent_population.outpath,'backup.pickle'), 'rb') as fin:
            backup_population, backup_gen = pickle.load(fin)
        return backup_population, backup_gen
    else:
        return parent_population, gen

#
def simulate(parent_population, start_gen, stop_gen, N_function, backup_gen,post_split=False):

    # assert that required directories exist
    if not os.path.exists(parent_population.outpath):
        os.makedirs(os.path.join(parent_population.outpath,''),exist_ok=True)

    # Check for backups
    parent_population, gen = check_for_backups(parent_population=parent_population,
                                               gen=start_gen)

    print('starting at ',gen)
    sys.stdout.flush()

    while gen > stop_gen:
        if gen <= 100: 
            backup_gen = 1
        if gen <= 50:
            break
        #Determine next generation's population size
        N = N_function(gen)

        # make next generation population
        parent_population.next_gen(N=N)
        # determine relevant statistics (mutator counts and which ones are fixed)
        mutator_counts, new_fixed = parent_population.get_mutator_counts()
        # update fixed loci
        parent_population.update_fixed_loci(new_fixed=new_fixed)
        # store mutator counts
        parent_population.store(a=mutator_counts)
        # update generation
        gen += -1

        # backup as needed
        if gen % backup_gen == 0:
            parent_population.write_out()

            print(f'Generation: {gen}, mean mutator freq: {np.mean(mutator_counts)}')
            sys.stdout.flush()

            with open(os.path.join(parent_population.outpath,'backup.pickle'), 'wb+') as fout:
                pickle.dump((parent_population, gen), fout)

    # after finishing simulations, save final population
    with open(os.path.join(parent_population.outpath,'final_population.pickle'), 'wb+') as fout:
        pickle.dump(parent_population,fout)

    return parent_population

# create joblist
def create_joblist():

    simpleParams = False
    humanParams = True
    variableMutatorEffect = False
    variableSelectedEffect = False

    default_params = {'N': 14000,  # population size
                      'M': 1000,  # number of modifier loci, M
                      'h': 0.5,  # h
                      's': 0.001,  # s - together hs are the average fitness effects of mutations at selected loci
                      'phi': 0,  # effect size of mutator alleles
                      'mutator_mutation_rate': 1.25E-8,  # Mutation rate at modifier sites
                      'mutation_rate': 1.25E-8,  # baseline mutation rate at selected sites, u0
                      'loci': 3E9 * 0.08 / 32,  # number of selected loci
                      'constant': int(False),  # is the population size constant
                      'split_gen': 10000,
                      # the generation at which the ancestral population is split into europeans and africans
                      'backup_gen': 100,  # backup the population every 100 generations
                      'ignore_gen': 0,  # stop simulations this many generations from the present
                      'total_gen': 100000,  # how many total generations to simulate
                      'outpath': '/home/ec2-user/Mutator/somatic/',  # where do we store results
                      'NE_path': '/home/ec2-user/Mutator/' + 'MSMC_NE_dict.pickle',
                      # where do we get population size estimates
                      'invariable_mutator_mutation_rate': int(True),
                      'variable_selective_effect': int(False),
                      'variable_mutator_effect': int(False),
                      'which_population': 'ancestral'}

    small_N = 1000
    scaling_factor = 20000/small_N
    if simpleParams or variableMutatorEffect or variableSelectedEffect:
        joblist_name = 'constant_params'
        default_params['N'] = small_N
        for key in ['s','mutator_mutation_rate','mutation_rate']:
            default_params[key] = default_params[key]*scaling_factor
        default_params['constant'] = int(True)
        default_params['split_gen'] = 0

    if simpleParams:
        default_params['outpath']='/home/ec2-user/Mutator/constantN/'
    elif variableMutatorEffect:
        joblist_name = 'variableMutatorEffect_params'
        default_params['variable_mutator_effect'] = int(True)
        default_params['outpath'] = '/home/ec2-user/Mutator/variableMutatorEffect/'
    elif variableSelectedEffect:
        joblist_name = 'variableSelectedEffect_params'
        default_params['variable_selective_effect'] = int(True)
        default_params['s'] = loguniform(20/(2*2000),1).mean()
        default_params['outpath'] = '/home/ec2-user/Mutator/variableSelectedEffect/'
    elif humanParams:
        joblist_name = 'human_params'

    jobs = []

    job_outpath = default_params['outpath']

    calc_phi = lambda p, S: S/(4 * p['N'] * p['loci'] * p['h'] * p['s'])

    for S in np.logspace(-2, 3, 21):
        for v in range(10):

            p = deepcopy(default_params)
            p['phi'] = calc_phi(p,S)
            p['outpath'] = os.path.join(os.path.join(job_outpath,'varyPHI_S{}'.format(round(np.log10(S), 3))),'V{}'.format(v))
            print(p['outpath'])

            if humanParams:
                for population in ['ancestral','CEU','YRI']:
                    p['which_population'] = population
                    jobstring = ''
                    for name,value in p.items():
                        jobstring += f'--{name}={value} '
                    jobstring += '\n'
                    jobs.append(jobstring)
            else:
                jobstring = ''
                for name, value in p.items():
                    jobstring += f'--{name}={value} '
                jobstring += '\n'
                jobs.append(jobstring)

    with open(os.path.join('.', f'{joblist_name}_joblist.txt'), 'w+') as fin:
        fin.writelines(jobs)


if __name__ == '__main__':
    main()
    #create_joblist()
