import numpy as np
from scipy.integrate import quad

import mutator_classes
from scipy import stats

# calculates the  stationary distribution for mutator alleles
# max_gen needs to be much greater than 1/s for strongly selected mutators
def get_SD(p,max_gen=int(1e5),phi=None,somatic=0):

    if not phi:
        phi = p.phi

    # calculate s*
    cum_survival = expected_extra_mutations_somatic(p = p, max_gen = max_gen, somatic=somatic, phi = phi)
    new_s = 1/sum(cum_survival)

    # keep track of the probability of transitioning to 0 or 1
    zero_transition_prob = 0
    one_transition_prob  = 0

    # will become a discrete stationary distribution to sample from
    sd = {}

    # iterate over possible discrete allele values between 1 and 2N-1
    for i in np.linspace(1, 2 * p.N - 1, 2 * p.N - 1):

        # calculate the probability of moving to lost or fixed state and the (not normalized) density at state i
        transition_prob_0, transition_prob_1, density_at_x = get_probability_of_transition(p = p,
                                                                                             x = i / (2 * p.N),
                                                                                             new_s = new_s)

        sd[i / (2 * p.N)] = density_at_x
        zero_transition_prob += transition_prob_0
        one_transition_prob  += transition_prob_1

    # density at lost (or fixed) state is sum of probabilities moving there divided by the probability of moving away (e.g. prob of not having 0 mutations)
    sd[0]  = zero_transition_prob / (1 - stats.binom.pmf(0, 2 * p.N, p.mutator_mutation_rate))
    sd[1]  = one_transition_prob / (1 - stats.binom.pmf(0, 2 * p.N, p.mutator_mutation_rate))

    # density not at boundaries
    integrand = lambda q: freq_dep(x=q, p=p, new_s=new_s)
    density_not_at_boundaries = quad(integrand, 1 / (4 * p.N), 1 - 1 / (4 * p.N))[0]

    # calculate normalizing constant
    nc = sd[0] + sd[1] + density_not_at_boundaries
    assert np.isclose(nc,sum(sd.values()))

    # creates a normalized, discrete stationary distribution to sample from
    for i in sd.keys():
        sd[i] = sd[i] / nc

    return sd

# calculate probability of moving to either lost or fixed state weighted by density at x
def get_probability_of_transition(p, x, new_s):

    # calculates that non-normalized stationary density around x (specifically from x-1/(4N) to x+1/(4N))
    integrand = lambda q: freq_dep(x = q, p = p, new_s=new_s)
    density_at_x = quad(integrand, x - 1 / (4 * p.N), x + 1 / (4 * p.N))[0]

    # relative fitness of mutator allele
    selection_prob = 1 - new_s

    # probability of moving to the zero state AND having no mutations to the mutator allele
    zero_transition_prob = stats.binom.pmf(0, int(2 * p.N), selection_prob * x) * \
                           stats.binom.pmf(0, int((1 - x) * 2 * p.N), p.mutator_mutation_rate)

    # probability of moving to the fixed state AND having no mutations away from the mutator allele
    one_transition_prob = stats.binom.pmf(int(2 * p.N), int(2 * p.N), selection_prob * x) * \
                          stats.binom.pmf(0, int((x) * 2 * p.N), p.mutator_mutation_rate)


    return zero_transition_prob * density_at_x, one_transition_prob * density_at_x, density_at_x


# calculates that non-normalized stationary density at x
def freq_dep(x, p, new_s):
    s = new_s
    u = p.mutator_mutation_rate
    f = np.exp(-4 * p.N * s * x * (1 - x/2)) * (x * (1 - x)) ** (4 * p.N * u - 1)

    return f

# calculate K(t)
def expected_extra_mutations_somatic(p, max_gen: int, phi, somatic=0):
    mutation_curve = [0]
    cum_survival = [1]

    for gen in np.arange(1, max_gen):
        mutation_curve.append(mutation_curve[gen - 1] * (1/2) + p.loci * phi)
        exp_x = np.exp(-(mutation_curve[gen])*p.h*p.s)*(1-somatic)
        cum_survival.append(cum_survival[-1] * exp_x)

    return cum_survival




