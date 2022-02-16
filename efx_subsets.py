import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import random
import time
import sys

from copy import deepcopy
from itertools import permutations, chain, combinations, product
from collections import Counter

from utils import *


def strong_envy_in_path(alloc, val_fns, edge_list):
    envies = []
    for e in edge_list:
        if alloc[e[1]]:
            val_self = np.sum(val_fns[e[0], alloc[e[0]]])
            val_other = np.sum(val_fns[e[0], alloc[e[1]]]) - np.min(val_fns[e[0], alloc[e[1]]])
            envies.append(max([val_other - val_self, 0]))
    return envies


def strong_envy_in_triplet(alloc, val_fns):
    total_envy = 0
    for pair in [(0, 1), (1, 2), (2, 1), (1, 0)]:
        # Add pair[0]'s envy for pair[1]'s bundle
        if alloc[pair[1]]:
            val_self = np.sum(val_fns[pair[0], alloc[pair[0]]])
            val_other = np.sum(val_fns[pair[0], alloc[pair[1]]]) - np.min(val_fns[pair[0], alloc[pair[1]]])
            total_envy += max([val_other - val_self, 0])
            # total_envy += val_other - val_self
    return total_envy


def envy_in_triplet(alloc, val_fns):
    # print("compute total envy")
    # print(alloc)
    total_envy = 0
    for pair in [(0, 1), (1, 2), (2, 1), (1, 0)]:
        # Add pair[0]'s envy for pair[1]'s bundle
        val_self = np.sum(val_fns[pair[0], alloc[pair[0]]])
        val_other = np.sum(val_fns[pair[0], alloc[pair[1]]])
        # print(val_self)
        # print(val_other)
        total_envy += max([val_other - val_self, 0])
        # total_envy += val_other - val_self

        # print(total_envy)
    return total_envy


def efx_among_triplet(val_fns):
    m = len(val_fns[0])

    alloc = (tuple(range(m)), (), ())

    already_visited = {alloc}

    # Fix L then fix R. Don't return to a previously seen state.
    strong_envy = np.inf
    envy = np.inf
    counter = -1 * np.inf
    while True:
        if counter > 0:
            print("counter: ", counter)
        # print(alloc)
        L_efx = is_efx(alloc[0], alloc[1], 0, 1, val_fns)
        R_efx = is_efx(alloc[1], alloc[2], 1, 2, val_fns)
        # print(L_efx)
        # print(R_efx)

        # print(alloc)

        if L_efx and R_efx:
            return alloc
        if not L_efx:
            L_goods = set(alloc[0]) | set(alloc[1])
            # print(L_goods)
            for a0 in powerset(L_goods):
                alloc_prime = (tuple(sorted(list(a0))), tuple(sorted(list(L_goods - set(a0)))), alloc[2])
                # print(alloc_prime)
                # print()

                if alloc_prime in already_visited and is_efx(alloc_prime[0], alloc_prime[1], 0, 1, val_fns):
                    print("in L")
                    print(val_fns)
                    print(alloc)
                    print(alloc_prime)

                if alloc_prime not in already_visited and is_efx(alloc_prime[0], alloc_prime[1], 0, 1, val_fns):
                    alloc = alloc_prime
                    already_visited.add(alloc)
                    break
            if not is_efx(alloc[0], alloc[1], 0, 1, val_fns):
                return None

        # print(alloc)
        R_efx = is_efx(alloc[1], alloc[2], 1, 2, val_fns)
        if not R_efx:
            R_goods = set(alloc[1]) | set(alloc[2])
            for a1 in powerset(R_goods):
                alloc_prime = (alloc[0], tuple(sorted(list(a1))), tuple(sorted(list(R_goods - set(a1)))))

                if alloc_prime in already_visited and is_efx(alloc_prime[1], alloc_prime[2], 1, 2, val_fns):
                    print("in R")
                    print(val_fns)
                    print(alloc)
                    print(alloc_prime)
                if alloc_prime not in already_visited and is_efx(alloc_prime[1], alloc_prime[2], 1, 2, val_fns):
                    alloc = alloc_prime
                    already_visited.add(alloc)
                    break
            if not is_efx(alloc[1], alloc[2], 1, 2, val_fns):
                return None
        curr_envy = envy_in_triplet(alloc, val_fns)
        curr_strong_envy = strong_envy_in_triplet(alloc, val_fns)
        # if curr_envy > envy:
        #     print("TOTAL ENVY INCREASED")
        #     print(val_fns)
        #     print(alloc)
        #     print(curr_envy)
        #     print(envy)
        #     print(curr_strong_envy > strong_envy)
        #     sys.exit(1)
        if curr_strong_envy > strong_envy:
            print("TOTAL STRONG ENVY INCREASED")
            print(val_fns)
            print(alloc)
            print(curr_envy)
            print(envy)
            print(curr_strong_envy > strong_envy)
            counter = 0
            # sys.exit(1)
        envy = curr_envy
        strong_envy = curr_strong_envy
        counter += 1


"""Returns True if v1 is leximin greater than or equal to v2"""


def leximin_comparator(v1, v2):
    _v1 = sorted(v1)
    _v2 = sorted(v2)

    if _v1[0] > _v2[0]:
        return True
    elif _v1[0] < _v2[0]:
        return False
    else:
        return _v1[1] >= _v2[1]


"""
Given this allocation and these valuations, rebalance the goods that e[0] and e[1] are currently assigned so 
that they are LEXIMIN with each other.
"""


def fix_edge_leximin(alloc, val_fns, e):
    good_set = set(alloc[e[0]]) | set(alloc[e[1]])
    norm0 = np.sum(val_fns[e[0], list(good_set)])
    norm1 = np.sum(val_fns[e[1], list(good_set)])

    best_vec = [-1, -1]
    best_alloc = None

    if not good_set:
        return alloc

    for a0 in powerset(good_set):
        g0 = list(a0)
        g1 = list(good_set - set(a0))

        v0 = np.sum(val_fns[e[0], g0]) / norm0
        v1 = np.sum(val_fns[e[1], g1]) / norm1

        if leximin_comparator([v0, v1], best_vec):
            best_vec = [v0, v1]
            best_alloc = [g0, g1]
    # Check if we actually haven't improved at all, if so, return what we originally had
    orig_v0 = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
    orig_v1 = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
    if leximin_comparator([orig_v0, orig_v1], best_vec) and leximin_comparator(best_vec, [orig_v0, orig_v1]):
        return alloc
    else:
        alloc[e[0]] = sorted(best_alloc[0])
        alloc[e[1]] = sorted(best_alloc[1])
        return alloc


"""
Given this allocation and these valuations, rebalance the goods that e[0] and e[1] are currently assigned so 
that they are EFX with each other.
"""


def fix_edge(alloc, val_fns, e):
    # Use the algorithm for 2 agents from the Plaut and Roughgarden paper.
    """Assume valns are identical. Do the following to distribute so EFX according to identical valns:
        Sort in decreasing order, always give to envious agent.
        Then let the agent who didn't use their valuations pick their favorite bundle."""
    good_set = sorted(list(set(alloc[e[0]]) | set(alloc[e[1]])))
    good_values = val_fns[e[0], good_set]
    ordered_goods = [good_set[i] for i in np.argsort(good_values)[::-1]]

    # Now divide the goods up by giving the next good to the most envious agent
    g0 = []
    g1 = []

    for g in ordered_goods:
        val_g0 = np.sum(val_fns[e[0], g0])
        val_g1 = np.sum(val_fns[e[0], g1])

        if val_g1 > val_g0:
            g0.append(g)
        else:
            g1.append(g)

    val_g0_to_e1 = np.sum(val_fns[e[1], g0])
    val_g1_to_e1 = np.sum(val_fns[e[1], g1])

    if val_g0_to_e1 > val_g1_to_e1:
        alloc[e[0]] = g1
        alloc[e[1]] = g0
    else:
        alloc[e[0]] = g0
        alloc[e[1]] = g1

    return alloc
    # for a0 in powerset(good_set):
    #     g0 = list(a0)
    #     g1 = list(good_set - set(a0))
    #
    #     if is_efx(g0, g1, e[0], e[1], val_fns):
    #         alloc[e[0]] = g0
    #         alloc[e[1]] = g1
    #         return alloc
    # if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
    #     return None


def calc_min_score(alloc, val_fns):
    scores = []
    for a, bundle in enumerate(alloc):
        scores.append(np.sum(val_fns[a, bundle]))
    # print(scores)
    return np.min(scores)


def calc_min_of_maxs(alloc, val_fns, edge_list):
    all_maxs = []
    for e in edge_list:
        good_set = set(alloc[e[0]]) | set(alloc[e[1]])
        norm0 = np.sum(val_fns[e[0], list(good_set)])
        norm1 = np.sum(val_fns[e[1], list(good_set)])

        vals = np.zeros(2)
        vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
        vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1

        if norm0 == 0:
            vals[0] = 0
        if norm1 == 0:
            vals[1] = 0

        all_maxs.append(np.max(vals))
    return min(all_maxs)


def calc_total_value_of_goods(alloc, val_fns, edge_list):
    all_denoms = []
    for e in edge_list:
        good_set = set(alloc[e[0]]) | set(alloc[e[1]])
        norm0 = np.sum(val_fns[e[0], list(good_set)])
        norm1 = np.sum(val_fns[e[1], list(good_set)])

        # if norm0 == 0 or norm1 == 0:
        #     all_mins.append(0)
        # else:
        #     vals = np.zeros(2)
        #     vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
        #     vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
        #
        #     all_mins.append(np.min(vals))
        # vals = np.zeros(2)
        # vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
        # vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
        all_denoms.append(norm0 + norm1)
    return sum(all_denoms)


# for each triplet, compute the min on both edges. Then compute the max of those 2. Then take the min over all triplets.
def calc_min_of_max_of_mins(alloc, val_fns, edge_list):
    all_triplets = []
    for e in edge_list:
        for e_prime in edge_list:
            if e != e_prime and (e[0] == e_prime[1] or e[1] == e_prime[0] or e[0] == e_prime[0] or e[1] == e_prime[1]):
                print(e, e_prime)
                #  Compute min on both edges
                good_set = set(alloc[e[0]]) | set(alloc[e[1]])
                norm0 = np.sum(val_fns[e[0], list(good_set)])
                norm1 = np.sum(val_fns[e[1], list(good_set)])
                vals = np.zeros(2)
                vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
                vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
                min1 = np.min(vals)

                good_set = set(alloc[e_prime[0]]) | set(alloc[e_prime[1]])
                norm0 = np.sum(val_fns[e_prime[0], list(good_set)])
                norm1 = np.sum(val_fns[e_prime[1], list(good_set)])
                vals = np.zeros(2)
                vals[0] = np.sum(val_fns[e_prime[0], alloc[e_prime[0]]]) / norm0
                vals[1] = np.sum(val_fns[e_prime[1], alloc[e_prime[1]]]) / norm1
                min2 = np.min(vals)

                all_triplets.append(np.max([min1, min2]))

    return min(all_triplets)


def calc_min_of_mins(alloc, val_fns, edge_list):
    all_mins = []
    for e in edge_list:
        # print(alloc)
        # print(e)
        good_set = set(alloc[e[0]]) | set(alloc[e[1]])
        norm0 = np.sum(val_fns[e[0], list(good_set)])
        norm1 = np.sum(val_fns[e[1], list(good_set)])

        # if norm0 == 0 or norm1 == 0:
        #     all_mins.append(0)
        # else:
        #     vals = np.zeros(2)
        #     vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
        #     vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
        #
        #     all_mins.append(np.min(vals))
        vals = np.zeros(2)
        vals[0] = np.sum(val_fns[e[0], alloc[e[0]]]) / norm0
        vals[1] = np.sum(val_fns[e[1], alloc[e[1]]]) / norm1
        all_mins.append(np.min(vals))
    return min(all_mins)


def compute_update(alloc, val_fns, e, current_invariant, done, edge_list):
    alloc_before = deepcopy(alloc)
    alloc = fix_edge_leximin(alloc, val_fns, e)
    if alloc != alloc_before:
        done = False

    # invar = calc_min_score(alloc, val_fns)
    invar = calc_min_of_mins(alloc, val_fns, edge_list)
    # invar = calc_total_value_of_goods(alloc, val_fns, edge_list)
    # invar = calc_min_of_max_of_mins(alloc, val_fns, edge_list)
    # if invar >= current_invariant or np.isclose(invar, current_invariant):
    current_invariant = invar
    # else:
    #     print("invariant decreased from %f to %f" % (current_invariant, invar))
    #     print(val_fns)
    #     print(alloc)
    #     time.sleep(5)
    return alloc, current_invariant, done


"""
val_fns: nxm numpy array, the ij element is the value of good j for agent i
edge_list: list of tuples (i, i'), which indicate the edges along which we must be leximin, and they also
            indicate the order for rebalancing goods under the iterative algorithm
"""


def pairwise_leximin_on_graph(val_fns, edge_list):
    n, m = val_fns.shape

    alloc = [list(range(m))] + [list() for _ in range(n - 1)]

    # Fix edges in edge_list in forward then reverse order.
    done = False
    current_invariant = -1
    prev_invariant = -1
    while not done:
        done = True
        for e in edge_list:
            alloc, current_invariant, done = compute_update(alloc, val_fns, e, current_invariant, done, edge_list)
            print("\t" + str(current_invariant))
            print(alloc)

        # Go backwards now. We just finished the last edge, so skip it.
        for e in edge_list[::-1][1:]:
            alloc, current_invariant, done = compute_update(alloc, val_fns, e, current_invariant, done, edge_list)
            print("\t" + str(current_invariant))
            print(alloc)

        if current_invariant < prev_invariant and not np.isclose(current_invariant, prev_invariant):
            print("current_invariant: ", current_invariant)
            print("prev_invariant: ", prev_invariant)
            sys.exit(0)
        else:
            prev_invariant = current_invariant

        # if check_pairwise_efx(alloc, val_fns, edge_list):
        #     done = True

    return alloc


def calculate_total_envy(alloc, edge_list, val_fns):
    total_envy = 0
    for e in edge_list:
        val_0_for_0 = np.sum(val_fns[e[0], alloc[e[0]]])
        val_0_for_1 = np.sum(val_fns[e[0], alloc[e[1]]])
        val_1_for_0 = np.sum(val_fns[e[1], alloc[e[0]]])
        val_1_for_1 = np.sum(val_fns[e[1], alloc[e[1]]])

        total_envy += max(0, val_0_for_1 - val_0_for_0)
        total_envy += max(0, val_1_for_0 - val_1_for_1)
    return total_envy


def calculate_total_strong_envy(alloc, edge_list, val_fns):
    total_envy = 0
    for e in edge_list:
        if len(alloc[e[0]]):
            val_0_for_0 = np.sum(val_fns[e[0], alloc[e[0]]])
            val_1_for_0 = np.sum(val_fns[e[1], alloc[e[0]]]) - np.min(val_fns[e[1], alloc[e[0]]])
        else:
            val_0_for_0 = 0
            val_1_for_0 = 0

        if len(alloc[e[1]]):
            val_0_for_1 = np.sum(val_fns[e[0], alloc[e[1]]]) - np.min(val_fns[e[0], alloc[e[1]]])
            val_1_for_1 = np.sum(val_fns[e[1], alloc[e[1]]])
        else:
            val_0_for_1 = 0
            val_1_for_1 = 0

        total_envy += max(0, val_0_for_1 - val_0_for_0)
        total_envy += max(0, val_1_for_0 - val_1_for_1)
    return total_envy


def calculate_min_value(alloc, edge_list, val_fns):
    min_val = 1000
    for i in range(val_fns.shape[0]):
        val_i = np.sum(val_fns[i, alloc[i]])
        if val_i < min_val:
            min_val = val_i
    return min_val


def calculate_strong_envy_vec(alloc, edge_list, val_fns):
    v = []
    for e in edge_list:
        if len(alloc[e[0]]):
            val_0_for_0 = np.sum(val_fns[e[0], alloc[e[0]]])
            val_1_for_0 = np.sum(val_fns[e[1], alloc[e[0]]]) - np.min(val_fns[e[1], alloc[e[0]]])
        else:
            val_0_for_0 = 0
            val_1_for_0 = 0

        if len(alloc[e[1]]):
            val_0_for_1 = np.sum(val_fns[e[0], alloc[e[1]]]) - np.min(val_fns[e[0], alloc[e[1]]])
            val_1_for_1 = np.sum(val_fns[e[1], alloc[e[1]]])
        else:
            val_0_for_1 = 0
            val_1_for_1 = 0

        v.append(max(0, val_0_for_1 - val_0_for_0) + max(0, val_1_for_0 - val_1_for_1))
    return v


def calculate_num_goods_to_drop(alloc, edge_list, val_fns):
    goods_to_drop = set()
    # print(alloc)
    # print(val_fns)
    for e in edge_list:
        # print(e)
        goods0 = alloc[e[0]]
        goods1 = alloc[e[1]]

        val_0_for_0 = np.sum(val_fns[e[0], alloc[e[0]]])
        val_0_for_1 = np.sum(val_fns[e[0], alloc[e[1]]])
        val_1_for_0 = np.sum(val_fns[e[1], alloc[e[0]]])
        val_1_for_1 = np.sum(val_fns[e[1], alloc[e[1]]])

        if val_0_for_0 < val_0_for_1:
            good_set = sorted(alloc[e[1]])
            good_values = val_fns[e[0], good_set]
            ordered_goods = [good_set[i] for i in np.argsort(good_values)]
            while val_0_for_0 < val_0_for_1 and ordered_goods:
                good_to_remove = ordered_goods.pop(0)
                val_0_for_1 -= val_fns[e[0], good_to_remove]
                if val_0_for_0 < val_0_for_1:
                    goods_to_drop.add(good_to_remove)

        if val_1_for_1 < val_1_for_0:
            good_set = sorted(alloc[e[0]])
            good_values = val_fns[e[1], good_set]
            ordered_goods = [good_set[i] for i in np.argsort(good_values)]
            while val_1_for_1 < val_1_for_0 and ordered_goods:
                good_to_remove = ordered_goods.pop(0)
                val_1_for_0 -= val_fns[e[1], good_to_remove]
                if val_1_for_1 < val_1_for_0:
                    goods_to_drop.add(good_to_remove)

    # print(goods_to_drop)
    return len(goods_to_drop)


def determine_g_efx(edge_list, val_fns, alloc):
    for e in edge_list:
        if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
            return False
    return True


def get_next_sequence(cutters_and_choosers, visited, terminating):
    if not len(cutters_and_choosers):
        if "R" not in visited:
            return "R"
        elif "L" not in visited:
            return "L"
    # If this was a terminating sequence, then we will just back up and check for a sequence we haven't checked
    if cutters_and_choosers[-1] == "L":
        if cutters_and_choosers[:-1] + "R" not in visited:


    # If this was a non-terminating sequence, then we need to append more things and check.
    return None

"""Find a sequence of cutters and choosers that will make strong envy monotonically non-increasing, if it exists"""


def efx_dfs_on_graph(val_fns, edge_list):
    # Search over strings maybe, so we can keep a list of visited states. At each state, just recompute the
    # full algorithm, since it's pretty fast.
    visited = set()
    cutters_and_choosers = ["L", "R"]

    done = False
    while cutters_and_choosers:
        # Run the sequence of fixes described by cutters_and_choosers
        current_cut_and_choose = cutters_and_choosers.pop()
        visited.add(current_cut_and_choose)


        alloc, strong_envy_over_passes = parametrized_efx_on_graph(current_cut_and_choose, val_fns, edge_list)
        if determine_g_efx(edge_list, val_fns, alloc):
            if monotone_dec(strong_envy_over_passes):
                return alloc, strong_envy_over_passes
        else:
            # Not a terminal node. Add the next sequences.
            if current_cut_and_choose + "L" not in visited:
                cutters_and_choosers.append(current_cut_and_choose + "L")
            if current_cut_and_choose + "R" not in visited:
                cutters_and_choosers.append(current_cut_and_choose + "R")





    n, m = val_fns.shape

    alloc = [list(range(m))] + [list() for _ in range(n - 1)]

    # envy_over_passes is the total envy after each full pass (starting at t=0)
    envy_over_passes = []
    strong_envy_over_passes = []
    min_over_passes = []
    strong_envy_vecs = []
    num_goods_to_drop = []
    # envy_over_steps is the total envy after each pairwise correction (starting at t=0)
    # envy_over_steps = []
    num_passes = 0

    # Was this necessary? Still unclear...
    # already_visited = {alloc}

    # Fix edges in edge_list in forward then reverse order.
    g_efx = False
    ctr = 0
    min_score = -1
    while not g_efx:
        num_passes += 1
        total_envy = calculate_total_envy(alloc, edge_list, val_fns)
        total_strong_envy = calculate_total_strong_envy(alloc, edge_list, val_fns)
        min_val = calculate_min_value(alloc, edge_list, val_fns)
        envy_over_passes.append(total_envy)
        strong_envy_over_passes.append(total_strong_envy)
        min_over_passes.append(min_val)
        strong_envy_vecs.append(calculate_strong_envy_vec(alloc, edge_list, val_fns))
        num_goods_to_drop.append(calculate_num_goods_to_drop(alloc, edge_list, val_fns))
        # print(alloc)
        # print(strong_envy_in_path(alloc, val_fns, edge_list))

        # print(ctr)
        # ctr += 1
        # if ctr > m / 2:
        #     print("ctr exceeded m")
        #     print(ctr)
        #     print(val_fns)
        #     sys.exit(1)
        for e in edge_list:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                alloc = fix_edge(alloc, val_fns, e)
                # print(alloc)
                # alloc = fix_edge_leximin(alloc, val_fns, e)

                # new_min = calc_min_score(alloc, val_fns)
                # if new_min >= min_score:
                #     min_score = new_min
                # else:
                #     print("min decreased from %d to %d" % (min_score, new_min))
                #     print(val_fns)
                #     print(alloc)
        # print(alloc)
        # Go backwards now. We just finished the last edge, so skip it.
        for e in edge_list[::-1][1:]:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                alloc = fix_edge(alloc, val_fns, e)
        # print(alloc)
        # alloc = fix_edge_leximin(alloc, val_fns, e)

        g_efx = True
        for e in edge_list:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                g_efx = False
        # print(alloc)

        # new_min = calc_min_score(alloc, val_fns)
        # if new_min >= min_score or np.isclose(new_min, min_score):
        #     min_score = new_min
        # else:
        #     print("min decreased from %d to %d" % (min_score, new_min))
        #     print(repr(val_fns))
        #     print(alloc)

    total_envy = calculate_total_envy(alloc, edge_list, val_fns)
    total_strong_envy = calculate_total_strong_envy(alloc, edge_list, val_fns)
    min_val = calculate_min_value(alloc, edge_list, val_fns)
    envy_over_passes.append(total_envy)
    strong_envy_over_passes.append(total_strong_envy)
    min_over_passes.append(min_val)
    strong_envy_vecs.append(calculate_strong_envy_vec(alloc, edge_list, val_fns))
    num_goods_to_drop.append(calculate_num_goods_to_drop(alloc, edge_list, val_fns))

    return alloc, envy_over_passes, strong_envy_over_passes, strong_envy_vecs, min_over_passes, num_goods_to_drop, num_passes



"""
val_fns: nxm numpy array, the ij element is the value of good j for agent i
edge_list: list of tuples (i, i'), which indicate the edges along which we must be EFX, and they also
            indicate the order for rebalancing goods under the iterative algorithm
"""


def efx_on_graph(val_fns, edge_list):
    n, m = val_fns.shape

    alloc = [list(range(m))] + [list() for _ in range(n - 1)]

    # envy_over_passes is the total envy after each full pass (starting at t=0)
    envy_over_passes = []
    strong_envy_over_passes = []
    min_over_passes = []
    strong_envy_vecs = []
    num_goods_to_drop = []
    # envy_over_steps is the total envy after each pairwise correction (starting at t=0)
    # envy_over_steps = []
    num_passes = 0

    # Was this necessary? Still unclear...
    # already_visited = {alloc}

    # Fix edges in edge_list in forward then reverse order.
    g_efx = False
    ctr = 0
    min_score = -1
    while not g_efx:
        num_passes += 1
        total_envy = calculate_total_envy(alloc, edge_list, val_fns)
        total_strong_envy = calculate_total_strong_envy(alloc, edge_list, val_fns)
        min_val = calculate_min_value(alloc, edge_list, val_fns)
        envy_over_passes.append(total_envy)
        strong_envy_over_passes.append(total_strong_envy)
        min_over_passes.append(min_val)
        strong_envy_vecs.append(calculate_strong_envy_vec(alloc, edge_list, val_fns))
        num_goods_to_drop.append(calculate_num_goods_to_drop(alloc, edge_list, val_fns))
        # print(alloc)
        # print(strong_envy_in_path(alloc, val_fns, edge_list))

        # print(ctr)
        # ctr += 1
        # if ctr > m / 2:
        #     print("ctr exceeded m")
        #     print(ctr)
        #     print(val_fns)
        #     sys.exit(1)
        for e in edge_list:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                alloc = fix_edge(alloc, val_fns, e)
                # print(alloc)
                # alloc = fix_edge_leximin(alloc, val_fns, e)

                # new_min = calc_min_score(alloc, val_fns)
                # if new_min >= min_score:
                #     min_score = new_min
                # else:
                #     print("min decreased from %d to %d" % (min_score, new_min))
                #     print(val_fns)
                #     print(alloc)
        # print(alloc)
        # Go backwards now. We just finished the last edge, so skip it.
        for e in edge_list[::-1][1:]:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                alloc = fix_edge(alloc, val_fns, e)
        # print(alloc)
        # alloc = fix_edge_leximin(alloc, val_fns, e)

        g_efx = True
        for e in edge_list:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                g_efx = False
        # print(alloc)

        # new_min = calc_min_score(alloc, val_fns)
        # if new_min >= min_score or np.isclose(new_min, min_score):
        #     min_score = new_min
        # else:
        #     print("min decreased from %d to %d" % (min_score, new_min))
        #     print(repr(val_fns))
        #     print(alloc)

    total_envy = calculate_total_envy(alloc, edge_list, val_fns)
    total_strong_envy = calculate_total_strong_envy(alloc, edge_list, val_fns)
    min_val = calculate_min_value(alloc, edge_list, val_fns)
    envy_over_passes.append(total_envy)
    strong_envy_over_passes.append(total_strong_envy)
    min_over_passes.append(min_val)
    strong_envy_vecs.append(calculate_strong_envy_vec(alloc, edge_list, val_fns))
    num_goods_to_drop.append(calculate_num_goods_to_drop(alloc, edge_list, val_fns))

    return alloc, envy_over_passes, strong_envy_over_passes, strong_envy_vecs, min_over_passes, num_goods_to_drop, num_passes


def check_pairwise_efx(alloc, val_fns, edges):
    for e in edges:
        if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
            # print(e[0], e[1])
            return False
    return True


def is_efx(alloc1, alloc2, a1, a2, val_fns):
    for i in alloc1:
        if np.sum(val_fns[a2, alloc2]) < np.sum(val_fns[a2, alloc1]) - val_fns[a2, i] and \
                not np.isclose(np.sum(val_fns[a2, alloc2]), np.sum(val_fns[a2, alloc1]) - val_fns[a2, i]):
            return False
    for i in alloc2:
        if np.sum(val_fns[a1, alloc1]) < np.sum(val_fns[a1, alloc2]) - val_fns[a1, i] \
                and not np.isclose(np.sum(val_fns[a1, alloc1]), np.sum(val_fns[a1, alloc2]) - val_fns[a1, i]):
            return False
    return True


def get_valuations(alloc1, alloc2, val_fns):
    return np.sum(val_fns[0, alloc1]), np.sum(val_fns[1, alloc2])


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# Is it the case that in any EFX allocation on 2 agents, for every subset of a1's goods there is a subset of a2's goods
# so the subsets give an EFX alloc? Answer is no, counterex from this code.
def for_every_subset_is_there_a_subset():
    np.random.seed(2)
    num_goods = 7
    valuations = np.random.randint(low=1, high=10, size=(2, num_goods))

    print(valuations)

    for p in permutations(range(num_goods)):
        for i in range(len(p)):
            alloc1 = p[:i]
            alloc2 = p[i:]
            print("alloc1 :", alloc1)
            print("alloc2: ", alloc2)
            if is_efx(alloc1, alloc2, valuations):
                print("is efx")
                # See if it's the case that for every subset of alloc1 there is a subset of alloc2 s.t. it's EFX
                for subset_alloc1 in powerset(alloc1):
                    print(subset_alloc1)
                    is_a_subset = False
                    for subset_alloc2 in powerset(alloc2):
                        if is_efx(subset_alloc1, subset_alloc2, valuations):
                            print(subset_alloc2)
                            is_a_subset = True
                            break
                    if not is_a_subset:
                        print("no subset")
                        print(subset_alloc1)
                        print(valuations)
                        sys.exit(0)


def exhaustive_sim_triplet(algo):
    val_fns = np.zeros((3, 8))
    max_val = 5
    agents = [0, 1, 2]
    goods = list(range(8))

    r = [range(1, max_val)] * (3 * len(goods))
    for idx, v in enumerate(product(*r)):
        i = 0
        for a in agents:
            for g in goods:
                val_fns[a, g] = v[i]
                i += 1

        if idx % 10000 == 0:
            print(idx)
            print(val_fns)

        if not algo(val_fns):
            print("FAIL")
            print(val_fns)
            sys.exit(0)


def monotone_dec(l):
    min_val = 1000
    for i in l:
        if i > min_val:
            return False
        else:
            min_val = i
    return True


def monotone_inc(l):
    max_val = 0
    for i in l:
        if i < max_val:
            return False
        else:
            max_val = i
    return True


if __name__ == "__main__":
    # sz_of_interest = 10
    # num_iters_per_problem = []
    # num_agents_per_problem = []
    #
    # plt_ct = 0
    #
    # for instance_id, valns in spliddit_generator():
    #     if valns.shape[0] > 2:
    #         num_agents_per_problem.append(valns.shape[0])
    #         # if instance_id == 85604:
    #         #     print(valns)
    #         # if valns.shape[0] == sz_of_interest:
    #         # print(instance_id)
    #         # print(valns)
    #         print(valns)
    #         valns += 0.001
    #
    #         # put them in a path
    #         edges = [(i, i + 1) for i in range(valns.shape[0] - 1)]
    #
    #         # run the algorithm
    #         alloc, envy_over_passes, strong_envy_over_passes, strong_envy_vecs, min_over_passes, num_goods_to_drop, num_passes = efx_on_graph(
    #             valns, edges)
    #         num_iters_per_problem.append(num_passes)
    #
    #         # Check for pairwise EFX
    #         if not check_pairwise_efx(alloc, valns, edges):
    #             print("Not EFX!")
    #             sys.exit(0)
    #
    #         if not monotone_dec(num_goods_to_drop):
    #             print("FAIL")
    #             print(valns)
    #             print(num_goods_to_drop)
    #             sys.exit(0)
    #             # plt_ct += 1
    #             # plt.plot(min_over_passes, label="id: %d" % instance_id)
    #             # plt_ct += 1
    #
    #         # Selected bumpy envy plots
    #         # if instance_id in [31633, 48854, 17399, 7672, 85604, 84774]:
    #         #     plt.plot(envy_over_passes, label="id: %d" % instance_id)
    #
    #         # if valns.shape[0] >= 15:
    #         #     plt.plot(envy_over_passes, label="n = 15, id: %d" % instance_id)
    #         #
    #         # if num_passes >= 3:
    #         #     plt.plot(envy_over_passes, label="passes >= 3, id: %d" % instance_id)
    #         #
    #         # if num_passes == 4:
    #         #     print(instance_id)
    #         #     print(valns)
    #
    # print(plt_ct)
    # print(max(num_agents_per_problem))
    # print(Counter(num_iters_per_problem))
    # plt.legend()
    # plt.xlabel("Number of Passes")
    # plt.ylabel("Minimum Valuation")
    # # plt.show()
    # # plt.savefig("bumpy_min.png")

    #
    # valns = np.load(os.path.join("spliddit_data", "matrices", "32723.npy"))
    # # print(valns)
    # valns = valns[:, ::2]
    # print(valns)
    # # valns = valns[:, random.sample(range(valns.shape[1]), valns.shape[1]*4//5)]
    # # print(valns)
    # edges = [(i, i + 1) for i in range(valns.shape[0] - 1)]
    # alloc, envy_over_passes, num_passes = efx_on_graph(valns, edges)
    # print(num_passes)

    np.random.seed(1)
    for _ in range(10000):
        n = 10
        m = 50

        valuations = np.random.randint(0, 50, size=(n, m))
        # valuations = np.array([[43, 9, 31, 29, 13, 33, 32],
        #                        [49, 22, 13, 3, 14, 15, 40],
        #                        [28, 30, 27, 19, 14, 32, 40],
        #                        [34, 16, 37, 3, 30, 28, 6]])
        # valuations = np.array([[2, 2, 3, 4, 7, 1],
        #    [1, 4, 2, 4, 8, 2],
        #    [6, 3, 1, 1, 5, 7]])
        # print(efx_on_graph(valuations, edges))

        # valuations = np.array([[3, 4, 8, 5, 6, 3, 1, 4, 1, 3],
        #                        [3, 1, 1, 9, 1, 5, 1, 7, 1, 2],
        #                        [3, 3, 9, 8, 3, 1, 3, 3, 2, 9],
        #                        [1, 8, 7, 7, 3, 2, 7, 2, 8, 9],
        #                        [2, 5, 4, 7, 3, 9, 5, 8, 3, 7]])
        edges = [(i, i + 1) for i in range(n - 1)]
        alloc, envy_over_passes, strong_envy_over_passes, strong_envy_vecs, min_over_passes, num_goods_to_drop, num_passes = efx_on_graph(
            valuations, edges)
        # print(strong_envy_over_passes)
        # print(strong_envy_vecs)
        # print(num_goods_to_drop)


        def check_lexically_non_increasing(vecs):
            def lexical_comp(v1, v2):
                for i in range(len(v1)):
                    if v1[i] < v2[i]:
                        return False
                    elif v1[i] > v2[i]:
                        return True
                return True

            curr = [1000] + [0] * (len(vecs[0]) - 1)
            for v in vecs:
                if not lexical_comp(curr, v):
                    return False
                curr = v.copy()
            return True


        if not monotone_dec(strong_envy_over_passes):
            print(valuations)
            print(strong_envy_over_passes)

            # See if there is any sequence of cutters and choosers that will make the strong envy monotonically
            # non-increasing.
            alloc, envy_over_passes, strong_envy_over_passes, strong_envy_vecs, min_over_passes, num_goods_to_drop, num_passes = efx_dfs_on_graph(
                valuations, edges)

            sys.exit(0)
    # alloc = pairwise_leximin_on_graph(valuations, edges)
    # print(alloc)
    # print(check_pairwise_efx(alloc, valuations, edges))
    # sys.exit(0)

    # def find_all_efx_allocations(n,m, val_fns):
    #     allocs = []
    #     for p in permutations(range(m)):
    #         for i in range(len(p)):
    #             alloc1 = p[:i]
    #             alloc2 = p[i:]
    #             if is_efx(alloc1, alloc2, 0, 1, val_fns):
    #                 allocs.append((alloc1, alloc2))
    #     return allocs
    #
    #
    # for s in range(10000):
    #     np.random.seed(s)
    #
    #     n = 2
    #     m = 4
    #
    #     valns = np.random.randint(0,5, size=(n, m))
    #     if len(find_all_efx_allocations(n, m, valns)) == 0:
    #         print(valns)
    #         print("unique")
    #         print()
    #     else:
    #         print(valns)
    #         print(find_all_efx_allocations(n, m, valns))
    #         print()

    #     edges = [(i, i + 1) for i in range(n - 1)]
    #
    #     # rng = default_rng()
    #     # vals = rng.standard_normal(10)
    #     # more_vals = rng.standard_normal(10)
    #     # valuations = rng.exponential(scale=10, size=(n, m))
    #     # valuations = np.random.randint(low=1, high=10, size=(n, m))
    #
    #     valuations = np.array([[4, 9, 6, 4, 2, 5, 1],
    #    [5, 6, 7, 7, 8, 3, 1],
    #    [6, 6, 2, 1, 6, 1, 2]])
    #
    #     # valuations = np.array([[9, 1, 6, 7, 2, 7, 9, 8, 7, 7, 9],
    #     #                        [6, 2, 9, 7, 6, 1, 3, 5, 9, 3, 9],
    #     #                        [4, 5, 8, 2, 7, 8, 9, 7, 8, 7, 8],
    #     #                        [5, 4, 3, 4, 1, 8, 8, 9, 8, 8, 5],
    #     #                        [8, 7, 5, 2, 9, 7, 9, 4, 9, 7, 5]])
    #     # The above example repeats if you try to make things leximin edge-by-edge. So there was never a potential
    #     # function argument to show that it always terminates. BUT we can say that it always terminates or repeats a
    #     # state. Is this helpful?
    #
    #     # valuations *= 1/np.sum(valuations, axis=1)
    #     # valuations = np.array([[4, 2, 5, 7, 1], [8, 7, 4, 1, 5], [8, 3, 5, 1, 6]])
    #
    #     # if not efx_on_graph(valuations, edges):
    #     print(repr(valuations))
    #     alloc = pairwise_leximin_on_graph(valuations, edges)
    #     if not check_pairwise_efx(alloc, valuations, edges):
    #         print(repr(valuations))
    #         sys.exit(1)
    #
    #     print("done")
    #
    #     if s % 100 == 0:
    #         print(s)
    #         # print(repr(valuations))
    #         # print(efx_on_graph(valuations, edges))
    #         # print(pairwise_leximin_on_graph(valuations, edges))
    #     # print(valuations)
    #     # if not efx_among_triplet(valuations):
    #     #     print(valuations)
    # # print(efx_among_triplet(valuations))
