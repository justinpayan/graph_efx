import numpy as np
from numpy.random import default_rng
import sys

from itertools import permutations, chain, combinations, product


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


"""
Given this allocation and these valuations, rebalance the goods that e[0] and e[1] are currently assigned so 
that they are EFX with each other.
"""


def fix_edge(alloc, val_fns, e):
    good_set = set(alloc[e[0]]) | set(alloc[e[1]])
    for a0 in powerset(good_set):
        g0 = list(a0)
        g1 = list(good_set - set(a0))

        if is_efx(g0, g1, e[0], e[1], val_fns):
            alloc[e[0]] = g0
            alloc[e[1]] = g1
            return alloc
    if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
        return None


"""
val_fns: nxm numpy array, the ij element is the value of good j for agent i
edge_list: list of tuples (i, i'), which indicate the edges along which we must be EFX, and they also
            indicate the order for rebalancing goods under the iterative algorithm
"""


def efx_on_graph(val_fns, edge_list):
    n, m = val_fns.shape

    alloc = [list(range(m))] + [list() for _ in range(n - 1)]

    # TODO: is this necessary?
    # already_visited = {alloc}

    # Fix edges in edge_list in forward then reverse order.
    g_efx = False
    ctr = 0
    while not g_efx:
        # print(ctr)
        ctr += 1
        g_efx = True
        for e in edge_list:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                g_efx = False
                alloc = fix_edge(alloc, val_fns, e)
        # Go backwards now. We just finished the last edge, so skip it.
        for e in edge_list[::-1][1:]:
            if not is_efx(alloc[e[0]], alloc[e[1]], e[0], e[1], val_fns):
                g_efx = False
                alloc = fix_edge(alloc, val_fns, e)

    return alloc


def is_efx(alloc1, alloc2, a1, a2, val_fns):
    for i in alloc1:
        if np.sum(val_fns[a2, alloc2]) < np.sum(val_fns[a2, alloc1]) - val_fns[a2, i]:
            return False
    for i in alloc2:
        if np.sum(val_fns[a1, alloc1]) < np.sum(val_fns[a1, alloc2]) - val_fns[a1, i]:
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


if __name__ == "__main__":
    # exhaustive_sim_triplet(efx_among_triplet)
    n = 6
    edges = [(i, i + 1) for i in range(n - 1)]

    m = 15

    # valuations = np.array([[2, 2, 6, 6, 8, 7, 7, 3, 4, 4],
    #                        [4, 5, 1, 3, 6, 4, 2, 8, 5, 7],
    #                        [4, 1, 5, 9, 2, 6, 8, 8, 8, 3],
    #                        [7, 3, 2, 6, 9, 4, 6, 6, 6, 5]])
    # print(efx_on_graph(valuations, edges))

    for s in range(100000):
        np.random.seed(s)

        rng = default_rng()
        vals = rng.standard_normal(10)
        more_vals = rng.standard_normal(10)
        valuations = rng.exponential(scale=10, size=(n, m))
        # valuations = np.random.randint(low=1, high=100, size=(n, m))
        # valuations *= 1/np.sum(valuations, axis=1)
        # valuations = np.array([[4, 2, 5, 7, 1], [8, 7, 4, 1, 5], [8, 3, 5, 1, 6]])

        if not efx_on_graph(valuations, edges):
            print(valuations)
            sys.exit(1)

        if s % 100 == 0:
            print(s)
            print(valuations)
            print(efx_on_graph(valuations, edges))
        # print(valuations)
        # if not efx_among_triplet(valuations):
        #     print(valuations)
    # print(efx_among_triplet(valuations))
