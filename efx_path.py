import os, sys
import numpy as np
from itertools import permutations, chain, combinations, product
from copy import deepcopy


def efx_path(val_fns):
    n, m = val_fns.shape

    alloc = [list()]*n
    alloc[0] = list(range(m))

    print("starting")
    # Trade goods left to right.
    while True:
        print(alloc)
        is_efx_path = True
        was_fix = False
        for i in range(n-1):
            if not is_efx(alloc[i], alloc[i+1], i, i+1, val_fns):
                is_efx_path = False

                # Fix it.
                L_goods = alloc[i]
                for a0 in powerset(L_goods):
                    nota0 = list(set(alloc[i]) - set(a0))
                    if is_efx(list(a0), alloc[i+1] + nota0, i, i+1, val_fns):
                        alloc[i] = list(a0)
                        alloc[i+1] = alloc[i+1] + nota0
                        was_fix = True
                        break

        if is_efx_path:
            return alloc
        elif not was_fix:
            print("fail")
            sys.exit(0)


def efx_path2(val_fns):
    n, m = val_fns.shape

    alloc = [list()]*n
    alloc[0] = list(range(m))

    print("starting")
    # Trade goods left to right.
    while True:
        print(alloc)
        is_efx_path = True
        was_fix = False
        for i in range(n-1):
            if not is_efx(alloc[i], alloc[i+1], i, i+1, val_fns):
                is_efx_path = False

                # Fix it.
                L_goods = alloc[i]
                best_value_L = -1
                alloc_a0 = None
                for a0 in powerset(L_goods):
                    nota0 = list(set(alloc[i]) - set(a0))
                    if is_efx(list(a0), alloc[i+1] + nota0, i, i+1, val_fns):
                        val_l = sum([val_fns[i, j] for j in a0])
                        if val_l > best_value_L:
                            best_value_L = val_l
                            alloc_a0 = a0
                if alloc_a0 is not None:
                    nota0 = list(set(alloc[i]) - set(alloc_a0))
                    alloc[i] = list(alloc_a0)
                    alloc[i+1] = alloc[i+1] + nota0
                    was_fix = True
                else:
                    print("fail")
                    sys.exit(0)

        if is_efx_path:
            return alloc
        elif not was_fix:
            print("fail")
            sys.exit(0)


def efx_path3(val_fns):
    n, m = val_fns.shape

    alloc = [list()]*n
    alloc[0] = list(range(m))

    print("starting")
    # Trade goods left to right.
    while True:
        print(alloc)
        is_efx_path = True
        was_fix = False
        for i in range(n-1):
            print(alloc)
            if not is_efx(alloc[i], alloc[i+1], i, i+1, val_fns):
                is_efx_path = False

                # Fix it.
                L_goods = alloc[i]
                prev_highest_efx_size = -1
                val_L = -1
                best_a0 = None
                for a0 in powerset(L_goods):
                    nota0 = list(set(alloc[i]) - set(a0))
                    if is_efx(list(a0), alloc[i+1] + nota0, i, i+1, val_fns):
                        print("efx option", a0)
                        highest_efx_size = len(a0)
                        v = sum([val_fns[i,j] for j in a0])
                        print(v, val_L, highest_efx_size, prev_highest_efx_size)
                        if v > val_L and highest_efx_size == prev_highest_efx_size or prev_highest_efx_size == -1:
                            val_L = v
                            best_a0 = a0

                        if prev_highest_efx_size > -1 and prev_highest_efx_size != highest_efx_size:
                            break
                        prev_highest_efx_size = highest_efx_size

                if best_a0 is not None:
                    nota0 = list(set(alloc[i]) - set(best_a0))
                    alloc[i] = list(best_a0)
                    alloc[i + 1] = alloc[i + 1] + nota0
                    was_fix = True
        if is_efx_path:
            return alloc
        elif not was_fix:
            print("fail")
            sys.exit(0)


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
    return chain.from_iterable(combinations(s, r) for r in list(range(len(s)+1))[::-1])


# Is it the case that in any EFX allocation on 2 agents, for every subset of a1's goods there is a subset of a2's goods
# so the subsets give an EFX alloc? Answer is no, counterex from this code.
def for_every_subset_is_there_a_subset():
    np.random.seed(2)
    num_goods = 7
    valuations = np.random.randint(low=1, high=10, size=(2,num_goods))

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


def load_valuations_spliddit():
    with open(os.path.join("C:\\", "Users", "Justin Payan", "Downloads", "valuations.csv")) as f:
        instance_id = -1
        instance = {}
        goods = set()
        agents = set()
        for l in f.readlines():
            a, g, iid, v = [int(l.strip().split()[i]) for i in range(1, 5)]
            if iid != instance_id and len(instance):
                processed_instance = []
                for agent in sorted(agents):
                    processed_instance.append(list())
                    for good in sorted(goods):
                        processed_instance[-1].append(instance[agent][good])

                yield np.array(processed_instance)
                instance_id = iid
                instance = {}
                goods = set()
                agents = set()

            instance_id = iid
            goods.add(g)
            agents.add(a)
            if a not in instance:
                instance[a] = {}
            if g not in instance[a]:
                instance[a][g] = v


if __name__ == "__main__":

    # valuations = np.array([[4, 2, 5, 7, 1], [8, 7, 4, 1, 5], [8, 3, 5, 1, 6], [8, 3, 5, 1, 6]])

    for val_fns in load_valuations_spliddit():
        if val_fns.shape[0] < 8 and val_fns.shape[1] < 8:
            print(val_fns)
            if not efx_path3(val_fns):
                print("fail")
                print(val_fns)

    print("done")
