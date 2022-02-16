import os

from collections import defaultdict

import numpy as np


def create_spliddit_matrices():
    # Create a dict with the pairs, and later form them all into matrices
    problem_instances = defaultdict(list)
    with open("spliddit_data/valuations.csv", "r") as f:
        for l in f.readlines():
            _, agent, good, instance_id, valn = l.strip().split("\t")
            problem_instances[instance_id].append((agent, good, valn))

    # Form into matrices, save in separate files as numpy arrays
    for instance_id in problem_instances:
        agents = sorted(set([a for a,_,_ in problem_instances[instance_id]]))
        goods = sorted(set([g for _,g,_ in problem_instances[instance_id]]))
        valn_map = {(a, g): v for a,g,v in problem_instances[instance_id]}

        matrx = np.zeros((len(agents), len(goods)))
        for a_idx, a in enumerate(agents):
            for g_idx, g in enumerate(goods):
                matrx[a_idx, g_idx] = valn_map[(a, g)]

        np.save(os.path.join("spliddit_data", "matrices", instance_id), matrx)


# Load the saved spliddit matrices
def spliddit_generator():
    mat_path = os.path.join("spliddit_data", "matrices")
    for matrix_file in os.listdir(mat_path):
        yield int(matrix_file.split(".")[0]), np.load(os.path.join(mat_path, matrix_file))