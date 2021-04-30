from sklearn.model_selection import train_test_split
from utils.utils import *
import numpy as np
import os


def count(traj, matrix1, lagtime):
    for j in range(len(traj) - lagtime):
        matrix1[int(traj[j])][int(traj[j + lagtime])] += 1
    return matrix1

def convert_microstate_to_indicator_function(assignment, nstate, lagtime):
    matrix = np.zeros([len(assignment), nstate])
    for j in range(len(assignment)):
        matrix[j][int(assignment[j])] = 1.0
    input_index = range(len(assignment) - lagtime)
    output_index = range(lagtime, len(assignment))
    input_sequence = matrix[input_index, :]
    output_sequence = matrix[output_index, :]
    return input_sequence, output_sequence


def gen_data(is_generate=True, path='', lagtime=10):
    dir_path = os.listdir(path)
    partition_curr_tot = []
    partition_next_tot = []
    for i in sorted(dir_path):
          path_name = os.path.join(path, i)
          traj = np.loadtxt(path_name)[0:]
          partition_curr, partition_next = convert_microstate_to_indicator_function(traj, nstate, lagtime)
          partition_curr_tot.append(partition_curr)
          partition_next_tot.append(partition_next)
    partition_curr_tot = np.array(partition_curr_tot)
    partition_next_tot = np.array(partition_next_tot)
    partition_next_tot = np.reshape(partition_next_tot, (-1, nstate))
    partition_curr_tot = np.reshape(partition_curr_tot, (-1, nstate))

    return partition_curr_tot, partition_next_tot


def split_data(partition_curr_tot, partition_next_tot):
    train_curr, test_curr, train_next, test_next = train_test_split(partition_curr_tot, partition_next_tot,
                                                                    test_size=0.01, random_state=42)
    tcm_train = (train_curr.transpose().dot(train_next) + train_next.transpose().dot(train_curr)) / 2
    tcm_test = (test_curr.transpose().dot(test_next) + test_next.transpose().dot(test_curr)) / 2

    return train_curr, train_next, test_curr, test_next, tcm_train
