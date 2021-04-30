import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import fractional_matrix_power
import pyemma
from utils.utils import *

def convert_lumping_matrix_loss(tcm, assign, n1, n2):
    lumping_matrix = np.zeros((n1, n2))
    #print(assign)
    for i in range(len(assign)):
        lumping_matrix[i][int(assign[i])] = 1
    TCM = lumping_matrix.transpose().dot(tcm).dot(lumping_matrix)
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)
    loss_matrix = np.abs(
        np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4])))
    print('lossmatrix:', loss_matrix)
    loss = np.mean(np.abs(loss_matrix - np.eye(n2)))
    print(loss)

    return loss


def convert_lumping_matrix_loss1(tcm, assign, n1, n2):
    lumping_matrix = np.zeros((n1, n2))
    #print(assign)
    for i in range(len(assign)):
        lumping_matrix[i][int(assign[i])] = 1
    TCM = lumping_matrix.transpose().dot(tcm).dot(lumping_matrix)
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)
    loss_matrix = np.abs(
        np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4])))
    print('lossmatrix:', loss_matrix)
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    loss = np.mean(np.abs(loss_matrix - B))
    print(loss)

    return loss


def ymatix_test(tcm, lumping_assignment):
    lumping_matrix = np.zeros((len(lumping_assignment), 4))
    for i in range(len(lumping_assignment)):
        lumping_matrix[i][int(lumping_assignment[i])] = 1

    TCM = lumping_matrix.transpose().dot(tcm).dot(lumping_matrix)
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)

    print('check identity:\n')
    print(
        np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4])))
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    PCCA_yloss = np.mean(
        np.abs(
            np.abs(np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4]))) - np.eye(4)))
    print(PCCA_yloss)
    return

def ymatix_test_new(tcm, lumping_matrix):

    TCM = lumping_matrix.transpose().dot(tcm).dot(lumping_matrix)
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)

    print('check identity:\n')
    print(
        np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4])))
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    PCCA_yloss = np.mean(
        np.abs(
            np.abs(np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:4]))) - np.eye(4)))
    print(PCCA_yloss)
    return


path = ''  ##Bpnet path
path1 = '' ##other method part
lagtime900 = np.load(path + 'alldata_eigenvector_eachepoch_Loss_lagtime900.npy')
plt.plot(np.arange(27), lagtime900, c='b', label='dl')
tcm = np.loadtxt('/home/share/xuhuihuang/lumping_data/ilona_data_interval900_tcm.txt')
lumping_matrix = np.load('../save_data/ilona_data/lumping_matrix_DL_lagtime900.npy')
ymatix_test_new(tcm, lumping_matrix)

lumping = pyemma.msm.PCCA(np.diag(np.sum(tcm, 0) ** (-1)).dot(tcm), 4)
assign_PCCA = lumping.metastable_assignment
PCCA_loss = convert_lumping_matrix_loss1(tcm, assign_PCCA, 100, 4)
plt.axhline(PCCA_loss, linestyle="--", c='r', label='PCCA')

assign_BACE = np.loadtxt(path1 + 'Ilona_BACE_4states.dat')
BACE_loss = convert_lumping_matrix_loss(tcm, assign_BACE, 100, 4)
plt.axhline(BACE_loss, linestyle="--", c='g', label='BACE')

assign_MPP = np.loadtxt(path1 + 'Ilona_MPP_q0.97_4states.dat')
MPP_loss = convert_lumping_matrix_loss(tcm, assign_MPP, 100, 4)
plt.axhline(MPP_loss, linestyle="--", c='y', label='MPP')

assign_ward = np.loadtxt(path1 + 'Ilona_Ward_4states.dat')
ward_loss = convert_lumping_matrix_loss(tcm, assign_ward, 100, 4)
plt.axhline(ward_loss, linestyle="--", c='orange', label='ward')

plt.legend()
plt.savefig(path + 'lagtime900_eigenvectorloss_eachepoch.png')
plt.close()

x = np.arange(5)
y = [lagtime900[-1], PCCA_loss, BACE_loss, MPP_loss, ward_loss]
plt.bar(x, y, color='grey')
plt.savefig(path + 'lagtime900_eigenvectorloss_bar.png')
plt.close()
