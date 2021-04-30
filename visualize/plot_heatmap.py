import torch

print(torch.cuda.is_available())
import math
from torch import nn
import numpy as np
from skimage.exposure import rescale_intensity

np.set_printoptions(threshold=np.inf)
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import pyemma
import matplotlib.pyplot as plt

data_path = ''
save_path = ''


def plot_psi_phi(matrix, figname, weights):
    position = np.loadtxt('ilona_daa_TIC.txt')
    color = ['b', 'orange', 'r', 'g']
    for i in range(100):
        index = int(np.argmax(matrix[i]))
        plt.scatter(position[i, 1], position[i, 2], s=weights[i] * 30, alpha=0.7,
                    c=color[index])
        # plt.text(position[i, 1], position[i, 2], s=str(i), fontsize=8)
    '''for j in np.linspace(0.5, 1, 2):
        for i in range(100):
            index = int(np.argmax(matrix[i]))
            plt.scatter(position[i, 1] + 10 * j, position[i, 2] + 10 * j, s=300, alpha=1 - j,
                        c=color[index])
    for j in np.linspace(0.5, 1, 3):
        for i in range(100):
            index = int(np.argmax(matrix[i]))
            plt.scatter(position[i, 1] - 10 * j, position[i, 2] - 10 * j, s=300, alpha=1 - j,
                        c=color[index])'''
    # plt.text(position[i, 1], position[i, 2], s=str(i), fontsize=8)
    plt.show()
    plt.close()


###ala2
'''tcm = np.loadtxt('ala2_interval50_tcm.txt')
weights = np.sum(tcm, 1)
weights = np.log(weights)
dl_lumping_matrix = np.load('lumping_matrix_DL_lagtime50.npy')
#plot_psi_phi(dl_lumping_matrix, '', weights)

lumping = pyemma.msm.PCCA(np.diag(np.sum(tcm, 0) ** (-1)).dot(tcm), 4)
assign_PCCA = lumping.metastable_assignment
pcca_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    pcca_lumping_matrix[i][assign_PCCA[i]] = 1
#plot_psi_phi(pcca_lumping_matrix, '', weights)

bace = np.loadtxt('../other_method/toGHL20210210/alanDip_BACE_4states.dat')
bace_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    bace_lumping_matrix[i][int(bace[i])] = 1
#plot_psi_phi(bace_lumping_matrix, '', weights)

mmp = np.loadtxt('../other_method/toGHL20210210/alanDip_MPP_q0.7_4states.dat')
mmp_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    mmp_lumping_matrix[i][int(mmp[i])] = 1
#plot_psi_phi(mmp_lumping_matrix, '', weights)

ward = np.loadtxt('../other_method/toGHL20210210/alanDip_Ward_4states.dat')
ward_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    ward_lumping_matrix[i][int(ward[i])] = 1
plot_psi_phi(ward_lumping_matrix, '', weights)'''

'''##ilona
tcm = np.loadtxt('ilona_data_interval900_tcm.txt')
weights = np.sum(tcm, 1)
weights = np.log(weights)
dl_lumping_matrix = np.load('lumping_matrix_DL_lagtime900.npy')
#plot_psi_phi(dl_lumping_matrix, '', weights)

lumping = pyemma.msm.PCCA(np.diag(np.sum(tcm, 0) ** (-1)).dot(tcm), 4)
assign_PCCA = lumping.metastable_assignment
pcca_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    pcca_lumping_matrix[i][assign_PCCA[i]] = 1
plot_psi_phi(pcca_lumping_matrix, '', weights)

bace = np.loadtxt('../other_method/toGHL20210210/Ilona_BACE_4states.dat')
bace_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    bace_lumping_matrix[i][int(bace[i])] = 1
#plot_psi_phi(bace_lumping_matrix, '', weights)

mmp = np.loadtxt('../other_method/toGHL20210210/Ilona_MPP_q0.97_4states.dat')
mmp_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    mmp_lumping_matrix[i][int(mmp[i])] = 1
#plot_psi_phi(mmp_lumping_matrix, '', weights)

ward = np.loadtxt('../other_method/toGHL20210210/Ilona_Ward_4states.dat')
ward_lumping_matrix = np.zeros((100, 4))
for i in range(100):
    ward_lumping_matrix[i][int(ward[i])] = 1
#plot_psi_phi(ward_lumping_matrix, '', weights)'''


###2D potential
###2D potential
def normalize_eigenvector_fromTCM(C):
    # D_norm = np.diag(np.sum(C, 0))

    D_norm_Nhalf = np.diag((np.sum(C, 0)) ** (-0.5))
    D_norm_Phalf = np.diag((np.sum(C, 0)) ** (0.5))
    u1, s1, v1 = np.linalg.svd(np.matmul(D_norm_Nhalf, C.dot(D_norm_Nhalf)))
    # s = s1
    u = D_norm_Phalf.dot(u1)
    v = v1.dot(D_norm_Nhalf)
    pi = u[:, 0] / sum(u[:, 0])
    D = np.diag(pi)
    a = np.matmul(v.dot(D), v.transpose())
    c = fractional_matrix_power(a, -0.5)
    b = fractional_matrix_power(a, 0.5)
    return u.dot(b), c.dot(v), s1


n = 10000000
k = 31
latent_dim = 4
traj = np.loadtxt('MicroAssignment.txt')[0:n]
position = np.loadtxt('overall_coordinates.txt')[0:n]
lag = []
for lagtime in lag:
    tcm = np.zeros((961, 961))
    for i in range(len(traj) - lagtime):
        tcm[int(traj[i])][int(traj[i + lagtime])] += 1
    tcm = (tcm + tcm.transpose()) / 2
    # PCCA
    lumping = pyemma.msm.PCCA(np.diag(np.sum(tcm, 0) ** (-1)).dot(tcm), 4)
    index = lumping.metastable_assignment

    lumping_matrix = np.zeros((961, 4))
    for i in range(len(index)):
        lumping_matrix[i, index[i]] = 1
    TCM = lumping_matrix.transpose().dot(tcm).dot(lumping_matrix)
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)
    print(lagtime)
    print('eigenvalue for TCM:\n')
    print(s_M)
    print('eigenvalue for tcm:\n')
    print(s_m)
    print('eigenvector for TCM:\n')
    print(u_M)
    print('eigenvector for tcm:\n')
    print(np.matmul(lumping_matrix.transpose(), u_m[:, 0:latent_dim]))
    print('check identity:\n')
    print(
        np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:latent_dim])))
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    PCCA_yloss = np.mean(
        np.abs(
            np.abs(np.matmul(v_M, np.matmul(lumping_matrix.transpose(), u_m[:, 0:latent_dim]))) - np.eye(4)))
    print(PCCA_yloss)

    traj_after = []
    data = np.zeros((k, k))
    colors = np.zeros((k, k, 2))
    for i in range(n):
        data[int(round(position[i, 0])), int(round(position[i, 1]))] += 1
        colors[int(round(position[i, 0])), int(round(position[i, 1])), 0] += int(index[int(traj[i])])
        colors[int(round(position[i, 0])), int(round(position[i, 1])), 1] += 1

    data = np.sqrt(data)
    data = rescale_intensity(1.0 * data, out_range=(0.301, 0.99))
    color = [plt.cm.Reds, plt.cm.Blues, plt.cm.YlOrRd, plt.cm.Greens]
    colorss = ['cyan', 'yellow', 'lime', 'red']
    x = []
    y = []
    color_list = []
    color_list1 = []
    for i in range(k):
        for j in range(k):
            if colors[i, j, 1] != 0:
                color_index = int(round(colors[i, j, 0] / colors[i, j, 1]))
                color_list1.append(colorss[color_index])
                x.append(i)
                y.append(j)
                if color_index == 2:
                    color_list.append(color[color_index](data[i, j] - 0.3))
                else:
                    color_list.append(color[color_index](data[i, j]))

    plt.scatter(x, y, c=color_list1, marker='s', s=100)
    plt.savefig('lagtime_%d.2dpotential.png' % lagtime)
    plt.close()

# plt.text(x, y, s=label, fontsize=6)'''
'''for i in range(k):
    for j in range(k):
        if colors[i, j, 1] != 0:
            plt.text(i, j, s=str(int(round(colors[i, j, 0] / colors[i, j, 1]))), fontsize=4)'''
