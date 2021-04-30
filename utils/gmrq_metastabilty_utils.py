from utils.utils import *
import numpy as np 
import pyemma
import os

def gmrq_torch(u, C, A, bs):
    D = torch.diag(torch.sum(C, 1))
    T = C.mm(torch.inverse(D))
    b = (1 / bs)
    ###u: the first fourth column of right eigenvector microstate 100 *4, A: lumping matrix 100*4, C:TCM
    return A.t().mm(u).t().mm(torch.inverse(D)).mm(T).mm(A.t()).mm(u) / b  ###whether over b


def gmrq_np(u, C, A, bs):
    D = np.diag(np.sum(C, 1))
    T = C.dot(np.linalg.inv(D))
    b = (1 / bs)
    ###u: the first fourth column of right eigenvector microstate 100 *4, A: lumping matrix 100*4, C:TCM
    return A.transpose().dot(u).transpose().dot(np.linalg.inv(D)).dot(T).dot(A.transpose()).dot(u) / b


def PCCA_GMRQ(train_curr, train_next, test_curr, test_next):
    tcm_train = (train_curr.transpose().dot(train_next) + train_next.transpose().dot(train_curr)) / 2
    tcm_test = (test_curr.transpose().dot(test_next) + test_next.transpose().dot(test_curr)) / 2
    u_train, v_train, s_train = normalize_eigenvector_fromTCM(tcm_train, int(train_curr.shape[0]))
    u_test, v_test, s_test = normalize_eigenvector_fromTCM(tcm_test, int(test_curr.shape[0]))
    tpm = np.dot(np.diag(np.sum(tcm_train, 1) ** (-1)), tcm_train)
    lumping = pyemma.msm.PCCA(tpm, latent_dim)
    TCM_PCCA_train = np.zeros((latent_dim, latent_dim))
    TCM_PCCA_test = np.zeros((latent_dim, latent_dim))
    lumping_PCCA = np.zeros((nstate, latent_dim))
    for i in range(nstate):
        lumping_PCCA[i][lumping.metastable_assignment[i]] = 1
    for i in range(nstate):
        for j in range(nstate):
            k = lumping.metastable_assignment[i]
            m = lumping.metastable_assignment[j]
            TCM_PCCA_train[k][m] += tcm_train[i][j]
            TCM_PCCA_test[k][m] += tcm_test[i][j]
    TCM_PCCA_train = (TCM_PCCA_train + TCM_PCCA_train.transpose()) / 2
    TCM_PCCA_test = (TCM_PCCA_test + TCM_PCCA_test.transpose()) / 2
    gmrq_train = gmrq_np(u_train[:, 0:latent_dim], TCM_PCCA_train, lumping_PCCA, int(train_curr.shape[0]))
    gmrq_test = gmrq_np(u_test[:, 0:latent_dim], TCM_PCCA_test, lumping_PCCA, int(test_curr.shape[0]))
    print(gmrq_train)
    print(gmrq_test)
    return gmrq_train, gmrq_test


def Othermethods_GMRQ(tcm_train, tcm_test, lumping_assignment, train_length, test_length):
    lumping_matrix = np.zeros((nstate, latent_dim))
    for i in range(nstate):
        lumping_matrix[i][int(lumping_assignment[i])] = 1
    TCM_train = lumping_matrix.transpose().dot(tcm_train).dot(lumping_matrix)
    TCM_test = lumping_matrix.transpose().dot(tcm_test).dot(lumping_matrix)
    u_train, v_train, s_train = normalize_eigenvector_fromTCM(tcm_train, int(train_length))
    u_test, v_test, s_test = normalize_eigenvector_fromTCM(tcm_test, int(test_length))
    gmrq_train = gmrq_np(u_train[:, 0:latent_dim], TCM_train, lumping_matrix, int(train_length))
    gmrq_test = gmrq_np(u_test[:, 0:latent_dim], TCM_test, lumping_matrix, int(test_length))
    return gmrq_train, gmrq_test


def DL_GMRQ(train_curr, train_next, test_curr, test_next, tcm_train, tcm_test, ucm_train, lumping_dl_matrix):
    TCM_train = lumping_matrix_dl.transpose().dot(tcm_train).dot(lumping_matrix_dl)
    TCM_test = lumping_matrix_dl.transpose().dot(tcm_test).dot(lumping_matrix_dl)
    ucm_test, vcm_test, scm_test = normalize_eigenvector_fromTCM(tcm_test, int(test_curr.shape[0]))
    gmrq_train = gmrq_np(ucm_train[:, 0:latent_dim], TCM_train, lumping_dl_matrix, int(train_curr.shape[0]))
    gmrq_test = gmrq_np(ucm_test[:, 0:latent_dim], TCM_test, lumping_dl_matrix, int(test_curr.shape[0]))
    print(gmrq_train)
    print(gmrq_test)
    return gmrq_train, gmrq_test


