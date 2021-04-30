import numpy as np
import math
from scipy.linalg import fractional_matrix_power


def count(traj, matrix1, lagtime):
    for j in range(len(traj) - lagtime):
        matrix1[int(traj[j])][int(traj[j + lagtime])] += 1
    return matrix1


def gmrq(u, C, A, bs):
    D = torch.diag(torch.sum(C, 1))
    T = C.mm(torch.inverse(D))
    b = (1 / bs)
    ###u: the first fourth column of right eigenvector microstate 100 *4, A: lumping matrix 100*4, C:TCM
    return A.t().mm(u).mm(torch.inverse(D)).t().mm(T).mm(A.t()).mm(u) / b  ###whether over b


def normalize_eigenvector_fromTCM(C):
    # D_norm = np.diag(np.sum(C, 0))

    D_norm_Nhalf = np.diag((np.sum(C, 0) + epison) ** (-0.5))
    D_norm_Phalf = np.diag((np.sum(C, 0) + epison) ** (0.5))
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

def normalize_eigenvector_fromTCM_torch(C, bs):
    # D_norm = np.diag(np.sum(C, 0))
    # print(torch.sum(C, 0))
    D_norm_Nhalf = torch.diag((torch.sum(C, 0) + epison) ** (-0.5))
    D_norm_Phalf = torch.diag((torch.sum(C, 0) + epison) ** (0.5))
    u1, s1, v1 = torch.svd(torch.mm(D_norm_Nhalf, C.mm(D_norm_Nhalf)))
    v1 = v1.t()
    # s = s1
    u = D_norm_Phalf.mm(u1)
    v = v1.mm(D_norm_Nhalf)
    pi = u[:, 0] / torch.sum((u[:, 0]))
    D = torch.diag(pi)
    a = torch.mm(v.mm(D), v.t())
    if bs != 0:
        b = (1 / bs) ** (0.5)
        c = (1 / bs) ** (-0.5)
        return u * b, v * c, s1
    else:
        ua, sa, va = torch.svd(a)
        c = torch.mm(torch.mm(ua, torch.diag(sa.pow(-0.5))), va.t())
        b = torch.mm(torch.mm(ua, torch.diag(sa.pow(0.5))), va.t())
        return u.mm(b), v.mm(c), s1


def PCCA_test(tcm, TCM, lumping_matrix):
    u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
    u_M, v_M, s_M = normalize_eigenvector_fromTCM(TCM)
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
    return PCCA_yloss

def PCCA(tcm):
    tpm = np.dot(np.diag(np.sum(tcm, 1) ** (-1)), tcm)
    lumping = pyemma.msm.PCCA(tpm, latent_dim)
    TCM_PCCA = np.zeros((latent_dim, latent_dim))
    lumping_PCCA = np.zeros((nstate, latent_dim))
    print(lumping.metastable_assignment)
    for i in range(nstate):
        lumping_PCCA[i][lumping.metastable_assignment[i]] = 1
    for i in range(nstate):
        for j in range(nstate):
            k = lumping.metastable_assignment[i]
            m = lumping.metastable_assignment[j]
            TCM_PCCA[k][m] += tcm[i][j]
    return (TCM_PCCA + TCM_PCCA.transpose()) / 2, lumping, lumping_PCCA


def abnormal_test(tcm):
    tpm = np.dot(np.diag(np.sum(tcm, 1) ** (-1)), tcm)
    lumping = pyemma.msm.PCCA(tpm, latent_dim)
    TCM_PCCA = np.zeros((latent_dim, latent_dim))
    lumping_PCCA = np.zeros((nstate, latent_dim))
    for s in range(100000):
        lumping.metastable_assignment[list1] = np.random.randint(2, size=len(list1)) + 2
        for i in range(nstate):
            lumping_PCCA[i][lumping.metastable_assignment[i]] = 1
        for i in range(nstate):
            for j in range(nstate):
                k = lumping.metastable_assignment[i]
                m = lumping.metastable_assignment[j]
                TCM_PCCA[k][m] += tcm[i][j]
        u_m, v_m, s_m = normalize_eigenvector_fromTCM(tcm)
        u_M, v_M, s_M = normalize_eigenvector_fromTCM((TCM_PCCA + TCM_PCCA.transpose()) / 2)
        if np.abs(np.matmul(v_M, np.matmul(lumping_PCCA.transpose(), u_m[:, 0:latent_dim])))[2, 2] > 0.90:
            print('good')
            print(np.matmul(v_M, np.matmul(lumping_PCCA.transpose(), u_m[:, 0:latent_dim])))
            print(lumping.metastable_assignment[list1])
        print(s)



def eigenvalue_comp(time1_M, time2_M):
    cov_M = (torch.mm(time1_M.t(), time2_M) + torch.mm(time2_M.t(), time1_M)) / 2
    D = torch.diag(torch.sum(cov_M, 1) ** (-0.5))
    print(torch.sum(cov_M, 1))
    cov = torch.mm(torch.mm(D, cov_M), D)
    u_M, s_M, v_M = torch.svd(cov)
    return s_M, torch.mean(torch.abs(s_M[0:4] - scm[0:latent_dim]))


def eigenvec_comp(time1_M, time2_M, projection, scm):
    Tensor1 = torch.tensor([10, 10, 10]).to(device=device, dtype=torch.float32)
    Tensor2 = torch.tensor([10, 100]).to(device=device, dtype=torch.float32)
    cov_M = (torch.mm(time1_M.t(), time2_M) + torch.mm(time2_M.t(), time1_M)) / 2
    u_M, v_M, s_M = normalize_eigenvector_fromTCM_torch(cov_M, batch_size)
    # u_M = torch.mm(u_M, torch.diag(torch.sign(u_M[0])))
    return s_M, u_M, v_M, cov_M, torch.mean(
        torch.abs(torch.abs(torch.mm(v_M, projection)[1:4, 1:4]) - torch.eye(3).cuda()) * Tensor1), torch.mean(
        torch.abs(s_M[2: 4] - scm[2:4]) * Tensor2.cuda())


