import numpy as np
import math

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
