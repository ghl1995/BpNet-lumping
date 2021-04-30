import numpy as np
from utils.utils import *
from utils.data_utils import star
from utils.args import parser_args
from models.Bpnet import Encoder
import os
import torch
from torch.nn import nn
import math
import pyemma
from skimage.exposure import rescale_intensity


##set parameter
nstate = 100
lag = [600]
# 2d potential [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 600, 700, 800, 900, 1000]
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
batch_size = 2000
input_dim = nstate
latent_dim = 4
show_iter = 400
learning_rate = 0.001
epison = 10e-8

###train
def train(save_path, train_curr, train_next, tcm, ucm, scm, model, optimizer, PCCA_yloss, lagtime):
    Loss1 = []
    Loss2 = []
    Tensor = np.identity(nstate)  ###idenity matrix 100*100
    Tensor = torch.from_numpy(Tensor).to(device=device, dtype=torch.float32)
    # position = np.loadtxt('../2D_potential/MOE_2D_Potential_961states/position.txt')
    # traj1 = np.loadtxt('../2D_potential/MOE_2D_Potential_961states/MicroAssignment.txt')
    for epoch in range(20):
        print('epoch', epoch)
        cov = torch.zeros((4, 4)).cuda()
        for i in range(int(train_curr.shape[0] / batch_size)):
            time1 = torch.from_numpy(train_curr[i * batch_size:i * batch_size + batch_size, ])
            time1 = time1.to(device=device, dtype=torch.float32)
            time1_M = model(time1)
            time2 = torch.from_numpy(train_next[i * batch_size:i * batch_size + batch_size, ])
            time2 = time2.to(device=device, dtype=torch.float32)
            time2_M = model(time2)

            ###
            lumping_matrix = model(Tensor)
            projection = torch.mm(lumping_matrix.t(), ucm[:, 0:latent_dim])
            # projection = torch.mm(projection, torch.diag(torch.sign(projection[0])))
            s, u, v, cov_M, loss1, loss2 = eigenvec_comp(time1_M, time2_M, projection, scm)
            cov += cov_M.data
            optimizer.zero_grad()
            if epoch < 15:
                loss = loss1
            else:
                loss = loss2 + loss1

            loss.backward()
            Loss1.append(torch.mean(
                torch.abs(
                    torch.abs(torch.mm(v, projection)) - torch.eye(latent_dim).cuda())).data.cpu().numpy())
            optimizer.step()
            # for name, parms in model.named_parameters():
            # print('-->name:', name[0], '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad[0])
            if i % show_iter == 0:
                print('cov', cov_M)
                print('eigenvalue:', s)
                print('eigenvec:',
                      torch.mean(torch.abs(torch.abs(torch.mm(v, projection)) - torch.eye(4).cuda())))
                print('eigenvalueloss:', torch.mean(torch.abs(s - scm[0:latent_dim])))
                print('BP:', torch.sum(cov_M, 0))
                # print('PCCA:', np.sum(TCM_PCCA_local[i], 0))
                # print('lumping matrix:', lumping_matrix)
                # print('vector', projection)
                # print(u)
                print('ymatrix:', torch.mm(v, projection))
                # print(torch.sum(lumping_matrix, 0))
                # print(torch.mm(v, projection))
                # print(torch.argmax(lumping_matrix, dim=1))
        cov1 = lumping_matrix.t().mm(tcm).mm(lumping_matrix)
        u_1, v_1, s_1 = normalize_eigenvector_fromTCM_torch(cov1, 0)
        proj = torch.mm(lumping_matrix.t(), ucm[:, 0:latent_dim])
        print(torch.sum(lumping_matrix, 0))
        # proj = torch.mm(proj, torch.diag(torch.sign(proj[0])))
        # u_1 = torch.mm(u_1, torch.diag(torch.sign(u_1[0])))
        print('total_eigenvalue', s_1)
        print('check Identity', torch.mm(v_1, proj))
        Loss2.append(
            torch.mean(torch.abs(torch.abs(torch.mm(v_1, proj)) - torch.eye(4).cuda())).data.cpu().numpy())
        print('eigenvector_loss', torch.mean(torch.abs(torch.abs(torch.mm(v_1, proj)) - torch.eye(4).cuda())))
        # if torch.mean(torch.abs(torch.abs(torch.mm(v_1, proj)) - torch.eye(4).cuda())) < 0.0120 and epoch > 20:
        # break
        total_eigenvector_loss = torch.mean(
            torch.abs(torch.abs(torch.mm(v_1, proj)) - torch.eye(4).cuda())).data.cpu().numpy()
        lumping_matrix = lumping_matrix.cpu().detach().numpy()
        # np.save(save_path + 'lumping_matrix_DL_epoch_%d_lagtime%d.npy' % (epoch, lagtime), lumping_matrix)
        # plot_2D_potential(lumping_matrix, position, traj1, epoch, save_path)
        # if s_1[3] < 0.01 and epoch == 8:
        # break
        if epoch == 4 and total_eigenvector_loss > 0.1:
            break

    torch.save(model.state_dict(),
               save_path + 'alldata_lumping_lagtime%d' % (lagtime))
    np.save(save_path + 'alldata_eigenvector_eachepoch_Loss_lagtime%d.npy' % (lagtime), Loss2)
    np.save(save_path + 'alldata_eigenvector_Loss_lagtime%d.npy' % (lagtime), Loss1)

    return lumping_matrix, total_eigenvector_loss

  

def main(args):
    for lagtime in lag:
        print('lagtime', lagtime)
        ###gen_data and tcm
        data_path = args.data_path
        save_path = args.save_data_path
        partition_curr_tot, partition_next_tot = gen_data(True, data_path, lagtime)

        train_curr, train_next, test_curr, test_next, tcm = split_data(partition_curr_tot, partition_next_tot)
        ucm, vcm, scm = normalize_eigenvector_fromTCM(tcm)

        ###perform PCCA
        # abnormal_test(tcm)
        TCM_PCCA, lumping_PCCA, lumping_assignment = PCCA(tcm)
        PCCA_yloss = PCCA_test(tcm, TCM_PCCA, lumping_assignment)

        ###set deep learning model
        ucm = np.real(ucm)
        scm = np.real(scm)
        tcm = torch.from_numpy(tcm).to(device=device, dtype=torch.float32)
        ucm = torch.from_numpy(ucm).to(device=device, dtype=torch.float32)
        scm = torch.from_numpy(scm).to(device=device, dtype=torch.float32)
        model = Encoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.weight_init(mean=0.0, std=0.8)
        ##model.load_state_dict(torch.load(save_path + 'pretrain.pth'))  #model pretrain
        model.cuda()
        lumping_matrix_dl = train(save_path, train_curr, train_next, tcm, ucm, scm, model, optimizer,
                                  PCCA_yloss)
        total_eigenvectot_loss = 10
        while total_eigenvectot_loss > 0.1:  # total_eigenvectot_loss > 0.002:
            model = Encoder().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            model.weight_init(mean=0.0, std=0.8)
            try:
                lumping_matrix_dl, total_eigenvectot_loss = train(save_path, train_curr, train_next, tcm, ucm, scm,
                                                                  model,
                                                                  optimizer,
                                                                  PCCA_yloss, lagtime)
            except:
                lumping_matrix_dl = 0
                total_eigenvectot_loss = 10
        np.save(save_path + 'lumping_matrix_DL_lagtime%d.npy' % lagtime, lumping_matrix_dl)
        del partition_curr_tot, partition_next_tot, train_curr, train_next, test_curr, test_next, tcm

 
if __name__ == '__main__':
  args = parser_args()
  print(args)
  
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  main(args)











