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



###train
def train(save_path, train_curr, train_next, ucm, scm, tcm, model, optimizer, exp, lagtime):
    Loss1 = []
    Loss2 = []
    Loss3 = []
    eigenvalue = []
    eigenvector = []
    Tensor = np.identity(nstate)  ###idenity matrix 100*100
    Tensor = torch.from_numpy(Tensor).to(device=device, dtype=torch.float32)
    tcm = torch.from_numpy(tcm).to(device=device, dtype=torch.float32)
    for epoch in range(20):
        print('epoch', epoch)
        for i in range(int(train_curr.shape[0] / batch_size)):
            time1 = train_curr[i * batch_size:i * batch_size + batch_size, ]
            time1 = torch.from_numpy(time1)
            time1 = time1.to(device=device, dtype=torch.float32)
            time1_M = model(time1)
            time2 = train_next[i * batch_size:i * batch_size + batch_size, ]
            time2 = torch.from_numpy(time2)
            time2 = time2.to(device=device, dtype=torch.float32)
            time2_M = model(time2)
            # time2_M = F.normalize(time2_M, dim=1)
            # time1_M = F.normalize(time1_M, dim=1)

            ###
            lumping_matrix = model(Tensor)
            projection = torch.mm(lumping_matrix.t(), ucm[:, 0:latent_dim])
            # projection = torch.mm(projection, torch.diag(torch.sign(projection[0])))
            s, u, v, cov_M, loss1, loss2 = eigenvec_comp(time1_M, time2_M, projection, scm)
            optimizer.zero_grad()
            if epoch < 15:
                loss = loss1
            else:
                loss = loss2

            loss.backward()
            eigenvalue.append(s.cpu().detach().numpy())
            eigenvector.append(torch.abs(torch.mm(v, projection)).cpu().detach().numpy())
            Loss2.append(torch.sum(s).cpu().detach().numpy())
            Loss1.append(torch.mean(
                torch.abs(torch.abs(torch.mm(v, projection)) - torch.eye(latent_dim).cuda())).data.cpu().numpy())
            optimizer.step()

            '''if i % show_iter == 0:
                print('eigenvalue:', s)
                print('eigenvec:',
                      torch.mean(torch.abs(torch.abs(torch.mm(v, projection)) - torch.eye(4).cuda())))
                print('eigenvalue:', torch.mean(torch.abs(s - scm[0:latent_dim])))
                print('BP:', torch.sum(cov_M, 0))
                # print('PCCA:', np.sum(TCM_PCCA_local[i], 0))
                print('lumping matrix:', lumping_matrix[79])
                print('vector', projection)
                print(u)
                print(torch.mm(v, projection))
                # print(torch.mm(v, projection))
                # print(torch.argmax(lumping_matrix, dim=1))'''
        cov = lumping_matrix.t().mm(tcm).mm(lumping_matrix)
        u_1, v_1, s_1 = normalize_eigenvector_fromTCM_torch(cov, int(train_curr.shape[0]))
        proj = torch.mm(lumping_matrix.t(), ucm[:, 0:latent_dim])
        total_eigenvector_loss = torch.mean(
            torch.abs(torch.abs(torch.mm(v_1, proj)) - torch.eye(4).cuda())).data.cpu().numpy()
        print('total_eigenvalue', s_1)
        print('check Identity', torch.mm(v_1, proj))
        print('eigenvector loss', total_eigenvector_loss)
        Loss3.append(total_eigenvector_loss)
        if epoch == 6 and s_1[3] < 0.2:
            break
    lumping_matrix = lumping_matrix.cpu().detach().numpy()
    np.save(save_path + 'lumping_matrix_DL_lagtime%d_exptime%d.npy' % (lagtime, exp), lumping_matrix)
    torch.save(model.state_dict(),
               save_path + 'lumping_lagtime%d_exptime%d' % (lagtime, exp))
    np.save(save_path + 'eigenvalue_lagtime%d_exptime%d.npy' % (lagtime, exp), eigenvalue)
    np.save(save_path + 'eigenvector_lagtime%d_exptime%d.npy' % (lagtime, exp), eigenvector)
    # np.save(save_path + 'eigenvalue_Loss_lagtime%d_exptime%d.npy' % (lagtime, exp), Loss2)
    # np.save(save_path + 'eigenvector_Loss_lagtime%d_exptime%d.npy' % (lagtime, exp), Loss1)
    # np.save(save_path + 'eigenvector_epoch_Loss_lagtime%d_exptime%d.npy' % (lagtime, exp), Loss3)
    return lumping_matrix, s_1[3]
  

def main(args):
    ##set parameter
    nstate = args.nstate
    lag = args.lag
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
       device = torch.device('cuda')
    else:
       device = torch.device('cpu')
    batch_size = args.bs
    input_dim = args.nstate
    latent_dim = args.latent_dim
    show_iter = args,show_iter
    learning_rate = args.lr
    epison = args.eps
    data_path = args.data_path
    save_path = args.save_data_path
    
    for lagtime in lag:
        partition_curr_tot, partition_next_tot = gen_data(True, data_path, lagtime)
        for i in range(opt.n_folder_epoch):
            print('exp_time', i)
            train_curr, train_next, test_curr, test_next, tcm_train, tcm_test = split_data(partition_curr_tot,
                                                                                           partition_next_tot, i)
            lumping_assignment = np.loadtxt(
                '../other_methods/forGHL20210410/experiment' + str(i) + '_tcm.txt-map_mpp_qMin0.950000_4states.dat')
            gmrq_mpp_train, gmrq_mpp_test = Othermethods_GMRQ(tcm_train, tcm_test, lumping_assignment,
                                                                train_next.shape[0], test_curr.shape[0])
            gmrq_MPP.append(gmrq_mpp_train)
            gmrq_MPP.append(gmrq_mpp_test)
            np.save(save_path + 'GMRQ_MPP_lagtime%d' % (lagtime), gmrq_MPP)
            ucm, vcm, scm = normalize_eigenvector_fromTCM(tcm_train, int(train_curr.shape[0]))
            # PCCA GMRQ
            gmrq_PCCA_train, gmrq_PCCA_test = PCCA_GMRQ(train_curr, train_next, test_curr, test_next)
            gmrq_PCCA.append(gmrq_PCCA_train)
            gmrq_PCCA.append(gmrq_PCCA_test)

            # dl GMRQ
            ucm = torch.from_numpy(ucm).to(device=device, dtype=torch.float32)
            scm = torch.from_numpy(scm).to(device=device, dtype=torch.float32)
            s_1 = 0
            while s_1 < 0.1:
                model = Encoder().cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                model.weight_init(mean=0.0, std=0.8)
                lumping_matrix_dl, s_1 = train(save_path, train_curr, train_next, ucm, scm, tcm_train, model, optimizer,
                                               i, lagtime)
            gmrq_DL_train, gmrq_DL_test = DL_GMRQ(train_curr, train_next, test_curr, test_next, tcm_train, tcm_test,
                                                  ucm, lumping_matrix_dl)
            gmrq_DL.append(gmrq_DL_train)
            gmrq_DL.append(gmrq_DL_test)
            np.save(save_path + 'GMRQ_PCCA_lagtime%d' % (lagtime), gmrq_PCCA)
            np.save(save_path + 'GMRQ_DL_lagtime%d' % (lagtime), gmrq_DL)
            del train_curr, train_next, test_curr, test_next

 
if __name__ == '__main__':
  args = parser_args()
  print(args)
  
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  main(args)
