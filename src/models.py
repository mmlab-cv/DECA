import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import re
import cv2
from random import shuffle

from utils import compute_distances
from utils import *
from layers import *

import pytorch_lightning as pl

import matplotlib.pyplot as plt
from datetime import datetime


class CapsulePose(pl.LightningModule):

    def __init__(self, FLAGS):
        super(CapsulePose, self).__init__()
        self.FLAGS = FLAGS

        self.P = self.FLAGS.pose_dim
        self.PP = int(np.max([2, self.P*self.P]))
        self.A, self.B, self.C, self.D = self.FLAGS.arch[:-1]
        self.n_classes = self.FLAGS.n_classes = self.FLAGS.arch[-1]
        self.in_channels = self.FLAGS.n_channels

        self.s1 = torch.nn.parameter.Parameter(torch.tensor(1., device=self.device), requires_grad=True)
        self.s2 = torch.nn.parameter.Parameter(torch.tensor(1., device=self.device), requires_grad=True)
        self.s3 = torch.nn.parameter.Parameter(torch.tensor(1., device=self.device), requires_grad=True)
        self.s4 = torch.nn.parameter.Parameter(torch.tensor(1., device=self.device), requires_grad=True)

        self.drop_rate = 0.5
        self.features = []

        #----------------------------------------------------------------------------
        
        self.Conv_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64,
                                kernel_size=9, stride=2, padding=4, bias=False)
        # self.Residual_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32,
        #                             kernel_size=1, stride=2, padding=0, bias=False)
        nn.init.xavier_uniform(self.Conv_1.weight)
        self.IN_1 = nn.InstanceNorm2d(64)
        self.Drop_1 = nn.Dropout(p=float(self.drop_rate))

        #----------------------------------------------------------------------------

        self.Conv_2 = nn.Conv2d(in_channels=64, out_channels=128,
                                kernel_size=9, stride=2, padding=4, bias=False)
        # self.Residual_2 = nn.Conv2d(in_channels=32, out_channels=64,
        #                             kernel_size=1, stride=2, padding=0, bias=False)
        nn.init.xavier_uniform(self.Conv_2.weight)
        self.IN_2 = nn.InstanceNorm2d(128)
        self.Drop_2 = nn.Dropout(p=float(self.drop_rate))

        #----------------------------------------------------------------------------

        self.Conv_3 = nn.Conv2d(in_channels=128, out_channels=256,
                                kernel_size=9, stride=2, padding=4, bias=False)
        # self.Residual_3 = nn.Conv2d(in_channels=64, out_channels=128,
        #                             kernel_size=1, stride=2, padding=0, bias=False)
        nn.init.xavier_uniform(self.Conv_3.weight)
        self.IN_3 = nn.InstanceNorm2d(256)
        self.Drop_3 = nn.Dropout(p=float(self.drop_rate))

        #----------------------------------------------------------------------------

        self.Conv_4 = nn.Conv2d(in_channels=256, out_channels=self.A,
                                kernel_size=9, stride=3, padding=7, bias=False)
        # self.Residual_4 = nn.Conv2d(in_channels=128, out_channels=self.A,
        #                             kernel_size=1, stride=3, padding=3, bias=False)
        nn.init.xavier_uniform(self.Conv_3.weight)
        self.IN_4 = nn.InstanceNorm2d(self.A)
        self.Drop_4 = nn.Dropout(p=float(self.drop_rate))

        #----------------------------------------------------------------------------

        self.PrimaryCaps = PrimaryCapsules2d(in_channels=self.A, out_caps=self.B,
                                             kernel_size=1, stride=1, pose_dim=self.P)

        #----------------------------------------------------------------------------

        self.ConvCaps_1 = ConvCapsules2d(in_caps=self.B, out_caps=self.C,
                                         kernel_size=3, stride=2, pose_dim=self.P)

        self.ConvRouting_1 = VariationalBayesRouting2d(in_caps=self.B, out_caps=self.C,
                                                       kernel_size=3, stride=2, pose_dim=self.P,
                                                       cov='diag', iter=self.FLAGS.routing_iter,
                                                       alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                       Psi0=torch.eye(self.PP), nu0=self.PP+1)

        #----------------------------------------------------------------------------

        self.ConvCaps_2 = ConvCapsules2d(in_caps=self.C, out_caps=self.D,
                                         kernel_size=3, stride=1, pose_dim=self.P)

        self.ConvRouting_2 = VariationalBayesRouting2d(in_caps=self.C, out_caps=self.D,
                                                       kernel_size=3, stride=1, pose_dim=self.P,
                                                       cov='diag', iter=self.FLAGS.routing_iter,
                                                       alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                       Psi0=torch.eye(self.PP), nu0=self.PP+1)

        #----------------------------------------------------------------------------

        self.ClassCaps = ConvCapsules2d(in_caps=self.D, out_caps=self.n_classes,
                                        kernel_size=1, stride=1, pose_dim=self.P, share_W_ij=True, coor_add=True)

        self.ClassRouting = VariationalBayesRouting2d(in_caps=self.D, out_caps=self.n_classes,
                                                      # adjust final kernel_size K depending on input H/W, for H=W=32, K=4.
                                                      kernel_size=4, stride=1, pose_dim=self.P,
                                                      cov='diag', iter=self.FLAGS.routing_iter,
                                                      alpha0=1., m0=torch.zeros(self.PP), kappa0=1.,
                                                      Psi0=torch.eye(self.PP), nu0=self.PP+1, class_caps=True)

        #----------------------------------------------------------------------------

        self.Entities = nn.Flatten()

        #----------------------------------------------------------------------------


        self.FC_2D = FullyConnected2d(
            self.FLAGS.batch_size, self.FLAGS.n_classes*self.FLAGS.P*self.FLAGS.P, rate=self.drop_rate)

        self.FC_3D = FullyConnected3d(
            self.FLAGS.batch_size, self.FLAGS.n_classes*self.FLAGS.P*self.FLAGS.P, rate=self.drop_rate)

        self.Depth_Recons = DepthReconstruction(
            self.FLAGS.batch_size, self.FLAGS.input_width, self.FLAGS.input_height, self.FLAGS.n_classes*self.FLAGS.P*self.FLAGS.P, rate=self.drop_rate)

    def forward(self, v):

        # Out ← [?, A, F, F]
        v = self.Conv_1(v) #+ self.Residual_1(v)
        v = self.IN_1(v)
        # v = self.Drop_1(v)
        v = F.gelu(v)

        # print(v.shape)

        v = self.Conv_2(v) #+ self.Residual_2(v)
        v = self.IN_2(v)
        # v = self.Drop_2(v)
        v = F.gelu(v)

        # print(v.shape)

        v = self.Conv_3(v) #+ self.Residual_3(v)
        v = self.IN_3(v)
        # v = self.Drop_3(v)
        v = F.gelu(v)

        # print(v.shape)

        v = self.Conv_4(v) #+ self.Residual_4(v)
        v = self.IN_4(v)
        # v = self.Drop_4(v)
        v = F.gelu(v)

        # print(v.shape)

        # Out ← a [?, B, F, F], v [?, B, P, P, F, F]
        a, v = self.PrimaryCaps(v)
        # print(v.shape)

        # Out ← a [?, B, 1, 1, 1, F, F, K, K], v [?, B, C, P*P, 1, F, F, K, K]
        a, v, _ = self.ConvCaps_1(a, v)
        # print(v.shape)

        # Out ← a [?, C, F, F], v [?, C, P, P, F, F]
        a, v = self.ConvRouting_1(a, v)
        # print(v.shape)

        # Out ← a [?, C, 1, 1, 1, F, F, K, K], v [?, C, D, P*P, 1, F, F, K, K]
        a, v, _ = self.ConvCaps_2(a, v)
        # print(v.shape)

        # Out ← a [?, D, F, F], v [?, D, P, P, F, F]
        a, v = self.ConvRouting_2(a, v)
        # print(v.shape)

        # Out ← a [?, D, 1, 1, 1, F, F, K, K], v [?, D, n_classes, P*P, 1, F, F, K, K]
        a, v, W_reg = self.ClassCaps(a, v)
        # print(v.shape)

        # Out ← yhat [?, n_classes], v [?, n_classes, P, P]
        #yhat, v = self.ClassRouting(a, v)
        _, v = self.ClassRouting(a, v)
        # print(v.shape)

        v = self.Entities(v)  # > (10, 272)
        # print(v.shape)

        # fc = self.FC_Base(entities)
        # print(v.shape)

        yhat2D = self.FC_2D(v)
        # print(yhat2D.shape)

        yhat3D = self.FC_3D(v)
        # print(yhat3D.shape)

        yhatD = self.Depth_Recons(v)
        # print(yhatD.shape)

        return yhat2D, yhat3D, yhatD, W_reg, v

    def training_step(self, train_batch, batch_idx):
        inputs, labels, _ = train_batch

        labels['msk'] = center_skeleton(labels['msk'])
        labels['msk'] = discretize(labels['msk'], 0, 1)

        yhat2D, yhat3D, yhatD, W_reg, _ = self.forward(inputs)
        loss, loss2D, loss3D, lossD, W_reg, ED_3D = self.pose_loss(
            yhat2D, yhat3D, yhatD, W_reg, labels)

        # viewpoint = "top"
        # save_3d_plot(labels['msk'], "gt_depth", display_labels=True, viewpoint=viewpoint)
        # save_3d_plot(yhat3D.cpu().detach().numpy(), "pred_depth", viewpoint=viewpoint)

        self.log('Training/loss', loss)
        self.log('Training/loss2D', loss2D)
        self.log('Training/loss3D', loss3D)
        self.log('Training/lossD', lossD)
        self.log('Training/W_reg', W_reg)
        self.log('Training/ED_3D', ED_3D)
        self.log('Training/mAP', calc_mAP(yhat3D, labels['msk'].type(
            torch.FloatTensor).unsqueeze(-1).cuda(non_blocking=True)))

        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, labels, _ = val_batch

        labels['msk'] = center_skeleton(labels['msk'])
        labels['msk'] = discretize(labels['msk'], 0, 1)

        yhat2D, yhat3D, yhatD, W_reg, _ = self.forward(inputs)
        loss, loss2D, loss3D, lossD, W_reg, ED_3D = self.pose_loss(
            yhat2D, yhat3D, yhatD, W_reg, labels)

        self.log('Validation/loss', loss)
        self.log('Validation/loss2D', loss2D)
        self.log('Validation/loss3D', loss3D)
        self.log('Validation/lossD', lossD)
        self.log('Validation/W_reg', W_reg)
        self.log('Validation/ED_3D', ED_3D)
        self.log('Validation/mAP', calc_mAP(yhat3D, labels['msk'].type(
            torch.FloatTensor).unsqueeze(-1).cuda(non_blocking=True)))
        
        return loss

    def test_step(self, val_batch, batch_idx):
        inputs, labels, _ = val_batch

        labels['msk'] = center_skeleton(labels['msk'])
        labels['msk'] = discretize(labels['msk'], 0, 1)

        yhat2D, yhat3D, yhatD, W_reg, feat_vec = self.forward(inputs)
        loss, loss2D, loss3D, lossD, W_reg, ED_3D = self.pose_loss(
            yhat2D, yhat3D, yhatD, W_reg, labels)

        self.features.extend(list(feat_vec.data.cpu().numpy()))

        # viewpoint = "top"
        # save_3d_plot(labels['msk'], "gt_depth", display_labels=True, viewpoint=viewpoint)
        # save_3d_plot(yhat3D.cpu().detach().numpy(), "pred_depth", viewpoint=viewpoint)

        self.log('Test/loss', loss)
        self.log('Test/loss2D', loss2D)
        self.log('Test/loss3D', loss3D)
        self.log('Test/lossD', lossD)
        self.log('Test/W_reg', W_reg)
        self.log('Test/ED_3D', ED_3D)
        MPJPE = calc_MPJPE(yhat3D, labels['msk'].type(
            torch.FloatTensor).unsqueeze(-1))
        self.log('Test/MPJPE_00_Neck', MPJPE[1])
        self.log('Test/MPJPE_01_Nose', MPJPE[2])
        self.log('Test/MPJPE_02_Body_center', MPJPE[3])
        self.log('Test/MPJPE_03_Shoulders', MPJPE[4])
        self.log('Test/MPJPE_04_Elbows', MPJPE[5])
        self.log('Test/MPJPE_05_Hands', MPJPE[6])
        self.log('Test/MPJPE_06_Hips', MPJPE[7])
        self.log('Test/MPJPE_07_Knees', MPJPE[8])
        self.log('Test/MPJPE_08_Feet', MPJPE[9])
        self.log('Test/MPJPE_09_Eyes', MPJPE[10])
        self.log('Test/MPJPE_10_Ears', MPJPE[11])
        self.log('Test/MPJPE_11_Upper_Body', MPJPE[12])
        self.log('Test/MPJPE_12_Lower_Body', MPJPE[13])
        self.log('Test/MPJPE_13_Mean', MPJPE[0])
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.FLAGS.learning_rate, weight_decay=self.FLAGS.weight_decay)
        # optimizer = adabound.AdaBound(
        #     self.parameters(), lr=self.FLAGS.learning_rate, final_lr=0.1)
        return optimizer

    def pose_loss(self, yhat2D, yhat3D, yhatD, W_reg, labels):
        msk2d = labels['msk2d'].type(
            torch.FloatTensor).unsqueeze(-1).cuda(non_blocking=True)  # > (10,19,2,1) (B,C,2,1)
        msk3d = labels['msk'].type(
            torch.FloatTensor).unsqueeze(-1).cuda(non_blocking=True)  # > (10,19,3,1) (B,C,3,1)

        # if(not int(datetime.now().strftime('%S')) % 10):
        #     plot_skeleton(msk3d.cpu().detach().numpy() * 20, "True")
        #     plot_skeleton(yhat3D.cpu().detach().numpy() * 20, "Predicted")
        
        # print("###########")
        # print(labels['depth'][0].cpu().numpy().shape)
        # cv2.imshow("PRE", labels['depth'][0].cpu().numpy())
        # cv2.waitKey(0)
        depth = labels['depth'].permute(0, 3, 1, 2).cuda(
            non_blocking=True) #/ 255.  # > (10,3,256,256) (B,C,H,W)
        # print(depth[0].permute(1, 2, 0).cpu().numpy().shape)
        # cv2.imshow("POST", depth[0].permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        loss3D, loss2D, EDs, EDs_2D, lossD = compute_distances(
            self.FLAGS, labels3D=msk3d, predictions3D=yhat3D, labels2D=msk2d, predictions2D=yhat2D, labelsD=depth, predictionsD=yhatD)
        
        loss = (0.5 * torch.exp(-self.s1)) * loss3D + self.s1 + \
            (0.5 * torch.exp(-self.s2)) * loss2D + self.s2 + \
            (0.5 * torch.exp(-self.s3)) * lossD + self.s3 + \
            (0.5 * torch.exp(-self.s4)) * W_reg/3000 + self.s4


        # loss = loss3D + loss2D + lossR + W_reg
        return loss, loss2D, loss3D, lossD, W_reg, 1000*EDs

def plot_skeleton(itop, name):
    itop_labels = ['Head','Neck','RShould','LShould',"RElbow","LElbow","RHand","LHand","Torso","RHip","LHip","RKnee","LKnee","RFoot","LFoot"]
    itop_connections = [[0,1],[1,2],[1,3],[2,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,10],[9,11],[10,12],[11,13],[12,14]]
    itop = itop[0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xdata = itop[:,0].flatten()
    ydata = itop[:,1].flatten()
    zdata = itop[:,2].flatten()

    for i in itop_connections:
        x1,x2,y1,y2,z1,z2 = connectpoints(xdata,ydata,zdata,i[0],i[1])
        ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

    ax.scatter3D(xdata, ydata, zdata, c=zdata)

    for x, y, z, label in zip(xdata,ydata,zdata, itop_labels):
        ax.text(x, y, z, label)

    ax.text2D(0.05, 0.95, "ITOP", transform=ax.transAxes)

    ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
    ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
    ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-2,2)
    ax.set_zlim3d(0,2)
    # plt.show(block=False)
    
    # redraw the canvas
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


    # display image with opencv or any operation you like
    cv2.imshow(name,img)
    cv2.waitKey(1)

def connectpoints(x,y,z,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1,z2 = z[p1],z[p2]
    return x1,x2,y1,y2,z1,z2

def calc_mAP(pred_or, gt_or, dist=0.1): 
    '''
    pred: (N, K, 3) 
    gt: (N, K, 3)
    '''

    pred = pred_or.cpu().detach().numpy()
    gt = gt_or.cpu().detach().numpy()

    pred = undiscretize(pred, 0, 1)
    gt = undiscretize(gt, 0, 1)

    pred = pred.squeeze()
    gt = gt.squeeze()

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    N, K = pred.shape[0], pred.shape[1] # [BS, 15, 3]
    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2)) # (N, K)

    acc_d = (err_dist < dist).sum(axis=0) / N

    return np.mean(acc_d)

def calc_mAP_procrustes(pred_or, gt_or, dist=0.1): 
    '''
    pred: (N, K, 3) 
    gt: (N, K, 3)
    '''


    pred = pred_or.cpu().detach().numpy()
    gt = gt_or.cpu().detach().numpy()

    pred = undiscretize(pred, 0, 1)
    gt = undiscretize(gt, 0, 1)

    pred = pred.squeeze(3)
    gt = gt.squeeze(3)

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    N, K = pred.shape[0], pred.shape[1] # [BS, 15, 3]
    # for i, p in enumerate(pred):
    #     d, Z, tform = procrustes(
    #                         gt[i], pred[i])
    #     pred[i] = Z

    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)
    acc_d = (err_dist < dist).sum(axis=0) / N
        
    err_dist_neck = np.sqrt(np.sum((pred[:,[0],:] - gt[:,[0],:])**2, axis=2))  # (N, K)
    acc_d_neck = (err_dist_neck < dist).sum(axis=0) / N
        
    err_dist_nose = np.sqrt(np.sum((pred[:,[1],:] - gt[:,[1],:])**2, axis=2))  # (N, K)
    acc_d_nose = (err_dist_nose < dist).sum(axis=0) / N
        
    err_dist_bodycenter = np.sqrt(np.sum((pred[:,[2],:] - gt[:,[2],:])**2, axis=2))  # (N, K)
    acc_d_bodycenter = (err_dist_bodycenter < dist).sum(axis=0) / N
        
    err_dist_shoulders = np.sqrt(np.sum((pred[:,[3,9],:] - gt[:,[3,9],:])**2, axis=2))  # (N, K)
    acc_d_shoulders = (err_dist_shoulders < dist).sum(axis=0) / N
        
    err_dist_elbows = np.sqrt(np.sum((pred[:,[4,10],:] - gt[:,[4,10],:])**2, axis=2))  # (N, K)
    acc_d_elbows = (err_dist_elbows < dist).sum(axis=0) / N
        
    err_dist_wrists = np.sqrt(np.sum((pred[:,[5,11],:] - gt[:,[5,11],:])**2, axis=2))  # (N, K)
    acc_d_wrists = (err_dist_wrists < dist).sum(axis=0) / N
        
    err_dist_hips = np.sqrt(np.sum((pred[:,[6,12],:] - gt[:,[6,12],:])**2, axis=2))  # (N, K)
    acc_d_hips = (err_dist_hips < dist).sum(axis=0) / N
        
    err_dist_knees = np.sqrt(np.sum((pred[:,[7,13],:] - gt[:,[7,13],:])**2, axis=2))  # (N, K)
    acc_d_knees = (err_dist_knees < dist).sum(axis=0) / N
        
    err_dist_ankles = np.sqrt(np.sum((pred[:,[8,14],:] - gt[:,[8,14],:])**2, axis=2))  # (N, K)
    acc_d_ankles = (err_dist_ankles < dist).sum(axis=0) / N
        
    err_dist_eyes = np.sqrt(np.sum((pred[:,[15,17],:] - gt[:,[15,17],:])**2, axis=2))  # (N, K)
    acc_d_eyes = (err_dist_eyes < dist).sum(axis=0) / N
        
    err_dist_ears = np.sqrt(np.sum((pred[:,[16,18],:] - gt[:,[16,18],:])**2, axis=2))  # (N, K)
    acc_d_ears = (err_dist_ears < dist).sum(axis=0) / N
        
    err_dist_upper_body = np.sqrt(np.sum((pred[:,[0,1,2,3,4,5,9,10,11,15,16,17,18],:] - gt[:,[0,1,2,3,4,5,9,10,11,15,16,17,18],:])**2, axis=2))  # (N, K)
    acc_d_upper_body = (err_dist_upper_body < dist).sum(axis=0) / N
        
    err_dist_lower_body = np.sqrt(np.sum((pred[:,[6,7,8,12,13,14],:] - gt[:,[6,7,8,12,13,14],:])**2, axis=2))  # (N, K)
    acc_d_lower_body = (err_dist_lower_body < dist).sum(axis=0) / N

    return np.mean(acc_d), np.mean(acc_d_neck), np.mean(acc_d_nose), np.mean(acc_d_bodycenter), \
        np.mean(acc_d_shoulders), np.mean(acc_d_elbows), np.mean(acc_d_wrists), np.mean(acc_d_hips), \
        np.mean(acc_d_knees), np.mean(acc_d_ankles), np.mean(acc_d_eyes), np.mean(acc_d_ears), \
        np.mean(acc_d_upper_body), np.mean(acc_d_lower_body)

def calc_MPJPE(pred_or, gt_or, procrustes_transform=True): 
    '''
    pred: (N, K, 3) 
    gt: (N, K, 3)
    '''


    pred = pred_or.cpu().detach().numpy()
    gt = gt_or.cpu().detach().numpy()

    pred = undiscretize(pred, 0, 1)
    gt = undiscretize(gt, 0, 1)

    pred = pred.squeeze(3)
    gt = gt.squeeze(3)

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    N, K = pred.shape[0], pred.shape[1] # [BS, 15, 3]

    if(procrustes_transform):
        for i, p in enumerate(pred):
            d, Z, tform = procrustes(gt[i], pred[i])
            pred[i] = Z

    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2)) * 100
        
    err_dist_neck = np.sqrt(np.sum((pred[:,[0],:] - gt[:,[0],:])**2, axis=2)) * 100
        
    err_dist_nose = np.sqrt(np.sum((pred[:,[1],:] - gt[:,[1],:])**2, axis=2)) * 100
        
    err_dist_bodycenter = np.sqrt(np.sum((pred[:,[2],:] - gt[:,[2],:])**2, axis=2)) * 100
        
    err_dist_shoulders = np.sqrt(np.sum((pred[:,[3,9],:] - gt[:,[3,9],:])**2, axis=2)) * 100
        
    err_dist_elbows = np.sqrt(np.sum((pred[:,[4,10],:] - gt[:,[4,10],:])**2, axis=2)) * 100
        
    err_dist_wrists = np.sqrt(np.sum((pred[:,[5,11],:] - gt[:,[5,11],:])**2, axis=2)) * 100
        
    err_dist_hips = np.sqrt(np.sum((pred[:,[6,12],:] - gt[:,[6,12],:])**2, axis=2)) * 100
        
    err_dist_knees = np.sqrt(np.sum((pred[:,[7,13],:] - gt[:,[7,13],:])**2, axis=2)) * 100
        
    err_dist_ankles = np.sqrt(np.sum((pred[:,[8,14],:] - gt[:,[8,14],:])**2, axis=2))* 100
        
    err_dist_eyes = np.sqrt(np.sum((pred[:,[15,17],:] - gt[:,[15,17],:])**2, axis=2)) * 100
        
    err_dist_ears = np.sqrt(np.sum((pred[:,[16,18],:] - gt[:,[16,18],:])**2, axis=2)) * 100
        
    err_dist_upper_body = np.sqrt(np.sum((pred[:,[0,1,2,3,4,5,9,10,11,15,16,17,18],:] - gt[:,[0,1,2,3,4,5,9,10,11,15,16,17,18],:])**2, axis=2)) * 100
        
    err_dist_lower_body = np.sqrt(np.sum((pred[:,[6,7,8,12,13,14],:] - gt[:,[6,7,8,12,13,14],:])**2, axis=2))  * 100

    return err_dist, err_dist_neck, err_dist_nose, err_dist_bodycenter, \
        err_dist_shoulders, err_dist_elbows, err_dist_wrists, err_dist_hips, \
        err_dist_knees, err_dist_ankles, err_dist_eyes, err_dist_ears, \
        err_dist_upper_body, err_dist_lower_body

    # 0: Neck
    # 1: Nose
    # 2: BodyCenter (center of hips)
    # 3: lShoulder
    # 4: lElbow
    # 5: lWrist,
    # 6: lHip
    # 7: lKnee
    # 8: lAnkle
    # 9: rShoulder
    # 10: rElbow
    # 11: rWrist
    # 12: rHip
    # 13: rKnee
    # 14: rAnkle
    # 15: lEye
    # 16: lEar
    # 17: rEye
    # 18: rEar