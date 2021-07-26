import torch
import numpy as np
from torch.utils.data import Dataset

import os, glob
import re
import cv2
import math
from random import shuffle
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from PIL import Image
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D

import time
import open3d as o3d
from queue import Queue

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def rotate(xyz):
    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        num = dotproduct(v1, v2)
        den = (length(v1) * length(v2))
        if den == 0:
            print('den = 0')
            print(length(v1))
            print(length(v2))
            print(num)
        ratio = num/den
        ratio = np.minimum(1, ratio)
        ratio = np.maximum(-1, ratio)

        return math.acos(ratio)

    p1 = np.float32(xyz[1, :])
    p2 = np.float32(xyz[6, :])
    v1 = np.subtract(p2, p1)
    mod_v1 = np.sqrt(np.sum(v1 ** 2))
    x = np.float32([1., 0., 0.])
    y = np.float32([0., 1., 0.])
    z = np.float32([0., 0., 1.])
    theta = math.acos(np.sum(v1 * z) / (mod_v1 * 1)) * 360 / (2 * math.pi)
    # M = cv2.getAffineTransform()
    p = np.cross(v1, z)
    # if sum(p)==0:
    #     p = np.cross(v1,y)
    p[2] = 0.
    # ang = -np.minimum(np.abs(angle(p, x)), 2 * math.pi - np.abs(angle(p, x)))
    ang = angle(x, p)

    if p[1] < 0:
        ang = -ang

    M = [[np.cos(ang), np.sin(ang), 0.],
         [-np.sin(ang), np.cos(ang), 0.], [0., 0., 1.]]
    M = np.reshape(M, [3, 3])
    xyz = np.transpose(xyz)
    xyz_ = np.matmul(M, xyz)
    xyz_ = np.transpose(xyz_)

    return xyz_


def flip_3d(msk):
    msk[:, 1] = -msk[:, 1]
    return msk


def compute_distances(FLAGS, labels3D, predictions3D, labels2D, predictions2D, labelsD, predictionsD):
    ED_list_3d = torch.sum(torch.square(predictions3D - labels3D), dim=2)
    ED_3d = torch.mean(ED_list_3d)
    EDs_3d = torch.mean(torch.sqrt(ED_list_3d))

    ED_list_2d = torch.sum(torch.square(predictions2D - labels2D), dim=2)
    ED_2d = torch.mean(ED_list_2d)
    EDs_2d = torch.mean(torch.sqrt(ED_list_2d))

    # print("P3D: ", predictions3D.shape)
    # print("L3D: ", labels3D.shape)
    # print("P2D: ", predictions2D.shape)
    # print("L2D: ", labels2D.shape)

    # print(torch.max(labelsD))
    # print(torch.min(labelsD))
    # print(torch.max(predictionsD))
    # print(torch.min(predictionsD))
    valid_mask = (labelsD > 0).detach()
    diff = (labelsD - predictionsD).abs()
    diff_masked = diff[valid_mask]
    ED_D = (diff_masked.mean() + diff.mean()) / 2.

    # cv2.imshow("Predicted", predictionsD.clone()[0].permute(1,2,0).cpu().detach().numpy())
    # cv2.imshow("Real", labelsD.clone()[0].permute(1,2,0).cpu().detach().numpy())
    # cv2.imshow("Diff", diff.clone()[0].permute(1,2,0).cpu().detach().numpy())
    # cv2.waitKey(1)

    return ED_3d, ED_2d, EDs_3d, EDs_2d, ED_D


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform

def plot_skeletons(FLAGS, fig, images_orig, links, preds_2D, gts_2D, preds_3D, gts_3D, preds_D, gts_D, writer, angle):
    plt.rcParams.update({'axes.labelsize': 'small'})
    for index in range(0, FLAGS.batch_size):

        plt.clf()
        angle = (angle + 1) % 360

        ax_bb = fig.add_subplot(331)
        ax_bb.set_title('Input image')

        ax_hat_3D = fig.add_subplot(338, projection='3d')
        ax_hat_3D.set_title('3D prediction')
        ax_hat_3D.set_xlabel('X')
        ax_hat_3D.set_ylabel('Y')
        ax_hat_3D.set_zlabel('Z')
        ax_hat_3D.set_xlim([-100, 100])
        ax_hat_3D.set_ylim([-100, 100])
        ax_hat_3D.set_zlim([-100, 100])
        ax_hat_3D.view_init(15, angle)
        ax_hat_3D.labelsize = 10

        ax_gt_3D = fig.add_subplot(339, projection='3d')
        ax_gt_3D.set_title('3D ground truth')
        ax_gt_3D.set_xlabel('X')
        ax_gt_3D.set_ylabel('Y')
        ax_gt_3D.set_zlabel('Z')
        ax_gt_3D.set_xlim([-100, 100])
        ax_gt_3D.set_ylim([-100, 100])
        ax_gt_3D.set_zlim([-100, 100])
        ax_gt_3D.view_init(15, angle)

        ax_hat_2D = fig.add_subplot(335)
        ax_hat_2D.set_title('2D prediction')
        ax_hat_2D.set_xlabel('X')
        ax_hat_2D.set_ylabel('Y')
        ax_hat_2D.set_xlim([0, 1])
        ax_hat_2D.set_ylim([0, 1])

        ax_gt_2D = fig.add_subplot(336)
        ax_gt_2D.set_title('2D ground truth')
        ax_gt_2D.set_xlabel('X')
        ax_gt_2D.set_ylabel('Y')
        ax_gt_2D.set_xlim([0, 1])
        ax_gt_2D.set_ylim([0, 1])

        ax_hat_D = fig.add_subplot(332)
        ax_hat_D.set_title('Depth prediction')

        ax_gt_D = fig.add_subplot(333)
        ax_gt_D.set_title('Depth ground truth')

        ax_bb.imshow(np.reshape(
            images_orig[index], (FLAGS.input_height, FLAGS.input_width, FLAGS.n_channels)))
        colormaps = [
            'Greys_r', 'Purples_r', 'Blues_r', 'Greens_r', 'Oranges_r', 'Reds_r',
            'YlOrBr_r', 'YlOrRd_r', 'OrRd_r', 'PuRd_r', 'RdPu_r', 'BuPu_r',
            'GnBu_r', 'PuBu_r', 'YlGnBu_r', 'PuBuGn_r', 'BuGn_r', 'YlGn_r']


        for i in range(len(links)):

            link = links[i]

            for j in range(len(link)):
                P2_hat_3D = preds_3D[index][i, :]
                P1_hat_3D = preds_3D[index][link[j], :]
                link_hat_3D = [list(x)
                               for x in list(zip(P1_hat_3D, P2_hat_3D))]
                ax_hat_3D.plot(
                    link_hat_3D[0], link_hat_3D[2], zs=[ -x for x in link_hat_3D[1]])
                P2_gt_3D = gts_3D[index][i, :]
                P1_gt_3D = gts_3D[index][link[j], :]
                link_gt_3D = [list(x) for x in list(zip(P1_gt_3D, P2_gt_3D))]
                ax_gt_3D.plot(link_gt_3D[0], link_gt_3D[2], zs=[ -x for x in link_gt_3D[1]])

                P2_hat_2D = preds_2D[index][i, :]
                P1_hat_2D = preds_2D[index][link[j], :]
                link_hat_2D = [list(x)
                               for x in list(zip(P1_hat_2D, P2_hat_2D))]
                ax_hat_2D.plot(
                    link_hat_2D[0], link_hat_2D[1])
                P2_gt_2D = gts_2D[index][i, :]
                P1_gt_2D = gts_2D[index][link[j], :]
                link_gt_2D = [list(x) for x in list(zip(P1_gt_2D, P2_gt_2D))]
                ax_gt_2D.plot(link_gt_2D[0], link_gt_2D[1])

                ax_gt_D.imshow(gts_D[index])
                # ax_hat_D.imshow(preds_D[index].cpu())
                ax_hat_D.imshow(preds_D[index])

        plt.draw()
        fig.canvas.flush_events()
        plt.show(block=False)

        writer.grab_frame()

    return angle


def eval_image(model):
    viewpoint = "top"
    sample = "05_00000000_rear"
    image = cv2.imread("/media/disi/New Volume/Datasets/PANOPTIC_CAPS/"+viewpoint+"/train/"+ sample +".png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            Standardize()])
    image = transform(image)
    image_tensor = image.unsqueeze(0)
    # image_tensor = image_tensor.permute(0,3,1,2)
    input = torch.autograd.Variable(image_tensor)
    input = input.cuda()
    input = torch.cat(128*[input])
    print("INPUT SHAPE: ", input.shape)
    yhat2D, yhat3D, yhatD, W_reg, _ = model(input)

    itop_labels = ['Head','Neck','LShould','RShould',"LElbow","RElbow","LHand","RHand","Torso","LHip","RHip","LKnee","RKnee","LFoot","RFoot"]
    
    import gzip
    msk3D = np.load("/media/disi/New Volume/Datasets/PANOPTIC_CAPS/"+viewpoint+"/train/"+sample+".npy")
    msk3D = torch.from_numpy(msk3D).float().unsqueeze(0).unsqueeze(-1)
    msk3D = torch.cat(128*[msk3D]) / 100.
    msk3D = center_skeleton(msk3D)
    msk3D = discretize(msk3D, 0, 1)
    print(msk3D.shape)

    pred = yhat3D.cpu().detach().numpy().squeeze(-1)
    gt = msk3D.cpu().detach().numpy().squeeze(-1)

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    msk3D = msk3D.squeeze(3)
    yhat3D = yhat3D.squeeze(3)
    for i, p in enumerate(pred):
        d, Z, tform = procrustes(
                            gt[i], pred[i])
        pred[i] = Z

    print(yhat3D.shape)
    print(pred.shape)

    yhat3D = torch.from_numpy(pred).float()

    # if(viewpoint=="top"):
    msk3D = msk3D[:,:,[2,0,1]]
    yhat3D = yhat3D[:,:,[2,0,1]]
    
    print("GT: ", msk3D.shape)
    print("PRED: ", yhat3D.shape)

    print("ERROR: ", np.mean(np.sqrt(np.sum((yhat3D.cpu().detach().numpy() - msk3D.cpu().detach().numpy())**2, axis=2))))

    save_3d_plot(msk3D, "gt_depth", display_labels=True, viewpoint=viewpoint)
    save_3d_plot(yhat3D.cpu().detach().numpy(), "pred_depth", viewpoint=viewpoint)
    
    index = 10
    image_2d = input[index].permute(1,2,0).cpu().detach().numpy()
    # # img_kps = np.zeros((256,256,3), np.uint8)
    # img_kps = cv2. cvtColor(image_2d, cv2.COLOR_GRAY2BGR)#.astype(np.uint8)
    # for i, kps in enumerate(yhat2D[index]): # (15,2,1)
    #     if(i == 8):
    #         color = (255,0,0)
    #     else:
    #         color = (0,255,0)
    #     cv2.circle(img_kps, (int(256*kps[0].cpu()), int(256*kps[1].cpu())), 2, color, 8, 0)
    #     # cv2.putText(img_kps, itop_labels[i], (int(256*kps[0].cpu()) + 10, int(256*kps[1].cpu())), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    # cv2.imshow("Kps", img_kps)
    cv2.imshow("Input", image_2d)
    cv2.waitKey(0)

def save_3d_plot(itop, name, azim=None, elev=None, gt=None, display_labels=False, viewpoint="top"):
    # itop_labels = ['Head','Neck','RShould','LShould',"RElbow","LElbow","RHand","LHand","Torso","RHip","LHip","RKnee","LKnee","RFoot","LFoot"]
    itop_labels = ['Head','Neck','LShould','RShould',"LElbow","RElbow","LHand","RHand","Torso","LHip","RHip","LKnee","RKnee","LFoot","RFoot"]
    itop_labels = ['0','1','2','3',"4","5","6","7","8","9","10","11","12","13","14"]

    itop_connections = [[0,1],[1,2],[1,3],[2,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,10],[9,11],[10,12],[11,13],[12,14]]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    index = 10

    itop_newjoints = change_format_from_19_joints_to_15_joints(itop[0])
    itop_newjoints = np.expand_dims(itop_newjoints, 0)
    itop = np.repeat(itop_newjoints, 128, axis=0)
    # print(itop.shape)


    xdata = itop[index,:,0].flatten()
    ydata = itop[index,:,1].flatten()
    zdata = itop[index,:,2].flatten()


    for i in itop_connections:    
        x1,x2,y1,y2,z1,z2 = connect(xdata,ydata,zdata,i[0],i[1])
        ax.plot([x1,x2],[y1,y2],[z1,z2],'k-')

    ax.scatter3D(xdata, ydata, zdata, c=zdata)

    if(gt is not None):
        pred = undiscretize(itop, 0, 1)[index]
        gt = undiscretize(gt, 0, 1)[index]

        pred = pred.squeeze()
        gt = gt.squeeze()

        assert(pred.shape == gt.shape)
        assert(len(pred.shape) == 2)

        err_dist = np.sqrt(np.sum((pred - gt)**2, axis=1))  # (N, K)

        errors = (err_dist < 0.1)

    for i, (x, y, z, label) in enumerate(zip(xdata,ydata,zdata, itop_labels)):
        error_color='black'
        if(gt is not None and not errors[i]):
            error_color='red'
        if(display_labels):
            ax.text(x, y, z, label, color=error_color)

    # ax.text2D(0.05, 0.95, "ITOP", transform=ax.transAxes)

    if(azim):
        ax.view_init(elev=elev, azim=azim)

    # ax.set_xlabel('x', rotation=0, fontsize=20, labelpad=20)
    # ax.set_ylabel('y', rotation=0, fontsize=20, labelpad=20)
    # ax.set_zlabel('z', rotation=0, fontsize=20, labelpad=20)

    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(-2,2)
    # ax.set_zlim3d(0,2)

    ax.set_xlim3d(0.2,1)
    ax.set_ylim3d(0,0.6)
    ax.set_zlim3d(0.8,0.2) 

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
    cv2.imshow(name, img)
    cv2.imwrite(name+".png", img)
    if(name=="True side"):
        cv2.waitKey(1)
    else:
        cv2.waitKey(1)

def connect(x,y,z,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1,z2 = z[p1],z[p2]
    return x1,x2,y1,y2,z1,z2


def center_skeleton(skeletons):
    for b, batch in enumerate(skeletons):
        skeletons[b,:,:] = skeletons[b,:,:] - skeletons[b,2,:]
    return skeletons

def change_format_from_19_joints_to_15_joints(joints):
    xdata = joints[:,0]
    ydata = joints[:,1]
    zdata = joints[:,2]

    panoptic_head = [(xdata[16]+xdata[18])/2,(ydata[16]+ydata[18])/2,(zdata[16]+zdata[18])/2]
    panoptic_torso = [(xdata[0]+xdata[2])/2,(ydata[0]+ydata[2])/2,(zdata[0]+zdata[2])/2]


    #                           head        neck      r shoulder l shoulder r elbow  l elbow     r hand    l hand      torso            r hip       l hip    r knee     l knee    r foot    l foot
    #xdata_new = np.array([panoptic_head[0], xdata[0], xdata[9], xdata[3], xdata[10], xdata[4], xdata[11], xdata[5], panoptic_torso[0], xdata[12], xdata[6], xdata[13], xdata[7], xdata[14], xdata[8]])
    #ydata_new = np.array([panoptic_head[1], ydata[0], ydata[9], ydata[3], ydata[10], ydata[4], ydata[11], ydata[5], panoptic_torso[1], ydata[12], ydata[6], ydata[13], ydata[7], ydata[14], ydata[8]])
    #zdata_new = np.array([panoptic_head[2], zdata[0], zdata[9], zdata[3], zdata[10], zdata[4], zdata[11], zdata[5], panoptic_torso[2], zdata[12], zdata[6], zdata[13], zdata[7], zdata[14], zdata[8]])

    xdata_new = np.array([panoptic_head[0], xdata[0], xdata[3], xdata[9], xdata[4], xdata[10], xdata[5], xdata[11], panoptic_torso[0], xdata[6], xdata[12], xdata[7], xdata[13], xdata[8], xdata[14]])
    ydata_new = np.array([panoptic_head[1], ydata[0], ydata[3], ydata[9], ydata[4], ydata[10], ydata[5], ydata[11], panoptic_torso[1], ydata[6], ydata[12], ydata[7], ydata[13], ydata[8], ydata[14]])
    zdata_new = np.array([panoptic_head[2], zdata[0], zdata[3], zdata[9], zdata[4], zdata[10], zdata[5], zdata[11], panoptic_torso[2], zdata[6], zdata[12], zdata[7], zdata[13], zdata[8], zdata[14]])

    panoptic_converted = np.empty(shape=(15, 3), dtype=float)
    for index in range(len(panoptic_converted)):
        panoptic_converted[index,0] = xdata_new[index]
        panoptic_converted[index,1] = ydata_new[index]
        panoptic_converted[index,2] = zdata_new[index]

    return panoptic_converted

def discretize(coord, a, b):
    
    normalizers_3D = [[-0.927149999999999, 1.4176299999999982], [-1.1949180000000008, 0.991252999999999], [-0.8993889999999993, 0.8777908000000015]]

    for i in range(3):
        coord[:,:,i] = (b - a) * (coord[:,:,i] - normalizers_3D[i][0]) / (normalizers_3D[i][1] - normalizers_3D[i][0]) + a

    return coord

def undiscretize(coord, a, b):
    
    normalizers_3D = [[-0.927149999999999, 1.4176299999999982], [-1.1949180000000008, 0.991252999999999], [-0.8993889999999993, 0.8777908000000015]]

    for i in range(3):
        coord[:,:,i] = ( (coord[:,:,i] - a) * (normalizers_3D[i][1] - normalizers_3D[i][0]) / (b - a) ) + normalizers_3D[i][0]

    return coord