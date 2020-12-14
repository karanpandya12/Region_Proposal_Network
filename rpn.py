import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), anchors_param=dict(ratio=0.8,scale= 256, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv5 = nn.Conv2d(128, 256, 5, padding=2)

        self.bnorm1 = nn.BatchNorm2d(16)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.bnorm3 = nn.BatchNorm2d(64)
        self.bnorm4 = nn.BatchNorm2d(128)
        self.bnorm5 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, stride=2)

        # TODO  Define Intermediate Layer

        self.inter_conv = nn.Conv2d(256,256,3, padding=1)
        self.inter_bnorm = nn.BatchNorm2d(256)

        # TODO  Define Proposal Classifier Head

        self.class_conv = nn.Conv2d(256, 1, 1)
        self.sigmoid = nn.Sigmoid()

        # TODO Define Proposal Regressor Head

        self.reg_conv = nn.Conv2d(256, 4, 1)

        #  find anchors
        self.anchors_param = anchors_param
        self.anchors = self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict = {}

    def forward(self, X):
        # Forward  the input through the backbone the intermediate layer and the RPN heads
        # Input:
        #       X: (bz,3,image_size[0],image_size[1])}
        # Ouput:
        #       logits: (bz,1,grid_size[0],grid_size[1])}
        #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}

        #TODO forward through the Backbone

        X = self.forward_backbone(X)

        #TODO forward through the Intermediate layer

        X = self.relu(self.inter_bnorm(self.inter_conv(X)))

        #TODO forward through the Classifier Head

        logits = self.sigmoid(self.class_conv(X))

        #TODO forward through the Regressor Head

        bbox_regs = self.reg_conv(X)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs

    def forward_backbone(self,X):
        # Forward input batch through the backbone
        # Input:
        #       X: (bz,3,image_size[0],image_size[1])}
        # Ouput:
        #       X: (bz,256,grid_size[0],grid_size[1])
        #####################################
        # TODO forward through the backbone
        #####################################
        X = self.max_pool(self.relu(self.bnorm1(self.conv1(X))))
        X = self.max_pool(self.relu(self.bnorm2(self.conv2(X))))
        X = self.max_pool(self.relu(self.bnorm3(self.conv3(X))))
        X = self.max_pool(self.relu(self.bnorm4(self.conv4(X))))
        X = self.relu(self.bnorm5(self.conv5(X)))

        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X

    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        # This function creates the anchor boxes
        # Output:
        #       anchors: (grid_size[0],grid_size[1],4)

        anchors = torch.zeros((grid_sizes[0], grid_sizes[1], 4), device=self.device)

        for i in range(grid_sizes[0]):
            anchors[i,:,1] = stride*i + stride/2

        for i in range(grid_sizes[1]):
            anchors[:,i,0] = stride*i + stride/2

        anchors[:,:,2] = torch.sqrt((scale**2)*aspect_ratio)
        anchors[:,:,3] = torch.sqrt((scale**2)/aspect_ratio)

        assert anchors.shape == (grid_sizes[0], grid_sizes[1], 4)

        return anchors

    def get_anchors(self):
        return self.anchors

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        # This function creates the ground truth for a batch of images by using
        # create_ground_truth internally
        # Input:
        #      bboxes_list: list:len(bz){(n_obj,4)}
        #      indexes:      list:len(bz)
        #      image_shape:  tuple:len(2)
        # Output:
        #      ground_class: (bz,1,grid_size[0],grid_size[1])
        #      ground_coord: (bz,4,grid_size[0],grid_size[1])
        #####################################
        # TODO create ground truth for a batch of images
        #####################################

        ground_class_list, ground_coord_list = self.MultiApply(self.create_ground_truth, bboxes_list, indexes, \
                                                               grid_sizes=(self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1]), \
                                                               anchors=self.anchors, image_size=image_shape)
        ground_class = torch.stack(ground_class_list)
        ground_coord = torch.stack(ground_coord_list)

        assert ground_class.shape[1:4] == (1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4] == (4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_class, ground_coord

    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        # This function creates the ground truth for one image
        # It also caches the ground truth for the image using its index
        # Input:
        #       bboxes:      (n_boxes,4)
        #       index:       scalar (the index of the image in the total dataset used for caching)
        #       grid_size:   tuple:len(2)
        #       anchors:     (grid_size[0],grid_size[1],4)
        # Output:
        #       ground_class:  (1,grid_sizes[0],grid_sizes[1])
        #       ground_coord: (4,grid_sizes[0],grid_sizes[1])

        ground_class = (-1)*torch.ones(grid_sizes[0], grid_sizes[1], device=self.device)

        x1 = anchors[:,:,0] - (anchors[:,:,2]/2)
        y1 = anchors[:,:,1] - (anchors[:,:,3]/2)
        x2 = anchors[:,:,0] + (anchors[:,:,2]/2)
        y2 = anchors[:,:,1] + (anchors[:,:,3]/2)

        iou_mat = IOU(torch.stack((x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)), dim=1), bboxes)

        pos_check = torch.sum(iou_mat > 0.7, dim=1, dtype=bool).view(grid_sizes[0], grid_sizes[1])
        max_values, max_ids = torch.max(iou_mat, dim=0)
        max_check = torch.sum(iou_mat >= 0.99*max_values, dim=1, dtype=bool).view(grid_sizes[0], grid_sizes[1])

        pos_check = torch.logical_or(pos_check, max_check)

        ground_class[pos_check] = 1

        neg_check = torch.logical_and(torch.logical_not(pos_check), torch.sum(iou_mat < 0.3, dim=1).view(grid_sizes[0], grid_sizes[1]) == bboxes.shape[0])
        ground_class[neg_check] = 0

        ground_class[torch.logical_or(torch.logical_or(x1 < 0, y1 < 0), \
                                      torch.logical_or(x2 > image_size[1], y2 > image_size[0]))] = -1

        ground_class = ground_class.unsqueeze(0)

        ground_coord = torch.zeros(4, grid_sizes[0], grid_sizes[1], device=self.device)

        max_gt_ids = torch.argmax(iou_mat, dim=1).view(grid_sizes[0], grid_sizes[1])

        ground_corners = bboxes[max_gt_ids]

        ground_coord[0,:,:] = (ground_corners[:,:,0] + ground_corners[:,:,2])/2
        ground_coord[1,:,:] = (ground_corners[:,:,1] + ground_corners[:,:,3])/2
        ground_coord[2,:,:] = ground_corners[:,:,2] - ground_corners[:,:,0]
        ground_coord[3,:,:] = ground_corners[:,:,3] - ground_corners[:,:,1]

        anchors = anchors.permute(2,0,1)

        ground_coord[0,:,:] = (ground_coord[0,:,:] - anchors[0,:,:])/anchors[2,:,:]
        ground_coord[1,:,:] = (ground_coord[1,:,:] - anchors[1,:,:])/anchors[3,:,:]
        ground_coord[2,:,:] = torch.log(ground_coord[2,:,:]/anchors[2,:,:])
        ground_coord[3,:,:] = torch.log(ground_coord[3,:,:]/anchors[3,:,:])

        ground_coord[:, torch.logical_not(pos_check)] = 0

        self.ground_dict[key] = (ground_class, ground_coord)

        assert ground_class.shape == (1,grid_sizes[0],grid_sizes[1])
        assert ground_coord.shape == (4,grid_sizes[0],grid_sizes[1])

        return ground_class, ground_coord

    def loss_class(self,pred_class, targ_class):
        # Compute the loss of the classifier
        # Input:
        #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
        #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels

        # TODO compute classifier's loss

        criterion = nn.BCELoss(reduction='sum')

        loss = criterion(pred_class, targ_class)

        return loss

    def loss_reg(self,pos_out_r,pos_target_coord):
        # Compute the loss of the regressor
        # Input:
        # pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
        # pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
        #torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss

        criterion = nn.SmoothL1Loss(reduction='sum')
        loss = criterion(pos_out_r, pos_target_coord)

        return loss

    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=1, effective_batch=50):
        # Compute the total loss
        # Input:
        #       clas_out: (bz,1,grid_size[0],grid_size[1])
        #       regr_out: (bz,4,grid_size[0],grid_size[1])
        #       targ_clas:(bz,1,grid_size[0],grid_size[1])
        #       targ_regr:(bz,4,grid_size[0],grid_size[1])
        #       l: lambda constant to weight between the two losses
        #       effective_batch: the number of anchors in the effective batch (M in the handout)
        #############################
        # TODO compute the total loss
        #############################

        clas_out = clas_out.view(-1, 1)
        targ_clas = targ_clas.view(-1, 1)
        regr_out = regr_out.view(-1, 4)
        targ_regr = targ_regr.view(-1, 4)

        pos_check = (targ_clas == 1).squeeze()
        neg_check = (targ_clas == 0).squeeze()

        targ_clas_pos = targ_clas[pos_check]
        targ_clas_neg = targ_clas[neg_check]

        clas_out_pos = clas_out[pos_check]
        clas_out_neg = clas_out[neg_check]

        regr_out_pos = regr_out[pos_check]

        targ_regr_pos = targ_regr[pos_check]

        pos_num = min(int(effective_batch/2), targ_clas_pos.shape[0])
        neg_num = effective_batch - pos_num

        pos_sample_ids = np.random.choice(targ_clas_pos.shape[0], pos_num, replace=False)
        neg_sample_ids = np.random.choice(targ_clas_neg.shape[0], neg_num, replace=False)

        pred_class = torch.cat((clas_out_pos[pos_sample_ids], clas_out_neg[neg_sample_ids]))
        gt_class = torch.cat((targ_clas_pos[pos_sample_ids], targ_clas_neg[neg_sample_ids]))
        pred_regr_pos = regr_out_pos[pos_sample_ids]
        gt_regr_pos = targ_regr_pos[pos_sample_ids]

        loss_c = self.loss_class(pred_class, gt_class)/effective_batch

        loss_r = self.loss_reg(pred_regr_pos, gt_regr_pos)/effective_batch

        loss = loss_c + l*loss_r

        return loss, loss_c, loss_r


    def NMS(self,clas,prebox,method='gauss',gauss_sigma=0.5):
        ##################################
        # TODO perform NSM
        ##################################
        # Input:
        #       clas: (top_k_boxes) (scores of the top k boxes)
        #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
        # Output:
        #       nms_clas: (Post_NMS_boxes)
        #       nms_prebox: (Post_NMS_boxes,4)

        ious = IOU(prebox,prebox).triu(diagonal=1)

        ious_cmax, ids = torch.max(ious, dim=0)
        ious_cmax = ious_cmax.expand(ious_cmax.shape[0], ious_cmax.shape[0]).T

        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)

        decay, ids = torch.min(decay, dim=0)

        nms_clas = clas*decay

        return nms_clas

# if __name__=="__main__":
