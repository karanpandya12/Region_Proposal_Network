import numpy as np
import torch
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## initilize device to cpu or cuda


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

def IOU(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    # This function computes the IOU between two set of boxes
    # Input: boxA :(n,4); boxB: (m,4)
    # Output: iou:(n,m)

    x_top_left = torch.max(boxA[:,0].view(-1,1), boxB[:,0].view(1,-1))
    y_top_left = torch.max(boxA[:,1].view(-1,1), boxB[:,1].view(1,-1))
    x_bottom_right = torch.min(boxA[:,2].view(-1,1), boxB[:,2].view(1,-1))
    y_bottom_right = torch.min(boxA[:,3].view(-1,1), boxB[:,3].view(1,-1))

    intersection_w = torch.max(torch.tensor([0.], device=device), x_bottom_right-x_top_left)
    intersection_h = torch.max(torch.tensor([0.], device=device), y_bottom_right-y_top_left)

    intersection_area = intersection_h*intersection_w

    union_area = ((boxA[:,2]-boxA[:,0])*(boxA[:,3]-boxA[:,1])).view(-1,1)\
               + ((boxB[:,2]-boxB[:,0])*(boxB[:,3]-boxB[:,1])).view(1,-1) - intersection_area

    iou = intersection_area/(union_area + 0.0001)

    return iou

def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    # This function flattens the output of the network and the corresponding anchors
    # in the sense that it concatenates  the outputs and the anchors from all the grid cells
    # from all the images into 2D matrices
    # Each row of the 2D matrices corresponds to a specific anchor/grid cell
    # Input:
    #       out_r: (bz,4,grid_size[0],grid_size[1])
    #       out_c: (bz,1,grid_size[0],grid_size[1])
    #       anchors: (grid_size[0],grid_size[1],4)
    # Output:
    #       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
    #       flatten_clas: (bz*grid_size[0]*grid_size[1])
    #       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)

    bz = out_r.shape[0]
    flatten_regr = out_r.permute(0,2,3,1).reshape(-1, 4)
    flatten_clas = out_c.permute(0,2,3,1).reshape(-1)

    if bz>1:

        flatten_anchors = torch.stack((anchors,anchors), dim=0)

        for i in range(bz-2):
            flatten_anchors = torch.stack((flatten_anchors, anchors), dim=0)

    elif bz == 1:
        flatten_anchors = anchors.unsqueeze(0)

    flatten_anchors = flatten_anchors.view(-1, 4)

    return flatten_regr, flatten_clas, flatten_anchors


def output_decoding(flatten_out, flatten_anchors):
    #######################################
    # TODO decode the output
    #######################################
    # This function decodes the output that is given in the encoded format (defined in the handout)
    # into box coordinates where it returns the upper left and lower right corner of the proposed box
    # Input:
    #       flatten_out: (total_number_of_anchors*bz,4)
    #       flatten_anchors: (total_number_of_anchors*bz,4)
    # Output:
    #       box: (total_number_of_anchors*bz,4)

    box = torch.zeros(flatten_anchors.shape, device=device)
    x_s = flatten_out[:,0]*flatten_anchors[:,2] + flatten_anchors[:,0]
    y_s = flatten_out[:,1]*flatten_anchors[:,3] + flatten_anchors[:,1]
    w_s = torch.exp(flatten_out[:,2])*flatten_anchors[:,2]
    h_s = torch.exp(flatten_out[:,3])*flatten_anchors[:,3]

    box[:,0] = x_s - w_s/2
    box[:,1] = y_s - h_s/2
    box[:,2] = x_s + w_s/2
    box[:,3] = y_s + h_s/2

    return box
