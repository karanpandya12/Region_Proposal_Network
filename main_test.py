import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import tqdm
import os
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from rpn import *


import torchvision

if __name__=="__main__":

    imgs_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)


    # Build the dataloader
    # Set 20% of the dataset as the training data
    batch_size = 16

    full_size = len(dataset)
    train_size = int(full_size * 0.8)  # DO NOT CHANGE
    test_size = full_size - train_size
    
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # push the randomized training data into the dataloader
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## initilize device to cpu or cuda

    anchors_param = dict(ratio=torch.tensor([0.7852]), scale=torch.tensor([360]), grid_size=(50, 68), stride=16)

    rcnn_model = RPNHead(device=device, anchors_param=anchors_param)
    rcnn_model.to(device)
    rcnn_model.eval()

    path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Mask_RCNN', 'saved_model',
                        'rcnn_epoch_39')

    checkpoint = torch.load(path)
    rcnn_model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize the variables
    pw_acc = 0
    pre_NMS_num = 50
    keep_num = 5

    plot_post_NMS = True

    for iter, data in enumerate(test_loader, 0):
        img, label, mask, bbox, index = data

        # del img
        # torch.cuda.empty_cache()

        clas_out, regr_out = rcnn_model.forward(img)

        targ_clas, targ_regr = rcnn_model.create_batch_truth(bbox, index, img.shape[-2:])

        hard_pred = torch.where(clas_out >= 0.5, torch.tensor([1.], device=device), torch.tensor([0.], device=device))
        hard_gt = torch.where(targ_clas == -1, torch.tensor([0.], device=device), targ_clas)

        pw_acc += torch.sum(hard_gt == hard_pred)/float(torch.numel(hard_gt))

        if iter==0:
            # flat_scores = clas_out.view(batch_size, -1)
            # flat_coords = regr_out.view(batch_size, 4, -1)
            # flat_anchors = rcnn_model.anchors.permute(2, 0, 1).view(4, -1)

            for j in range(batch_size):
                image = img[j]
                scores = clas_out[j]
                coords = regr_out[j]

                # Flatten the ground truth and the anchors
                flat_coords, flat_scores, flat_anchors = output_flattening(coords.unsqueeze(0), scores.unsqueeze(0), rcnn_model.anchors)

                indices = torch.argsort(flat_scores, descending=True)
                top_scores = flat_scores[indices[:pre_NMS_num]]
                top_coords = flat_coords[indices[:pre_NMS_num], :]
                top_anchors = flat_anchors[indices[:pre_NMS_num], :]

                # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
                boxes_to_plot = output_decoding(top_coords, top_anchors)

                boxes_to_plot[:,[0,2]] = torch.clamp(boxes_to_plot[:,[0,2]], min=0, max=image.shape[2]-1)
                boxes_to_plot[:,[1,3]] = torch.clamp(boxes_to_plot[:,[1,3]], min=0, max=image.shape[1]-1)

                if plot_post_NMS:

                    NMS_clas = rcnn_model.NMS(top_scores, boxes_to_plot)

                    indices = torch.argsort(NMS_clas, descending=True)
                    boxes_to_plot = boxes_to_plot[indices[:keep_num], :]


                # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
                image = transforms.functional.normalize(image,
                                                         [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                         [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
                image=image.cpu()
                plt.figure(j)
                ax1 = plt.gca()
                plt.imshow(image.permute(1, 2, 0))
                title = 'Pre NMS Top ' + str(pre_NMS_num)
                if plot_post_NMS:
                    title = 'Post NMS Top ' + str(keep_num)

                plt.title(title)

                for elem in range(boxes_to_plot.shape[0]):
                    coord = boxes_to_plot[elem, :].view(-1)

                    col = 'b'
                    rect = patches.Rectangle((coord[0], coord[1]), coord[2] - coord[0], coord[3] - coord[1], fill=False,
                                             color=col)
                    ax1.add_patch(rect)

                plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/post_NMS_images/prop_" + str(j) + ".png")
                plt.close('all')

                del image,scores,coords, flat_coords,flat_scores,flat_anchors,indices,top_anchors,top_coords,boxes_to_plot,coord,NMS_clas
                torch.cuda.empty_cache()

        # del bbox,label,mask
        # torch.cuda.empty_cache()

        del img, bbox, label, mask, index, clas_out, regr_out, targ_clas, targ_regr

        # del fpn_feat_list,cate_pred_list,ins_pred_list,ins_gts_list,\
        # ins_ind_gts_list,cate_gts_list,cate_loss,mask_loss,total_loss
        torch.cuda.empty_cache()

        pass

    pw_acc /= (iter+1)
    print("Point-wise Test accuracy: ", pw_acc)
    print()

