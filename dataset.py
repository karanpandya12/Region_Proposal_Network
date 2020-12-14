import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.patches as patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        #############################################
        # Initialize  Dataset
        #############################################

        image_set = h5py.File(paths[0], 'r')
        mask_set = h5py.File(paths[1], 'r')
        self.labels = np.load(paths[2], allow_pickle=True)
        self.bboxes = np.load(paths[3], allow_pickle=True)

        self.images = image_set['data']
        masks = mask_set['data']

        self.masks = []
        j = 0
        for i in range(self.images.shape[0]):
            num_obj = self.labels[i].shape[0]
            self.masks.append(masks[j:j + num_obj])
            j += num_obj

        self.masks = np.array(self.masks, dtype=object)

        pass

    def __getitem__(self, index):
        ################################
        # Return transformed images,labels,masks,boxes,index
        ################################
        # In this function for given index we rescale the image and the corresponding  masks, boxes
        # and we return them as output
        # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index

        transed_img, transed_mask, transed_bbox = self.pre_process_batch(self.images[index], self.masks[index],
                                                                         self.bboxes[index])
        label = self.labels[index]
        label = torch.tensor(label, device=device)

        # assert transed_img.shape == (3,800,1088)
        # assert transed_bbox.shape[0] == transed_mask.shape[0]
        
        return transed_img, label, transed_mask, transed_bbox, index

    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # Apply the correct transformation to the images,masks,boxes
        ######################################
        # This function preprocesses the given image, mask, box by rescaling them appropriately
        # output:
        #        img: (3,800,1088)
        #        mask: (n_box,800,1088)
        #        box: (n_box,4)

        img = img / 255.0

        img = torch.tensor(img, device=device, dtype=torch.float32)
        mask = torch.tensor(mask.astype(float), device=device)
        bbox = torch.tensor(bbox, device=device)

        img = F.interpolate(img.unsqueeze(dim=0), size=(800, 1066))
        img = img.squeeze(dim=0)

        mask = F.interpolate(mask.unsqueeze(dim=0), size=(800, 1066))
        mask = mask.squeeze(dim=0)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = transforms.functional.normalize(img, mean, std)

        img = F.pad(img, (11, 11))
        mask = F.pad(mask, (11, 11))

        bbox[:, [0, 2]] = ((bbox[:, [0, 2]] / 400) * 1066) + 11
        bbox[:, [1, 3]] = (bbox[:, [1, 3]] / 300) * 800

        # assert img.squeeze(0).shape == (3, 800, 1088)
        # assert bbox.shape[0] == mask.squeeze(0).shape[0]

        return img, mask, bbox
    
    def __len__(self):
        return len(self.images)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def collect_fn(self, batch):
        # output:
        #  dict{images: (bz, 3, 800, 1088)
        #       labels: list:len(bz)
        #       masks: list:len(bz){(n_obj, 800,1088)}
        #       bbox: list:len(bz){(n_obj, 4)}
        #       index: list:len(bz)

        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        index_list = []

        for transed_img, label, transed_mask, transed_bbox, index in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
            index_list.append(index)

        return (torch.stack(transed_img_list, dim=0)), label_list, transed_mask_list, transed_bbox_list, index_list

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


if __name__ == '__main__':
    # file path and make a list
    imgs_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    # Randomly split the dataset into training and testset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # push the randomized training data into the dataloader
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    # Define parameters for anchors
    anchors_param = dict(ratio=torch.tensor([0.7852]), scale=torch.tensor([360]), grid_size=(50, 68), stride=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## initilize device to cpu or cuda

    # Initialize the network
    rpn_net = RPNHead(device=device, anchors_param=anchors_param)
    rpn_net.to(device)

    # Define colour lists for mask and bounding box
    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # mask_color_list = ["Reds", "Blues", "Greens", "Oranges", "Purples"]
    box_color_list = ["Blue", "Green", "Red", "Orange", "Purple"]


    for i,batch in enumerate(train_loader,0):
        images, label, mask, boxes, indexes = batch
        gt, ground_coord = rpn_net.create_batch_truth(boxes, indexes, images.shape[-2:])

        images = images[0,:,:,:]
        boxes = boxes[0]
        label = label[0]
        # Flatten the ground truth and the anchors
        flatten_coord, flatten_gt, flatten_anchors = output_flattening(ground_coord, gt, rpn_net.get_anchors())
        
        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord = output_decoding(flatten_coord, flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        plt.figure(1)
        ax1 = plt.gca()
        plt.imshow(images.permute(1, 2, 0))

        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax1.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax1.add_patch(rect)

        plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/gt_images_decode/gt_"+str(i)+".png")

        # Plot the images and masks from the dataset
        # plt.figure(2)
        # ax2 = plt.gca()
        # plt.imshow(images.permute(1, 2, 0))
        #
        # mask = mask[0].cpu()
        #
        # for j in range(len(boxes)):
        #     rect = patches.Rectangle((boxes[j][0], boxes[j][1]), boxes[j][2] - boxes[j][0], boxes[j][3] - boxes[j][1], fill=False,
        #                              color=box_color_list[label[j]-1])
        #     ax2.add_patch(rect)
        #
        #     m = np.ma.masked_where(mask[j] == 0, mask[j])
        #     plt.imshow(m, cmap=mask_color_list[label[j] - 1], alpha=0.7)
        #
        # plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/data_images/data_"+str(i)+".png")

        plt.show()
 
        if(i>20):
            break
