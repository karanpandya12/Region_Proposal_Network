import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *
import tqdm
import os
import time

import torchvision

if __name__=="__main__":

    imgs_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "/content/drive/My Drive/CIS 680/Mask_RCNN/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    batch_size = 32

    full_size = len(dataset)
    train_size = int(full_size * 0.8)  # DO NOT CHANGE
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ## initilize device to cpu or cuda

    anchors_param = dict(ratio=torch.tensor([0.7852]), scale=torch.tensor([360]), grid_size=(50, 68), stride=16)

    rcnn_model = RPNHead(device=device, anchors_param=anchors_param)
    rcnn_model.to(device)

    num_epochs = 40  ## intialize this, atleast 36 epoch required for validation
    validation_total_loss = torch.zeros(num_epochs)
    validation_class_loss = torch.zeros(num_epochs)
    validation_regr_loss = torch.zeros(num_epochs)
    # avg_precision_values = torch.zeros(num_epochs)
    # learning_rate = (0.01 / 16) * batch_size
    learning_rate = 0.001
    start_epoch = 0

    # Intialize optimizer
    optimizer = torch.optim.Adam(rcnn_model.parameters(), lr=learning_rate)

    effective_batch = batch_size*40
    l = 5


    for epoch in tqdm.tqdm(range(start_epoch, num_epochs)):
        start_time = time.time()
        print("Epoch %d/%d" % (epoch + 1, num_epochs))

        path = os.path.join('.', 'drive', 'My Drive', 'CIS 680', 'Mask_RCNN', 'saved_final_model',
                            'rcnn_epoch_' + str(epoch))  # Set this to where the checkpoint is saved
        checkpoint = torch.load(path)
        rcnn_model.load_state_dict(checkpoint['model_state_dict'])

        rcnn_model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                # img, label, mask, bbox = [data[k] for k in range(len(data))]
                img, label, mask, bbox, index = data
                # label = [label_img.to(device) for label_img in label]
                # mask = [mask_img.to(device) for mask_img in mask]
                # bbox = [bbox_img.to(device) for bbox_img in bbox]
                # img = img.float().to(device)
                # img = img.float()

                # del img
                # torch.cuda.empty_cache()

                clas_out, regr_out = rcnn_model.forward(img)

                targ_clas, targ_regr = rcnn_model.create_batch_truth(bbox, index, img.shape[-2:])

                total_loss, class_loss, regr_loss = rcnn_model.compute_loss(clas_out,regr_out, targ_clas, targ_regr, l, effective_batch)

                # del bbox,label,mask
                # torch.cuda.empty_cache()

                if (i + 1) % 200 == 0:
                    print("Batch: ", i + 1, "\tClass_loss: ", class_loss, "\tRegression_loss: ", regr_loss, "\tTotal_loss: ",
                          total_loss)

                validation_total_loss[epoch] += total_loss.item()
                validation_class_loss[epoch] += class_loss.item()
                validation_regr_loss[epoch] += regr_loss.item()

                del img, bbox, label, mask, index, clas_out, regr_out, targ_clas, targ_regr, class_loss, regr_loss, total_loss

                # del fpn_feat_list,cate_pred_list,ins_pred_list,ins_gts_list,\
                # ins_ind_gts_list,cate_gts_list,cate_loss,mask_loss,total_loss
                torch.cuda.empty_cache()

                pass

        end_time = time.time()
        print("Time taken for epoch " + str(epoch + 1), end_time - start_time)

    # Dividing by number of batches
    validation_total_loss = validation_total_loss / (i + 1)
    validation_class_loss = validation_class_loss / (i + 1)
    validation_regr_loss = validation_regr_loss / (i + 1)
    print("Total loss: ", validation_total_loss)
    print("Class loss: ", validation_class_loss)
    print("Regression loss: ", validation_regr_loss)

    plt.figure(1)
    plt.title("Total Validation Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Total Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, validation_total_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/total_vloss.png")

    plt.figure(2)
    plt.title("Class Validation Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Class Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, validation_class_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/class_vloss.png")

    plt.figure(3)
    plt.title("Regression Validation Loss Curve")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Regression Loss per Batch")
    plt.plot(np.arange(num_epochs) + 1, validation_regr_loss)
    plt.savefig("/content/drive/My Drive/CIS 680/Mask_RCNN/regr_vloss.png")

    plt.show()


