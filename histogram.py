from dataset import *

if __name__ == '__main__':
    # file path and make a list
    # imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    # masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    # labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    # bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'

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
    train_size = int(full_size)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    # test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # test_loader = test_build_loader.loader()

    boxes = []
    for i, batch in enumerate(train_loader, 0):
      img, label, mask, bbox, index = batch
      boxes.extend(bbox)

    boxes = torch.cat(boxes)

    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    aspect_ratio = w/h
    scale = torch.sqrt(w*h)

    min_ar = torch.min(aspect_ratio)
    max_ar = torch.max(aspect_ratio)
    min_scale = torch.min(scale)
    max_scale = torch.max(scale)

    print("AR Min: ", min_ar)
    print("Scale Min: ", min_scale)

    num_bins = 20

    ar_bin_size = (max_ar - min_ar)/num_bins
    scale_bin_size = (max_scale - min_scale)/num_bins

    print("Aspect ratio bin size: ", ar_bin_size.cpu())
    print("Scale bin size: ", scale_bin_size.cpu())

    ar_bins = (aspect_ratio-min_ar)//ar_bin_size
    scale_bins = (scale - min_scale)//scale_bin_size

    plt.figure(0)
    ax0 = plt.gca()
    plt.hist(aspect_ratio.cpu(), bins = 20)
    plt.title("Histogram of Aspect Ratio")
    plt.ylabel("Number of occurunces")
    plt.xlabel("Aspect Ratio")
    plt.show()

    plt.figure(1)
    ax1 = plt.gca()
    plt.hist(scale.cpu(), bins =20)
    plt.title("Histogram of Scale")
    plt.ylabel("Number of occurunces")
    plt.xlabel("Scale")
    plt.show()
