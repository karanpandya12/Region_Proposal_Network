# Region_Proposal_Network
Implementation of RPN using PyTorch

## What are Region Proposal Networks?
Region Proposal Networks (RPNs) are "attention mechanisms" for the object detection task, performing a crude but inexpensive first estimation of where the bounding boxes of the objects should be. They work through classifying the initial anchor boxes into object/background and refine the coordinates for the boxes with objects. Later, these
boxes can further refined and tightened by the instance segmentation heads as well as classified in their corresponding classes.

## Dataset
A subset of the COCO dataset was used containing data of 3 classess namely, Vehicles, People and Animals. Here are some example images from the dataset: <br>
<img src = "/Images/dataset_1.png" height = 300> <img src = "/Images/dataset_2.png" height = 300>
<img src = "/Images/dataset_3.png" height = 300> <img src = "/Images/dataset_4.png" height = 300>

## Results
The RPN provides initial estimation of the bounding boxes which can be further refined by using applying a FasterRCNN head on these proposals. <br>
<img src = "/Images/pre_NMS_1.png" height = 300> <img src = "/Images/post_NMS_1.png" height = 300>
<img src = "/Images/pre_NMS_2.png" height = 300> <img src = "/Images/post_NMS_2.png" height = 300>
