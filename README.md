# People Counter App at the Edge

Detect people in a designated area, providing the number of people and average duration they spend in a frame.


## Choosing a Model

It would probably be enough to use a classification model for this task since we need just to count people and only one person can appear in a frame at the same time. However, bounding boxes could be a useful indicator of application correctness, therefore I decided to use a detection model. 

This project supports 1- and 2-input topologies from TensorFlow Object Detection Models Zoo. Tested with:
* [SSD MobileNet V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
* [SSD Lite MobileNet V2 COCO](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
* [SSD Inception V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
* [Faster R-CNN Inception V2 COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
* [Faster R-CNN ResNet 50 COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)
* [Faster R-CNN ResNet 50 Low Proposals COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz)
* [Faster R-CNN NasNet Low Proposals COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz)
