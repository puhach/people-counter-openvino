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

Faster R-CNN models proved to be way more accurate as compared to SSD models, but due to their slow inference I ended up with [SSD Lite MobileNet V2 COCO](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz).


## Model Conversion

The major problem of SSD Lite model was long series of frames with undetected people in a few parts of the video. In order to improve accuracy I changed the shape of the input while converting the model using the `--input_shape` parameter of the model optimizer. It turned out that increasing the expected frame size from 300x300 (default) to 400x400 aids in making much more stable detections.

Since we load images in BGR channels order whereas TensorFlow models were trained with images in RGB order, the `--reverse_input_channels` parameter is used to make input data match the channel order of the model training dataset.

```
/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --model_name ssdlite_mobilenet_v2_coco_custom_shape --input_model  ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --reverse_input_channels --input_shape [1,400,400,3]
```
