# People Counter App at the Edge

![people-counter-python](./images/people-counter-image.png)

Detect people in a designated area, providing the number of people and average duration they spend in a frame.


## How it Works

The counter exploits the Inference Engine included in the Intel Distribution of OpenVINO Toolkit. The Inference Engine identifies people in a video frame by means of an object detection model. Assuming that several people can't be present in the same frame, the app looks if there is a person and increments the total count. When that person leaves, it measures the duration they spent in the frame (time elapsed between entering and exiting a frame) and calculates the average duration. It then sends the data to a local web server using the Paho MQTT Python package.

![architectural diagram](./images/arch_diagram.png)

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

The major problem of SSD Lite model was long series of frames with undetected people in some parts of the video. In order to improve accuracy I changed the shape of the input while converting the model using the `--input_shape` parameter of the model optimizer. It turned out that increasing the expected frame size from 300x300 (default) to 400x400 aids in making much more stable detections.

Since we load images in BGR channels order, whereas TensorFlow models were trained with images in RGB order, the `--reverse_input_channels` parameter is used to make input data match the channel order of the model training dataset.

```
/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --model_name ssdlite_mobilenet_v2_coco_custom_shape --input_model  ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config --reverse_input_channels --input_shape [1,400,400,3]
```

## Usage

Although reshaped SSD Lite model is pretty accurate, occasionally fails still occur. To solve that issue the volatility threshold is used to account for short misdetections. Unfortunately, there are a couple of places where the sequence of failures is rather long, therefore it is recommended to decrease the probability threshold for detections to 0.3 or increase the volatility threshold to a value around 20.

All supported command line arguments are described below.

Parameter&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Meaning 
------------ | ------ 
--model, -m | [Required] The path to an xml file with a trained model.
--input, -i | [Required] The path to an image or a video file. To get the input from the camera, use "CAM" as an argument.
--device, -d | [Optional] The target device to infer on: CPU, GPU, FPGA, or MYRIAD is acceptable. Defaults to CPU.
--cpu_extension, -l | [Optional] The absolute path to a shared library with MKLDNN kernel implementations.
--concurrency, -cy | [Optional] Specifies the number of asynchronous infer requests to perform at the same time. Default is 4.
--batch, -b | [Optional] Defines the frame batch size. Default is 1.
--volatility, -vt | [Optional] The maximal number of consecutive frames allowed to have a detection value different from the last stable one. Exceeding this limit results in their detection to be considered a new stable value. Default is 10.
--prob_threshold, -pt | [Optional] Probability threshold for detections filtering. Default is 0.3.
--crowd_alarm, -ca | [Optional] The number of people after which the warning message will be displayed. Default is 5.
--duration_alarm, -da | [Optional] The duration of stay (in seconds) after which the warning message will be displayed. Default is 15.

The script outputs processed frames, so they can be pipelined to FFMPEG server. Output frames are of the same size as original ones, hence it is important to specify correct input resolution in the `-video_size` parameter of the ffmpeg command. For example, the following command can be used to perform inference on the input video 768x432 pixels in size:
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssdlite_mobilenet_v2_coco_custom_shape.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 2 -i - http://0.0.0.0:3004/fac.ffm
```



