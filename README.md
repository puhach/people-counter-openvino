# People Counter App at the Edge

![people-counter-python](./images/people-counter-image.png)

Detect people in a designated area, providing the number of people and average duration they spend in a frame.


## How it Works

The counter exploits the Inference Engine included in the Intel Distribution of OpenVINO Toolkit. The Inference Engine identifies people in a video frame by means of an object detection model. Assuming that several people can't be present in the same frame, the app looks if there is a person and increments the total count. When that person leaves, it measures the duration they spent in the frame (time elapsed between entering and exiting a frame) and calculates the average duration. It then sends the data to a local web server using the Paho MQTT Python package.

![architectural diagram](./images/arch_diagram.png)


## Requirements

### Hardware

* 6th to 10th generation Intel Core processor with Iris Pro graphics or Intel HD Graphics
* Or Intel Neural Compute Stick 2 (NCS2)
* 4GB RAM on a development machine

### Software

* Python 3.5
* Intel Distribution of OpenVINO toolkit 2019 R3 release
* Node v6.17.1
* Npm v3.10.10
* CMake
* FFmpeg
* MQTT Mosca server
  

## Setup

* Initial setup instructions are platform-specific. Please, refer to the relevant guidelines for your operating system:

    - [Linux/Ubuntu](./setup/linux-setup.md)
    - [Mac](./setup/mac-setup.md)
    - [Windows](./setup/windows-setup.md)

* Install Intel Distribution of OpenVINO toolkit
* Install Node.js and its dependencies
* Install FFmpeg
* Install MQTT/Mosca Sever

    From the setup directory run:
    ```
    cd webservice/server
    npm install
    ```

* Install the Web Server

    From the current directory run:
    ```
    cd ../ui
    npm install
    ```
    **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
    ```
    sudo npm install npm -g 
    rm -rf node_modules
    npm cache clean
    npm config set registry "http://registry.npmjs.org"
    npm install
    ```


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

## Launch

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server

### Start the Mosca server

From the setup directory:

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Start the GUI

Open a new terminal and run the below commands from the setup directory.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Start the FFmpeg Server

Open a new terminal and the commands below.
```
sudo ffserver -f ./ffmpeg/server.conf
```


### Configure the environment

Before running the code the environment must be configured to use the Intel Distribution of OpenVINO toolkit one time per session by executing the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.


### Usage

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

#### Running on the CPU

When running inference on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m path-to-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3000](http://0.0.0.0:3000/) in a browser.



#### Running on the Intel Neural Compute Stick

To run on the Intel Neural Compute Stick, use the ```-d MYRIAD``` command-line argument:

```
python main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m path-to-model.xml -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

**Note:** The Intel Neural Compute Stick can only run FP16 models at this time. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.


#### Using a camera stream instead of a video file

To get the input video from the camera, use the `-i CAM` command-line argument. Specify the resolution of the camera using the `-video_size` command line argument.

For example:
```
python main.py -i CAM -m path-to-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```


## Credits 

This project is based on the assignment from the [Intel Edge AI for IoT Developers](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131) program. 