"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from collections import deque

import logging as log
import paho.mqtt.client as mqtt

from inference import Network
from argparse import ArgumentParser


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")

    parser.add_argument("-vt", "--volatility", type=int, default=10,
                       help="The maximal number of frames "
                           "allowed to have a detection value "
                           "different from the last stable one. "
                           "Exceeding this limit results in "
                           "their detection to be considered "
                           "a new stable value.")

    parser.add_argument("-b", "--batch", type=int, default=1,
                       help="Defines the frame batch size")

    parser.add_argument("-cy", "--concurrency", type=int, default=4,
                       help="Specifies the number of asynchronous "
                            "infer requests to perform at the same time")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")


    return parser



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    """
        
    # Read command line arguments
    
    model = args.model  # path to the model IR       
    device = args.device # device name to perform inference on    
    cpu_ext = args.cpu_extension # CPU extension
    prob_threshold = args.prob_threshold # threshold for detections
    volatility = args.volatility # volatility threshold        
    concurrency = args.concurrency # number of concurrent infer requests
    batch_size = args.batch # set the batch size    
    
    ### Load the model through `infer_network` ###

    infer_network = Network()
    infer_network.load_model(model, batch_size, concurrency, device, cpu_ext)
    net_input_shape = infer_network.get_input_shape()


    ### Handle the input stream ###
    
    if args.input is None or args.input.lower() == 'cam':
        input = 0
    else:
        input = args.input

    cap = cv2.VideoCapture(input)
    assert cap.isOpened(), "Failed to open the input"
    
    frames = [] # frames to batch
    q = deque() # infer request queue
    
    ### Loop until stream is over ###    
    while cap.isOpened():
        
        ### Read from the video capture ###
        
        captured, next_frame = cap.read()
        
        if captured:            
            frames.append(next_frame)
        
        
        ### Pre-process the image as needed ###        
                
        if len(frames)>=batch_size or not captured and frames:
            h = net_input_shape[2]
            w = net_input_shape[3]            
            resized_frames = [cv2.resize(f,(w, h))
                              .transpose(2,0,1)[None,...] 
                              for f in frames]
            frame_batch = np.concatenate(resized_frames, axis=0)
            request = infer_network.exec_net(frame_batch)
            q.append((request, frames))
            frames = []


        ### Start asynchronous inference for specified request ###
                
        # If the number of concurrent requests hit the limit,
        # we have to wait. Also if the end of the stream has 
        # been reached, process what we have in the queue.
        if len(q) >= concurrency or not captured: 
            if not q:
                break

            prev_request, prev_frames = q.popleft()

            ### Wait for the result ###        
            ### Get the results of the inference request ###
            
            detected, boxes = infer_network.get_output(
                request=prev_request, 
                class_id=1, 
                confidence=prob_threshold)

            for i,prev_frame in enumerate(prev_frames):           
                if detected[i]:
                    box = boxes[i]
                    x_min = int(box[0]*prev_frame.shape[1])
                    y_min = int(box[1]*prev_frame.shape[0])
                    x_max = int(box[2]*prev_frame.shape[1])
                    y_max = int(box[3]*prev_frame.shape[0])
                    output_frame = cv2.rectangle(prev_frame, 
                                                 (x_min,y_min),
                                                 (x_max,y_max), 
                                                 (0,255,0))
                    
                else: # nothing detected or error
                    output_frame = prev_frame

            ### Send the frame or image to the FFMPEG server ###
            sys.stdout.buffer.write(output_frame)
            sys.stdout.buffer.flush()        
        
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    """

    # Grab command line args
    args = build_argparser().parse_args()

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
