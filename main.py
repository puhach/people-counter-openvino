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

from argparse import ArgumentParser
from inference import Network

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
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-ca", "--crowd_alarm", type=int, default=5,
                       help="The number of people after which "
                            "the warning message will be displayed.")
    parser.add_argument("-da", "--duration_alarm", 
                        type=float, default=15,
                       help="The duration of stay after which "
                            "the warning message will be displayed.")
    parser.add_argument("-cy", "--concurrency", type=int, default=4,
                       help="Specifies the number of asynchronous "
                            "infer requests to perform at the same time")
    parser.add_argument("-b", "--batch", type=int, default=1,
                       help="Defines the frame batch size")
    parser.add_argument("-vt", "--volatility", type=int, default=10,
                       help="The maximal number of consecutive frames "
                           "allowed to have a detection value "
                           "different from the last stable one. "
                           "Exceeding this limit results in "
                           "their detection to be considered "
                           "a new stable value.")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT server ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    client.loop_start()
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    """
        
    # Read command line arguments
    
    model = args.model  # path to the model IR       
    batch_size = args.batch # set the batch size    
    device = args.device # device name to perform inference on    
    cpu_ext = args.cpu_extension # CPU extension
    concurrency = args.concurrency # number of concurrent infer requests
    volatility = args.volatility # volatility threshold    
    prob_threshold = args.prob_threshold # threshold for detections
    duration_alarm_threshold = args.duration_alarm # longest stay allowed
    crowd_alarm_threshold = args.crowd_alarm # max people allowed
    
    ### Load the model through `infer_network` ###

    infer_network = Network()
    infer_network.load_model(model, batch_size, 
                             concurrency, 
                             device, cpu_ext)
    net_input_shape = infer_network.get_input_shape()
    
    ### Handle the input stream ###
    
    if args.input is None or args.input.lower() == 'cam':
        input = 0
    else:
        input = args.input

    # VideoCapture supports images too
    cap = cv2.VideoCapture(input)
    assert cap.isOpened(), "Failed to open the input"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    total_people_count = 0
    last_stable_people_count = 0
    mismatch_count = 0 # deviation from the last stable detection
    total_duration = 0 # total duration in frames
    current_duration = 0 # current person's stay duration in frames
    
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
                ### Extract stats from the results ###
                    
                cur_people_count = int(detected[i])
                if last_stable_people_count != cur_people_count:
                    mismatch_count += 1
                else:
                    mismatch_count = 0
                
                
                # Check if we have a new stable value
                if mismatch_count>volatility or frame_count<=1:
                    
                    ### Calculate and send relevant information ###
                    ### on current_count, total_count and duration ### 
                    ### to the MQTT server ###

                    last_stable_people_count = cur_people_count
                    total_people_count += last_stable_people_count
                    mismatch_count = 0
                    
                    if last_stable_people_count > 0: # person entered
                        current_duration = 1
                        total_duration += 1                        
                    else: # person left
                        current_duration = 0
                        
                        # Send average duration to the server
                        # (average duration is calculated in  
                        # terms of the original video and doesn't
                        # depend on inference time or network delays)
                        ### Topic "person/duration": key of "duration" ###
                        
                        avg_duration_payload = json.dumps({'duration' 
                            : total_duration / total_people_count / fps })
                        
                        client.publish(
                            topic='person/duration',
                            payload=avg_duration_payload)

                    # Send new people count to the server
                    ### Topic "person": keys of "count" and "total" ###
                    
                    people_count_payload = json.dumps(
                        {
                            'count' : last_stable_people_count,
                            'total' : total_people_count
                        })
                    
                    client.publish(topic='person',
                        payload=people_count_payload)
                    
                else: # Last stable count remains the same
                    if last_stable_people_count > 0:
                        current_duration += 1
                        total_duration += 1
            
            
                # Prepare the output frame
                        
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

                    
                # Alarms 
                
                if duration_alarm_threshold >= 0 \
                    and current_duration/fps > duration_alarm_threshold:
                        cv2.putText(output_frame, 
                                    text="Chop-chop! "
                                    "Don't stay for too long. "
                                    "Life is short!",
                                    org=(20,output_frame.shape[0]-60),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.9,
                                    color=(100,250,250),
                                    thickness=2)
                    
                if crowd_alarm_threshold >= 0 \
                    and total_people_count > crowd_alarm_threshold:
                        cv2.putText(output_frame, 
                                text="Too many people. "
                                "Beware COVID-19!",
                                org=(20,output_frame.shape[0]-20),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1,
                                color=(0,0,255),
                                thickness=2)    
                        
                ### Send the frame to the FFMPEG server ###
                ### Write an output image if `single_image_mode` ###
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
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)

    # Disconnect from the MQTT server
    client.loop_stop()
    client.disconnect()


if __name__ == '__main__':
    main()
