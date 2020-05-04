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

import logging as log

from argparse import ArgumentParser



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

    parser.add_argument("-cy", "--concurrency", type=int, default=4,
                       help="Specifies the number of asynchronous "
                            "infer requests to perform at the same time")

    parser.add_argument("-b", "--batch", type=int, default=1,
                       help="Defines the frame batch size")


    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
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
    concurrency = args.concurrency # number of concurrent infer requests
    batch_size = args.batch # set the batch size    
    
    ### Load the model through `infer_network` ###

    infer_network = Network()
    infer_network.load_model(model, concurrency, device, cpu_ext)
    net_input_shape = infer_network.get_input_shape()


    ### Handle the input stream ###
    
    if args.input is None or args.input.lower() == 'cam':
        input = 0
    else:
        input = args.input

    cap = cv2.VideoCapture(input)
    assert cap.isOpened(), "Failed to open the input"
    
    
    ### Loop until stream is over ###    
    while cap.isOpened():
        
        ### Read from the video capture ###
        
        captured, next_frame = cap.read()
        
        if captured:       
            h = net_input_shape[2]
            w = net_input_shape[3]            
            resized_frame = cv2.resize(next_frame,(w, h))
            resized_frame = resized_frame.transpose(2,0,1)
            frame_batch = resized_frame[None,...]        
            request = infer_network.exec_net(frame_batch)
            q.append((request, next_frame))

        if len(q) >= concurrency or not captured:
            if not q:
                break

            detected, box = infer_network.get_output(
                request=prev_request, 
                class_id=1, 
                confidence=prob_threshold)

            if detected:
                x_min = int(box[0]*next_frame.shape[1])
                y_min = int(box[1]*next_frame.shape[0])
                x_max = int(box[2]*next_frame.shape[1])
                y_max = int(box[3]*next_frame.shape[0])
                output_frame = cv2.rectangle(next_frame, 
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
