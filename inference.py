#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
import numpy as np

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        #self.infer_request_handle = None
        self.request_id = 0
        self.request_count = 0

    def load_model(self, model, device, cpu_ext = None):
                        
        # Initialize the Inference Engine
        self.plugin = IECore()
        #print(self.plugin.available_devices)
        
        ### Add any necessary extensions ###
        if cpu_ext and device.lower() == 'cpu':
            self.plugin.add_extension(cpu_ext, device)
        
        # Initialize IENetwork object from IR files
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + '.bin'        
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Support topologies with 1 and 2 inputs
        
        self.image_tensor_blob = None
        self.image_info_blob = None
                
        for input_key, input_val in self.network.inputs.items():
            if len(input_val.shape) == 4: # image tensor
                self.image_tensor_blob = input_key
            elif len(input_val.shape) == 2: # image info
                self.image_info_blob = input_key
        
        assert self.image_tensor_blob is not None, \
            "Failed to find the input image specification"
        
        self.output_blob = next(iter(self.network.outputs))
        
        ### Check for unsupported layers ###        
        
        supported_layers = self.plugin.query_network(
            network=self.network, device_name=device)
        supported_layers = set(supported_layers.keys())
        net_layers = set(self.network.layers.keys())
        unsupported_layers = net_layers.difference(supported_layers)
        
        if unsupported_layers:
            raise Exception('Unsupported layers: ' +
                ', '.join(unsupported_layers))
        
        ### Load the model ###
        self.exec_network = self.plugin.load_network(
            network=self.network, device_name=device)
        



    def get_input_shape(self):
        ### Return the shape of the input layer ###        
        return self.network.inputs[self.image_tensor_blob].shape

