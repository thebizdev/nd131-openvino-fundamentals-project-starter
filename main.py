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
  
# -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()
   
    return args
    
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    return parser




def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

 #def alarm(self):
     #Not finished yet
     #if more than exp. 4 people went into the room a alarm is send.
    
 #def writeincsv(self):
     #Not finished yet
     #Here the output text is writen in a text file
        
        

    


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
   
  
    

    #Marking first for the single image
    single_image_mode = False
    
    # Initialise the class
    ## from mqtt.app.py course work
    def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names
    
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model

    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape = infer_network.get_input_shape()
    in_shape = network_shape['image_tensor']
    
     
    
    ### TODO: Handle the input stream ###
    
    # Get and open video capture source from course work
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    # Create a flag for single images
    image_flag = False
    
    # Check if the input is CAM
    if args.i == 'CAM':
        args.i = 0
    
    #Check for image type(jpg, bmp or png)
    elif args.i.endswith('.jpg') or args.i.endswith('.bmp') or args.i.endswith(".png"):
         single_image_flag = True
         input_stream = args.i 
    
    #Check for CAM
    if args.i =="CAM":
        input_stream = 0
        
    #Check for video
    else:
        input_stream = args.i
        assert os.path.isfile(args.i), "video file does not exist"
        

    ### TODO: Loop until stream is over ###
    
    # Get and open video capture from mqtt.py course work
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    # Grab the shape of the input from mqtt.py course work
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    
    
    
    # Process frames until the video ends, or process is exited from mqtt.py course work
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
    
    
    #Iniatilize Variables
    
    total_count = 0
    duration = 0
    person_on_s = False
    person_count = 0
    no_person_count = 0
    people_count = 0
    duration_time = 0
    
   
    
    durration_flag = 0
    new_person_flag = 0
    person_leaves_flag = 0
    request_id = 0
    i_start = 0
    person_detected = 0
    
    ### TODO: Read from the video capture ###
        while cap.isOpened():
            flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

    ### TODO: Pre-process the image as needed ###
    image = cv2.resize(frame, (w, h))
        # HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)    

    ### TODO: Start asynchronous inference for specified request ###
    net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
    duration_report = None
    infer_network.exec_net(net_input, request_id)


    ### TODO: Wait for the result ###
    if infer_network.wait(request_id) == 0:
            
    ### TODO: Get the results of the inference request ###
    net_output = infer_network.get_output()

    ### TODO: Extract any desired stats from the results ###
    pointer = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > prob_threshold:
                    pointer += 1
                    box = net_output[0, 0, i, 3:]
                    p1 = (int(box[0] * w), int(box[1] * h))
                    p2 = (int(box[2] * w), int(box[3] * h))
                    frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
        
            if pointer != counter:
                counter_prev = counter
                counter = pointer
                if dur >= 3:
                    duration_prev = dur
                    dur = 0
                else:
                    dur = duration_prev + dur
                    duration_prev = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > counter_prev:
                        counter_total += counter - counter_prev
                    elif dur == 3 and counter < counter_prev:
                        duration_report = int((duration_prev / 10.0) * 1000)

    ### TODO: Calculate and send relevant information on ###
    ### current_count, total_count and duration to the MQTT server ###
    ### Topic "person": keys of "count" and "total" ###
    ### Topic "person/duration": key of "duration" ###
    
    client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
    

    ### TODO: Send the frame to the FFMPEG server ###
    frame = cv2.resize(frame, (320, 320))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

    ### TODO: Write an output image if `single_image_mode` ###
    
    if single_image_mode:
            cv2.write(".jpg", frame)
            
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
