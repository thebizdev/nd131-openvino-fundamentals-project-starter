# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The Intel® Distribution of OpenVINO™ toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*. The list of known layers is different for each of the supported frameworks. 
To see the layers supported by your framework, refer to supported frameworks.
https://docs.openvinotoolkit.org/latest/openvino_docs_HOWTO_Custom_Layers_Guide.html

Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

I choose from https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#freeze-the-tensorflow-model to compare models 

faster_rcnn_inception_v2_coco_2018_01_28 and ssd_mobilenet_v2_coco_2018_03_29.

Because their performance on openvino more efficient.

Firstly begin faster_rcnn_inception_v2_coco_2018_01_28

1 wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

2 tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

3 cd   faster_rcnn_inception_v2_coco_2018_01_28

## converting tensorflow model to inference engine
4 python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

Secondly begin ssd_mobilenet_v2_coco_2018_03_29.tar.gz 

1- wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

2- tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

3- cd ssd_mobilenet_v2_coco_2018_03.29

##converting tensorflow model to inference engine
4- python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ./frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
Terms used in this guide

Layer — The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional). This is one of a sequential series of building blocks within the neural network.

Kernel — The implementation of a layer function, in this case, the math programmed (in C++ and Python) to perform the layer operation for target hardware (CPU or GPU)

Intermediate Representation (IR) — Neural Network used only by the Inference Engine in OpenVINO abstracting the different frameworks and describing topology, layer parameters and weights. The original format will be a supported framework such as TensorFlow, Caffe, or MXNet.

Model Extension Generator — Generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.

Inference Engine Extension — Device-specific module implementing custom layers (a set of kernels).

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although you shouldn’t have to use it very often, if at all, due to all of the supported layers. However, it’s useful to know a little about its existence and how to use it if the need arises.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference. 

It is expected to be used, when you have the model that has complex structure and it is not an easy task to write extensions for internal subgraphs. 

however, for each such subgraph, TensorFlow library is called that is not optimized for inference. Then, you start replacing each subgraph with extension and remove its offloading to TensorFlow during inference until all the models are converted by Model Optimizer and inferred by Inference Engine only with the maximum performance.
    
## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...  faster_rcnn_inception_v2_coco_2018_01_28 and ssd_mobilenet_v2_coco_2018_03_29.



The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are currently will more efficiently useful about on healthcare, public health security. Predicting to the COVID-19 virus symptom from people. Especially will be use in airports, international congress, business fairs etc.

Each of these use cases would be useful because image, pose detection, object detection automatize system.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
