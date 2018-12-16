# Neural Network Tools: Converter, Constructor and Analyser

 Providing a tool for some fashion neural network frameworks.
 
 The nn_tools is released under the MIT License (refer to the LICENSE file for details).

### features

1. Converting a model between different frameworks.
2. Some convenient tools of manipulate caffemodel and prototxt quickly(like get or set weights of layers), 
see [nn_tools.Caffe](https://github.com/hahnyuan/nn_tools/tree/master/Caffe).
3. Analysing a model, get the operations number(ops) in every layers.

### requirements

- Python2.7 or Python3.x
- Each functions in this tools requires corresponding neural network python package (tensorflow pytorch and so on).

# Analyser

The analyser can analyse all the model layers' [input_size, output_size, multiplication ops, addition ops, 
comparation ops, tot ops, weight size and so on] given a input tensor size, which is convenint for model deploy analyse.

## Caffe
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), in caffe image shape should be: 
batch_size, channel, image_height, image_width.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,3,224,224`

## Pytorch
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path name shape`
- The path is the python file path which contaning your class.
- The name is the class name or instance name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be:
batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py example/resnet_pytorch_analysis_example.py resnet18 1,3,224,224`


## Mxnet
Supporting analyse the inheritors of mxnet.sym.

Command：`mxnet_analyser.py [-h] [--out OUT] [--func_args ARGS] [--func_kwargs FUNC_KWARGS] path name shape`
- The path is the python file path which contaning your symbol definition.
- the symbol object name or function that generate the symbol in your python file.
- The shape is the input shape of the network(split by comma `,`), in mxnet image shape should be:
batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/mxnet_analyse.csv'.
- The func_args (optional) is the args to init the class in python file, default is empty.

For example `python mxnet_analyser.py example/mobilenet_mxnet_symbol.py get_symbol 1,3,224,224`

# Converter

## Pytorch to Caffe

The new version of pytorch_to_caffe supporting the newest version(from 0.2.0 to 0.4.0) of pytorch.
NOTICE: The transfer output will be somewhat different with the original model, caused by implementation difference.

- Supporting layers types: conv2d, linear, max_pool2d, avg_pool2d, dropout,
 relu, prelu, threshold(only value=0),softmax, batch_norm
- Supporting operations: torch.split, torch.max, torch.cat
- Supporting tensor Variable operations: var.view, var.mean, + (add), += (iadd), -(sub), -=(isub)
 \* (mul) *= (imul)

The supported above can transfer many kinds of nets, 
such as AlexNet(tested), VGG(tested), ResNet(tested), Inception_V3(tested).

The supported layers concluded the most popular layers and operations.
 The other layer types will be added soon, you can ask me to add them in issues.

Example: please see file `example/alexnet_pytorch_to_caffe.py`. Just Run `python3 example/alexnet_pytorch_to_caffe.py`


# Some common functions

## funcs.py

- **get_iou(box_a, box_b)** intersection over union of two boxes
- **nms(bboxs,scores,thresh)** Non-maximum suppression
- **Logger** print some str to a file and stdout with H M S

