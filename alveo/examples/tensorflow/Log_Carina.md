## [Quantize Model](https://github.com/passlab/Vitis-AI/tree/master/alveo/examples/tensorflow#quantize-model)

```
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ python run.py --quantize --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --input_shapes 1,224,224,3 --batch_size 16

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:36: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:41: The name tf.NodeDef is deprecated. Please use tf.compat.v1.NodeDef instead.

INFO: Checking Float Graph...
2020-11-29 11:58:41.421887: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
2020-11-29 11:58:41.456377: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2020-11-29 11:58:41.460207: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c2d32de7e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-29 11:58:41.460241: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-11-29 11:58:41.462559: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-11-29 11:58:41.560174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:58:41.560437: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:58:41.561725: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:58:41.563068: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:58:41.563398: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:58:41.564932: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:58:41.566058: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:58:41.569849: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:58:41.574698: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:58:41.574751: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:58:41.706378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1180] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-29 11:58:41.706404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1186]      0 
2020-11-29 11:58:41.706413: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1199] 0:   N 
2020-11-29 11:58:41.713658: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0)
2020-11-29 11:58:41.716302: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c2d8713c40 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-11-29 11:58:41.716318: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-PCIE-32GB, Compute Capability 7.0
2020-11-29 11:58:42.936894: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:58:43.191850: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:58:44.553378: W tensorflow/stream_executor/cuda/redzone_allocator.cc:312] Not found: ./bin/ptxas not found
Relying on driver to perform ptx compilation. This message will be only logged once.
INFO: Float Graph Check Done.
INFO: Calibrating for 10 iterations...
2020-11-29 11:58:45.578868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:58:45.578936: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:58:45.578950: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:58:45.578959: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:58:45.578970: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:58:45.578981: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:58:45.578995: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:58:45.579006: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:58:45.580378: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:58:45.580408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1180] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-29 11:58:45.580415: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1186]      0 
2020-11-29 11:58:45.580421: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1199] 0:   N 
2020-11-29 11:58:45.581854: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0)
100% (10 of 10) |##################################################################################################################################################################################################################| Elapsed Time: 0:00:03 Time:  0:00:03
INFO: Calibration Done.
INFO: Generating Deploy Model...
2020-11-29 11:58:49.411332: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:58:49.411426: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:58:49.411451: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:58:49.411471: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:58:49.411494: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:58:49.411514: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:58:49.411535: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:58:49.411556: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:58:49.413365: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:58:49.413400: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1180] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-29 11:58:49.413407: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1186]      0 
2020-11-29 11:58:49.413413: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1199] 0:   N 
2020-11-29 11:58:49.415069: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0)
INFO: Deploy Model Generated.
********************* Quantization Summary *********************      
INFO: Output:       
  quantize_eval_model: work/quantize_eval_model.pb       
  deploy_model: work/deploy_model.pb
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
WARNING: arch/dpuv1/ALVEO/ALVEO.json is deprecated.  Replacing with arch/DPUCADX8G/ALVEO/arch.json
Please specify a quantization file
Please specify the input shapes
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:36: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

GenerateCode: work/fix_info.txt
Weights: None
PngFile: None
ConcatStrategy: None
Strategy: all
ScheduleFile: None
DDR: 256
DSP: 96
Verbose: False
FromTF: True
Memory: 9
Byte per Pixel: 1
Start compiling

**************************************************
* BUILDING DATA FLOW GRAPH
**************************************************
Reading pre-build graph

######### load_graph arguments #############
networkfile               work/deploy_model.pb
loadmode                  binary
startnode                 None
finalnode                 None
inclusive                 False
batch_sz                  None
fixinputnames             None
placeholdershape          None
remove_training_nodes     None
remove_redundant_nodes    None
freeze_blacklist          None
freeze_whitelist          None
graph_savepath            None
#############################################

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:345: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
freeze model
.... node count 141
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:247: remove_training_nodes (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.remove_training_nodes`
.... node count after removing training nodes 141
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/utils/xdnn_util_tf.py:251: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
.... node count after removing redundant nodes 141
.... node count after removing blacklisted nodes 141
NodeDef mentions attr 'opos' not in Op<name=Placeholder; signature= -> output:dtype; attr=dtype:type; attr=shape:shape,default=<unknown>>; NodeDef: {{node data}}. (Check whether your GraphDef-interpreting binary is up to date with your GraphDef-generating binary.).
<class 'tensorflow.core.framework.graph_pb2.GraphDef'>
data
data [] list {
  i: 8
  i: -1
}
 [] []
1 data 8 -1

conv1_7x7_s2/Conv2D
conv1_7x7_s2/Conv2D list {
  i: 8
  i: -1
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 13
}
 list {
  i: 8
  i: 5
}

2 conv1_7x7_s2/Conv2D 8 -1 8 4 8 13 8 5

conv1_7x7_s2/conv1_7x7_s2
conv1_7x7_s2/conv1_7x7_s2 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
3 conv1_7x7_s2/conv1_7x7_s2 8 4 8 4

pool1_3x3_s2
pool1_3x3_s2 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
4 pool1_3x3_s2 8 4 8 4

conv2_3x3_reduce/Conv2D
conv2_3x3_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 4
}

5 conv2_3x3_reduce/Conv2D 8 4 8 5 8 6 8 4

conv2_3x3_reduce/conv2_3x3_reduce
conv2_3x3_reduce/conv2_3x3_reduce list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
6 conv2_3x3_reduce/conv2_3x3_reduce 8 5 8 5

conv2_3x3/Conv2D
conv2_3x3/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 4
}

7 conv2_3x3/Conv2D 8 5 8 5 8 8 8 4

conv2_3x3/conv2_3x3
conv2_3x3/conv2_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
8 conv2_3x3/conv2_3x3 8 5 8 5

pool2_3x3_s2
pool2_3x3_s2 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
9 pool2_3x3_s2 8 5 8 5

inception_3a_pool
inception_3a_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
10 inception_3a_pool 8 5 8 5

inception_3a_pool_proj/Conv2D
inception_3a_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 4
}

11 inception_3a_pool_proj/Conv2D 8 5 8 4 8 7 8 4

inception_3a_pool_proj/inception_3a_pool_proj
inception_3a_pool_proj/inception_3a_pool_proj list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
12 inception_3a_pool_proj/inception_3a_pool_proj 8 4 8 4

inception_3a_5x5_reduce/Conv2D
inception_3a_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

13 inception_3a_5x5_reduce/Conv2D 8 5 8 5 8 8 8 5

inception_3a_5x5_reduce/inception_3a_5x5_reduce
inception_3a_5x5_reduce/inception_3a_5x5_reduce list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
14 inception_3a_5x5_reduce/inception_3a_5x5_reduce 8 5 8 5

inception_3a_5x5/Conv2D
inception_3a_5x5/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 4
}

15 inception_3a_5x5/Conv2D 8 5 8 4 8 7 8 4

inception_3a_5x5/inception_3a_5x5
inception_3a_5x5/inception_3a_5x5 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
16 inception_3a_5x5/inception_3a_5x5 8 4 8 4

inception_3a_3x3_reduce/Conv2D
inception_3a_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

17 inception_3a_3x3_reduce/Conv2D 8 5 8 4 8 8 8 5

inception_3a_3x3_reduce/inception_3a_3x3_reduce
inception_3a_3x3_reduce/inception_3a_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
18 inception_3a_3x3_reduce/inception_3a_3x3_reduce 8 4 8 4

inception_3a_3x3/Conv2D
inception_3a_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

19 inception_3a_3x3/Conv2D 8 4 8 4 8 8 8 5

inception_3a_3x3/inception_3a_3x3
inception_3a_3x3/inception_3a_3x3 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
20 inception_3a_3x3/inception_3a_3x3 8 4 8 4

inception_3a_1x1/Conv2D
inception_3a_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 5
}

21 inception_3a_1x1/Conv2D 8 5 8 4 8 7 8 5

inception_3a_1x1/inception_3a_1x1
inception_3a_1x1/inception_3a_1x1 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
22 inception_3a_1x1/inception_3a_1x1 8 4 8 4

inception_3a_output
inception_3a_output list {
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
23 inception_3a_output 8 4 8 4 8 4 8 4 8 4

inception_3b_pool
inception_3b_pool list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
24 inception_3b_pool 8 4 8 4

inception_3b_pool_proj/Conv2D
inception_3b_pool_proj/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 4
}

25 inception_3b_pool_proj/Conv2D 8 4 8 5 8 8 8 4

inception_3b_pool_proj/inception_3b_pool_proj
inception_3b_pool_proj/inception_3b_pool_proj list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
26 inception_3b_pool_proj/inception_3b_pool_proj 8 5 8 5

inception_3b_5x5_reduce/Conv2D
inception_3b_5x5_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 6
}

27 inception_3b_5x5_reduce/Conv2D 8 4 8 4 8 7 8 6

inception_3b_5x5_reduce/inception_3b_5x5_reduce
inception_3b_5x5_reduce/inception_3b_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
28 inception_3b_5x5_reduce/inception_3b_5x5_reduce 8 4 8 4

inception_3b_5x5/Conv2D
inception_3b_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

29 inception_3b_5x5/Conv2D 8 4 8 5 8 9 8 5

inception_3b_5x5/inception_3b_5x5
inception_3b_5x5/inception_3b_5x5 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
30 inception_3b_5x5/inception_3b_5x5 8 5 8 5

inception_3b_3x3_reduce/Conv2D
inception_3b_3x3_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 4
}

31 inception_3b_3x3_reduce/Conv2D 8 4 8 4 8 7 8 4

inception_3b_3x3_reduce/inception_3b_3x3_reduce
inception_3b_3x3_reduce/inception_3b_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
32 inception_3b_3x3_reduce/inception_3b_3x3_reduce 8 4 8 4

inception_3b_3x3/Conv2D
inception_3b_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

33 inception_3b_3x3/Conv2D 8 4 8 5 8 9 8 5

inception_3b_3x3/inception_3b_3x3
inception_3b_3x3/inception_3b_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
34 inception_3b_3x3/inception_3b_3x3 8 5 8 5

inception_3b_1x1/Conv2D
inception_3b_1x1/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 6
}

35 inception_3b_1x1/Conv2D 8 4 8 5 8 8 8 6

inception_3b_1x1/inception_3b_1x1
inception_3b_1x1/inception_3b_1x1 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
36 inception_3b_1x1/inception_3b_1x1 8 5 8 5

inception_3b_output
inception_3b_output list {
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
37 inception_3b_output 8 5 8 5 8 5 8 5 8 5

pool3_3x3_s2
pool3_3x3_s2 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
38 pool3_3x3_s2 8 5 8 5

inception_4a_pool
inception_4a_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
39 inception_4a_pool 8 5 8 5

inception_4a_pool_proj/Conv2D
inception_4a_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 5
}

40 inception_4a_pool_proj/Conv2D 8 5 8 5 8 7 8 5

inception_4a_pool_proj/inception_4a_pool_proj
inception_4a_pool_proj/inception_4a_pool_proj list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
41 inception_4a_pool_proj/inception_4a_pool_proj 8 5 8 5

inception_4a_5x5_reduce/Conv2D
inception_4a_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 6
}

42 inception_4a_5x5_reduce/Conv2D 8 5 8 4 8 7 8 6

inception_4a_5x5_reduce/inception_4a_5x5_reduce
inception_4a_5x5_reduce/inception_4a_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
43 inception_4a_5x5_reduce/inception_4a_5x5_reduce 8 4 8 4

inception_4a_5x5/Conv2D
inception_4a_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

44 inception_4a_5x5/Conv2D 8 4 8 5 8 8 8 5

inception_4a_5x5/inception_4a_5x5
inception_4a_5x5/inception_4a_5x5 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
45 inception_4a_5x5/inception_4a_5x5 8 5 8 5

inception_4a_3x3_reduce/Conv2D
inception_4a_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 5
}

46 inception_4a_3x3_reduce/Conv2D 8 5 8 4 8 7 8 5

inception_4a_3x3_reduce/inception_4a_3x3_reduce
inception_4a_3x3_reduce/inception_4a_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
47 inception_4a_3x3_reduce/inception_4a_3x3_reduce 8 4 8 4

inception_4a_3x3/Conv2D
inception_4a_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

48 inception_4a_3x3/Conv2D 8 4 8 5 8 9 8 5

inception_4a_3x3/inception_4a_3x3
inception_4a_3x3/inception_4a_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
49 inception_4a_3x3/inception_4a_3x3 8 5 8 5

inception_4a_1x1/Conv2D
inception_4a_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

50 inception_4a_1x1/Conv2D 8 5 8 5 8 8 8 5

inception_4a_1x1/inception_4a_1x1
inception_4a_1x1/inception_4a_1x1 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
51 inception_4a_1x1/inception_4a_1x1 8 5 8 5

inception_4a_output
inception_4a_output list {
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
52 inception_4a_output 8 5 8 5 8 5 8 5 8 5

inception_4b_pool
inception_4b_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
53 inception_4b_pool 8 5 8 5

inception_4b_pool_proj/Conv2D
inception_4b_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

54 inception_4b_pool_proj/Conv2D 8 5 8 5 8 9 8 5

inception_4b_pool_proj/inception_4b_pool_proj
inception_4b_pool_proj/inception_4b_pool_proj list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
55 inception_4b_pool_proj/inception_4b_pool_proj 8 5 8 5

inception_4b_5x5_reduce/Conv2D
inception_4b_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 6
}

56 inception_4b_5x5_reduce/Conv2D 8 5 8 4 8 7 8 6

inception_4b_5x5_reduce/inception_4b_5x5_reduce
inception_4b_5x5_reduce/inception_4b_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
57 inception_4b_5x5_reduce/inception_4b_5x5_reduce 8 4 8 4

inception_4b_5x5/Conv2D
inception_4b_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

58 inception_4b_5x5/Conv2D 8 4 8 5 8 9 8 5

inception_4b_5x5/inception_4b_5x5
inception_4b_5x5/inception_4b_5x5 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
59 inception_4b_5x5/inception_4b_5x5 8 5 8 5

inception_4b_3x3_reduce/Conv2D
inception_4b_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 7
}

60 inception_4b_3x3_reduce/Conv2D 8 5 8 4 8 7 8 7

inception_4b_3x3_reduce/inception_4b_3x3_reduce
inception_4b_3x3_reduce/inception_4b_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
61 inception_4b_3x3_reduce/inception_4b_3x3_reduce 8 4 8 4

inception_4b_3x3/Conv2D
inception_4b_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 4
}

62 inception_4b_3x3/Conv2D 8 4 8 5 8 9 8 4

inception_4b_3x3/inception_4b_3x3
inception_4b_3x3/inception_4b_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
63 inception_4b_3x3/inception_4b_3x3 8 5 8 5

inception_4b_1x1/Conv2D
inception_4b_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 7
}

64 inception_4b_1x1/Conv2D 8 5 8 5 8 7 8 7

inception_4b_1x1/inception_4b_1x1
inception_4b_1x1/inception_4b_1x1 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
65 inception_4b_1x1/inception_4b_1x1 8 5 8 5

inception_4b_output
inception_4b_output list {
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
66 inception_4b_output 8 5 8 5 8 5 8 5 8 5

inception_4c_pool
inception_4c_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
67 inception_4c_pool 8 5 8 5

inception_4c_pool_proj/Conv2D
inception_4c_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

68 inception_4c_pool_proj/Conv2D 8 5 8 5 8 9 8 5

inception_4c_pool_proj/inception_4c_pool_proj
inception_4c_pool_proj/inception_4c_pool_proj list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
69 inception_4c_pool_proj/inception_4c_pool_proj 8 5 8 5

inception_4c_5x5_reduce/Conv2D
inception_4c_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 6
}

70 inception_4c_5x5_reduce/Conv2D 8 5 8 4 8 8 8 6

inception_4c_5x5_reduce/inception_4c_5x5_reduce
inception_4c_5x5_reduce/inception_4c_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
71 inception_4c_5x5_reduce/inception_4c_5x5_reduce 8 4 8 4

inception_4c_5x5/Conv2D
inception_4c_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

72 inception_4c_5x5/Conv2D 8 4 8 5 8 9 8 5

inception_4c_5x5/inception_4c_5x5
inception_4c_5x5/inception_4c_5x5 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
73 inception_4c_5x5/inception_4c_5x5 8 5 8 5

inception_4c_3x3_reduce/Conv2D
inception_4c_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 6
}

74 inception_4c_3x3_reduce/Conv2D 8 5 8 4 8 8 8 6

inception_4c_3x3_reduce/inception_4c_3x3_reduce
inception_4c_3x3_reduce/inception_4c_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
75 inception_4c_3x3_reduce/inception_4c_3x3_reduce 8 4 8 4

inception_4c_3x3/Conv2D
inception_4c_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

76 inception_4c_3x3/Conv2D 8 4 8 5 8 9 8 5

inception_4c_3x3/inception_4c_3x3
inception_4c_3x3/inception_4c_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
77 inception_4c_3x3/inception_4c_3x3 8 5 8 5

inception_4c_1x1/Conv2D
inception_4c_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 7
}

78 inception_4c_1x1/Conv2D 8 5 8 5 8 8 8 7

inception_4c_1x1/inception_4c_1x1
inception_4c_1x1/inception_4c_1x1 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
79 inception_4c_1x1/inception_4c_1x1 8 5 8 5

inception_4c_output
inception_4c_output list {
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
80 inception_4c_output 8 5 8 5 8 5 8 5 8 5

inception_4d_pool
inception_4d_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
81 inception_4d_pool 8 5 8 5

inception_4d_pool_proj/Conv2D
inception_4d_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 7
}

82 inception_4d_pool_proj/Conv2D 8 5 8 5 8 9 8 7

inception_4d_pool_proj/inception_4d_pool_proj
inception_4d_pool_proj/inception_4d_pool_proj list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
83 inception_4d_pool_proj/inception_4d_pool_proj 8 5 8 5

inception_4d_5x5_reduce/Conv2D
inception_4d_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 7
}

84 inception_4d_5x5_reduce/Conv2D 8 5 8 4 8 8 8 7

inception_4d_5x5_reduce/inception_4d_5x5_reduce
inception_4d_5x5_reduce/inception_4d_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
85 inception_4d_5x5_reduce/inception_4d_5x5_reduce 8 4 8 4

inception_4d_5x5/Conv2D
inception_4d_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

86 inception_4d_5x5/Conv2D 8 4 8 5 8 10 8 6

inception_4d_5x5/inception_4d_5x5
inception_4d_5x5/inception_4d_5x5 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
87 inception_4d_5x5/inception_4d_5x5 8 5 8 5

inception_4d_3x3_reduce/Conv2D
inception_4d_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 5
}

88 inception_4d_3x3_reduce/Conv2D 8 5 8 4 8 8 8 5

inception_4d_3x3_reduce/inception_4d_3x3_reduce
inception_4d_3x3_reduce/inception_4d_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
89 inception_4d_3x3_reduce/inception_4d_3x3_reduce 8 4 8 4

inception_4d_3x3/Conv2D
inception_4d_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

90 inception_4d_3x3/Conv2D 8 4 8 5 8 9 8 5

inception_4d_3x3/inception_4d_3x3
inception_4d_3x3/inception_4d_3x3 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
91 inception_4d_3x3/inception_4d_3x3 8 5 8 5

inception_4d_1x1/Conv2D
inception_4d_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 7
}

92 inception_4d_1x1/Conv2D 8 5 8 5 8 8 8 7

inception_4d_1x1/inception_4d_1x1
inception_4d_1x1/inception_4d_1x1 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
93 inception_4d_1x1/inception_4d_1x1 8 5 8 5

inception_4d_output
inception_4d_output list {
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
94 inception_4d_output 8 5 8 5 8 5 8 5 8 5

inception_4e_pool
inception_4e_pool list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
95 inception_4e_pool 8 5 8 5

inception_4e_pool_proj/Conv2D
inception_4e_pool_proj/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 6
}

96 inception_4e_pool_proj/Conv2D 8 5 8 4 8 8 8 6

inception_4e_pool_proj/inception_4e_pool_proj
inception_4e_pool_proj/inception_4e_pool_proj list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
97 inception_4e_pool_proj/inception_4e_pool_proj 8 4 8 4

inception_4e_5x5_reduce/Conv2D
inception_4e_5x5_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 7
}

98 inception_4e_5x5_reduce/Conv2D 8 5 8 4 8 8 8 7

inception_4e_5x5_reduce/inception_4e_5x5_reduce
inception_4e_5x5_reduce/inception_4e_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
99 inception_4e_5x5_reduce/inception_4e_5x5_reduce 8 4 8 4

inception_4e_5x5/Conv2D
inception_4e_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

100 inception_4e_5x5/Conv2D 8 4 8 4 8 9 8 5

inception_4e_5x5/inception_4e_5x5
inception_4e_5x5/inception_4e_5x5 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
101 inception_4e_5x5/inception_4e_5x5 8 4 8 4

inception_4e_3x3_reduce/Conv2D
inception_4e_3x3_reduce/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 7
}

102 inception_4e_3x3_reduce/Conv2D 8 5 8 4 8 8 8 7

inception_4e_3x3_reduce/inception_4e_3x3_reduce
inception_4e_3x3_reduce/inception_4e_3x3_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
103 inception_4e_3x3_reduce/inception_4e_3x3_reduce 8 4 8 4

inception_4e_3x3/Conv2D
inception_4e_3x3/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

104 inception_4e_3x3/Conv2D 8 4 8 4 8 9 8 5

inception_4e_3x3/inception_4e_3x3
inception_4e_3x3/inception_4e_3x3 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
105 inception_4e_3x3/inception_4e_3x3 8 4 8 4

inception_4e_1x1/Conv2D
inception_4e_1x1/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 8
}
 list {
  i: 8
  i: 6
}

106 inception_4e_1x1/Conv2D 8 5 8 4 8 8 8 6

inception_4e_1x1/inception_4e_1x1
inception_4e_1x1/inception_4e_1x1 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
107 inception_4e_1x1/inception_4e_1x1 8 4 8 4

inception_4e_output
inception_4e_output list {
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
108 inception_4e_output 8 4 8 4 8 4 8 4 8 4

pool4_3x3_s2
pool4_3x3_s2 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
109 pool4_3x3_s2 8 4 8 4

inception_5a_pool
inception_5a_pool list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
110 inception_5a_pool 8 4 8 4

inception_5a_pool_proj/Conv2D
inception_5a_pool_proj/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 6
}

111 inception_5a_pool_proj/Conv2D 8 4 8 4 8 10 8 6

inception_5a_pool_proj/inception_5a_pool_proj
inception_5a_pool_proj/inception_5a_pool_proj list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
112 inception_5a_pool_proj/inception_5a_pool_proj 8 4 8 4

inception_5a_5x5_reduce/Conv2D
inception_5a_5x5_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 6
}

113 inception_5a_5x5_reduce/Conv2D 8 4 8 4 8 9 8 6

inception_5a_5x5_reduce/inception_5a_5x5_reduce
inception_5a_5x5_reduce/inception_5a_5x5_reduce list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
114 inception_5a_5x5_reduce/inception_5a_5x5_reduce 8 4 8 4

inception_5a_5x5/Conv2D
inception_5a_5x5/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 5
}

115 inception_5a_5x5/Conv2D 8 4 8 4 8 9 8 5

inception_5a_5x5/inception_5a_5x5
inception_5a_5x5/inception_5a_5x5 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
116 inception_5a_5x5/inception_5a_5x5 8 4 8 4

inception_5a_3x3_reduce/Conv2D
inception_5a_3x3_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 6
}

117 inception_5a_3x3_reduce/Conv2D 8 4 8 5 8 9 8 6

inception_5a_3x3_reduce/inception_5a_3x3_reduce
inception_5a_3x3_reduce/inception_5a_3x3_reduce list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 5
}
 [] []
118 inception_5a_3x3_reduce/inception_5a_3x3_reduce 8 5 8 5

inception_5a_3x3/Conv2D
inception_5a_3x3/Conv2D list {
  i: 8
  i: 5
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 5
}

119 inception_5a_3x3/Conv2D 8 5 8 4 8 10 8 5

inception_5a_3x3/inception_5a_3x3
inception_5a_3x3/inception_5a_3x3 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
120 inception_5a_3x3/inception_5a_3x3 8 4 8 4

inception_5a_1x1/Conv2D
inception_5a_1x1/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 6
}

121 inception_5a_1x1/Conv2D 8 4 8 4 8 9 8 6

inception_5a_1x1/inception_5a_1x1
inception_5a_1x1/inception_5a_1x1 list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
122 inception_5a_1x1/inception_5a_1x1 8 4 8 4

inception_5a_output
inception_5a_output list {
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
123 inception_5a_output 8 4 8 4 8 4 8 4 8 4

inception_5b_pool
inception_5b_pool list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
124 inception_5b_pool 8 4 8 4

inception_5b_pool_proj/Conv2D
inception_5b_pool_proj/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 5
}

125 inception_5b_pool_proj/Conv2D 8 4 8 3 8 7 8 5

inception_5b_pool_proj/inception_5b_pool_proj
inception_5b_pool_proj/inception_5b_pool_proj list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 3
}
 [] []
126 inception_5b_pool_proj/inception_5b_pool_proj 8 3 8 3

inception_5b_5x5_reduce/Conv2D
inception_5b_5x5_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 7
}

127 inception_5b_5x5_reduce/Conv2D 8 4 8 6 8 9 8 7

inception_5b_5x5_reduce/inception_5b_5x5_reduce
inception_5b_5x5_reduce/inception_5b_5x5_reduce list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 6
}
 [] []
128 inception_5b_5x5_reduce/inception_5b_5x5_reduce 8 6 8 6

inception_5b_5x5/Conv2D
inception_5b_5x5/Conv2D list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 7
}
 list {
  i: 8
  i: 6
}

129 inception_5b_5x5/Conv2D 8 6 8 3 8 7 8 6

inception_5b_5x5/inception_5b_5x5
inception_5b_5x5/inception_5b_5x5 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 3
}
 [] []
130 inception_5b_5x5/inception_5b_5x5 8 3 8 3

inception_5b_3x3_reduce/Conv2D
inception_5b_3x3_reduce/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 9
}
 list {
  i: 8
  i: 7
}

131 inception_5b_3x3_reduce/Conv2D 8 4 8 6 8 9 8 7

inception_5b_3x3_reduce/inception_5b_3x3_reduce
inception_5b_3x3_reduce/inception_5b_3x3_reduce list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 6
}
 [] []
132 inception_5b_3x3_reduce/inception_5b_3x3_reduce 8 6 8 6

inception_5b_3x3/Conv2D
inception_5b_3x3/Conv2D list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 6
}

133 inception_5b_3x3/Conv2D 8 6 8 3 8 6 8 6

inception_5b_3x3/inception_5b_3x3
inception_5b_3x3/inception_5b_3x3 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 3
}
 [] []
134 inception_5b_3x3/inception_5b_3x3 8 3 8 3

inception_5b_1x1/Conv2D
inception_5b_1x1/Conv2D list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 6
}
 list {
  i: 8
  i: 6
}

135 inception_5b_1x1/Conv2D 8 4 8 3 8 6 8 6

inception_5b_1x1/inception_5b_1x1
inception_5b_1x1/inception_5b_1x1 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 3
}
 [] []
136 inception_5b_1x1/inception_5b_1x1 8 3 8 3

inception_5b_output
inception_5b_output list {
  i: 8
  i: 3
  i: 8
  i: 3
  i: 8
  i: 3
  i: 8
  i: 3
}
 list {
  i: 8
  i: 3
}
 [] []
137 inception_5b_output 8 3 8 3 8 3 8 3 8 3

pool5_7x7_s1
pool5_7x7_s1 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 4
}
 [] []
138 pool5_7x7_s1 8 3 8 4

loss3_classifier/Reshape
loss3_classifier/Reshape list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 4
}
 [] []
139 loss3_classifier/Reshape 8 4 8 4

loss3_classifier/loss3_classifier/MatMul
loss3_classifier/loss3_classifier/MatMul list {
  i: 8
  i: 4
}
 list {
  i: 8
  i: 3
}
 list {
  i: 8
  i: 10
}
 list {
  i: 8
  i: 7
}

140 loss3_classifier/loss3_classifier/MatMul 8 4 8 3 8 10 8 7

loss3_loss3
loss3_loss3 list {
  i: 8
  i: 3
}
 [] [] []
141 loss3_loss3 8 3

Thank you, we wrote the deephi quantization file work/fix_info.txt
Generated model artifacts in /workspace/alveo/examples/tensorflow/work
  quantize_eval_model.pb
  fix_info.txt
  deploy_model.pb
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ ls work/
deploy_model.pb  fix_info.txt  quantize_eval_model.pb
```
## [Partition, compile and run inference](https://github.com/passlab/Vitis-AI/tree/master/alveo/examples/tensorflow#partition-compile-and-run-inference)
```
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ python run.py --validate --model models/inception_v1_baseline.pb --pre_process inception_v1 --output_dir work --input_nodes data --output_nodes loss3_loss3 --c_input_nodes data --c_output_nodes pool5_7x7_s1 --input_shapes ?,224,224,3
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:36: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:41: The name tf.NodeDef is deprecated. Please use tf.compat.v1.NodeDef instead.


######### load_graph arguments #############
networkfile               models/inception_v1_baseline.pb
loadmode                  None
startnode                 None
finalnode                 None
inclusive                 True
batch_sz                  1
fixinputnames             None
placeholdershape          {'data': [1, 224, 224, 3]}
remove_training_nodes     True
remove_redundant_nodes    True
freeze_blacklist          None
freeze_whitelist          None
graph_savepath            None
#############################################

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:345: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
change palceholder data shape to [1, 224, 224, 3]
freeze model
.... node count 325
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:247: remove_training_nodes (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.remove_training_nodes`
.... node count after removing training nodes 325
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:251: extract_sub_graph (from tensorflow.python.framework.graph_util_impl) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.graph_util.extract_sub_graph`
.... node count after removing redundant nodes 325
.... node count after removing blacklisted nodes 325

######### load_graph arguments #############
networkfile               ALREADY LOADED
loadmode                  None
startnode                 ['data']
finalnode                 ['pool5_7x7_s1']
inclusive                 None
batch_sz                  1
fixinputnames             None
placeholdershape          {'data': [1, 224, 224, 3]}
remove_training_nodes     None
remove_redundant_nodes    None
freeze_blacklist          None
freeze_whitelist          None
graph_savepath            None
#############################################

change palceholder data shape to [1, 224, 224, 3]
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/tensorflow_core/python/util/decorator_utils.py:145: GraphKeys.VARIABLES (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.GraphKeys.GLOBAL_VARIABLES` instead.
freeze model
.... node count 318
.... node count after removing training nodes 318
.... node count after removing redundant nodes 318
.... node count after removing blacklisted nodes 318

#####################################
adding nodes to graph
#####################################

build graph connections ....
build graph connections [DONE]


color input path ....
color input path [DONE]


color output path ....
color output path [DONE]


color constants ....
color constants [DONE]

.... is_supported, layer_index, layer_name
.... False,   0, data
.... True ,   1, conv1_7x7_s2/Conv2D
.... True ,   2, conv1_7x7_s2/BiasAdd
.... True ,   3, conv1_7x7_s2/conv1_7x7_s2
.... True ,   4, pool1_3x3_s2
.... True ,   5, conv2_3x3_reduce/Conv2D
.... True ,   6, conv2_3x3_reduce/BiasAdd
.... True ,   7, conv2_3x3_reduce/conv2_3x3_reduce
.... True ,   8, conv2_3x3/Conv2D
.... True ,   9, conv2_3x3/BiasAdd
.... True ,  10, conv2_3x3/conv2_3x3
.... True ,  11, pool2_3x3_s2
.... True ,  12, inception_3a_pool
.... True ,  13, inception_3a_pool_proj/Conv2D
.... True ,  14, inception_3a_pool_proj/BiasAdd
.... True ,  15, inception_3a_pool_proj/inception_3a_pool_proj
.... True ,  16, inception_3a_5x5_reduce/Conv2D
.... True ,  17, inception_3a_5x5_reduce/BiasAdd
.... True ,  18, inception_3a_5x5_reduce/inception_3a_5x5_reduce
.... True ,  19, inception_3a_5x5/Conv2D
.... True ,  20, inception_3a_5x5/BiasAdd
.... True ,  21, inception_3a_5x5/inception_3a_5x5
.... True ,  22, inception_3a_3x3_reduce/Conv2D
.... True ,  23, inception_3a_3x3_reduce/BiasAdd
.... True ,  24, inception_3a_3x3_reduce/inception_3a_3x3_reduce
.... True ,  25, inception_3a_3x3/Conv2D
.... True ,  26, inception_3a_3x3/BiasAdd
.... True ,  27, inception_3a_3x3/inception_3a_3x3
.... True ,  28, inception_3a_1x1/Conv2D
.... True ,  29, inception_3a_1x1/BiasAdd
.... True ,  30, inception_3a_1x1/inception_3a_1x1
.... True ,  31, inception_3a_output
.... True ,  32, inception_3b_pool
.... True ,  33, inception_3b_pool_proj/Conv2D
.... True ,  34, inception_3b_pool_proj/BiasAdd
.... True ,  35, inception_3b_pool_proj/inception_3b_pool_proj
.... True ,  36, inception_3b_5x5_reduce/Conv2D
.... True ,  37, inception_3b_5x5_reduce/BiasAdd
.... True ,  38, inception_3b_5x5_reduce/inception_3b_5x5_reduce
.... True ,  39, inception_3b_5x5/Conv2D
.... True ,  40, inception_3b_5x5/BiasAdd
.... True ,  41, inception_3b_5x5/inception_3b_5x5
.... True ,  42, inception_3b_3x3_reduce/Conv2D
.... True ,  43, inception_3b_3x3_reduce/BiasAdd
.... True ,  44, inception_3b_3x3_reduce/inception_3b_3x3_reduce
.... True ,  45, inception_3b_3x3/Conv2D
.... True ,  46, inception_3b_3x3/BiasAdd
.... True ,  47, inception_3b_3x3/inception_3b_3x3
.... True ,  48, inception_3b_1x1/Conv2D
.... True ,  49, inception_3b_1x1/BiasAdd
.... True ,  50, inception_3b_1x1/inception_3b_1x1
.... True ,  51, inception_3b_output
.... True ,  52, pool3_3x3_s2
.... True ,  53, inception_4a_pool
.... True ,  54, inception_4a_pool_proj/Conv2D
.... True ,  55, inception_4a_pool_proj/BiasAdd
.... True ,  56, inception_4a_pool_proj/inception_4a_pool_proj
.... True ,  57, inception_4a_5x5_reduce/Conv2D
.... True ,  58, inception_4a_5x5_reduce/BiasAdd
.... True ,  59, inception_4a_5x5_reduce/inception_4a_5x5_reduce
.... True ,  60, inception_4a_5x5/Conv2D
.... True ,  61, inception_4a_5x5/BiasAdd
.... True ,  62, inception_4a_5x5/inception_4a_5x5
.... True ,  63, inception_4a_3x3_reduce/Conv2D
.... True ,  64, inception_4a_3x3_reduce/BiasAdd
.... True ,  65, inception_4a_3x3_reduce/inception_4a_3x3_reduce
.... True ,  66, inception_4a_3x3/Conv2D
.... True ,  67, inception_4a_3x3/BiasAdd
.... True ,  68, inception_4a_3x3/inception_4a_3x3
.... True ,  69, inception_4a_1x1/Conv2D
.... True ,  70, inception_4a_1x1/BiasAdd
.... True ,  71, inception_4a_1x1/inception_4a_1x1
.... True ,  72, inception_4a_output
.... True ,  73, inception_4b_pool
.... True ,  74, inception_4b_pool_proj/Conv2D
.... True ,  75, inception_4b_pool_proj/BiasAdd
.... True ,  76, inception_4b_pool_proj/inception_4b_pool_proj
.... True ,  77, inception_4b_5x5_reduce/Conv2D
.... True ,  78, inception_4b_5x5_reduce/BiasAdd
.... True ,  79, inception_4b_5x5_reduce/inception_4b_5x5_reduce
.... True ,  80, inception_4b_5x5/Conv2D
.... True ,  81, inception_4b_5x5/BiasAdd
.... True ,  82, inception_4b_5x5/inception_4b_5x5
.... True ,  83, inception_4b_3x3_reduce/Conv2D
.... True ,  84, inception_4b_3x3_reduce/BiasAdd
.... True ,  85, inception_4b_3x3_reduce/inception_4b_3x3_reduce
.... True ,  86, inception_4b_3x3/Conv2D
.... True ,  87, inception_4b_3x3/BiasAdd
.... True ,  88, inception_4b_3x3/inception_4b_3x3
.... True ,  89, inception_4b_1x1/Conv2D
.... True ,  90, inception_4b_1x1/BiasAdd
.... True ,  91, inception_4b_1x1/inception_4b_1x1
.... True ,  92, inception_4b_output
.... True ,  93, inception_4c_pool
.... True ,  94, inception_4c_pool_proj/Conv2D
.... True ,  95, inception_4c_pool_proj/BiasAdd
.... True ,  96, inception_4c_pool_proj/inception_4c_pool_proj
.... True ,  97, inception_4c_5x5_reduce/Conv2D
.... True ,  98, inception_4c_5x5_reduce/BiasAdd
.... True ,  99, inception_4c_5x5_reduce/inception_4c_5x5_reduce
.... True , 100, inception_4c_5x5/Conv2D
.... True , 101, inception_4c_5x5/BiasAdd
.... True , 102, inception_4c_5x5/inception_4c_5x5
.... True , 103, inception_4c_3x3_reduce/Conv2D
.... True , 104, inception_4c_3x3_reduce/BiasAdd
.... True , 105, inception_4c_3x3_reduce/inception_4c_3x3_reduce
.... True , 106, inception_4c_3x3/Conv2D
.... True , 107, inception_4c_3x3/BiasAdd
.... True , 108, inception_4c_3x3/inception_4c_3x3
.... True , 109, inception_4c_1x1/Conv2D
.... True , 110, inception_4c_1x1/BiasAdd
.... True , 111, inception_4c_1x1/inception_4c_1x1
.... True , 112, inception_4c_output
.... True , 113, inception_4d_pool
.... True , 114, inception_4d_pool_proj/Conv2D
.... True , 115, inception_4d_pool_proj/BiasAdd
.... True , 116, inception_4d_pool_proj/inception_4d_pool_proj
.... True , 117, inception_4d_5x5_reduce/Conv2D
.... True , 118, inception_4d_5x5_reduce/BiasAdd
.... True , 119, inception_4d_5x5_reduce/inception_4d_5x5_reduce
.... True , 120, inception_4d_5x5/Conv2D
.... True , 121, inception_4d_5x5/BiasAdd
.... True , 122, inception_4d_5x5/inception_4d_5x5
.... True , 123, inception_4d_3x3_reduce/Conv2D
.... True , 124, inception_4d_3x3_reduce/BiasAdd
.... True , 125, inception_4d_3x3_reduce/inception_4d_3x3_reduce
.... True , 126, inception_4d_3x3/Conv2D
.... True , 127, inception_4d_3x3/BiasAdd
.... True , 128, inception_4d_3x3/inception_4d_3x3
.... True , 129, inception_4d_1x1/Conv2D
.... True , 130, inception_4d_1x1/BiasAdd
.... True , 131, inception_4d_1x1/inception_4d_1x1
.... True , 132, inception_4d_output
.... True , 133, inception_4e_pool
.... True , 134, inception_4e_pool_proj/Conv2D
.... True , 135, inception_4e_pool_proj/BiasAdd
.... True , 136, inception_4e_pool_proj/inception_4e_pool_proj
.... True , 137, inception_4e_5x5_reduce/Conv2D
.... True , 138, inception_4e_5x5_reduce/BiasAdd
.... True , 139, inception_4e_5x5_reduce/inception_4e_5x5_reduce
.... True , 140, inception_4e_5x5/Conv2D
.... True , 141, inception_4e_5x5/BiasAdd
.... True , 142, inception_4e_5x5/inception_4e_5x5
.... True , 143, inception_4e_3x3_reduce/Conv2D
.... True , 144, inception_4e_3x3_reduce/BiasAdd
.... True , 145, inception_4e_3x3_reduce/inception_4e_3x3_reduce
.... True , 146, inception_4e_3x3/Conv2D
.... True , 147, inception_4e_3x3/BiasAdd
.... True , 148, inception_4e_3x3/inception_4e_3x3
.... True , 149, inception_4e_1x1/Conv2D
.... True , 150, inception_4e_1x1/BiasAdd
.... True , 151, inception_4e_1x1/inception_4e_1x1
.... True , 152, inception_4e_output
.... True , 153, pool4_3x3_s2
.... True , 154, inception_5a_pool
.... True , 155, inception_5a_pool_proj/Conv2D
.... True , 156, inception_5a_pool_proj/BiasAdd
.... True , 157, inception_5a_pool_proj/inception_5a_pool_proj
.... True , 158, inception_5a_5x5_reduce/Conv2D
.... True , 159, inception_5a_5x5_reduce/BiasAdd
.... True , 160, inception_5a_5x5_reduce/inception_5a_5x5_reduce
.... True , 161, inception_5a_5x5/Conv2D
.... True , 162, inception_5a_5x5/BiasAdd
.... True , 163, inception_5a_5x5/inception_5a_5x5
.... True , 164, inception_5a_3x3_reduce/Conv2D
.... True , 165, inception_5a_3x3_reduce/BiasAdd
.... True , 166, inception_5a_3x3_reduce/inception_5a_3x3_reduce
.... True , 167, inception_5a_3x3/Conv2D
.... True , 168, inception_5a_3x3/BiasAdd
.... True , 169, inception_5a_3x3/inception_5a_3x3
.... True , 170, inception_5a_1x1/Conv2D
.... True , 171, inception_5a_1x1/BiasAdd
.... True , 172, inception_5a_1x1/inception_5a_1x1
.... True , 173, inception_5a_output
.... True , 174, inception_5b_pool
.... True , 175, inception_5b_pool_proj/Conv2D
.... True , 176, inception_5b_pool_proj/BiasAdd
.... True , 177, inception_5b_pool_proj/inception_5b_pool_proj
.... True , 178, inception_5b_5x5_reduce/Conv2D
.... True , 179, inception_5b_5x5_reduce/BiasAdd
.... True , 180, inception_5b_5x5_reduce/inception_5b_5x5_reduce
.... True , 181, inception_5b_5x5/Conv2D
.... True , 182, inception_5b_5x5/BiasAdd
.... True , 183, inception_5b_5x5/inception_5b_5x5
.... True , 184, inception_5b_3x3_reduce/Conv2D
.... True , 185, inception_5b_3x3_reduce/BiasAdd
.... True , 186, inception_5b_3x3_reduce/inception_5b_3x3_reduce
.... True , 187, inception_5b_3x3/Conv2D
.... True , 188, inception_5b_3x3/BiasAdd
.... True , 189, inception_5b_3x3/inception_5b_3x3
.... True , 190, inception_5b_1x1/Conv2D
.... True , 191, inception_5b_1x1/BiasAdd
.... True , 192, inception_5b_1x1/inception_5b_1x1
.... True , 193, inception_5b_output
.... True , 194, pool5_7x7_s1

Partition FPGA (un)supported layers from compiler schedule ....
Partition FPGA (un)supported layers from compiler schedule [DONE]

Refine Graph Partitions ....
.... partition (  0, False) --> [(1, True)]
.... partition (  1, True ) --> []
....
.... partition (  0, False) <-- []
.... partition (  1, True ) <-- [(0, False)]
....

SUMMARY:
.... partition_index "0" - SUPPORTED: False
........ inputs:          ['data']
........ inputs actual:   ['data']
........ outputs:         ['data']
........ outputs actual:  ['data']
.... partition_index "1" - SUPPORTED: True
........ inputs:          ['data']
........ inputs actual:   ['data']
........ outputs:         ['pool5_7x7_s1']
........ outputs actual:  ['pool5_7x7_s1']
Refine Graph Partitions [DONE]
Re-compile partition_index "1"
GenerateCode: ./inception_v1_baseline_partition_01.pb
Weights: ./inception_v1_baseline_partition_01.pb
PngFile: None
ConcatStrategy: None
Strategy: all
ScheduleFile: None
DDR: 256
DSP: 96
Verbose: False
FromTF: True
Memory: 9
Byte per Pixel: 1
Start compiling

**************************************************
* BUILDING DATA FLOW GRAPH
**************************************************
Reading pre-build graph

######### load_graph arguments #############
networkfile               ALREADY LOADED
loadmode                  binary
startnode                 ['data']
finalnode                 ['pool5_7x7_s1']
inclusive                 False
batch_sz                  1
fixinputnames             None
placeholdershape          {'data': [1, 224, 224, 3]}
remove_training_nodes     None
remove_redundant_nodes    None
freeze_blacklist          None
freeze_whitelist          None
graph_savepath            None
#############################################

change palceholder data shape to [1, 224, 224, 3]
freeze model
.... node count 318
.... node count after removing training nodes 318
.... node count after removing redundant nodes 318
.... node count after removing blacklisted nodes 318
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vaic/dpuv1/bin/xfdnn_compiler_tensorflow.py:204: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2020-11-29 11:59:31.995420: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-11-29 11:59:32.043898: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:59:32.044141: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:59:32.045733: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:59:32.046889: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:59:32.047145: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:59:32.048473: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:59:32.049385: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:59:32.052233: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:59:32.055692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:59:32.057925: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
2020-11-29 11:59:32.096386: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300000000 Hz
2020-11-29 11:59:32.099438: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5615c191c300 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-11-29 11:59:32.099455: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-11-29 11:59:32.102403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:59:32.102445: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:59:32.102460: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:59:32.102474: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:59:32.102488: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:59:32.102503: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:59:32.102517: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:59:32.102533: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:59:32.107088: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:59:32.107122: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:59:32.250103: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1180] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-29 11:59:32.250129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1186]      0 
2020-11-29 11:59:32.250134: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1199] 0:   N 
2020-11-29 11:59:32.257157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0)
2020-11-29 11:59:32.259735: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5615bd8881d0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-11-29 11:59:32.259750: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-PCIE-32GB, Compute Capability 7.0

**************************************************
* CONVERTING GRAPH TO SCHEDULE
**************************************************
Schedule Idx 0 TF Operation Name: conv1_7x7_s2/weights type Const
WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

Schedule Idx 1 TF Operation Name: data type Placeholder
Schedule Idx 2 TF Operation Name: conv1_7x7_s2/Conv2D type Conv2D
Asimmetric Padding 2 3 2 3 SizeType(batches=1, channels=64, height=112, width=112) SizeType(batches=64, channels=3, height=7, width=7) SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=3, height=224, width=224)
Schedule Idx 3 TF Operation Name: conv1_7x7_s2/biases type Const
Schedule Idx 4 TF Operation Name: conv1_7x7_s2/BiasAdd type BiasAdd
Schedule Idx 5 TF Operation Name: conv1_7x7_s2/conv1_7x7_s2 type Relu
Schedule Idx 6 TF Operation Name: pool1_3x3_s2 type MaxPool
Asimmetric Padding 0 1 0 1 SizeType(batches=1, channels=64, height=56, width=56) SizeType(batches=1, channels=1, height=3, width=3) SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=64, height=112, width=112)
Schedule Idx 7 TF Operation Name: conv2_3x3_reduce/weights type Const
Schedule Idx 8 TF Operation Name: conv2_3x3_reduce/Conv2D type Conv2D
Schedule Idx 9 TF Operation Name: conv2_3x3_reduce/biases type Const
Schedule Idx 10 TF Operation Name: conv2_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 11 TF Operation Name: conv2_3x3_reduce/conv2_3x3_reduce type Relu
Schedule Idx 12 TF Operation Name: conv2_3x3/weights type Const
Schedule Idx 13 TF Operation Name: conv2_3x3/Conv2D type Conv2D
Schedule Idx 14 TF Operation Name: conv2_3x3/biases type Const
Schedule Idx 15 TF Operation Name: conv2_3x3/BiasAdd type BiasAdd
Schedule Idx 16 TF Operation Name: conv2_3x3/conv2_3x3 type Relu
Schedule Idx 17 TF Operation Name: pool2_3x3_s2 type MaxPool
Asimmetric Padding 0 1 0 1 SizeType(batches=1, channels=192, height=28, width=28) SizeType(batches=1, channels=1, height=3, width=3) SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=192, height=56, width=56)
Schedule Idx 18 TF Operation Name: inception_3a_1x1/weights type Const
Schedule Idx 19 TF Operation Name: inception_3a_1x1/Conv2D type Conv2D
Schedule Idx 20 TF Operation Name: inception_3a_1x1/biases type Const
Schedule Idx 21 TF Operation Name: inception_3a_1x1/BiasAdd type BiasAdd
Schedule Idx 22 TF Operation Name: inception_3a_1x1/inception_3a_1x1 type Relu
Schedule Idx 23 TF Operation Name: inception_3a_3x3_reduce/weights type Const
Schedule Idx 24 TF Operation Name: inception_3a_3x3_reduce/Conv2D type Conv2D
Schedule Idx 25 TF Operation Name: inception_3a_3x3_reduce/biases type Const
Schedule Idx 26 TF Operation Name: inception_3a_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 27 TF Operation Name: inception_3a_3x3_reduce/inception_3a_3x3_reduce type Relu
Schedule Idx 28 TF Operation Name: inception_3a_3x3/weights type Const
Schedule Idx 29 TF Operation Name: inception_3a_3x3/Conv2D type Conv2D
Schedule Idx 30 TF Operation Name: inception_3a_3x3/biases type Const
Schedule Idx 31 TF Operation Name: inception_3a_3x3/BiasAdd type BiasAdd
Schedule Idx 32 TF Operation Name: inception_3a_3x3/inception_3a_3x3 type Relu
Schedule Idx 33 TF Operation Name: inception_3a_5x5_reduce/weights type Const
Schedule Idx 34 TF Operation Name: inception_3a_5x5_reduce/Conv2D type Conv2D
Schedule Idx 35 TF Operation Name: inception_3a_5x5_reduce/biases type Const
Schedule Idx 36 TF Operation Name: inception_3a_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 37 TF Operation Name: inception_3a_5x5_reduce/inception_3a_5x5_reduce type Relu
Schedule Idx 38 TF Operation Name: inception_3a_5x5/weights type Const
Schedule Idx 39 TF Operation Name: inception_3a_5x5/Conv2D type Conv2D
Schedule Idx 40 TF Operation Name: inception_3a_5x5/biases type Const
Schedule Idx 41 TF Operation Name: inception_3a_5x5/BiasAdd type BiasAdd
Schedule Idx 42 TF Operation Name: inception_3a_5x5/inception_3a_5x5 type Relu
Schedule Idx 43 TF Operation Name: inception_3a_pool type MaxPool
Schedule Idx 44 TF Operation Name: inception_3a_pool_proj/weights type Const
Schedule Idx 45 TF Operation Name: inception_3a_pool_proj/Conv2D type Conv2D
Schedule Idx 46 TF Operation Name: inception_3a_pool_proj/biases type Const
Schedule Idx 47 TF Operation Name: inception_3a_pool_proj/BiasAdd type BiasAdd
Schedule Idx 48 TF Operation Name: inception_3a_pool_proj/inception_3a_pool_proj type Relu
Schedule Idx 49 TF Operation Name: inception_3a_output/axis type Const
Schedule Idx 50 TF Operation Name: inception_3a_output type ConcatV2
Schedule Idx 51 TF Operation Name: inception_3b_1x1/weights type Const
Schedule Idx 52 TF Operation Name: inception_3b_1x1/Conv2D type Conv2D
Schedule Idx 53 TF Operation Name: inception_3b_1x1/biases type Const
Schedule Idx 54 TF Operation Name: inception_3b_1x1/BiasAdd type BiasAdd
Schedule Idx 55 TF Operation Name: inception_3b_1x1/inception_3b_1x1 type Relu
Schedule Idx 56 TF Operation Name: inception_3b_3x3_reduce/weights type Const
Schedule Idx 57 TF Operation Name: inception_3b_3x3_reduce/Conv2D type Conv2D
Schedule Idx 58 TF Operation Name: inception_3b_3x3_reduce/biases type Const
Schedule Idx 59 TF Operation Name: inception_3b_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 60 TF Operation Name: inception_3b_3x3_reduce/inception_3b_3x3_reduce type Relu
Schedule Idx 61 TF Operation Name: inception_3b_3x3/weights type Const
Schedule Idx 62 TF Operation Name: inception_3b_3x3/Conv2D type Conv2D
Schedule Idx 63 TF Operation Name: inception_3b_3x3/biases type Const
Schedule Idx 64 TF Operation Name: inception_3b_3x3/BiasAdd type BiasAdd
Schedule Idx 65 TF Operation Name: inception_3b_3x3/inception_3b_3x3 type Relu
Schedule Idx 66 TF Operation Name: inception_3b_5x5_reduce/weights type Const
Schedule Idx 67 TF Operation Name: inception_3b_5x5_reduce/Conv2D type Conv2D
Schedule Idx 68 TF Operation Name: inception_3b_5x5_reduce/biases type Const
Schedule Idx 69 TF Operation Name: inception_3b_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 70 TF Operation Name: inception_3b_5x5_reduce/inception_3b_5x5_reduce type Relu
Schedule Idx 71 TF Operation Name: inception_3b_5x5/weights type Const
Schedule Idx 72 TF Operation Name: inception_3b_5x5/Conv2D type Conv2D
Schedule Idx 73 TF Operation Name: inception_3b_5x5/biases type Const
Schedule Idx 74 TF Operation Name: inception_3b_5x5/BiasAdd type BiasAdd
Schedule Idx 75 TF Operation Name: inception_3b_5x5/inception_3b_5x5 type Relu
Schedule Idx 76 TF Operation Name: inception_3b_pool type MaxPool
Schedule Idx 77 TF Operation Name: inception_3b_pool_proj/weights type Const
Schedule Idx 78 TF Operation Name: inception_3b_pool_proj/Conv2D type Conv2D
Schedule Idx 79 TF Operation Name: inception_3b_pool_proj/biases type Const
Schedule Idx 80 TF Operation Name: inception_3b_pool_proj/BiasAdd type BiasAdd
Schedule Idx 81 TF Operation Name: inception_3b_pool_proj/inception_3b_pool_proj type Relu
Schedule Idx 82 TF Operation Name: inception_3b_output/axis type Const
Schedule Idx 83 TF Operation Name: inception_3b_output type ConcatV2
Schedule Idx 84 TF Operation Name: pool3_3x3_s2 type MaxPool
Asimmetric Padding 0 1 0 1 SizeType(batches=1, channels=480, height=14, width=14) SizeType(batches=1, channels=1, height=3, width=3) SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=480, height=28, width=28)
Schedule Idx 85 TF Operation Name: inception_4a_1x1/weights type Const
Schedule Idx 86 TF Operation Name: inception_4a_1x1/Conv2D type Conv2D
Schedule Idx 87 TF Operation Name: inception_4a_1x1/biases type Const
Schedule Idx 88 TF Operation Name: inception_4a_1x1/BiasAdd type BiasAdd
Schedule Idx 89 TF Operation Name: inception_4a_1x1/inception_4a_1x1 type Relu
Schedule Idx 90 TF Operation Name: inception_4a_3x3_reduce/weights type Const
Schedule Idx 91 TF Operation Name: inception_4a_3x3_reduce/Conv2D type Conv2D
Schedule Idx 92 TF Operation Name: inception_4a_3x3_reduce/biases type Const
Schedule Idx 93 TF Operation Name: inception_4a_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 94 TF Operation Name: inception_4a_3x3_reduce/inception_4a_3x3_reduce type Relu
Schedule Idx 95 TF Operation Name: inception_4a_3x3/weights type Const
Schedule Idx 96 TF Operation Name: inception_4a_3x3/Conv2D type Conv2D
Schedule Idx 97 TF Operation Name: inception_4a_3x3/biases type Const
Schedule Idx 98 TF Operation Name: inception_4a_3x3/BiasAdd type BiasAdd
Schedule Idx 99 TF Operation Name: inception_4a_3x3/inception_4a_3x3 type Relu
Schedule Idx 100 TF Operation Name: inception_4a_5x5_reduce/weights type Const
Schedule Idx 101 TF Operation Name: inception_4a_5x5_reduce/Conv2D type Conv2D
Schedule Idx 102 TF Operation Name: inception_4a_5x5_reduce/biases type Const
Schedule Idx 103 TF Operation Name: inception_4a_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 104 TF Operation Name: inception_4a_5x5_reduce/inception_4a_5x5_reduce type Relu
Schedule Idx 105 TF Operation Name: inception_4a_5x5/weights type Const
Schedule Idx 106 TF Operation Name: inception_4a_5x5/Conv2D type Conv2D
Schedule Idx 107 TF Operation Name: inception_4a_5x5/biases type Const
Schedule Idx 108 TF Operation Name: inception_4a_5x5/BiasAdd type BiasAdd
Schedule Idx 109 TF Operation Name: inception_4a_5x5/inception_4a_5x5 type Relu
Schedule Idx 110 TF Operation Name: inception_4a_pool type MaxPool
Schedule Idx 111 TF Operation Name: inception_4a_pool_proj/weights type Const
Schedule Idx 112 TF Operation Name: inception_4a_pool_proj/Conv2D type Conv2D
Schedule Idx 113 TF Operation Name: inception_4a_pool_proj/biases type Const
Schedule Idx 114 TF Operation Name: inception_4a_pool_proj/BiasAdd type BiasAdd
Schedule Idx 115 TF Operation Name: inception_4a_pool_proj/inception_4a_pool_proj type Relu
Schedule Idx 116 TF Operation Name: inception_4a_output/axis type Const
Schedule Idx 117 TF Operation Name: inception_4a_output type ConcatV2
Schedule Idx 118 TF Operation Name: inception_4b_1x1/weights type Const
Schedule Idx 119 TF Operation Name: inception_4b_1x1/Conv2D type Conv2D
Schedule Idx 120 TF Operation Name: inception_4b_1x1/biases type Const
Schedule Idx 121 TF Operation Name: inception_4b_1x1/BiasAdd type BiasAdd
Schedule Idx 122 TF Operation Name: inception_4b_1x1/inception_4b_1x1 type Relu
Schedule Idx 123 TF Operation Name: inception_4b_3x3_reduce/weights type Const
Schedule Idx 124 TF Operation Name: inception_4b_3x3_reduce/Conv2D type Conv2D
Schedule Idx 125 TF Operation Name: inception_4b_3x3_reduce/biases type Const
Schedule Idx 126 TF Operation Name: inception_4b_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 127 TF Operation Name: inception_4b_3x3_reduce/inception_4b_3x3_reduce type Relu
Schedule Idx 128 TF Operation Name: inception_4b_3x3/weights type Const
Schedule Idx 129 TF Operation Name: inception_4b_3x3/Conv2D type Conv2D
Schedule Idx 130 TF Operation Name: inception_4b_3x3/biases type Const
Schedule Idx 131 TF Operation Name: inception_4b_3x3/BiasAdd type BiasAdd
Schedule Idx 132 TF Operation Name: inception_4b_3x3/inception_4b_3x3 type Relu
Schedule Idx 133 TF Operation Name: inception_4b_5x5_reduce/weights type Const
Schedule Idx 134 TF Operation Name: inception_4b_5x5_reduce/Conv2D type Conv2D
Schedule Idx 135 TF Operation Name: inception_4b_5x5_reduce/biases type Const
Schedule Idx 136 TF Operation Name: inception_4b_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 137 TF Operation Name: inception_4b_5x5_reduce/inception_4b_5x5_reduce type Relu
Schedule Idx 138 TF Operation Name: inception_4b_5x5/weights type Const
Schedule Idx 139 TF Operation Name: inception_4b_5x5/Conv2D type Conv2D
Schedule Idx 140 TF Operation Name: inception_4b_5x5/biases type Const
Schedule Idx 141 TF Operation Name: inception_4b_5x5/BiasAdd type BiasAdd
Schedule Idx 142 TF Operation Name: inception_4b_5x5/inception_4b_5x5 type Relu
Schedule Idx 143 TF Operation Name: inception_4b_pool type MaxPool
Schedule Idx 144 TF Operation Name: inception_4b_pool_proj/weights type Const
Schedule Idx 145 TF Operation Name: inception_4b_pool_proj/Conv2D type Conv2D
Schedule Idx 146 TF Operation Name: inception_4b_pool_proj/biases type Const
Schedule Idx 147 TF Operation Name: inception_4b_pool_proj/BiasAdd type BiasAdd
Schedule Idx 148 TF Operation Name: inception_4b_pool_proj/inception_4b_pool_proj type Relu
Schedule Idx 149 TF Operation Name: inception_4b_output/axis type Const
Schedule Idx 150 TF Operation Name: inception_4b_output type ConcatV2
Schedule Idx 151 TF Operation Name: inception_4c_1x1/weights type Const
Schedule Idx 152 TF Operation Name: inception_4c_1x1/Conv2D type Conv2D
Schedule Idx 153 TF Operation Name: inception_4c_1x1/biases type Const
Schedule Idx 154 TF Operation Name: inception_4c_1x1/BiasAdd type BiasAdd
Schedule Idx 155 TF Operation Name: inception_4c_1x1/inception_4c_1x1 type Relu
Schedule Idx 156 TF Operation Name: inception_4c_3x3_reduce/weights type Const
Schedule Idx 157 TF Operation Name: inception_4c_3x3_reduce/Conv2D type Conv2D
Schedule Idx 158 TF Operation Name: inception_4c_3x3_reduce/biases type Const
Schedule Idx 159 TF Operation Name: inception_4c_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 160 TF Operation Name: inception_4c_3x3_reduce/inception_4c_3x3_reduce type Relu
Schedule Idx 161 TF Operation Name: inception_4c_3x3/weights type Const
Schedule Idx 162 TF Operation Name: inception_4c_3x3/Conv2D type Conv2D
Schedule Idx 163 TF Operation Name: inception_4c_3x3/biases type Const
Schedule Idx 164 TF Operation Name: inception_4c_3x3/BiasAdd type BiasAdd
Schedule Idx 165 TF Operation Name: inception_4c_3x3/inception_4c_3x3 type Relu
Schedule Idx 166 TF Operation Name: inception_4c_5x5_reduce/weights type Const
Schedule Idx 167 TF Operation Name: inception_4c_5x5_reduce/Conv2D type Conv2D
Schedule Idx 168 TF Operation Name: inception_4c_5x5_reduce/biases type Const
Schedule Idx 169 TF Operation Name: inception_4c_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 170 TF Operation Name: inception_4c_5x5_reduce/inception_4c_5x5_reduce type Relu
Schedule Idx 171 TF Operation Name: inception_4c_5x5/weights type Const
Schedule Idx 172 TF Operation Name: inception_4c_5x5/Conv2D type Conv2D
Schedule Idx 173 TF Operation Name: inception_4c_5x5/biases type Const
Schedule Idx 174 TF Operation Name: inception_4c_5x5/BiasAdd type BiasAdd
Schedule Idx 175 TF Operation Name: inception_4c_5x5/inception_4c_5x5 type Relu
Schedule Idx 176 TF Operation Name: inception_4c_pool type MaxPool
Schedule Idx 177 TF Operation Name: inception_4c_pool_proj/weights type Const
Schedule Idx 178 TF Operation Name: inception_4c_pool_proj/Conv2D type Conv2D
Schedule Idx 179 TF Operation Name: inception_4c_pool_proj/biases type Const
Schedule Idx 180 TF Operation Name: inception_4c_pool_proj/BiasAdd type BiasAdd
Schedule Idx 181 TF Operation Name: inception_4c_pool_proj/inception_4c_pool_proj type Relu
Schedule Idx 182 TF Operation Name: inception_4c_output/axis type Const
Schedule Idx 183 TF Operation Name: inception_4c_output type ConcatV2
Schedule Idx 184 TF Operation Name: inception_4d_1x1/weights type Const
Schedule Idx 185 TF Operation Name: inception_4d_1x1/Conv2D type Conv2D
Schedule Idx 186 TF Operation Name: inception_4d_1x1/biases type Const
Schedule Idx 187 TF Operation Name: inception_4d_1x1/BiasAdd type BiasAdd
Schedule Idx 188 TF Operation Name: inception_4d_1x1/inception_4d_1x1 type Relu
Schedule Idx 189 TF Operation Name: inception_4d_3x3_reduce/weights type Const
Schedule Idx 190 TF Operation Name: inception_4d_3x3_reduce/Conv2D type Conv2D
Schedule Idx 191 TF Operation Name: inception_4d_3x3_reduce/biases type Const
Schedule Idx 192 TF Operation Name: inception_4d_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 193 TF Operation Name: inception_4d_3x3_reduce/inception_4d_3x3_reduce type Relu
Schedule Idx 194 TF Operation Name: inception_4d_3x3/weights type Const
Schedule Idx 195 TF Operation Name: inception_4d_3x3/Conv2D type Conv2D
Schedule Idx 196 TF Operation Name: inception_4d_3x3/biases type Const
Schedule Idx 197 TF Operation Name: inception_4d_3x3/BiasAdd type BiasAdd
Schedule Idx 198 TF Operation Name: inception_4d_3x3/inception_4d_3x3 type Relu
Schedule Idx 199 TF Operation Name: inception_4d_5x5_reduce/weights type Const
Schedule Idx 200 TF Operation Name: inception_4d_5x5_reduce/Conv2D type Conv2D
Schedule Idx 201 TF Operation Name: inception_4d_5x5_reduce/biases type Const
Schedule Idx 202 TF Operation Name: inception_4d_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 203 TF Operation Name: inception_4d_5x5_reduce/inception_4d_5x5_reduce type Relu
Schedule Idx 204 TF Operation Name: inception_4d_5x5/weights type Const
Schedule Idx 205 TF Operation Name: inception_4d_5x5/Conv2D type Conv2D
Schedule Idx 206 TF Operation Name: inception_4d_5x5/biases type Const
Schedule Idx 207 TF Operation Name: inception_4d_5x5/BiasAdd type BiasAdd
Schedule Idx 208 TF Operation Name: inception_4d_5x5/inception_4d_5x5 type Relu
Schedule Idx 209 TF Operation Name: inception_4d_pool type MaxPool
Schedule Idx 210 TF Operation Name: inception_4d_pool_proj/weights type Const
Schedule Idx 211 TF Operation Name: inception_4d_pool_proj/Conv2D type Conv2D
Schedule Idx 212 TF Operation Name: inception_4d_pool_proj/biases type Const
Schedule Idx 213 TF Operation Name: inception_4d_pool_proj/BiasAdd type BiasAdd
Schedule Idx 214 TF Operation Name: inception_4d_pool_proj/inception_4d_pool_proj type Relu
Schedule Idx 215 TF Operation Name: inception_4d_output/axis type Const
Schedule Idx 216 TF Operation Name: inception_4d_output type ConcatV2
Schedule Idx 217 TF Operation Name: inception_4e_1x1/weights type Const
Schedule Idx 218 TF Operation Name: inception_4e_1x1/Conv2D type Conv2D
Schedule Idx 219 TF Operation Name: inception_4e_1x1/biases type Const
Schedule Idx 220 TF Operation Name: inception_4e_1x1/BiasAdd type BiasAdd
Schedule Idx 221 TF Operation Name: inception_4e_1x1/inception_4e_1x1 type Relu
Schedule Idx 222 TF Operation Name: inception_4e_3x3_reduce/weights type Const
Schedule Idx 223 TF Operation Name: inception_4e_3x3_reduce/Conv2D type Conv2D
Schedule Idx 224 TF Operation Name: inception_4e_3x3_reduce/biases type Const
Schedule Idx 225 TF Operation Name: inception_4e_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 226 TF Operation Name: inception_4e_3x3_reduce/inception_4e_3x3_reduce type Relu
Schedule Idx 227 TF Operation Name: inception_4e_3x3/weights type Const
Schedule Idx 228 TF Operation Name: inception_4e_3x3/Conv2D type Conv2D
Schedule Idx 229 TF Operation Name: inception_4e_3x3/biases type Const
Schedule Idx 230 TF Operation Name: inception_4e_3x3/BiasAdd type BiasAdd
Schedule Idx 231 TF Operation Name: inception_4e_3x3/inception_4e_3x3 type Relu
Schedule Idx 232 TF Operation Name: inception_4e_5x5_reduce/weights type Const
Schedule Idx 233 TF Operation Name: inception_4e_5x5_reduce/Conv2D type Conv2D
Schedule Idx 234 TF Operation Name: inception_4e_5x5_reduce/biases type Const
Schedule Idx 235 TF Operation Name: inception_4e_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 236 TF Operation Name: inception_4e_5x5_reduce/inception_4e_5x5_reduce type Relu
Schedule Idx 237 TF Operation Name: inception_4e_5x5/weights type Const
Schedule Idx 238 TF Operation Name: inception_4e_5x5/Conv2D type Conv2D
Schedule Idx 239 TF Operation Name: inception_4e_5x5/biases type Const
Schedule Idx 240 TF Operation Name: inception_4e_5x5/BiasAdd type BiasAdd
Schedule Idx 241 TF Operation Name: inception_4e_5x5/inception_4e_5x5 type Relu
Schedule Idx 242 TF Operation Name: inception_4e_pool type MaxPool
Schedule Idx 243 TF Operation Name: inception_4e_pool_proj/weights type Const
Schedule Idx 244 TF Operation Name: inception_4e_pool_proj/Conv2D type Conv2D
Schedule Idx 245 TF Operation Name: inception_4e_pool_proj/biases type Const
Schedule Idx 246 TF Operation Name: inception_4e_pool_proj/BiasAdd type BiasAdd
Schedule Idx 247 TF Operation Name: inception_4e_pool_proj/inception_4e_pool_proj type Relu
Schedule Idx 248 TF Operation Name: inception_4e_output/axis type Const
Schedule Idx 249 TF Operation Name: inception_4e_output type ConcatV2
Schedule Idx 250 TF Operation Name: pool4_3x3_s2 type MaxPool
Asimmetric Padding 0 1 0 1 SizeType(batches=1, channels=832, height=7, width=7) SizeType(batches=1, channels=1, height=3, width=3) SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=832, height=14, width=14)
Schedule Idx 251 TF Operation Name: inception_5a_1x1/weights type Const
Schedule Idx 252 TF Operation Name: inception_5a_1x1/Conv2D type Conv2D
Schedule Idx 253 TF Operation Name: inception_5a_1x1/biases type Const
Schedule Idx 254 TF Operation Name: inception_5a_1x1/BiasAdd type BiasAdd
Schedule Idx 255 TF Operation Name: inception_5a_1x1/inception_5a_1x1 type Relu
Schedule Idx 256 TF Operation Name: inception_5a_3x3_reduce/weights type Const
Schedule Idx 257 TF Operation Name: inception_5a_3x3_reduce/Conv2D type Conv2D
Schedule Idx 258 TF Operation Name: inception_5a_3x3_reduce/biases type Const
Schedule Idx 259 TF Operation Name: inception_5a_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 260 TF Operation Name: inception_5a_3x3_reduce/inception_5a_3x3_reduce type Relu
Schedule Idx 261 TF Operation Name: inception_5a_3x3/weights type Const
Schedule Idx 262 TF Operation Name: inception_5a_3x3/Conv2D type Conv2D
Schedule Idx 263 TF Operation Name: inception_5a_3x3/biases type Const
Schedule Idx 264 TF Operation Name: inception_5a_3x3/BiasAdd type BiasAdd
Schedule Idx 265 TF Operation Name: inception_5a_3x3/inception_5a_3x3 type Relu
Schedule Idx 266 TF Operation Name: inception_5a_5x5_reduce/weights type Const
Schedule Idx 267 TF Operation Name: inception_5a_5x5_reduce/Conv2D type Conv2D
Schedule Idx 268 TF Operation Name: inception_5a_5x5_reduce/biases type Const
Schedule Idx 269 TF Operation Name: inception_5a_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 270 TF Operation Name: inception_5a_5x5_reduce/inception_5a_5x5_reduce type Relu
Schedule Idx 271 TF Operation Name: inception_5a_5x5/weights type Const
Schedule Idx 272 TF Operation Name: inception_5a_5x5/Conv2D type Conv2D
Schedule Idx 273 TF Operation Name: inception_5a_5x5/biases type Const
Schedule Idx 274 TF Operation Name: inception_5a_5x5/BiasAdd type BiasAdd
Schedule Idx 275 TF Operation Name: inception_5a_5x5/inception_5a_5x5 type Relu
Schedule Idx 276 TF Operation Name: inception_5a_pool type MaxPool
Schedule Idx 277 TF Operation Name: inception_5a_pool_proj/weights type Const
Schedule Idx 278 TF Operation Name: inception_5a_pool_proj/Conv2D type Conv2D
Schedule Idx 279 TF Operation Name: inception_5a_pool_proj/biases type Const
Schedule Idx 280 TF Operation Name: inception_5a_pool_proj/BiasAdd type BiasAdd
Schedule Idx 281 TF Operation Name: inception_5a_pool_proj/inception_5a_pool_proj type Relu
Schedule Idx 282 TF Operation Name: inception_5a_output/axis type Const
Schedule Idx 283 TF Operation Name: inception_5a_output type ConcatV2
Schedule Idx 284 TF Operation Name: inception_5b_1x1/weights type Const
Schedule Idx 285 TF Operation Name: inception_5b_1x1/Conv2D type Conv2D
Schedule Idx 286 TF Operation Name: inception_5b_1x1/biases type Const
Schedule Idx 287 TF Operation Name: inception_5b_1x1/BiasAdd type BiasAdd
Schedule Idx 288 TF Operation Name: inception_5b_1x1/inception_5b_1x1 type Relu
Schedule Idx 289 TF Operation Name: inception_5b_3x3_reduce/weights type Const
Schedule Idx 290 TF Operation Name: inception_5b_3x3_reduce/Conv2D type Conv2D
Schedule Idx 291 TF Operation Name: inception_5b_3x3_reduce/biases type Const
Schedule Idx 292 TF Operation Name: inception_5b_3x3_reduce/BiasAdd type BiasAdd
Schedule Idx 293 TF Operation Name: inception_5b_3x3_reduce/inception_5b_3x3_reduce type Relu
Schedule Idx 294 TF Operation Name: inception_5b_3x3/weights type Const
Schedule Idx 295 TF Operation Name: inception_5b_3x3/Conv2D type Conv2D
Schedule Idx 296 TF Operation Name: inception_5b_3x3/biases type Const
Schedule Idx 297 TF Operation Name: inception_5b_3x3/BiasAdd type BiasAdd
Schedule Idx 298 TF Operation Name: inception_5b_3x3/inception_5b_3x3 type Relu
Schedule Idx 299 TF Operation Name: inception_5b_5x5_reduce/weights type Const
Schedule Idx 300 TF Operation Name: inception_5b_5x5_reduce/Conv2D type Conv2D
Schedule Idx 301 TF Operation Name: inception_5b_5x5_reduce/biases type Const
Schedule Idx 302 TF Operation Name: inception_5b_5x5_reduce/BiasAdd type BiasAdd
Schedule Idx 303 TF Operation Name: inception_5b_5x5_reduce/inception_5b_5x5_reduce type Relu
Schedule Idx 304 TF Operation Name: inception_5b_5x5/weights type Const
Schedule Idx 305 TF Operation Name: inception_5b_5x5/Conv2D type Conv2D
Schedule Idx 306 TF Operation Name: inception_5b_5x5/biases type Const
Schedule Idx 307 TF Operation Name: inception_5b_5x5/BiasAdd type BiasAdd
Schedule Idx 308 TF Operation Name: inception_5b_5x5/inception_5b_5x5 type Relu
Schedule Idx 309 TF Operation Name: inception_5b_pool type MaxPool
Schedule Idx 310 TF Operation Name: inception_5b_pool_proj/weights type Const
Schedule Idx 311 TF Operation Name: inception_5b_pool_proj/Conv2D type Conv2D
Schedule Idx 312 TF Operation Name: inception_5b_pool_proj/biases type Const
Schedule Idx 313 TF Operation Name: inception_5b_pool_proj/BiasAdd type BiasAdd
Schedule Idx 314 TF Operation Name: inception_5b_pool_proj/inception_5b_pool_proj type Relu
Schedule Idx 315 TF Operation Name: inception_5b_output/axis type Const
Schedule Idx 316 TF Operation Name: inception_5b_output type ConcatV2
Schedule Idx 317 TF Operation Name: pool5_7x7_s1 type AvgPool
**************************************************
* Introduction DeepPhi (aka deeppy) factors
**************************************************
Phase 0.0 Removed batch norm 57 ['conv1_7x7_s2/BiasAdd', 'conv2_3x3_reduce/BiasAdd', 'conv2_3x3/BiasAdd', 'inception_3a_1x1/BiasAdd', 'inception_3a_3x3_reduce/BiasAdd', 'inception_3a_3x3/BiasAdd', 'inception_3a_5x5_reduce/BiasAdd', 'inception_3a_5x5/BiasAdd', 'inception_3a_pool_proj/BiasAdd', 'inception_3b_1x1/BiasAdd', 'inception_3b_3x3_reduce/BiasAdd', 'inception_3b_3x3/BiasAdd', 'inception_3b_5x5_reduce/BiasAdd', 'inception_3b_5x5/BiasAdd', 'inception_3b_pool_proj/BiasAdd', 'inception_4a_1x1/BiasAdd', 'inception_4a_3x3_reduce/BiasAdd', 'inception_4a_3x3/BiasAdd', 'inception_4a_5x5_reduce/BiasAdd', 'inception_4a_5x5/BiasAdd', 'inception_4a_pool_proj/BiasAdd', 'inception_4b_1x1/BiasAdd', 'inception_4b_3x3_reduce/BiasAdd', 'inception_4b_3x3/BiasAdd', 'inception_4b_5x5_reduce/BiasAdd', 'inception_4b_5x5/BiasAdd', 'inception_4b_pool_proj/BiasAdd', 'inception_4c_1x1/BiasAdd', 'inception_4c_3x3_reduce/BiasAdd', 'inception_4c_3x3/BiasAdd', 'inception_4c_5x5_reduce/BiasAdd', 'inception_4c_5x5/BiasAdd', 'inception_4c_pool_proj/BiasAdd', 'inception_4d_1x1/BiasAdd', 'inception_4d_3x3_reduce/BiasAdd', 'inception_4d_3x3/BiasAdd', 'inception_4d_5x5_reduce/BiasAdd', 'inception_4d_5x5/BiasAdd', 'inception_4d_pool_proj/BiasAdd', 'inception_4e_1x1/BiasAdd', 'inception_4e_3x3_reduce/BiasAdd', 'inception_4e_3x3/BiasAdd', 'inception_4e_5x5_reduce/BiasAdd', 'inception_4e_5x5/BiasAdd', 'inception_4e_pool_proj/BiasAdd', 'inception_5a_1x1/BiasAdd', 'inception_5a_3x3_reduce/BiasAdd', 'inception_5a_3x3/BiasAdd', 'inception_5a_5x5_reduce/BiasAdd', 'inception_5a_5x5/BiasAdd', 'inception_5a_pool_proj/BiasAdd', 'inception_5b_1x1/BiasAdd', 'inception_5b_3x3_reduce/BiasAdd', 'inception_5b_3x3/BiasAdd', 'inception_5b_5x5_reduce/BiasAdd', 'inception_5b_5x5/BiasAdd', 'inception_5b_pool_proj/BiasAdd']
Phase 0.1 Removed batch norm 0 []
Phase 0 Done Graphing
Phase 1 Computation Islands 1
Phase 2 Memory Islands 1
Phase 3 Removed Identities 0
Phase 4 Removed in place 0
Phase 5 Removed Scale+ BIAS 0
Phase 8 Removed Pad 0 []
**************************************************
* General Graph Manipulations
**************************************************
**************************************************
* Fully connected layers as Convolutions
**************************************************
{'version': 3, 'name': 'rule3', 'images': {'width': range(1, 4095), 'height': range(1, 4095), 'channels': range(3, 4097)}, 'kernels': {'frames': [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15]], 'strides': [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15]], 'padding': [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [1, 15], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [14, 1], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 1], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12], [15, 13], [15, 14], [15, 15]], 'dilation': [[1, 1], [1, 2], [1, 4], [1, 8], [1, 16], [2, 1], [2, 2], [2, 4], [2, 8], [2, 16], [4, 1], [4, 2], [4, 4], [4, 8], [4, 16], [8, 1], [8, 2], [8, 4], [8, 8], [8, 16], [16, 1], [16, 2], [16, 4], [16, 8], [16, 16]]}, 'max_filter': 9792, 'dsp': [[96, 32]], 'operations': ['Convolution', 'MaxPool', 'AvgPool', 'EltwiseAdd', 'UpSample', 'Download', 'Upload'], 'upsample': [[1, 1], [1, 2], [1, 4], [1, 8], [2, 1], [2, 2], [2, 4], [2, 8], [4, 1], [4, 2], [4, 4], [4, 8], [8, 1], [8, 2], [8, 4], [8, 8]]}
Inner Product Found 0
Schedule Name: 
{1} -|-0 name data type Placeholder fpga False bottoms None [Extras None]-  Past [] -> Future []
{1} -|-1 name conv1_7x7_s2/Conv2D type Convolution fpga True bottoms ['data'] [Extras ['conv1_7x7_s2/BiasAdd']]-  Past [] -> Future ['conv1_7x7_s2/BiasAdd']
{1} -|-2 name conv1_7x7_s2/conv1_7x7_s2 type ReLU fpga False bottoms ['conv1_7x7_s2/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-3 name pool1_3x3_s2 type Pooling fpga True bottoms ['conv1_7x7_s2/conv1_7x7_s2'] [Extras None]-  Past [] -> Future []
{1} -|-4 name conv2_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool1_3x3_s2'] [Extras ['conv2_3x3_reduce/BiasAdd']]-  Past [] -> Future ['conv2_3x3_reduce/BiasAdd']
{1} -|-5 name conv2_3x3_reduce/conv2_3x3_reduce type ReLU fpga False bottoms ['conv2_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-6 name conv2_3x3/Conv2D type Convolution fpga True bottoms ['conv2_3x3_reduce/conv2_3x3_reduce'] [Extras ['conv2_3x3/BiasAdd']]-  Past [] -> Future ['conv2_3x3/BiasAdd']
{1} -|-7 name conv2_3x3/conv2_3x3 type ReLU fpga False bottoms ['conv2_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-8 name pool2_3x3_s2 type Pooling fpga True bottoms ['conv2_3x3/conv2_3x3'] [Extras None]-  Past [] -> Future []
{1} -|-9 name inception_3a_1x1/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_1x1/BiasAdd']]-  Past [] -> Future ['inception_3a_1x1/BiasAdd']
{1} -|-10 name inception_3a_1x1/inception_3a_1x1 type ReLU fpga False bottoms ['inception_3a_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-11 name inception_3a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_3a_3x3_reduce/BiasAdd']
{1} -|-12 name inception_3a_3x3_reduce/inception_3a_3x3_reduce type ReLU fpga False bottoms ['inception_3a_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-13 name inception_3a_3x3/Conv2D type Convolution fpga True bottoms ['inception_3a_3x3_reduce/inception_3a_3x3_reduce'] [Extras ['inception_3a_3x3/BiasAdd']]-  Past [] -> Future ['inception_3a_3x3/BiasAdd']
{1} -|-14 name inception_3a_3x3/inception_3a_3x3 type ReLU fpga False bottoms ['inception_3a_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-15 name inception_3a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_3a_5x5_reduce/BiasAdd']
{1} -|-16 name inception_3a_5x5_reduce/inception_3a_5x5_reduce type ReLU fpga False bottoms ['inception_3a_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-17 name inception_3a_5x5/Conv2D type Convolution fpga True bottoms ['inception_3a_5x5_reduce/inception_3a_5x5_reduce'] [Extras ['inception_3a_5x5/BiasAdd']]-  Past [] -> Future ['inception_3a_5x5/BiasAdd']
{1} -|-18 name inception_3a_5x5/inception_3a_5x5 type ReLU fpga False bottoms ['inception_3a_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-19 name inception_3a_pool type Pooling fpga True bottoms ['pool2_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-20 name inception_3a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3a_pool'] [Extras ['inception_3a_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_3a_pool_proj/BiasAdd']
{1} -|-21 name inception_3a_pool_proj/inception_3a_pool_proj type ReLU fpga False bottoms ['inception_3a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-22 name inception_3a_output type Concat fpga True bottoms ['inception_3a_1x1/inception_3a_1x1', 'inception_3a_3x3/inception_3a_3x3', 'inception_3a_5x5/inception_3a_5x5', 'inception_3a_pool_proj/inception_3a_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-23 name inception_3b_1x1/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_1x1/BiasAdd']]-  Past [] -> Future ['inception_3b_1x1/BiasAdd']
{1} -|-24 name inception_3b_1x1/inception_3b_1x1 type ReLU fpga False bottoms ['inception_3b_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-25 name inception_3b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_3b_3x3_reduce/BiasAdd']
{1} -|-26 name inception_3b_3x3_reduce/inception_3b_3x3_reduce type ReLU fpga False bottoms ['inception_3b_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-27 name inception_3b_3x3/Conv2D type Convolution fpga True bottoms ['inception_3b_3x3_reduce/inception_3b_3x3_reduce'] [Extras ['inception_3b_3x3/BiasAdd']]-  Past [] -> Future ['inception_3b_3x3/BiasAdd']
{1} -|-28 name inception_3b_3x3/inception_3b_3x3 type ReLU fpga False bottoms ['inception_3b_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-29 name inception_3b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_3b_5x5_reduce/BiasAdd']
{1} -|-30 name inception_3b_5x5_reduce/inception_3b_5x5_reduce type ReLU fpga False bottoms ['inception_3b_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-31 name inception_3b_5x5/Conv2D type Convolution fpga True bottoms ['inception_3b_5x5_reduce/inception_3b_5x5_reduce'] [Extras ['inception_3b_5x5/BiasAdd']]-  Past [] -> Future ['inception_3b_5x5/BiasAdd']
{1} -|-32 name inception_3b_5x5/inception_3b_5x5 type ReLU fpga False bottoms ['inception_3b_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-33 name inception_3b_pool type Pooling fpga True bottoms ['inception_3a_output'] [Extras None]-  Past [] -> Future []
{1} -|-34 name inception_3b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3b_pool'] [Extras ['inception_3b_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_3b_pool_proj/BiasAdd']
{1} -|-35 name inception_3b_pool_proj/inception_3b_pool_proj type ReLU fpga False bottoms ['inception_3b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-36 name inception_3b_output type Concat fpga True bottoms ['inception_3b_1x1/inception_3b_1x1', 'inception_3b_3x3/inception_3b_3x3', 'inception_3b_5x5/inception_3b_5x5', 'inception_3b_pool_proj/inception_3b_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-37 name pool3_3x3_s2 type Pooling fpga True bottoms ['inception_3b_output'] [Extras None]-  Past [] -> Future []
{1} -|-38 name inception_4a_1x1/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_1x1/BiasAdd']]-  Past [] -> Future ['inception_4a_1x1/BiasAdd']
{1} -|-39 name inception_4a_1x1/inception_4a_1x1 type ReLU fpga False bottoms ['inception_4a_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-40 name inception_4a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_4a_3x3_reduce/BiasAdd']
{1} -|-41 name inception_4a_3x3_reduce/inception_4a_3x3_reduce type ReLU fpga False bottoms ['inception_4a_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-42 name inception_4a_3x3/Conv2D type Convolution fpga True bottoms ['inception_4a_3x3_reduce/inception_4a_3x3_reduce'] [Extras ['inception_4a_3x3/BiasAdd']]-  Past [] -> Future ['inception_4a_3x3/BiasAdd']
{1} -|-43 name inception_4a_3x3/inception_4a_3x3 type ReLU fpga False bottoms ['inception_4a_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-44 name inception_4a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_4a_5x5_reduce/BiasAdd']
{1} -|-45 name inception_4a_5x5_reduce/inception_4a_5x5_reduce type ReLU fpga False bottoms ['inception_4a_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-46 name inception_4a_5x5/Conv2D type Convolution fpga True bottoms ['inception_4a_5x5_reduce/inception_4a_5x5_reduce'] [Extras ['inception_4a_5x5/BiasAdd']]-  Past [] -> Future ['inception_4a_5x5/BiasAdd']
{1} -|-47 name inception_4a_5x5/inception_4a_5x5 type ReLU fpga False bottoms ['inception_4a_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-48 name inception_4a_pool type Pooling fpga True bottoms ['pool3_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-49 name inception_4a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4a_pool'] [Extras ['inception_4a_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_4a_pool_proj/BiasAdd']
{1} -|-50 name inception_4a_pool_proj/inception_4a_pool_proj type ReLU fpga False bottoms ['inception_4a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-51 name inception_4a_output type Concat fpga True bottoms ['inception_4a_1x1/inception_4a_1x1', 'inception_4a_3x3/inception_4a_3x3', 'inception_4a_5x5/inception_4a_5x5', 'inception_4a_pool_proj/inception_4a_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-52 name inception_4b_1x1/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_1x1/BiasAdd']]-  Past [] -> Future ['inception_4b_1x1/BiasAdd']
{1} -|-53 name inception_4b_1x1/inception_4b_1x1 type ReLU fpga False bottoms ['inception_4b_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-54 name inception_4b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_4b_3x3_reduce/BiasAdd']
{1} -|-55 name inception_4b_3x3_reduce/inception_4b_3x3_reduce type ReLU fpga False bottoms ['inception_4b_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-56 name inception_4b_3x3/Conv2D type Convolution fpga True bottoms ['inception_4b_3x3_reduce/inception_4b_3x3_reduce'] [Extras ['inception_4b_3x3/BiasAdd']]-  Past [] -> Future ['inception_4b_3x3/BiasAdd']
{1} -|-57 name inception_4b_3x3/inception_4b_3x3 type ReLU fpga False bottoms ['inception_4b_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-58 name inception_4b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_4b_5x5_reduce/BiasAdd']
{1} -|-59 name inception_4b_5x5_reduce/inception_4b_5x5_reduce type ReLU fpga False bottoms ['inception_4b_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-60 name inception_4b_5x5/Conv2D type Convolution fpga True bottoms ['inception_4b_5x5_reduce/inception_4b_5x5_reduce'] [Extras ['inception_4b_5x5/BiasAdd']]-  Past [] -> Future ['inception_4b_5x5/BiasAdd']
{1} -|-61 name inception_4b_5x5/inception_4b_5x5 type ReLU fpga False bottoms ['inception_4b_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-62 name inception_4b_pool type Pooling fpga True bottoms ['inception_4a_output'] [Extras None]-  Past [] -> Future []
{1} -|-63 name inception_4b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4b_pool'] [Extras ['inception_4b_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_4b_pool_proj/BiasAdd']
{1} -|-64 name inception_4b_pool_proj/inception_4b_pool_proj type ReLU fpga False bottoms ['inception_4b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-65 name inception_4b_output type Concat fpga True bottoms ['inception_4b_1x1/inception_4b_1x1', 'inception_4b_3x3/inception_4b_3x3', 'inception_4b_5x5/inception_4b_5x5', 'inception_4b_pool_proj/inception_4b_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-66 name inception_4c_1x1/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_1x1/BiasAdd']]-  Past [] -> Future ['inception_4c_1x1/BiasAdd']
{1} -|-67 name inception_4c_1x1/inception_4c_1x1 type ReLU fpga False bottoms ['inception_4c_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-68 name inception_4c_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_4c_3x3_reduce/BiasAdd']
{1} -|-69 name inception_4c_3x3_reduce/inception_4c_3x3_reduce type ReLU fpga False bottoms ['inception_4c_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-70 name inception_4c_3x3/Conv2D type Convolution fpga True bottoms ['inception_4c_3x3_reduce/inception_4c_3x3_reduce'] [Extras ['inception_4c_3x3/BiasAdd']]-  Past [] -> Future ['inception_4c_3x3/BiasAdd']
{1} -|-71 name inception_4c_3x3/inception_4c_3x3 type ReLU fpga False bottoms ['inception_4c_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-72 name inception_4c_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_4c_5x5_reduce/BiasAdd']
{1} -|-73 name inception_4c_5x5_reduce/inception_4c_5x5_reduce type ReLU fpga False bottoms ['inception_4c_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-74 name inception_4c_5x5/Conv2D type Convolution fpga True bottoms ['inception_4c_5x5_reduce/inception_4c_5x5_reduce'] [Extras ['inception_4c_5x5/BiasAdd']]-  Past [] -> Future ['inception_4c_5x5/BiasAdd']
{1} -|-75 name inception_4c_5x5/inception_4c_5x5 type ReLU fpga False bottoms ['inception_4c_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-76 name inception_4c_pool type Pooling fpga True bottoms ['inception_4b_output'] [Extras None]-  Past [] -> Future []
{1} -|-77 name inception_4c_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4c_pool'] [Extras ['inception_4c_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_4c_pool_proj/BiasAdd']
{1} -|-78 name inception_4c_pool_proj/inception_4c_pool_proj type ReLU fpga False bottoms ['inception_4c_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-79 name inception_4c_output type Concat fpga True bottoms ['inception_4c_1x1/inception_4c_1x1', 'inception_4c_3x3/inception_4c_3x3', 'inception_4c_5x5/inception_4c_5x5', 'inception_4c_pool_proj/inception_4c_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-80 name inception_4d_1x1/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_1x1/BiasAdd']]-  Past [] -> Future ['inception_4d_1x1/BiasAdd']
{1} -|-81 name inception_4d_1x1/inception_4d_1x1 type ReLU fpga False bottoms ['inception_4d_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-82 name inception_4d_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_4d_3x3_reduce/BiasAdd']
{1} -|-83 name inception_4d_3x3_reduce/inception_4d_3x3_reduce type ReLU fpga False bottoms ['inception_4d_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-84 name inception_4d_3x3/Conv2D type Convolution fpga True bottoms ['inception_4d_3x3_reduce/inception_4d_3x3_reduce'] [Extras ['inception_4d_3x3/BiasAdd']]-  Past [] -> Future ['inception_4d_3x3/BiasAdd']
{1} -|-85 name inception_4d_3x3/inception_4d_3x3 type ReLU fpga False bottoms ['inception_4d_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-86 name inception_4d_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_4d_5x5_reduce/BiasAdd']
{1} -|-87 name inception_4d_5x5_reduce/inception_4d_5x5_reduce type ReLU fpga False bottoms ['inception_4d_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-88 name inception_4d_5x5/Conv2D type Convolution fpga True bottoms ['inception_4d_5x5_reduce/inception_4d_5x5_reduce'] [Extras ['inception_4d_5x5/BiasAdd']]-  Past [] -> Future ['inception_4d_5x5/BiasAdd']
{1} -|-89 name inception_4d_5x5/inception_4d_5x5 type ReLU fpga False bottoms ['inception_4d_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-90 name inception_4d_pool type Pooling fpga True bottoms ['inception_4c_output'] [Extras None]-  Past [] -> Future []
{1} -|-91 name inception_4d_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4d_pool'] [Extras ['inception_4d_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_4d_pool_proj/BiasAdd']
{1} -|-92 name inception_4d_pool_proj/inception_4d_pool_proj type ReLU fpga False bottoms ['inception_4d_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-93 name inception_4d_output type Concat fpga True bottoms ['inception_4d_1x1/inception_4d_1x1', 'inception_4d_3x3/inception_4d_3x3', 'inception_4d_5x5/inception_4d_5x5', 'inception_4d_pool_proj/inception_4d_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-94 name inception_4e_1x1/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_1x1/BiasAdd']]-  Past [] -> Future ['inception_4e_1x1/BiasAdd']
{1} -|-95 name inception_4e_1x1/inception_4e_1x1 type ReLU fpga False bottoms ['inception_4e_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-96 name inception_4e_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_4e_3x3_reduce/BiasAdd']
{1} -|-97 name inception_4e_3x3_reduce/inception_4e_3x3_reduce type ReLU fpga False bottoms ['inception_4e_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-98 name inception_4e_3x3/Conv2D type Convolution fpga True bottoms ['inception_4e_3x3_reduce/inception_4e_3x3_reduce'] [Extras ['inception_4e_3x3/BiasAdd']]-  Past [] -> Future ['inception_4e_3x3/BiasAdd']
{1} -|-99 name inception_4e_3x3/inception_4e_3x3 type ReLU fpga False bottoms ['inception_4e_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-100 name inception_4e_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_4e_5x5_reduce/BiasAdd']
{1} -|-101 name inception_4e_5x5_reduce/inception_4e_5x5_reduce type ReLU fpga False bottoms ['inception_4e_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-102 name inception_4e_5x5/Conv2D type Convolution fpga True bottoms ['inception_4e_5x5_reduce/inception_4e_5x5_reduce'] [Extras ['inception_4e_5x5/BiasAdd']]-  Past [] -> Future ['inception_4e_5x5/BiasAdd']
{1} -|-103 name inception_4e_5x5/inception_4e_5x5 type ReLU fpga False bottoms ['inception_4e_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-104 name inception_4e_pool type Pooling fpga True bottoms ['inception_4d_output'] [Extras None]-  Past [] -> Future []
{1} -|-105 name inception_4e_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4e_pool'] [Extras ['inception_4e_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_4e_pool_proj/BiasAdd']
{1} -|-106 name inception_4e_pool_proj/inception_4e_pool_proj type ReLU fpga False bottoms ['inception_4e_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-107 name inception_4e_output type Concat fpga True bottoms ['inception_4e_1x1/inception_4e_1x1', 'inception_4e_3x3/inception_4e_3x3', 'inception_4e_5x5/inception_4e_5x5', 'inception_4e_pool_proj/inception_4e_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-108 name pool4_3x3_s2 type Pooling fpga True bottoms ['inception_4e_output'] [Extras None]-  Past [] -> Future []
{1} -|-109 name inception_5a_1x1/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_1x1/BiasAdd']]-  Past [] -> Future ['inception_5a_1x1/BiasAdd']
{1} -|-110 name inception_5a_1x1/inception_5a_1x1 type ReLU fpga False bottoms ['inception_5a_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-111 name inception_5a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_5a_3x3_reduce/BiasAdd']
{1} -|-112 name inception_5a_3x3_reduce/inception_5a_3x3_reduce type ReLU fpga False bottoms ['inception_5a_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-113 name inception_5a_3x3/Conv2D type Convolution fpga True bottoms ['inception_5a_3x3_reduce/inception_5a_3x3_reduce'] [Extras ['inception_5a_3x3/BiasAdd']]-  Past [] -> Future ['inception_5a_3x3/BiasAdd']
{1} -|-114 name inception_5a_3x3/inception_5a_3x3 type ReLU fpga False bottoms ['inception_5a_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-115 name inception_5a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_5a_5x5_reduce/BiasAdd']
{1} -|-116 name inception_5a_5x5_reduce/inception_5a_5x5_reduce type ReLU fpga False bottoms ['inception_5a_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-117 name inception_5a_5x5/Conv2D type Convolution fpga True bottoms ['inception_5a_5x5_reduce/inception_5a_5x5_reduce'] [Extras ['inception_5a_5x5/BiasAdd']]-  Past [] -> Future ['inception_5a_5x5/BiasAdd']
{1} -|-118 name inception_5a_5x5/inception_5a_5x5 type ReLU fpga False bottoms ['inception_5a_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-119 name inception_5a_pool type Pooling fpga True bottoms ['pool4_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-120 name inception_5a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5a_pool'] [Extras ['inception_5a_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_5a_pool_proj/BiasAdd']
{1} -|-121 name inception_5a_pool_proj/inception_5a_pool_proj type ReLU fpga False bottoms ['inception_5a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-122 name inception_5a_output type Concat fpga True bottoms ['inception_5a_1x1/inception_5a_1x1', 'inception_5a_3x3/inception_5a_3x3', 'inception_5a_5x5/inception_5a_5x5', 'inception_5a_pool_proj/inception_5a_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-123 name inception_5b_1x1/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_1x1/BiasAdd']]-  Past [] -> Future ['inception_5b_1x1/BiasAdd']
{1} -|-124 name inception_5b_1x1/inception_5b_1x1 type ReLU fpga False bottoms ['inception_5b_1x1/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-125 name inception_5b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_3x3_reduce/BiasAdd']]-  Past [] -> Future ['inception_5b_3x3_reduce/BiasAdd']
{1} -|-126 name inception_5b_3x3_reduce/inception_5b_3x3_reduce type ReLU fpga False bottoms ['inception_5b_3x3_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-127 name inception_5b_3x3/Conv2D type Convolution fpga True bottoms ['inception_5b_3x3_reduce/inception_5b_3x3_reduce'] [Extras ['inception_5b_3x3/BiasAdd']]-  Past [] -> Future ['inception_5b_3x3/BiasAdd']
{1} -|-128 name inception_5b_3x3/inception_5b_3x3 type ReLU fpga False bottoms ['inception_5b_3x3/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-129 name inception_5b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_5x5_reduce/BiasAdd']]-  Past [] -> Future ['inception_5b_5x5_reduce/BiasAdd']
{1} -|-130 name inception_5b_5x5_reduce/inception_5b_5x5_reduce type ReLU fpga False bottoms ['inception_5b_5x5_reduce/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-131 name inception_5b_5x5/Conv2D type Convolution fpga True bottoms ['inception_5b_5x5_reduce/inception_5b_5x5_reduce'] [Extras ['inception_5b_5x5/BiasAdd']]-  Past [] -> Future ['inception_5b_5x5/BiasAdd']
{1} -|-132 name inception_5b_5x5/inception_5b_5x5 type ReLU fpga False bottoms ['inception_5b_5x5/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-133 name inception_5b_pool type Pooling fpga True bottoms ['inception_5a_output'] [Extras None]-  Past [] -> Future []
{1} -|-134 name inception_5b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5b_pool'] [Extras ['inception_5b_pool_proj/BiasAdd']]-  Past [] -> Future ['inception_5b_pool_proj/BiasAdd']
{1} -|-135 name inception_5b_pool_proj/inception_5b_pool_proj type ReLU fpga False bottoms ['inception_5b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-136 name inception_5b_output type Concat fpga True bottoms ['inception_5b_1x1/inception_5b_1x1', 'inception_5b_3x3/inception_5b_3x3', 'inception_5b_5x5/inception_5b_5x5', 'inception_5b_pool_proj/inception_5b_pool_proj'] [Extras None]-  Past [] -> Future []
{1} -|-137 name pool5_7x7_s1 type Pooling fpga True bottoms ['inception_5b_output'] [Extras None]-  Past [] -> Future []
####################################
**************************************************
* Deconv to (Upsample)? + Conv
**************************************************
Added Updamples 0 []
**************************************************
* Introduction of identity Scale by type or name
**************************************************
WITH schedule
Removal Dropout
Removed Dropout ? 0 []
Bathnorm-Scale telescoping
Removed SC 0 []
Pre-Bathnorm or Pre-Scale
Removed Pre 0 []
Pre-Bathnorm or Pre-Scale Inner-product
Removed Pre 0 []
Post-Bathnorm or Post-Scale
Removed Post 0 []
Removed ReLU? 57 ['conv1_7x7_s2/conv1_7x7_s2', 'conv2_3x3_reduce/conv2_3x3_reduce', 'conv2_3x3/conv2_3x3', 'inception_3a_1x1/inception_3a_1x1', 'inception_3a_3x3_reduce/inception_3a_3x3_reduce', 'inception_3a_3x3/inception_3a_3x3', 'inception_3a_5x5_reduce/inception_3a_5x5_reduce', 'inception_3a_5x5/inception_3a_5x5', 'inception_3a_pool_proj/inception_3a_pool_proj', 'inception_3b_1x1/inception_3b_1x1', 'inception_3b_3x3_reduce/inception_3b_3x3_reduce', 'inception_3b_3x3/inception_3b_3x3', 'inception_3b_5x5_reduce/inception_3b_5x5_reduce', 'inception_3b_5x5/inception_3b_5x5', 'inception_3b_pool_proj/inception_3b_pool_proj', 'inception_4a_1x1/inception_4a_1x1', 'inception_4a_3x3_reduce/inception_4a_3x3_reduce', 'inception_4a_3x3/inception_4a_3x3', 'inception_4a_5x5_reduce/inception_4a_5x5_reduce', 'inception_4a_5x5/inception_4a_5x5', 'inception_4a_pool_proj/inception_4a_pool_proj', 'inception_4b_1x1/inception_4b_1x1', 'inception_4b_3x3_reduce/inception_4b_3x3_reduce', 'inception_4b_3x3/inception_4b_3x3', 'inception_4b_5x5_reduce/inception_4b_5x5_reduce', 'inception_4b_5x5/inception_4b_5x5', 'inception_4b_pool_proj/inception_4b_pool_proj', 'inception_4c_1x1/inception_4c_1x1', 'inception_4c_3x3_reduce/inception_4c_3x3_reduce', 'inception_4c_3x3/inception_4c_3x3', 'inception_4c_5x5_reduce/inception_4c_5x5_reduce', 'inception_4c_5x5/inception_4c_5x5', 'inception_4c_pool_proj/inception_4c_pool_proj', 'inception_4d_1x1/inception_4d_1x1', 'inception_4d_3x3_reduce/inception_4d_3x3_reduce', 'inception_4d_3x3/inception_4d_3x3', 'inception_4d_5x5_reduce/inception_4d_5x5_reduce', 'inception_4d_5x5/inception_4d_5x5', 'inception_4d_pool_proj/inception_4d_pool_proj', 'inception_4e_1x1/inception_4e_1x1', 'inception_4e_3x3_reduce/inception_4e_3x3_reduce', 'inception_4e_3x3/inception_4e_3x3', 'inception_4e_5x5_reduce/inception_4e_5x5_reduce', 'inception_4e_5x5/inception_4e_5x5', 'inception_4e_pool_proj/inception_4e_pool_proj', 'inception_5a_1x1/inception_5a_1x1', 'inception_5a_3x3_reduce/inception_5a_3x3_reduce', 'inception_5a_3x3/inception_5a_3x3', 'inception_5a_5x5_reduce/inception_5a_5x5_reduce', 'inception_5a_5x5/inception_5a_5x5', 'inception_5a_pool_proj/inception_5a_pool_proj', 'inception_5b_1x1/inception_5b_1x1', 'inception_5b_3x3_reduce/inception_5b_3x3_reduce', 'inception_5b_3x3/inception_5b_3x3', 'inception_5b_5x5_reduce/inception_5b_5x5_reduce', 'inception_5b_5x5/inception_5b_5x5', 'inception_5b_pool_proj/inception_5b_pool_proj']
Removed PreReLU? 0 []
**************************************************
*  Ping Pong Failure preventions 
**************************************************
	 Fatty convolutions:  0 []

**************************************************
* Concat Alignment verification to mod 8             
**************************************************
Bs ['Concat']
**************************************************
* Removal of Bridges by identity Scale
**************************************************
**************************************************
* Pipelining Convolution and Pooling
**************************************************
Convolution  conv1_7x7_s2/Conv2D Merged with the pool  conv1_7x7_s2/Conv2D
Convolution  conv2_3x3/Conv2D Merged with the pool  conv2_3x3/Conv2D

**************************************************
* Concat Alignment verification              
**************************************************

**************************************************
* Concat Alignment verification READ        
**************************************************
**************************************************
* CPU Layer will be REMOVED
**************************************************
Enforce Convexity of the FPGA Computation
Schedule Name: 
{1} -|-0 name data type Placeholder fpga False bottoms None [Extras ['conv1_7x7_s2/Conv2D', 'conv1_7x7_s2/BiasAdd', 'conv1_7x7_s2/conv1_7x7_s2']]-  Past [] -> Future []
{1} -|-1 name pool1_3x3_s2 type Pooling fpga True bottoms ['data'] [Extras None]-  Past ['conv1_7x7_s2/Conv2D', 'conv1_7x7_s2/BiasAdd', 'conv1_7x7_s2/conv1_7x7_s2'] -> Future []
{1} -|-2 name conv2_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool1_3x3_s2'] [Extras ['conv2_3x3_reduce/BiasAdd', 'conv2_3x3_reduce/conv2_3x3_reduce', 'conv2_3x3/Conv2D', 'conv2_3x3/BiasAdd', 'conv2_3x3/conv2_3x3']]-  Past [] -> Future ['conv2_3x3_reduce/BiasAdd', 'conv2_3x3_reduce/conv2_3x3_reduce']
{1} -|-3 name pool2_3x3_s2 type Pooling fpga True bottoms ['conv2_3x3_reduce/Conv2D'] [Extras None]-  Past ['conv2_3x3/Conv2D', 'conv2_3x3/BiasAdd', 'conv2_3x3/conv2_3x3'] -> Future []
{1} -|-4 name inception_3a_1x1/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_1x1/BiasAdd', 'inception_3a_1x1/inception_3a_1x1']]-  Past [] -> Future ['inception_3a_1x1/BiasAdd', 'inception_3a_1x1/inception_3a_1x1']
{1} -|-5 name inception_3a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_3x3_reduce/BiasAdd', 'inception_3a_3x3_reduce/inception_3a_3x3_reduce']]-  Past [] -> Future ['inception_3a_3x3_reduce/BiasAdd', 'inception_3a_3x3_reduce/inception_3a_3x3_reduce']
{1} -|-6 name inception_3a_3x3/Conv2D type Convolution fpga True bottoms ['inception_3a_3x3_reduce/Conv2D'] [Extras ['inception_3a_3x3/BiasAdd', 'inception_3a_3x3/inception_3a_3x3']]-  Past [] -> Future ['inception_3a_3x3/BiasAdd', 'inception_3a_3x3/inception_3a_3x3']
{1} -|-7 name inception_3a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_5x5_reduce/BiasAdd', 'inception_3a_5x5_reduce/inception_3a_5x5_reduce']]-  Past [] -> Future ['inception_3a_5x5_reduce/BiasAdd', 'inception_3a_5x5_reduce/inception_3a_5x5_reduce']
{1} -|-8 name inception_3a_5x5/Conv2D type Convolution fpga True bottoms ['inception_3a_5x5_reduce/Conv2D'] [Extras ['inception_3a_5x5/BiasAdd', 'inception_3a_5x5/inception_3a_5x5']]-  Past [] -> Future ['inception_3a_5x5/BiasAdd', 'inception_3a_5x5/inception_3a_5x5']
{1} -|-9 name inception_3a_pool type Pooling fpga True bottoms ['pool2_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-10 name inception_3a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3a_pool'] [Extras ['inception_3a_pool_proj/BiasAdd', 'inception_3a_pool_proj/inception_3a_pool_proj']]-  Past [] -> Future ['inception_3a_pool_proj/BiasAdd', 'inception_3a_pool_proj/inception_3a_pool_proj']
{1} -|-11 name inception_3a_output type Concat fpga True bottoms ['inception_3a_1x1/Conv2D', 'inception_3a_3x3/Conv2D', 'inception_3a_5x5/Conv2D', 'inception_3a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-12 name inception_3b_1x1/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_1x1/BiasAdd', 'inception_3b_1x1/inception_3b_1x1']]-  Past [] -> Future ['inception_3b_1x1/BiasAdd', 'inception_3b_1x1/inception_3b_1x1']
{1} -|-13 name inception_3b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_3x3_reduce/BiasAdd', 'inception_3b_3x3_reduce/inception_3b_3x3_reduce']]-  Past [] -> Future ['inception_3b_3x3_reduce/BiasAdd', 'inception_3b_3x3_reduce/inception_3b_3x3_reduce']
{1} -|-14 name inception_3b_3x3/Conv2D type Convolution fpga True bottoms ['inception_3b_3x3_reduce/Conv2D'] [Extras ['inception_3b_3x3/BiasAdd', 'inception_3b_3x3/inception_3b_3x3']]-  Past [] -> Future ['inception_3b_3x3/BiasAdd', 'inception_3b_3x3/inception_3b_3x3']
{1} -|-15 name inception_3b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_5x5_reduce/BiasAdd', 'inception_3b_5x5_reduce/inception_3b_5x5_reduce']]-  Past [] -> Future ['inception_3b_5x5_reduce/BiasAdd', 'inception_3b_5x5_reduce/inception_3b_5x5_reduce']
{1} -|-16 name inception_3b_5x5/Conv2D type Convolution fpga True bottoms ['inception_3b_5x5_reduce/Conv2D'] [Extras ['inception_3b_5x5/BiasAdd', 'inception_3b_5x5/inception_3b_5x5']]-  Past [] -> Future ['inception_3b_5x5/BiasAdd', 'inception_3b_5x5/inception_3b_5x5']
{1} -|-17 name inception_3b_pool type Pooling fpga True bottoms ['inception_3a_output'] [Extras None]-  Past [] -> Future []
{1} -|-18 name inception_3b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3b_pool'] [Extras ['inception_3b_pool_proj/BiasAdd', 'inception_3b_pool_proj/inception_3b_pool_proj']]-  Past [] -> Future ['inception_3b_pool_proj/BiasAdd', 'inception_3b_pool_proj/inception_3b_pool_proj']
{1} -|-19 name inception_3b_output type Concat fpga True bottoms ['inception_3b_1x1/Conv2D', 'inception_3b_3x3/Conv2D', 'inception_3b_5x5/Conv2D', 'inception_3b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-20 name pool3_3x3_s2 type Pooling fpga True bottoms ['inception_3b_output'] [Extras None]-  Past [] -> Future []
{1} -|-21 name inception_4a_1x1/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_1x1/BiasAdd', 'inception_4a_1x1/inception_4a_1x1']]-  Past [] -> Future ['inception_4a_1x1/BiasAdd', 'inception_4a_1x1/inception_4a_1x1']
{1} -|-22 name inception_4a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_3x3_reduce/BiasAdd', 'inception_4a_3x3_reduce/inception_4a_3x3_reduce']]-  Past [] -> Future ['inception_4a_3x3_reduce/BiasAdd', 'inception_4a_3x3_reduce/inception_4a_3x3_reduce']
{1} -|-23 name inception_4a_3x3/Conv2D type Convolution fpga True bottoms ['inception_4a_3x3_reduce/Conv2D'] [Extras ['inception_4a_3x3/BiasAdd', 'inception_4a_3x3/inception_4a_3x3']]-  Past [] -> Future ['inception_4a_3x3/BiasAdd', 'inception_4a_3x3/inception_4a_3x3']
{1} -|-24 name inception_4a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_5x5_reduce/BiasAdd', 'inception_4a_5x5_reduce/inception_4a_5x5_reduce']]-  Past [] -> Future ['inception_4a_5x5_reduce/BiasAdd', 'inception_4a_5x5_reduce/inception_4a_5x5_reduce']
{1} -|-25 name inception_4a_5x5/Conv2D type Convolution fpga True bottoms ['inception_4a_5x5_reduce/Conv2D'] [Extras ['inception_4a_5x5/BiasAdd', 'inception_4a_5x5/inception_4a_5x5']]-  Past [] -> Future ['inception_4a_5x5/BiasAdd', 'inception_4a_5x5/inception_4a_5x5']
{1} -|-26 name inception_4a_pool type Pooling fpga True bottoms ['pool3_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-27 name inception_4a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4a_pool'] [Extras ['inception_4a_pool_proj/BiasAdd', 'inception_4a_pool_proj/inception_4a_pool_proj']]-  Past [] -> Future ['inception_4a_pool_proj/BiasAdd', 'inception_4a_pool_proj/inception_4a_pool_proj']
{1} -|-28 name inception_4a_output type Concat fpga True bottoms ['inception_4a_1x1/Conv2D', 'inception_4a_3x3/Conv2D', 'inception_4a_5x5/Conv2D', 'inception_4a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-29 name inception_4b_1x1/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_1x1/BiasAdd', 'inception_4b_1x1/inception_4b_1x1']]-  Past [] -> Future ['inception_4b_1x1/BiasAdd', 'inception_4b_1x1/inception_4b_1x1']
{1} -|-30 name inception_4b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_3x3_reduce/BiasAdd', 'inception_4b_3x3_reduce/inception_4b_3x3_reduce']]-  Past [] -> Future ['inception_4b_3x3_reduce/BiasAdd', 'inception_4b_3x3_reduce/inception_4b_3x3_reduce']
{1} -|-31 name inception_4b_3x3/Conv2D type Convolution fpga True bottoms ['inception_4b_3x3_reduce/Conv2D'] [Extras ['inception_4b_3x3/BiasAdd', 'inception_4b_3x3/inception_4b_3x3']]-  Past [] -> Future ['inception_4b_3x3/BiasAdd', 'inception_4b_3x3/inception_4b_3x3']
{1} -|-32 name inception_4b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_5x5_reduce/BiasAdd', 'inception_4b_5x5_reduce/inception_4b_5x5_reduce']]-  Past [] -> Future ['inception_4b_5x5_reduce/BiasAdd', 'inception_4b_5x5_reduce/inception_4b_5x5_reduce']
{1} -|-33 name inception_4b_5x5/Conv2D type Convolution fpga True bottoms ['inception_4b_5x5_reduce/Conv2D'] [Extras ['inception_4b_5x5/BiasAdd', 'inception_4b_5x5/inception_4b_5x5']]-  Past [] -> Future ['inception_4b_5x5/BiasAdd', 'inception_4b_5x5/inception_4b_5x5']
{1} -|-34 name inception_4b_pool type Pooling fpga True bottoms ['inception_4a_output'] [Extras None]-  Past [] -> Future []
{1} -|-35 name inception_4b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4b_pool'] [Extras ['inception_4b_pool_proj/BiasAdd', 'inception_4b_pool_proj/inception_4b_pool_proj']]-  Past [] -> Future ['inception_4b_pool_proj/BiasAdd', 'inception_4b_pool_proj/inception_4b_pool_proj']
{1} -|-36 name inception_4b_output type Concat fpga True bottoms ['inception_4b_1x1/Conv2D', 'inception_4b_3x3/Conv2D', 'inception_4b_5x5/Conv2D', 'inception_4b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-37 name inception_4c_1x1/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_1x1/BiasAdd', 'inception_4c_1x1/inception_4c_1x1']]-  Past [] -> Future ['inception_4c_1x1/BiasAdd', 'inception_4c_1x1/inception_4c_1x1']
{1} -|-38 name inception_4c_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_3x3_reduce/BiasAdd', 'inception_4c_3x3_reduce/inception_4c_3x3_reduce']]-  Past [] -> Future ['inception_4c_3x3_reduce/BiasAdd', 'inception_4c_3x3_reduce/inception_4c_3x3_reduce']
{1} -|-39 name inception_4c_3x3/Conv2D type Convolution fpga True bottoms ['inception_4c_3x3_reduce/Conv2D'] [Extras ['inception_4c_3x3/BiasAdd', 'inception_4c_3x3/inception_4c_3x3']]-  Past [] -> Future ['inception_4c_3x3/BiasAdd', 'inception_4c_3x3/inception_4c_3x3']
{1} -|-40 name inception_4c_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_5x5_reduce/BiasAdd', 'inception_4c_5x5_reduce/inception_4c_5x5_reduce']]-  Past [] -> Future ['inception_4c_5x5_reduce/BiasAdd', 'inception_4c_5x5_reduce/inception_4c_5x5_reduce']
{1} -|-41 name inception_4c_5x5/Conv2D type Convolution fpga True bottoms ['inception_4c_5x5_reduce/Conv2D'] [Extras ['inception_4c_5x5/BiasAdd', 'inception_4c_5x5/inception_4c_5x5']]-  Past [] -> Future ['inception_4c_5x5/BiasAdd', 'inception_4c_5x5/inception_4c_5x5']
{1} -|-42 name inception_4c_pool type Pooling fpga True bottoms ['inception_4b_output'] [Extras None]-  Past [] -> Future []
{1} -|-43 name inception_4c_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4c_pool'] [Extras ['inception_4c_pool_proj/BiasAdd', 'inception_4c_pool_proj/inception_4c_pool_proj']]-  Past [] -> Future ['inception_4c_pool_proj/BiasAdd', 'inception_4c_pool_proj/inception_4c_pool_proj']
{1} -|-44 name inception_4c_output type Concat fpga True bottoms ['inception_4c_1x1/Conv2D', 'inception_4c_3x3/Conv2D', 'inception_4c_5x5/Conv2D', 'inception_4c_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-45 name inception_4d_1x1/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_1x1/BiasAdd', 'inception_4d_1x1/inception_4d_1x1']]-  Past [] -> Future ['inception_4d_1x1/BiasAdd', 'inception_4d_1x1/inception_4d_1x1']
{1} -|-46 name inception_4d_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_3x3_reduce/BiasAdd', 'inception_4d_3x3_reduce/inception_4d_3x3_reduce']]-  Past [] -> Future ['inception_4d_3x3_reduce/BiasAdd', 'inception_4d_3x3_reduce/inception_4d_3x3_reduce']
{1} -|-47 name inception_4d_3x3/Conv2D type Convolution fpga True bottoms ['inception_4d_3x3_reduce/Conv2D'] [Extras ['inception_4d_3x3/BiasAdd', 'inception_4d_3x3/inception_4d_3x3']]-  Past [] -> Future ['inception_4d_3x3/BiasAdd', 'inception_4d_3x3/inception_4d_3x3']
{1} -|-48 name inception_4d_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_5x5_reduce/BiasAdd', 'inception_4d_5x5_reduce/inception_4d_5x5_reduce']]-  Past [] -> Future ['inception_4d_5x5_reduce/BiasAdd', 'inception_4d_5x5_reduce/inception_4d_5x5_reduce']
{1} -|-49 name inception_4d_5x5/Conv2D type Convolution fpga True bottoms ['inception_4d_5x5_reduce/Conv2D'] [Extras ['inception_4d_5x5/BiasAdd', 'inception_4d_5x5/inception_4d_5x5']]-  Past [] -> Future ['inception_4d_5x5/BiasAdd', 'inception_4d_5x5/inception_4d_5x5']
{1} -|-50 name inception_4d_pool type Pooling fpga True bottoms ['inception_4c_output'] [Extras None]-  Past [] -> Future []
{1} -|-51 name inception_4d_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4d_pool'] [Extras ['inception_4d_pool_proj/BiasAdd', 'inception_4d_pool_proj/inception_4d_pool_proj']]-  Past [] -> Future ['inception_4d_pool_proj/BiasAdd', 'inception_4d_pool_proj/inception_4d_pool_proj']
{1} -|-52 name inception_4d_output type Concat fpga True bottoms ['inception_4d_1x1/Conv2D', 'inception_4d_3x3/Conv2D', 'inception_4d_5x5/Conv2D', 'inception_4d_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-53 name inception_4e_1x1/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_1x1/BiasAdd', 'inception_4e_1x1/inception_4e_1x1']]-  Past [] -> Future ['inception_4e_1x1/BiasAdd', 'inception_4e_1x1/inception_4e_1x1']
{1} -|-54 name inception_4e_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_3x3_reduce/BiasAdd', 'inception_4e_3x3_reduce/inception_4e_3x3_reduce']]-  Past [] -> Future ['inception_4e_3x3_reduce/BiasAdd', 'inception_4e_3x3_reduce/inception_4e_3x3_reduce']
{1} -|-55 name inception_4e_3x3/Conv2D type Convolution fpga True bottoms ['inception_4e_3x3_reduce/Conv2D'] [Extras ['inception_4e_3x3/BiasAdd', 'inception_4e_3x3/inception_4e_3x3']]-  Past [] -> Future ['inception_4e_3x3/BiasAdd', 'inception_4e_3x3/inception_4e_3x3']
{1} -|-56 name inception_4e_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_5x5_reduce/BiasAdd', 'inception_4e_5x5_reduce/inception_4e_5x5_reduce']]-  Past [] -> Future ['inception_4e_5x5_reduce/BiasAdd', 'inception_4e_5x5_reduce/inception_4e_5x5_reduce']
{1} -|-57 name inception_4e_5x5/Conv2D type Convolution fpga True bottoms ['inception_4e_5x5_reduce/Conv2D'] [Extras ['inception_4e_5x5/BiasAdd', 'inception_4e_5x5/inception_4e_5x5']]-  Past [] -> Future ['inception_4e_5x5/BiasAdd', 'inception_4e_5x5/inception_4e_5x5']
{1} -|-58 name inception_4e_pool type Pooling fpga True bottoms ['inception_4d_output'] [Extras None]-  Past [] -> Future []
{1} -|-59 name inception_4e_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4e_pool'] [Extras ['inception_4e_pool_proj/BiasAdd', 'inception_4e_pool_proj/inception_4e_pool_proj']]-  Past [] -> Future ['inception_4e_pool_proj/BiasAdd', 'inception_4e_pool_proj/inception_4e_pool_proj']
{1} -|-60 name inception_4e_output type Concat fpga True bottoms ['inception_4e_1x1/Conv2D', 'inception_4e_3x3/Conv2D', 'inception_4e_5x5/Conv2D', 'inception_4e_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-61 name pool4_3x3_s2 type Pooling fpga True bottoms ['inception_4e_output'] [Extras None]-  Past [] -> Future []
{1} -|-62 name inception_5a_1x1/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_1x1/BiasAdd', 'inception_5a_1x1/inception_5a_1x1']]-  Past [] -> Future ['inception_5a_1x1/BiasAdd', 'inception_5a_1x1/inception_5a_1x1']
{1} -|-63 name inception_5a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_3x3_reduce/BiasAdd', 'inception_5a_3x3_reduce/inception_5a_3x3_reduce']]-  Past [] -> Future ['inception_5a_3x3_reduce/BiasAdd', 'inception_5a_3x3_reduce/inception_5a_3x3_reduce']
{1} -|-64 name inception_5a_3x3/Conv2D type Convolution fpga True bottoms ['inception_5a_3x3_reduce/Conv2D'] [Extras ['inception_5a_3x3/BiasAdd', 'inception_5a_3x3/inception_5a_3x3']]-  Past [] -> Future ['inception_5a_3x3/BiasAdd', 'inception_5a_3x3/inception_5a_3x3']
{1} -|-65 name inception_5a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_5x5_reduce/BiasAdd', 'inception_5a_5x5_reduce/inception_5a_5x5_reduce']]-  Past [] -> Future ['inception_5a_5x5_reduce/BiasAdd', 'inception_5a_5x5_reduce/inception_5a_5x5_reduce']
{1} -|-66 name inception_5a_5x5/Conv2D type Convolution fpga True bottoms ['inception_5a_5x5_reduce/Conv2D'] [Extras ['inception_5a_5x5/BiasAdd', 'inception_5a_5x5/inception_5a_5x5']]-  Past [] -> Future ['inception_5a_5x5/BiasAdd', 'inception_5a_5x5/inception_5a_5x5']
{1} -|-67 name inception_5a_pool type Pooling fpga True bottoms ['pool4_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-68 name inception_5a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5a_pool'] [Extras ['inception_5a_pool_proj/BiasAdd', 'inception_5a_pool_proj/inception_5a_pool_proj']]-  Past [] -> Future ['inception_5a_pool_proj/BiasAdd', 'inception_5a_pool_proj/inception_5a_pool_proj']
{1} -|-69 name inception_5a_output type Concat fpga True bottoms ['inception_5a_1x1/Conv2D', 'inception_5a_3x3/Conv2D', 'inception_5a_5x5/Conv2D', 'inception_5a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-70 name inception_5b_1x1/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_1x1/BiasAdd', 'inception_5b_1x1/inception_5b_1x1']]-  Past [] -> Future ['inception_5b_1x1/BiasAdd', 'inception_5b_1x1/inception_5b_1x1']
{1} -|-71 name inception_5b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_3x3_reduce/BiasAdd', 'inception_5b_3x3_reduce/inception_5b_3x3_reduce']]-  Past [] -> Future ['inception_5b_3x3_reduce/BiasAdd', 'inception_5b_3x3_reduce/inception_5b_3x3_reduce']
{1} -|-72 name inception_5b_3x3/Conv2D type Convolution fpga True bottoms ['inception_5b_3x3_reduce/Conv2D'] [Extras ['inception_5b_3x3/BiasAdd', 'inception_5b_3x3/inception_5b_3x3']]-  Past [] -> Future ['inception_5b_3x3/BiasAdd', 'inception_5b_3x3/inception_5b_3x3']
{1} -|-73 name inception_5b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_5x5_reduce/BiasAdd', 'inception_5b_5x5_reduce/inception_5b_5x5_reduce']]-  Past [] -> Future ['inception_5b_5x5_reduce/BiasAdd', 'inception_5b_5x5_reduce/inception_5b_5x5_reduce']
{1} -|-74 name inception_5b_5x5/Conv2D type Convolution fpga True bottoms ['inception_5b_5x5_reduce/Conv2D'] [Extras ['inception_5b_5x5/BiasAdd', 'inception_5b_5x5/inception_5b_5x5']]-  Past [] -> Future ['inception_5b_5x5/BiasAdd', 'inception_5b_5x5/inception_5b_5x5']
{1} -|-75 name inception_5b_pool type Pooling fpga True bottoms ['inception_5a_output'] [Extras None]-  Past [] -> Future []
{1} -|-76 name inception_5b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5b_pool'] [Extras ['inception_5b_pool_proj/BiasAdd', 'inception_5b_pool_proj/inception_5b_pool_proj']]-  Past [] -> Future ['inception_5b_pool_proj/BiasAdd', 'inception_5b_pool_proj/inception_5b_pool_proj']
{1} -|-77 name inception_5b_output type Concat fpga True bottoms ['inception_5b_1x1/Conv2D', 'inception_5b_3x3/Conv2D', 'inception_5b_5x5/Conv2D', 'inception_5b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-78 name pool5_7x7_s1 type Pooling fpga True bottoms ['inception_5b_output'] [Extras None]-  Past [] -> Future []
####################################

 convex_fpga 

OUTPUTZ pool5_7x7_s1 in  ['pool5_7x7_s1']
Schedule Name: 
{1} -|-0 name data type Input fpga True bottoms [] [Extras ['conv1_7x7_s2/Conv2D', 'conv1_7x7_s2/BiasAdd', 'conv1_7x7_s2/conv1_7x7_s2']]-  Past [] -> Future []
{1} -|-1 name pool1_3x3_s2 type Pooling fpga True bottoms ['data'] [Extras None]-  Past ['conv1_7x7_s2/Conv2D', 'conv1_7x7_s2/BiasAdd', 'conv1_7x7_s2/conv1_7x7_s2'] -> Future []
{1} -|-2 name conv2_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool1_3x3_s2'] [Extras ['conv2_3x3_reduce/BiasAdd', 'conv2_3x3_reduce/conv2_3x3_reduce', 'conv2_3x3/Conv2D', 'conv2_3x3/BiasAdd', 'conv2_3x3/conv2_3x3']]-  Past [] -> Future ['conv2_3x3_reduce/BiasAdd', 'conv2_3x3_reduce/conv2_3x3_reduce']
{1} -|-3 name pool2_3x3_s2 type Pooling fpga True bottoms ['conv2_3x3_reduce/Conv2D'] [Extras None]-  Past ['conv2_3x3/Conv2D', 'conv2_3x3/BiasAdd', 'conv2_3x3/conv2_3x3'] -> Future []
{1} -|-4 name inception_3a_1x1/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_1x1/BiasAdd', 'inception_3a_1x1/inception_3a_1x1']]-  Past [] -> Future ['inception_3a_1x1/BiasAdd', 'inception_3a_1x1/inception_3a_1x1']
{1} -|-5 name inception_3a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_3x3_reduce/BiasAdd', 'inception_3a_3x3_reduce/inception_3a_3x3_reduce']]-  Past [] -> Future ['inception_3a_3x3_reduce/BiasAdd', 'inception_3a_3x3_reduce/inception_3a_3x3_reduce']
{1} -|-6 name inception_3a_3x3/Conv2D type Convolution fpga True bottoms ['inception_3a_3x3_reduce/Conv2D'] [Extras ['inception_3a_3x3/BiasAdd', 'inception_3a_3x3/inception_3a_3x3']]-  Past [] -> Future ['inception_3a_3x3/BiasAdd', 'inception_3a_3x3/inception_3a_3x3']
{1} -|-7 name inception_3a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool2_3x3_s2'] [Extras ['inception_3a_5x5_reduce/BiasAdd', 'inception_3a_5x5_reduce/inception_3a_5x5_reduce']]-  Past [] -> Future ['inception_3a_5x5_reduce/BiasAdd', 'inception_3a_5x5_reduce/inception_3a_5x5_reduce']
{1} -|-8 name inception_3a_5x5/Conv2D type Convolution fpga True bottoms ['inception_3a_5x5_reduce/Conv2D'] [Extras ['inception_3a_5x5/BiasAdd', 'inception_3a_5x5/inception_3a_5x5']]-  Past [] -> Future ['inception_3a_5x5/BiasAdd', 'inception_3a_5x5/inception_3a_5x5']
{1} -|-9 name inception_3a_pool type Pooling fpga True bottoms ['pool2_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-10 name inception_3a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3a_pool'] [Extras ['inception_3a_pool_proj/BiasAdd', 'inception_3a_pool_proj/inception_3a_pool_proj']]-  Past [] -> Future ['inception_3a_pool_proj/BiasAdd', 'inception_3a_pool_proj/inception_3a_pool_proj']
{1} -|-11 name inception_3a_output type Concat fpga True bottoms ['inception_3a_1x1/Conv2D', 'inception_3a_3x3/Conv2D', 'inception_3a_5x5/Conv2D', 'inception_3a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-12 name inception_3b_1x1/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_1x1/BiasAdd', 'inception_3b_1x1/inception_3b_1x1']]-  Past [] -> Future ['inception_3b_1x1/BiasAdd', 'inception_3b_1x1/inception_3b_1x1']
{1} -|-13 name inception_3b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_3x3_reduce/BiasAdd', 'inception_3b_3x3_reduce/inception_3b_3x3_reduce']]-  Past [] -> Future ['inception_3b_3x3_reduce/BiasAdd', 'inception_3b_3x3_reduce/inception_3b_3x3_reduce']
{1} -|-14 name inception_3b_3x3/Conv2D type Convolution fpga True bottoms ['inception_3b_3x3_reduce/Conv2D'] [Extras ['inception_3b_3x3/BiasAdd', 'inception_3b_3x3/inception_3b_3x3']]-  Past [] -> Future ['inception_3b_3x3/BiasAdd', 'inception_3b_3x3/inception_3b_3x3']
{1} -|-15 name inception_3b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_3a_output'] [Extras ['inception_3b_5x5_reduce/BiasAdd', 'inception_3b_5x5_reduce/inception_3b_5x5_reduce']]-  Past [] -> Future ['inception_3b_5x5_reduce/BiasAdd', 'inception_3b_5x5_reduce/inception_3b_5x5_reduce']
{1} -|-16 name inception_3b_5x5/Conv2D type Convolution fpga True bottoms ['inception_3b_5x5_reduce/Conv2D'] [Extras ['inception_3b_5x5/BiasAdd', 'inception_3b_5x5/inception_3b_5x5']]-  Past [] -> Future ['inception_3b_5x5/BiasAdd', 'inception_3b_5x5/inception_3b_5x5']
{1} -|-17 name inception_3b_pool type Pooling fpga True bottoms ['inception_3a_output'] [Extras None]-  Past [] -> Future []
{1} -|-18 name inception_3b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_3b_pool'] [Extras ['inception_3b_pool_proj/BiasAdd', 'inception_3b_pool_proj/inception_3b_pool_proj']]-  Past [] -> Future ['inception_3b_pool_proj/BiasAdd', 'inception_3b_pool_proj/inception_3b_pool_proj']
{1} -|-19 name inception_3b_output type Concat fpga True bottoms ['inception_3b_1x1/Conv2D', 'inception_3b_3x3/Conv2D', 'inception_3b_5x5/Conv2D', 'inception_3b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-20 name pool3_3x3_s2 type Pooling fpga True bottoms ['inception_3b_output'] [Extras None]-  Past [] -> Future []
{1} -|-21 name inception_4a_1x1/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_1x1/BiasAdd', 'inception_4a_1x1/inception_4a_1x1']]-  Past [] -> Future ['inception_4a_1x1/BiasAdd', 'inception_4a_1x1/inception_4a_1x1']
{1} -|-22 name inception_4a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_3x3_reduce/BiasAdd', 'inception_4a_3x3_reduce/inception_4a_3x3_reduce']]-  Past [] -> Future ['inception_4a_3x3_reduce/BiasAdd', 'inception_4a_3x3_reduce/inception_4a_3x3_reduce']
{1} -|-23 name inception_4a_3x3/Conv2D type Convolution fpga True bottoms ['inception_4a_3x3_reduce/Conv2D'] [Extras ['inception_4a_3x3/BiasAdd', 'inception_4a_3x3/inception_4a_3x3']]-  Past [] -> Future ['inception_4a_3x3/BiasAdd', 'inception_4a_3x3/inception_4a_3x3']
{1} -|-24 name inception_4a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool3_3x3_s2'] [Extras ['inception_4a_5x5_reduce/BiasAdd', 'inception_4a_5x5_reduce/inception_4a_5x5_reduce']]-  Past [] -> Future ['inception_4a_5x5_reduce/BiasAdd', 'inception_4a_5x5_reduce/inception_4a_5x5_reduce']
{1} -|-25 name inception_4a_5x5/Conv2D type Convolution fpga True bottoms ['inception_4a_5x5_reduce/Conv2D'] [Extras ['inception_4a_5x5/BiasAdd', 'inception_4a_5x5/inception_4a_5x5']]-  Past [] -> Future ['inception_4a_5x5/BiasAdd', 'inception_4a_5x5/inception_4a_5x5']
{1} -|-26 name inception_4a_pool type Pooling fpga True bottoms ['pool3_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-27 name inception_4a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4a_pool'] [Extras ['inception_4a_pool_proj/BiasAdd', 'inception_4a_pool_proj/inception_4a_pool_proj']]-  Past [] -> Future ['inception_4a_pool_proj/BiasAdd', 'inception_4a_pool_proj/inception_4a_pool_proj']
{1} -|-28 name inception_4a_output type Concat fpga True bottoms ['inception_4a_1x1/Conv2D', 'inception_4a_3x3/Conv2D', 'inception_4a_5x5/Conv2D', 'inception_4a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-29 name inception_4b_1x1/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_1x1/BiasAdd', 'inception_4b_1x1/inception_4b_1x1']]-  Past [] -> Future ['inception_4b_1x1/BiasAdd', 'inception_4b_1x1/inception_4b_1x1']
{1} -|-30 name inception_4b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_3x3_reduce/BiasAdd', 'inception_4b_3x3_reduce/inception_4b_3x3_reduce']]-  Past [] -> Future ['inception_4b_3x3_reduce/BiasAdd', 'inception_4b_3x3_reduce/inception_4b_3x3_reduce']
{1} -|-31 name inception_4b_3x3/Conv2D type Convolution fpga True bottoms ['inception_4b_3x3_reduce/Conv2D'] [Extras ['inception_4b_3x3/BiasAdd', 'inception_4b_3x3/inception_4b_3x3']]-  Past [] -> Future ['inception_4b_3x3/BiasAdd', 'inception_4b_3x3/inception_4b_3x3']
{1} -|-32 name inception_4b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4a_output'] [Extras ['inception_4b_5x5_reduce/BiasAdd', 'inception_4b_5x5_reduce/inception_4b_5x5_reduce']]-  Past [] -> Future ['inception_4b_5x5_reduce/BiasAdd', 'inception_4b_5x5_reduce/inception_4b_5x5_reduce']
{1} -|-33 name inception_4b_5x5/Conv2D type Convolution fpga True bottoms ['inception_4b_5x5_reduce/Conv2D'] [Extras ['inception_4b_5x5/BiasAdd', 'inception_4b_5x5/inception_4b_5x5']]-  Past [] -> Future ['inception_4b_5x5/BiasAdd', 'inception_4b_5x5/inception_4b_5x5']
{1} -|-34 name inception_4b_pool type Pooling fpga True bottoms ['inception_4a_output'] [Extras None]-  Past [] -> Future []
{1} -|-35 name inception_4b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4b_pool'] [Extras ['inception_4b_pool_proj/BiasAdd', 'inception_4b_pool_proj/inception_4b_pool_proj']]-  Past [] -> Future ['inception_4b_pool_proj/BiasAdd', 'inception_4b_pool_proj/inception_4b_pool_proj']
{1} -|-36 name inception_4b_output type Concat fpga True bottoms ['inception_4b_1x1/Conv2D', 'inception_4b_3x3/Conv2D', 'inception_4b_5x5/Conv2D', 'inception_4b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-37 name inception_4c_1x1/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_1x1/BiasAdd', 'inception_4c_1x1/inception_4c_1x1']]-  Past [] -> Future ['inception_4c_1x1/BiasAdd', 'inception_4c_1x1/inception_4c_1x1']
{1} -|-38 name inception_4c_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_3x3_reduce/BiasAdd', 'inception_4c_3x3_reduce/inception_4c_3x3_reduce']]-  Past [] -> Future ['inception_4c_3x3_reduce/BiasAdd', 'inception_4c_3x3_reduce/inception_4c_3x3_reduce']
{1} -|-39 name inception_4c_3x3/Conv2D type Convolution fpga True bottoms ['inception_4c_3x3_reduce/Conv2D'] [Extras ['inception_4c_3x3/BiasAdd', 'inception_4c_3x3/inception_4c_3x3']]-  Past [] -> Future ['inception_4c_3x3/BiasAdd', 'inception_4c_3x3/inception_4c_3x3']
{1} -|-40 name inception_4c_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4b_output'] [Extras ['inception_4c_5x5_reduce/BiasAdd', 'inception_4c_5x5_reduce/inception_4c_5x5_reduce']]-  Past [] -> Future ['inception_4c_5x5_reduce/BiasAdd', 'inception_4c_5x5_reduce/inception_4c_5x5_reduce']
{1} -|-41 name inception_4c_5x5/Conv2D type Convolution fpga True bottoms ['inception_4c_5x5_reduce/Conv2D'] [Extras ['inception_4c_5x5/BiasAdd', 'inception_4c_5x5/inception_4c_5x5']]-  Past [] -> Future ['inception_4c_5x5/BiasAdd', 'inception_4c_5x5/inception_4c_5x5']
{1} -|-42 name inception_4c_pool type Pooling fpga True bottoms ['inception_4b_output'] [Extras None]-  Past [] -> Future []
{1} -|-43 name inception_4c_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4c_pool'] [Extras ['inception_4c_pool_proj/BiasAdd', 'inception_4c_pool_proj/inception_4c_pool_proj']]-  Past [] -> Future ['inception_4c_pool_proj/BiasAdd', 'inception_4c_pool_proj/inception_4c_pool_proj']
{1} -|-44 name inception_4c_output type Concat fpga True bottoms ['inception_4c_1x1/Conv2D', 'inception_4c_3x3/Conv2D', 'inception_4c_5x5/Conv2D', 'inception_4c_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-45 name inception_4d_1x1/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_1x1/BiasAdd', 'inception_4d_1x1/inception_4d_1x1']]-  Past [] -> Future ['inception_4d_1x1/BiasAdd', 'inception_4d_1x1/inception_4d_1x1']
{1} -|-46 name inception_4d_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_3x3_reduce/BiasAdd', 'inception_4d_3x3_reduce/inception_4d_3x3_reduce']]-  Past [] -> Future ['inception_4d_3x3_reduce/BiasAdd', 'inception_4d_3x3_reduce/inception_4d_3x3_reduce']
{1} -|-47 name inception_4d_3x3/Conv2D type Convolution fpga True bottoms ['inception_4d_3x3_reduce/Conv2D'] [Extras ['inception_4d_3x3/BiasAdd', 'inception_4d_3x3/inception_4d_3x3']]-  Past [] -> Future ['inception_4d_3x3/BiasAdd', 'inception_4d_3x3/inception_4d_3x3']
{1} -|-48 name inception_4d_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4c_output'] [Extras ['inception_4d_5x5_reduce/BiasAdd', 'inception_4d_5x5_reduce/inception_4d_5x5_reduce']]-  Past [] -> Future ['inception_4d_5x5_reduce/BiasAdd', 'inception_4d_5x5_reduce/inception_4d_5x5_reduce']
{1} -|-49 name inception_4d_5x5/Conv2D type Convolution fpga True bottoms ['inception_4d_5x5_reduce/Conv2D'] [Extras ['inception_4d_5x5/BiasAdd', 'inception_4d_5x5/inception_4d_5x5']]-  Past [] -> Future ['inception_4d_5x5/BiasAdd', 'inception_4d_5x5/inception_4d_5x5']
{1} -|-50 name inception_4d_pool type Pooling fpga True bottoms ['inception_4c_output'] [Extras None]-  Past [] -> Future []
{1} -|-51 name inception_4d_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4d_pool'] [Extras ['inception_4d_pool_proj/BiasAdd', 'inception_4d_pool_proj/inception_4d_pool_proj']]-  Past [] -> Future ['inception_4d_pool_proj/BiasAdd', 'inception_4d_pool_proj/inception_4d_pool_proj']
{1} -|-52 name inception_4d_output type Concat fpga True bottoms ['inception_4d_1x1/Conv2D', 'inception_4d_3x3/Conv2D', 'inception_4d_5x5/Conv2D', 'inception_4d_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-53 name inception_4e_1x1/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_1x1/BiasAdd', 'inception_4e_1x1/inception_4e_1x1']]-  Past [] -> Future ['inception_4e_1x1/BiasAdd', 'inception_4e_1x1/inception_4e_1x1']
{1} -|-54 name inception_4e_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_3x3_reduce/BiasAdd', 'inception_4e_3x3_reduce/inception_4e_3x3_reduce']]-  Past [] -> Future ['inception_4e_3x3_reduce/BiasAdd', 'inception_4e_3x3_reduce/inception_4e_3x3_reduce']
{1} -|-55 name inception_4e_3x3/Conv2D type Convolution fpga True bottoms ['inception_4e_3x3_reduce/Conv2D'] [Extras ['inception_4e_3x3/BiasAdd', 'inception_4e_3x3/inception_4e_3x3']]-  Past [] -> Future ['inception_4e_3x3/BiasAdd', 'inception_4e_3x3/inception_4e_3x3']
{1} -|-56 name inception_4e_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_4d_output'] [Extras ['inception_4e_5x5_reduce/BiasAdd', 'inception_4e_5x5_reduce/inception_4e_5x5_reduce']]-  Past [] -> Future ['inception_4e_5x5_reduce/BiasAdd', 'inception_4e_5x5_reduce/inception_4e_5x5_reduce']
{1} -|-57 name inception_4e_5x5/Conv2D type Convolution fpga True bottoms ['inception_4e_5x5_reduce/Conv2D'] [Extras ['inception_4e_5x5/BiasAdd', 'inception_4e_5x5/inception_4e_5x5']]-  Past [] -> Future ['inception_4e_5x5/BiasAdd', 'inception_4e_5x5/inception_4e_5x5']
{1} -|-58 name inception_4e_pool type Pooling fpga True bottoms ['inception_4d_output'] [Extras None]-  Past [] -> Future []
{1} -|-59 name inception_4e_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_4e_pool'] [Extras ['inception_4e_pool_proj/BiasAdd', 'inception_4e_pool_proj/inception_4e_pool_proj']]-  Past [] -> Future ['inception_4e_pool_proj/BiasAdd', 'inception_4e_pool_proj/inception_4e_pool_proj']
{1} -|-60 name inception_4e_output type Concat fpga True bottoms ['inception_4e_1x1/Conv2D', 'inception_4e_3x3/Conv2D', 'inception_4e_5x5/Conv2D', 'inception_4e_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-61 name pool4_3x3_s2 type Pooling fpga True bottoms ['inception_4e_output'] [Extras None]-  Past [] -> Future []
{1} -|-62 name inception_5a_1x1/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_1x1/BiasAdd', 'inception_5a_1x1/inception_5a_1x1']]-  Past [] -> Future ['inception_5a_1x1/BiasAdd', 'inception_5a_1x1/inception_5a_1x1']
{1} -|-63 name inception_5a_3x3_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_3x3_reduce/BiasAdd', 'inception_5a_3x3_reduce/inception_5a_3x3_reduce']]-  Past [] -> Future ['inception_5a_3x3_reduce/BiasAdd', 'inception_5a_3x3_reduce/inception_5a_3x3_reduce']
{1} -|-64 name inception_5a_3x3/Conv2D type Convolution fpga True bottoms ['inception_5a_3x3_reduce/Conv2D'] [Extras ['inception_5a_3x3/BiasAdd', 'inception_5a_3x3/inception_5a_3x3']]-  Past [] -> Future ['inception_5a_3x3/BiasAdd', 'inception_5a_3x3/inception_5a_3x3']
{1} -|-65 name inception_5a_5x5_reduce/Conv2D type Convolution fpga True bottoms ['pool4_3x3_s2'] [Extras ['inception_5a_5x5_reduce/BiasAdd', 'inception_5a_5x5_reduce/inception_5a_5x5_reduce']]-  Past [] -> Future ['inception_5a_5x5_reduce/BiasAdd', 'inception_5a_5x5_reduce/inception_5a_5x5_reduce']
{1} -|-66 name inception_5a_5x5/Conv2D type Convolution fpga True bottoms ['inception_5a_5x5_reduce/Conv2D'] [Extras ['inception_5a_5x5/BiasAdd', 'inception_5a_5x5/inception_5a_5x5']]-  Past [] -> Future ['inception_5a_5x5/BiasAdd', 'inception_5a_5x5/inception_5a_5x5']
{1} -|-67 name inception_5a_pool type Pooling fpga True bottoms ['pool4_3x3_s2'] [Extras None]-  Past [] -> Future []
{1} -|-68 name inception_5a_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5a_pool'] [Extras ['inception_5a_pool_proj/BiasAdd', 'inception_5a_pool_proj/inception_5a_pool_proj']]-  Past [] -> Future ['inception_5a_pool_proj/BiasAdd', 'inception_5a_pool_proj/inception_5a_pool_proj']
{1} -|-69 name inception_5a_output type Concat fpga True bottoms ['inception_5a_1x1/Conv2D', 'inception_5a_3x3/Conv2D', 'inception_5a_5x5/Conv2D', 'inception_5a_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-70 name inception_5b_1x1/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_1x1/BiasAdd', 'inception_5b_1x1/inception_5b_1x1']]-  Past [] -> Future ['inception_5b_1x1/BiasAdd', 'inception_5b_1x1/inception_5b_1x1']
{1} -|-71 name inception_5b_3x3_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_3x3_reduce/BiasAdd', 'inception_5b_3x3_reduce/inception_5b_3x3_reduce']]-  Past [] -> Future ['inception_5b_3x3_reduce/BiasAdd', 'inception_5b_3x3_reduce/inception_5b_3x3_reduce']
{1} -|-72 name inception_5b_3x3/Conv2D type Convolution fpga True bottoms ['inception_5b_3x3_reduce/Conv2D'] [Extras ['inception_5b_3x3/BiasAdd', 'inception_5b_3x3/inception_5b_3x3']]-  Past [] -> Future ['inception_5b_3x3/BiasAdd', 'inception_5b_3x3/inception_5b_3x3']
{1} -|-73 name inception_5b_5x5_reduce/Conv2D type Convolution fpga True bottoms ['inception_5a_output'] [Extras ['inception_5b_5x5_reduce/BiasAdd', 'inception_5b_5x5_reduce/inception_5b_5x5_reduce']]-  Past [] -> Future ['inception_5b_5x5_reduce/BiasAdd', 'inception_5b_5x5_reduce/inception_5b_5x5_reduce']
{1} -|-74 name inception_5b_5x5/Conv2D type Convolution fpga True bottoms ['inception_5b_5x5_reduce/Conv2D'] [Extras ['inception_5b_5x5/BiasAdd', 'inception_5b_5x5/inception_5b_5x5']]-  Past [] -> Future ['inception_5b_5x5/BiasAdd', 'inception_5b_5x5/inception_5b_5x5']
{1} -|-75 name inception_5b_pool type Pooling fpga True bottoms ['inception_5a_output'] [Extras None]-  Past [] -> Future []
{1} -|-76 name inception_5b_pool_proj/Conv2D type Convolution fpga True bottoms ['inception_5b_pool'] [Extras ['inception_5b_pool_proj/BiasAdd', 'inception_5b_pool_proj/inception_5b_pool_proj']]-  Past [] -> Future ['inception_5b_pool_proj/BiasAdd', 'inception_5b_pool_proj/inception_5b_pool_proj']
{1} -|-77 name inception_5b_output type Concat fpga True bottoms ['inception_5b_1x1/Conv2D', 'inception_5b_3x3/Conv2D', 'inception_5b_5x5/Conv2D', 'inception_5b_pool_proj/Conv2D'] [Extras None]-  Past [] -> Future []
{1} -|-78 name pool5_7x7_s1 type Pooling fpga True bottoms ['inception_5b_output'] [Extras None]-  Past [] -> Future []
{1} -|-78 name pool5_7x7_s1_output type Output fpga True bottoms ['pool5_7x7_s1'] [Extras None]-  Past [] -> Future []
####################################
**************************************************
* * AVG + Scale -> AVG with different scaling  
**************************************************
**************************************************
* * Enrich the graph with quantization information  
**************************************************
inception_3a_pool
inception_3b_pool
pool3_3x3_s2
inception_4a_pool
inception_4b_pool
inception_4c_pool
inception_4d_pool
inception_4e_pool
pool4_3x3_s2
inception_5a_pool
inception_5b_pool
Optimizing 1 schedules
**************************************************
* COMPUTING MEMORY REQUIREMENTS
**************************************************
IO @@@ data_blob MemoryAllocation(start=0, end=4816896, size=4816896, extra=[], strategy=[], layout=-1, timestamp=-1, slice=1, shapes=SizeType(batches=1, channels=3, height=224, width=224), replication=Replication(full_sect_num=0, repl_sect_num=0, repl_unit_num=0, repl_unit_width=0, channels_division=0), written=False, specifier='', IO=True)
IO @@@ pool5_7x7_s1_output_blob MemoryAllocation(start=0, end=1056, size=1056, extra=[], strategy=[], layout=-1, timestamp=-1, slice=1, shapes=SizeType(batches=1, channels=1024, height=1, width=1), replication=Replication(full_sect_num=0, repl_sect_num=0, repl_unit_num=0, repl_unit_width=0, channels_division=0), written=False, specifier='', IO=True)
Minimum Memory __________
1 ['pool1_3x3_s2'] size:5117952 remap:[] data movement:[]
1	data_blob M[0,4816896] Z=4816896 F=[1] B=[0] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=3, height=224, width=224)
1	pool1_3x3_s2_blob M[0,301056] Z=301056 F=[2] B=[1] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=64, height=56, width=56)
MAX  1
TOP 5
__________
1 ['pool1_3x3_s2'] size:5117952 remap:[] data movement:[]
1	data_blob M[0,4816896] Z=4816896 F=[1] B=[0] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=3, height=224, width=224)
1	pool1_3x3_s2_blob M[0,301056] Z=301056 F=[2] B=[1] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=64, height=56, width=56)
__________
0 ['data'] size:4816896 remap:[] data movement:[]
0	data_blob M[0,4816896] Z=4816896 F=[1] B=[0] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=3, height=224, width=224)
__________
17 ['inception_3b_pool'] size:827904 remap:[] data movement:[]
17	inception_3a_output_blob M[0,225792] Z=225792 F=[12, 13, 15, 17] B=[11] E=[] S=['replace_layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=256, height=28, width=28)
17	inception_3b_pool_blob M[0,225792] Z=225792 F=[18] B=[17] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=256, height=28, width=28)
17	inception_3b_1x1/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[12] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=128, height=28, width=28)
17	inception_3b_3x3/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[14] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=192, height=28, width=28)
17	inception_3b_5x5/Conv2D_blob M[0,75264] Z=75264 F=[19] B=[16] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=96, height=28, width=28)
__________
14 ['inception_3b_3x3/Conv2D'] size:677376 remap:[] data movement:[]
14	inception_3b_3x3_reduce/Conv2D_blob M[0,150528] Z=150528 F=[14] B=[13] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=128, height=28, width=28)
14	inception_3a_output_blob M[0,225792] Z=225792 F=[12, 13, 15, 17] B=[11] E=[] S=['replace_layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=256, height=28, width=28)
14	inception_3b_1x1/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[12] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=128, height=28, width=28)
14	inception_3b_3x3/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[14] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=192, height=28, width=28)
__________
16 ['inception_3b_5x5/Conv2D'] size:677376 remap:[] data movement:[]
16	inception_3b_5x5_reduce/Conv2D_blob M[0,75264] Z=75264 F=[16] B=[15] E=[] S=['layer'] [] L=-1 T=SizeType(batches=1, channels=32, height=28, width=28)
16	inception_3a_output_blob M[0,225792] Z=225792 F=[12, 13, 15, 17] B=[11] E=[] S=['replace_layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=256, height=28, width=28)
16	inception_3b_1x1/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[12] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=128, height=28, width=28)
16	inception_3b_3x3/Conv2D_blob M[0,150528] Z=150528 F=[19] B=[14] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=192, height=28, width=28)
16	inception_3b_5x5/Conv2D_blob M[0,75264] Z=75264 F=[19] B=[16] E=[] S=['layer'] ['concat'] L=-1 T=SizeType(batches=1, channels=96, height=28, width=28)
Using Hardware Version [3, 3]
**************************************************
* REPLICATION 
**************************************************
Simple Replication
optimized replication: 32 pool1_3x3_s2
optimized replication: 32 pool2_3x3_s2
optimized replication: 32 inception_3a_5x5/Conv2D
optimized replication: 32 inception_3b_3x3/Conv2D
optimized replication: 32 inception_3b_5x5/Conv2D
optimized replication: 32 inception_4a_5x5/Conv2D
optimized replication: 32 inception_4b_3x3/Conv2D
optimized replication: 32 inception_4b_5x5/Conv2D
optimized replication: 32 inception_4c_3x3/Conv2D
optimized replication: 32 inception_4c_5x5/Conv2D
optimized replication: 48 inception_4d_3x3/Conv2D
optimized replication: 32 inception_4d_5x5/Conv2D
optimized replication: 32 inception_4e_3x3/Conv2D
optimized replication: 32 inception_4e_5x5/Conv2D
optimized replication: 32 inception_5a_3x3/Conv2D
optimized replication: 32 inception_5a_5x5/Conv2D
optimized replication: 48 inception_5b_5x5/Conv2D
replicaiton done
replicaiton done 2
**************************************************
* ALLOCATING DYNAMIC MEMORY SCHEDULE
**************************************************
Allocating Memory all
Trying no-DDR strategies...
You tell me there must be DDR, Skip the AM only
Trying DDR strategies with 0 MB ...
Reset Memory
Trying strategy top (DDR: 0 MB)
Performing two level schedule strategy all
Reference {'Convolution': ['ddr_to_am', 'ddr_to_ddr', 'am_to_am']}
outs 1
inss 1
 ?????  <vaic.dpuv1.codegeneration.hardware.DDR object at 0x7f2930200668> <vaic.dpuv1.codegeneration.hardware.DDR object at 0x7f29303411d0>
Done I instruction:inputs  [] outputs [0] call data
None
BlobInformation(size=4816896, name='data_blob', memory=MemoryAllocation(start=0, end=1605632, size=1605632, extra=[1], strategy=[], layout=1, timestamp=1, slice=1, shapes=SizeType(batches=1, channels=3, height=224, width=224), replication=Replication(full_sect_num=0, repl_sect_num=1, repl_unit_num=3, repl_unit_width=32, channels_division=[3]), written=False, specifier='Input', IO=True), dag=ColorForDAG(active=[3], schedule=-1, forward=[1], backward=[0], extra=None, hook=[]), layer_type=['layer'], data_movement_operations=[], data_movement_operation_costs=[])
Successful Strategy top (DDR: 256 MB)
Done schedule 80 STEPS
**************************************************
* ADDED weight information weights_bit 
**************************************************
**************************************************
* GENERATING OUTPUT REPORTS
**************************************************
schedule_and_parallelism
Minimum Memory 79 ['pool5_7x7_s1_output'] 9437184
pool5_7x7_s1_blob M[9436128,9437184] Z=1056 F=[79] B=[78] E=[1] S=['layer'] [] L=0 T=SizeType(batches=1, channels=1024, height=1, width=1)
pool5_7x7_s1_output_blob M[0,16384] Z=16384 F=[] B=[79] E=[1] S=['layer'] [] L=1 T=SizeType(batches=1, channels=1024, height=1, width=1)
**************************************************
* GENERATING OUTPUT FILES
**************************************************
XDNN Command file: ./inception_v1_baseline_partition_01.pb
XDNN JSON Report file: ./inception_v1_baseline_partition_01.pb.json
.
Path to generatefile exists...
***** Inst JSON
[1] <class 'vaic.dpuv1.utils.xdnn_util.dict2attr'>
ddr_to_am REP V BestReplication(FSN=0, RSN=1, RUN=12, RUW=8, FSC=0, RSC=7) SpaceAndTime(space=4816896, time=523, replication=Replication(full_sect_num=0, repl_sect_num=1, repl_unit_num=12, repl_unit_width=8, channels_division=[3])) pool1_3x3_s2
AM_BUFF_1 9437184 6 1572864 16384 MemoryAllocation(start=0, end=1605632, size=4816896, extra=[1], strategy=[], layout=1, timestamp=523, slice=1, shapes=SizeType(batches=1, channels=3, height=224, width=224), replication=Replication(full_sect_num=0, repl_sect_num=1, repl_unit_num=12, repl_unit_width=8, channels_division=[3]), written=True, specifier='Input', IO=True)
PIPELINE POOL S SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=64, height=27, width=32)
V3 Tile pool1_3x3_s2 (19.12037037037037, -82944)
output ddr {(16384, 268435456): MemoryAllocation(start=16384, end=268435456, size=268419072, extra=[], strategy=[], layout=-1, timestamp=-1, slice=-1, shapes=None, replication=Replication(full_sect_num=0, repl_sect_num=0, repl_unit_num=0, repl_unit_width=0, channels_division=0), written=False, specifier='', IO=False)}
input ddr {(1605632, 268435456): MemoryAllocation(start=1605632, end=268435456, size=266829824, extra=[], strategy=[], layout=-1, timestamp=-1, slice=-1, shapes=None, replication=Replication(full_sect_num=0, repl_sect_num=0, repl_unit_num=0, repl_unit_width=0, channels_division=0), written=False, specifier='', IO=False)}
OUTPUT REPORT:
Unsupported Layers: 0
***** Inst JSON Done
* XDNN QUANT JSON  ./inception_v1_baseline_partition_01.pb_quant.json
***** Inst FILE
ddr_to_am REP V BestReplication(FSN=0, RSN=1, RUN=12, RUW=8, FSC=0, RSC=7) SpaceAndTime(space=4816896, time=1282, replication=Replication(full_sect_num=0, repl_sect_num=1, repl_unit_num=12, repl_unit_width=8, channels_division=[3])) pool1_3x3_s2
AM_BUFF_1 9437184 6 1572864 16384 MemoryAllocation(start=0, end=1605632, size=4816896, extra=[1], strategy=[], layout=1, timestamp=1282, slice=1, shapes=SizeType(batches=1, channels=3, height=224, width=224), replication=Replication(full_sect_num=0, repl_sect_num=1, repl_unit_num=12, repl_unit_width=8, channels_division=[3]), written=True, specifier='Input', IO=True)
PIPELINE POOL S SizeType(batches=1, channels=1, height=2, width=2) SizeType(batches=1, channels=64, height=27, width=32)
V3 Tile pool1_3x3_s2 (19.12037037037037, -82944)
***** Inst FILE OUT
***** Inst COLLECT
***** COLLECT CODES 95
# template XNConv id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu prelu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset 
# template XNDeconv id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu prelu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset 
# template XNConvP id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt  pool_kernel_w pool_kernel_h pool_strides_w pool_strides_h pool_paddings_w pool_paddings_h pool_fcmode
# template XNUpload id XNOp inaddr insize inchan
# template XNConcat pound id XNOp name start end size dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width MUTE
# template XNInner id XNOp name relu prelu preshift scale postshift matrixheight matrixwidthh inaddr inheight inwidth outaddr outheight outwidth
# template XNGather id XNOp uram_dest ddr_src insize_w insize_h inchan a0 b1 c1 start_row end_row slice full_sect_num repl_sect_num repl_unit_num repl_unit_width srcAddrReadFromImgQ sep comment
# template XNScatter id XNOp uram_src ddr_dest outsize_w outsize_h outchan a0 b1 c1 start_row end_row slice full_sect_num repl_sect_num repl_unit_num repl_unit_width destAddrReadFromImgQ
# template XNEltwise id XNOp name add bn relu prelu inaddrA inaddrB insize_w insize_h inchan outaddr weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset EWA_2nd-Src_AM 
# template XNAvgPool id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h inaddr insize_w insize_h inchan outaddr outsize_w outsize_h weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset 
# template XNMaxPool id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h inaddr insize_w insize_h inchan outaddr outsize_w outsize_h weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset 
# template XNUpsample id XNOp name kernel_h kernel_w   inaddr insize_h  insize_w  inchan outaddr outsize_w outsize_h method weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset 
# template XNAvgPoolPipelined id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h inaddr insize_w insize_h inchan outaddr outsize_w outsize_h weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset conv_name conv_kernel_w conv_kernel_h conv_strides_w conv_strides_h conv_padding_w conv_padding_h conv_dilation_w conv_dilation_h conv_relu conv_prelu conv_bias pool_insize_w pool_insize_h pool_inchan conv_outsize_w
# template XNMaxPoolPipelined id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h inaddr insize_w insize_h inchan outaddr outsize_w outsize_h weights_bit slice src_full_sect_num src_repl_sect_num src_repl_unit_num src_repl_unit_width dst_full_sect_num dst_repl_sect_num dst_repl_unit_num dst_repl_unit_width concat_i_am_your_father concat_full_sect_num concat_repl_sect_num concat_repl_unit_num concat_repl_unit_width concat_starting_ch concat_channels wait_download wait_upload wait_conv wait_pool wait_ew wait_upsmpl parallel_read prerelu en_pingpong_weight en_halfrate_mode en_inlinemaxpool srcAddrReadFromImgQ destAddrReadFromImgQ srcAddDDR dstAddDDR tile_width tile_height HAT SRCAM-Buffer_0 SRCAM-Buffer_1 DEST_AM-Buffer_Offset conv_name conv_kernel_w conv_kernel_h conv_strides_w conv_strides_h conv_padding_w conv_padding_h conv_dilation_w conv_dilation_h conv_relu conv_prelu conv_bias pool_insize_w pool_insize_h pool_inchan conv_outsize_w
# 1 XNGather 0x0 0x0 224 224 3 0 1 1 0 223 1 0 1 3 32 1 # Input data data: type=Input, sizes=None, shapes=None, sched 0 Kernel None Strides None Padding None MUTE CODE
# spLiT pool1_3x3_s2 9437184 ;
2 XNMaxPoolPipelined pool1_3x3_s2 3 3 2 2 0 0 0x0 224 224 3 0x173c0 56 56 1 1 0 1 12 8 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 1 1 0 1 0 32 27 1 0 16384 65536 conv1_7x7_s2/Conv2D 7 7 2 2 2 2 1 1 1 0 1 112 112 64 56 # V3 SPLIT Code :)ddr_to_am
5 XNConv conv2_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 20 1 0 1 0x173c0 56 56 64 0x15b40 56 56 64 1 1 1 0 0 0 0 2 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
7 XNMaxPoolPipelined pool2_3x3_s2 3 3 2 2 0 0 0x15b40 56 56 64 0x179e0 28 28 1 1 0 2 3 32 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 conv2_3x3/Conv2D 3 3 1 1 1 1 1 1 1 0 1 56 56 192 28 # am_to_am
9 XNConv inception_3a_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x179e0 28 28 192 0x170b0 28 28 64 1 1 2 0 0 0 1 0 0 0 1 3 0 0 0 0 256 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
11 XNConv inception_3a_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x179e0 28 28 192 0x16da0 28 28 96 1 1 2 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
13 XNConv inception_3a_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x16da0 28 28 96 0x170b0 28 28 128 1 1 1 0 0 0 2 0 0 0 1 3 0 0 0 64 256 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
15 XNConv inception_3a_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x179e0 28 28 192 0x16da0 28 28 16 1 1 2 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
17 XNConv inception_3a_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 23 1 0 1 0x16da0 28 28 16 0x176d0 28 28 32 1 1 0 1 3 32 1 0 0 0 1 3 0 0 0 192 256 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
19 XNMaxPool inception_3a_pool 3 3 1 1 1 1 0x179e0 28 28 192 0x16a90 28 28 1 1 2 0 0 0 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
21 XNConv inception_3a_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x16a90 28 28 192 0x176d0 28 28 32 1 1 2 0 0 0 1 0 0 0 1 3 0 0 0 224 256 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 23 XNConcat inception_3a_output  0x170b0 0x179e0 225792 3 0 0 0 MUTE CODE inception_3a_output: type=Concat, sizes=None, shapes=None, sched 11 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
25 XNConv inception_3b_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x170b0 28 28 256 0x16160 28 28 128 1 1 3 0 0 0 2 0 0 0 1 5 0 0 0 0 480 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
27 XNConv inception_3b_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x170b0 28 28 256 0x179e0 28 28 128 1 1 3 0 0 0 1 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
29 XNConv inception_3b_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x179e0 28 28 128 0x16470 28 28 192 1 1 1 1 3 32 2 0 0 0 1 5 0 0 0 128 480 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
31 XNConv inception_3b_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x170b0 28 28 256 0x17cf0 28 28 32 1 1 3 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
33 XNConv inception_3b_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 23 1 0 1 0x17cf0 28 28 32 0x16a90 28 28 96 1 1 0 1 3 32 1 0 0 0 1 5 0 0 0 320 480 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
35 XNMaxPool inception_3b_pool 3 3 1 1 1 1 0x170b0 28 28 256 0x15830 28 28 1 1 3 0 0 0 3 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
37 XNConv inception_3b_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x15830 28 28 256 0x16da0 28 28 64 1 1 3 0 0 0 1 0 0 0 1 5 0 0 0 416 480 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 39 XNConcat inception_3b_output  0x16160 0x170b0 376320 5 0 0 0 MUTE CODE inception_3b_output: type=Concat, sizes=None, shapes=None, sched 19 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
41 XNMaxPool pool3_3x3_s2 3 3 2 2 0 0 0x16160 28 28 480 0x17c2c 14 14 1 1 5 0 0 0 5 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
43 XNConv inception_4a_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17c2c 14 14 480 0x17794 14 14 192 1 1 5 0 0 0 2 0 0 0 1 6 0 0 0 0 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
45 XNConv inception_4a_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17c2c 14 14 480 0x176d0 14 14 96 1 1 5 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
47 XNConv inception_4a_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x176d0 14 14 96 0x1791c 14 14 208 1 1 1 0 0 0 3 0 0 0 1 6 0 0 0 192 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
49 XNConv inception_4a_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17c2c 14 14 480 0x176d0 14 14 16 1 1 5 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
51 XNConv inception_4a_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 22 1 0 1 0x176d0 14 14 16 0x17aa4 14 14 48 1 1 0 1 3 32 1 0 0 0 1 6 0 0 0 400 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
53 XNMaxPool inception_4a_pool 3 3 1 1 1 1 0x17c2c 14 14 480 0x173c0 14 14 1 1 5 0 0 0 5 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
55 XNConv inception_4a_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x173c0 14 14 480 0x17aa4 14 14 64 1 1 5 0 0 0 1 0 0 0 1 6 0 0 0 448 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 57 XNConcat inception_4a_output  0x17794 0x17c2c 112896 6 0 0 0 MUTE CODE inception_4a_output: type=Concat, sizes=None, shapes=None, sched 28 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
59 XNConv inception_4b_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x17794 14 14 512 0x172fc 14 14 160 1 1 6 0 0 0 2 0 0 0 1 6 0 0 0 0 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
61 XNConv inception_4b_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17794 14 14 512 0x17e78 14 14 112 1 1 6 0 0 0 1 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
63 XNConv inception_4b_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x17e78 14 14 112 0x173c0 14 14 224 1 1 1 1 3 32 3 0 0 0 1 6 0 0 0 160 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
65 XNConv inception_4b_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17794 14 14 512 0x17f3c 14 14 24 1 1 6 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
67 XNConv inception_4b_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 23 1 0 1 0x17f3c 14 14 24 0x1760c 14 14 64 1 1 0 1 3 32 1 0 0 0 1 6 0 0 0 384 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
69 XNMaxPool inception_4b_pool 3 3 1 1 1 1 0x17794 14 14 512 0x16e64 14 14 1 1 6 0 0 0 6 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
71 XNConv inception_4b_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x16e64 14 14 512 0x1760c 14 14 64 1 1 6 0 0 0 1 0 0 0 1 6 0 0 0 448 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 73 XNConcat inception_4b_output  0x172fc 0x17794 112896 6 0 0 0 MUTE CODE inception_4b_output: type=Concat, sizes=None, shapes=None, sched 36 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
75 XNConv inception_4c_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x172fc 14 14 512 0x17b68 14 14 128 1 1 6 0 0 0 2 0 0 0 1 6 0 0 0 0 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
77 XNConv inception_4c_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x172fc 14 14 512 0x179e0 14 14 128 1 1 6 0 0 0 1 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
79 XNConv inception_4c_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x179e0 14 14 128 0x17c2c 14 14 256 1 1 1 1 3 32 3 0 0 0 1 6 0 0 0 128 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
81 XNConv inception_4c_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x172fc 14 14 512 0x17aa4 14 14 24 1 1 6 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
83 XNConv inception_4c_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 23 1 0 1 0x17aa4 14 14 24 0x17e78 14 14 64 1 1 0 1 3 32 1 0 0 0 1 6 0 0 0 384 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
85 XNMaxPool inception_4c_pool 3 3 1 1 1 1 0x172fc 14 14 512 0x16e64 14 14 1 1 6 0 0 0 6 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
87 XNConv inception_4c_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x16e64 14 14 512 0x17e78 14 14 64 1 1 6 0 0 0 1 0 0 0 1 6 0 0 0 448 512 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 89 XNConcat inception_4c_output  0x17b68 0x18000 112896 6 0 0 0 MUTE CODE inception_4c_output: type=Concat, sizes=None, shapes=None, sched 44 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
91 XNConv inception_4d_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17b68 14 14 512 0x176d0 14 14 112 1 1 6 0 0 0 2 0 0 0 1 6 0 0 0 0 528 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
93 XNConv inception_4d_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17b68 14 14 512 0x17548 14 14 144 1 1 6 0 0 0 1 1 2 48 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
95 XNConv inception_4d_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 23 1 0 1 0x17548 14 14 144 0x17794 14 14 288 1 1 1 1 2 48 3 0 0 0 1 6 0 0 0 112 528 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
97 XNConv inception_4d_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17b68 14 14 512 0x1760c 14 14 32 1 1 6 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
99 XNConv inception_4d_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 24 1 0 1 0x1760c 14 14 32 0x179e0 14 14 64 1 1 0 1 3 32 1 0 0 0 1 6 0 0 0 400 528 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
101 XNMaxPool inception_4d_pool 3 3 1 1 1 1 0x17b68 14 14 512 0x17238 14 14 1 1 6 0 0 0 6 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
103 XNConv inception_4d_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17238 14 14 512 0x179e0 14 14 64 1 1 6 0 0 0 1 0 0 0 1 6 0 0 0 464 528 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 105 XNConcat inception_4d_output  0x176d0 0x17b68 112896 6 0 0 0 MUTE CODE inception_4d_output: type=Concat, sizes=None, shapes=None, sched 52 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
107 XNConv inception_4e_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x176d0 14 14 528 0x16fec 14 14 256 1 1 6 0 0 0 3 0 0 0 1 9 0 0 0 0 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
109 XNConv inception_4e_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x176d0 14 14 528 0x17db4 14 14 160 1 1 6 0 0 0 1 2 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
111 XNConv inception_4e_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 24 1 0 1 0x17db4 14 14 160 0x17174 14 14 320 1 1 1 2 3 32 4 0 0 0 1 9 0 0 0 256 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
113 XNConv inception_4e_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x176d0 14 14 528 0x17f3c 14 14 32 1 1 6 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
115 XNConv inception_4e_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 24 1 0 1 0x17f3c 14 14 32 0x17484 14 14 128 1 1 0 1 3 32 2 0 0 0 1 9 0 0 0 576 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
117 XNMaxPool inception_4e_pool 3 3 1 1 1 1 0x176d0 14 14 528 0x17b68 14 14 1 1 6 0 0 0 6 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
119 XNConv inception_4e_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17b68 14 14 528 0x17548 14 14 128 1 1 6 0 0 0 2 0 0 0 1 9 0 0 0 704 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 121 XNConcat inception_4e_output  0x16fec 0x176d0 169344 9 0 0 0 MUTE CODE inception_4e_output: type=Concat, sizes=None, shapes=None, sched 60 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
123 XNMaxPool pool4_3x3_s2 3 3 2 2 0 0 0x16fec 14 14 832 0x17e47 7 7 1 1 9 0 0 0 9 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
125 XNConv inception_5a_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17e47 7 7 832 0x17c8e 7 7 256 1 1 9 0 0 0 3 0 0 0 1 9 0 0 0 0 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
127 XNConv inception_5a_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17e47 7 7 832 0x17bfb 7 7 160 1 1 9 0 0 0 1 2 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
129 XNConv inception_5a_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 26 1 0 1 0x17bfb 7 7 160 0x17cf0 7 7 320 1 1 1 2 3 32 4 0 0 0 1 9 0 0 0 256 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
131 XNConv inception_5a_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 24 1 0 1 0x17e47 7 7 832 0x17c5d 7 7 32 1 1 9 0 0 0 0 1 3 32 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
133 XNConv inception_5a_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 24 1 0 1 0x17c5d 7 7 32 0x17db4 7 7 128 1 1 0 1 3 32 2 0 0 0 1 9 0 0 0 576 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
135 XNMaxPool inception_5a_pool 3 3 1 1 1 1 0x17e47 7 7 832 0x17ad5 7 7 1 1 9 0 0 0 9 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
137 XNConv inception_5a_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 25 1 0 1 0x17ad5 7 7 832 0x17de5 7 7 128 1 1 9 0 0 0 2 0 0 0 1 9 0 0 0 704 832 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 139 XNConcat inception_5a_output  0x17c8e 0x17e47 42336 9 0 0 0 MUTE CODE inception_5a_output: type=Concat, sizes=None, shapes=None, sched 69 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
141 XNConv inception_5b_1x1/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x17c8e 7 7 832 0x17a73 7 7 384 1 1 9 0 0 0 4 0 0 0 1 11 0 0 0 0 1024 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
143 XNConv inception_5b_3x3_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x17c8e 7 7 832 0x17f9e 7 7 192 1 1 9 0 0 0 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
145 XNConv inception_5b_3x3/Conv2D 3 3 1 1 1 1 1 1 0 32768 24 1 0 1 0x17f9e 7 7 192 0x17b37 7 7 384 1 1 2 0 0 0 4 0 0 0 1 11 0 0 0 384 1024 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
147 XNConv inception_5b_5x5_reduce/Conv2D 1 1 1 1 0 0 1 1 0 32768 22 1 0 1 0x17c8e 7 7 832 0x17fcf 7 7 48 1 1 9 0 0 0 0 1 2 48 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
149 XNConv inception_5b_5x5/Conv2D 5 5 1 1 2 2 1 1 0 32768 25 1 0 1 0x17fcf 7 7 48 0x17bfb 7 7 128 1 1 0 1 2 48 2 0 0 0 1 11 0 0 0 768 1024 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
151 XNMaxPool inception_5b_pool 3 3 1 1 1 1 0x17c8e 7 7 832 0x17e47 7 7 1 1 9 0 0 0 9 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
153 XNConv inception_5b_pool_proj/Conv2D 1 1 1 1 0 0 1 1 0 32768 23 1 0 1 0x17e47 7 7 832 0x17c2c 7 7 128 1 1 9 0 0 0 2 0 0 0 1 11 0 0 0 896 1024 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
# # 155 XNConcat inception_5b_output  0x17a73 0x17c8e 51744 11 0 0 0 MUTE CODE inception_5b_output: type=Concat, sizes=None, shapes=None, sched 77 Kernel None Strides None Padding None  REDUNDANT CODE OMMITTED 
157 XNAvgPool pool5_7x7_s1 7 7 1 1 0 0 0x17a73 7 7 1024 0x17ff5 1 1 1 1 11 0 0 0 11 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 # am_to_am
159 XNScatter 0x17ff5 0x0 1 1 1024 0 1 1 0 0 1 11 0 0 0 1 # Output pool5_7x7_s1_output
**************************************************
* CLEANING PREVIOUS WEIGHTS
**************************************************
**************************************************
* WRITING WEIGHTS
**************************************************
Weight HDF5: ./inception_v1_baseline_partition_01.pb_data.h5
Processing weights for 80 schedule steps: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80
Done writing weights.
**************************************************
* EUREKA Schedule  PASSED
**************************************************
SUCCESSFUL COMPILATION
.... is_supported, layer_index, layer_name
.... False,   0, data
.... True ,   1, pool1_3x3_s2
.... True ,   2, conv2_3x3_reduce/Conv2D
.... True ,   3, pool2_3x3_s2
.... True ,   4, inception_3a_1x1/Conv2D
.... True ,   5, inception_3a_3x3_reduce/Conv2D
.... True ,   6, inception_3a_3x3/Conv2D
.... True ,   7, inception_3a_5x5_reduce/Conv2D
.... True ,   8, inception_3a_5x5/Conv2D
.... True ,   9, inception_3a_pool
.... True ,  10, inception_3a_pool_proj/Conv2D
.... True ,  11, inception_3a_output
.... True ,  12, inception_3b_1x1/Conv2D
.... True ,  13, inception_3b_3x3_reduce/Conv2D
.... True ,  14, inception_3b_3x3/Conv2D
.... True ,  15, inception_3b_5x5_reduce/Conv2D
.... True ,  16, inception_3b_5x5/Conv2D
.... True ,  17, inception_3b_pool
.... True ,  18, inception_3b_pool_proj/Conv2D
.... True ,  19, inception_3b_output
.... True ,  20, pool3_3x3_s2
.... True ,  21, inception_4a_1x1/Conv2D
.... True ,  22, inception_4a_3x3_reduce/Conv2D
.... True ,  23, inception_4a_3x3/Conv2D
.... True ,  24, inception_4a_5x5_reduce/Conv2D
.... True ,  25, inception_4a_5x5/Conv2D
.... True ,  26, inception_4a_pool
.... True ,  27, inception_4a_pool_proj/Conv2D
.... True ,  28, inception_4a_output
.... True ,  29, inception_4b_1x1/Conv2D
.... True ,  30, inception_4b_3x3_reduce/Conv2D
.... True ,  31, inception_4b_3x3/Conv2D
.... True ,  32, inception_4b_5x5_reduce/Conv2D
.... True ,  33, inception_4b_5x5/Conv2D
.... True ,  34, inception_4b_pool
.... True ,  35, inception_4b_pool_proj/Conv2D
.... True ,  36, inception_4b_output
.... True ,  37, inception_4c_1x1/Conv2D
.... True ,  38, inception_4c_3x3_reduce/Conv2D
.... True ,  39, inception_4c_3x3/Conv2D
.... True ,  40, inception_4c_5x5_reduce/Conv2D
.... True ,  41, inception_4c_5x5/Conv2D
.... True ,  42, inception_4c_pool
.... True ,  43, inception_4c_pool_proj/Conv2D
.... True ,  44, inception_4c_output
.... True ,  45, inception_4d_1x1/Conv2D
.... True ,  46, inception_4d_3x3_reduce/Conv2D
.... True ,  47, inception_4d_3x3/Conv2D
.... True ,  48, inception_4d_5x5_reduce/Conv2D
.... True ,  49, inception_4d_5x5/Conv2D
.... True ,  50, inception_4d_pool
.... True ,  51, inception_4d_pool_proj/Conv2D
.... True ,  52, inception_4d_output
.... True ,  53, inception_4e_1x1/Conv2D
.... True ,  54, inception_4e_3x3_reduce/Conv2D
.... True ,  55, inception_4e_3x3/Conv2D
.... True ,  56, inception_4e_5x5_reduce/Conv2D
.... True ,  57, inception_4e_5x5/Conv2D
.... True ,  58, inception_4e_pool
.... True ,  59, inception_4e_pool_proj/Conv2D
.... True ,  60, inception_4e_output
.... True ,  61, pool4_3x3_s2
.... True ,  62, inception_5a_1x1/Conv2D
.... True ,  63, inception_5a_3x3_reduce/Conv2D
.... True ,  64, inception_5a_3x3/Conv2D
.... True ,  65, inception_5a_5x5_reduce/Conv2D
.... True ,  66, inception_5a_5x5/Conv2D
.... True ,  67, inception_5a_pool
.... True ,  68, inception_5a_pool_proj/Conv2D
.... True ,  69, inception_5a_output
.... True ,  70, inception_5b_1x1/Conv2D
.... True ,  71, inception_5b_3x3_reduce/Conv2D
.... True ,  72, inception_5b_3x3/Conv2D
.... True ,  73, inception_5b_5x5_reduce/Conv2D
.... True ,  74, inception_5b_5x5/Conv2D
.... True ,  75, inception_5b_pool
.... True ,  76, inception_5b_pool_proj/Conv2D
.... True ,  77, inception_5b_output
.... True ,  78, pool5_7x7_s1
.... False,  79, pool5_7x7_s1_output

Partition FPGA (un)supported layers from compiler schedule ....
Partition FPGA (un)supported layers from compiler schedule [DONE]

Refine Graph Partitions ....
.... partition (  0, False) --> [(1, True)]
.... partition (  1, True ) --> []
....
.... partition (  0, False) <-- []
.... partition (  1, True ) <-- [(0, False)]
....

SUMMARY:
.... partition_index "0" - SUPPORTED: False
........ inputs:          ['data']
........ inputs actual:   ['data']
........ outputs:         ['data']
........ outputs actual:  ['data']
.... partition_index "1" - SUPPORTED: True
........ inputs:          ['data']
........ inputs actual:   ['data']
........ outputs:         ['pool5_7x7_s1']
........ outputs actual:  ['pool5_7x7_s1']
Refine Graph Partitions [DONE]
Transorm partition index "1"
Creating "FPGA" transformation....
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:236: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:261: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
freeze model
.... node count 330
.... node count after removing redundant nodes 14
.... node count after removing blacklisted nodes 14
save graph at ./inception_v1_baseline-fpga.pb
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_util_tf.py:281: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

load partitioned graph
WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:401: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

WARNING:tensorflow:From /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib/python3.6/site-packages/vai/dpuv1/rt/xdnn_rt_tf.py:403: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

2020-11-29 11:59:48.711741: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties: 
name: Tesla V100-PCIE-32GB major: 7 minor: 0 memoryClockRate(GHz): 1.38
pciBusID: 0000:61:00.0
2020-11-29 11:59:48.712001: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-11-29 11:59:48.712033: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-11-29 11:59:48.712054: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-11-29 11:59:48.712079: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-11-29 11:59:48.712096: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-11-29 11:59:48.712114: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-11-29 11:59:48.712136: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-11-29 11:59:48.713654: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
2020-11-29 11:59:48.713723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1180] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-11-29 11:59:48.713730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1186]      0 
2020-11-29 11:59:48.713736: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1199] 0:   N 
2020-11-29 11:59:48.715305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1325] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 30555 MB memory) -> physical GPU (device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:61:00.0, compute capability: 7.0)
N/A% (0 of 500) |                                                                                                                                                                                                                  | Elapsed Time: 0:00:00 ETA:  --:--:--2020-11-29 11:59:49.126735: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
-------------------
Speaking to Butler 
Response from Butler is: 
errCode: errCode: 0
errCode String: SUCCESS
myHandle: 11
valid: 1

[XDNN] loading xclbin settings from /opt/xilinx/overlaybins/xdnnv3/xdnn_v3_96x16_2pe_8b_9mb_bank03_2.xclbin.json
[XDNN] using custom DDR banks 0,3
Path ./inception_v1_baseline_partition_01-data.h5 is a file.
Loading weights/bias/quant_params to FPGA...

[XRT]    git hash                   : 2d6bfe4ce91051d4e5b499d38fc493586dd4859a
[XDNN]   git hash                   : bba7f201475d7e280ff3d2ba2355aa6dd6fdb6a0
[XDNN] kernel configuration
[XDNN]   num cores                  : 2
[XDNN]   dsp array width            : 96
[XDNN]   axi data width (in 32bits) : 16
[XDNN]   img mem size               : 9 MB
[XDNN]   max instr num              : 1536
[XDNN]   max xbar entries           : 4096
[XDNN]   version                    : 3.2
[XDNN]   8-bit mode                 : 1
Finished 'execute' in 0.0054 secs
  0% (1 of 500) |                                                                                                                                                                                                                  | Elapsed Time: 0:00:06 ETA:   0:54:15Finished 'execute' in 0.0048 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
  0% (4 of 500) |#                                                                                                                                                                                                                 | Elapsed Time: 0:00:06 ETA:   0:53:55Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
  1% (8 of 500) |###                                                                                                                                                                                                               | Elapsed Time: 0:00:06 ETA:   0:06:49Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
  2% (12 of 500) |#####                                                                                                                                                                                                            | Elapsed Time: 0:00:06 ETA:   0:06:46Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
  3% (16 of 500) |######                                                                                                                                                                                                           | Elapsed Time: 0:00:06 ETA:   0:03:25Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
  3% (19 of 500) |#######                                                                                                                                                                                                          | Elapsed Time: 0:00:06 ETA:   0:03:24Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
  4% (23 of 500) |#########                                                                                                                                                                                                        | Elapsed Time: 0:00:06 ETA:   0:02:23Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0042 secs
  5% (27 of 500) |###########                                                                                                                                                                                                      | Elapsed Time: 0:00:07 ETA:   0:02:22Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
  6% (31 of 500) |############                                                                                                                                                                                                     | Elapsed Time: 0:00:07 ETA:   0:01:47Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0040 secs
  7% (36 of 500) |###############                                                                                                                                                                                                  | Elapsed Time: 0:00:07 ETA:   0:01:45Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
  8% (40 of 500) |################                                                                                                                                                                                                 | Elapsed Time: 0:00:07 ETA:   0:01:23Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
  8% (44 of 500) |##################                                                                                                                                                                                               | Elapsed Time: 0:00:07 ETA:   0:01:22Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
  9% (48 of 500) |####################                                                                                                                                                                                             | Elapsed Time: 0:00:07 ETA:   0:01:09Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
 10% (51 of 500) |#####################                                                                                                                                                                                            | Elapsed Time: 0:00:07 ETA:   0:01:08Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0049 secs
 11% (55 of 500) |######################                                                                                                                                                                                           | Elapsed Time: 0:00:07 ETA:   0:01:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 11% (59 of 500) |########################                                                                                                                                                                                         | Elapsed Time: 0:00:07 ETA:   0:01:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 12% (63 of 500) |##########################                                                                                                                                                                                       | Elapsed Time: 0:00:07 ETA:   0:00:53Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 13% (67 of 500) |############################                                                                                                                                                                                     | Elapsed Time: 0:00:07 ETA:   0:00:52Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0046 secs
Finished 'execute' in 0.0041 secs
 14% (70 of 500) |#############################                                                                                                                                                                                    | Elapsed Time: 0:00:07 ETA:   0:00:47Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 14% (74 of 500) |##############################                                                                                                                                                                                   | Elapsed Time: 0:00:07 ETA:   0:00:47Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 15% (78 of 500) |################################                                                                                                                                                                                 | Elapsed Time: 0:00:07 ETA:   0:00:43Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 16% (80 of 500) |#################################                                                                                                                                                                                | Elapsed Time: 0:00:08 ETA:   0:00:42Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
 16% (84 of 500) |###################################                                                                                                                                                                              | Elapsed Time: 0:00:08 ETA:   0:00:40Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 17% (89 of 500) |#####################################                                                                                                                                                                            | Elapsed Time: 0:00:08 ETA:   0:00:39Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 18% (93 of 500) |######################################                                                                                                                                                                           | Elapsed Time: 0:00:08 ETA:   0:00:36Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
 19% (97 of 500) |########################################                                                                                                                                                                         | Elapsed Time: 0:00:08 ETA:   0:00:35Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 20% (101 of 500) |##########################################                                                                                                                                                                      | Elapsed Time: 0:00:08 ETA:   0:00:33Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 20% (104 of 500) |###########################################                                                                                                                                                                     | Elapsed Time: 0:00:08 ETA:   0:00:32Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0037 secs
 21% (108 of 500) |############################################                                                                                                                                                                    | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0048 secs
 22% (112 of 500) |##############################################                                                                                                                                                                  | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
 23% (116 of 500) |################################################                                                                                                                                                                | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
 23% (119 of 500) |#################################################                                                                                                                                                               | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0037 secs
 24% (123 of 500) |###################################################                                                                                                                                                             | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
 25% (127 of 500) |####################################################                                                                                                                                                            | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 26% (131 of 500) |######################################################                                                                                                                                                          | Elapsed Time: 0:00:08 ETA:   0:00:07Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
 26% (134 of 500) |#######################################################                                                                                                                                                         | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 27% (138 of 500) |#########################################################                                                                                                                                                       | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 28% (142 of 500) |###########################################################                                                                                                                                                     | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0037 secs
 29% (146 of 500) |############################################################                                                                                                                                                    | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 30% (150 of 500) |##############################################################                                                                                                                                                  | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0043 secs
 31% (155 of 500) |################################################################                                                                                                                                                | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0036 secs
Finished 'execute' in 0.0039 secs
 31% (159 of 500) |##################################################################                                                                                                                                              | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 32% (163 of 500) |###################################################################                                                                                                                                             | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 33% (167 of 500) |#####################################################################                                                                                                                                           | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 34% (170 of 500) |######################################################################                                                                                                                                          | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 34% (174 of 500) |########################################################################                                                                                                                                        | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
 35% (178 of 500) |##########################################################################                                                                                                                                      | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 36% (182 of 500) |###########################################################################                                                                                                                                     | Elapsed Time: 0:00:09 ETA:   0:00:06Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 37% (185 of 500) |############################################################################                                                                                                                                    | Elapsed Time: 0:00:10 ETA:   0:00:06Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0037 secs
 37% (189 of 500) |##############################################################################                                                                                                                                  | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 38% (193 of 500) |################################################################################                                                                                                                                | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
 39% (197 of 500) |#################################################################################                                                                                                                               | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0048 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 40% (201 of 500) |###################################################################################                                                                                                                             | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
 40% (204 of 500) |####################################################################################                                                                                                                            | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 41% (208 of 500) |######################################################################################                                                                                                                          | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0042 secs
 42% (212 of 500) |########################################################################################                                                                                                                        | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0038 secs
 43% (216 of 500) |#########################################################################################                                                                                                                       | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 43% (218 of 500) |##########################################################################################                                                                                                                      | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 44% (223 of 500) |############################################################################################                                                                                                                    | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0044 secs
 45% (227 of 500) |##############################################################################################                                                                                                                  | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0043 secs
Finished 'execute' in 0.0041 secs
 46% (231 of 500) |################################################################################################                                                                                                                | Elapsed Time: 0:00:10 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0044 secs
Finished 'execute' in 0.0041 secs
 47% (236 of 500) |##################################################################################################                                                                                                              | Elapsed Time: 0:00:11 ETA:   0:00:05Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0042 secs
 47% (238 of 500) |###################################################################################################                                                                                                             | Elapsed Time: 0:00:11 ETA:   0:00:05Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0048 secs
Finished 'execute' in 0.0039 secs
 48% (242 of 500) |####################################################################################################                                                                                                            | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 49% (246 of 500) |######################################################################################################                                                                                                          | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 50% (250 of 500) |########################################################################################################                                                                                                        | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 50% (253 of 500) |#########################################################################################################                                                                                                       | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0043 secs
 51% (257 of 500) |##########################################################################################################                                                                                                      | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 52% (261 of 500) |############################################################################################################                                                                                                    | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
 53% (265 of 500) |##############################################################################################################                                                                                                  | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 53% (268 of 500) |###############################################################################################################                                                                                                 | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
 54% (272 of 500) |#################################################################################################################                                                                                               | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0037 secs
 55% (276 of 500) |##################################################################################################################                                                                                              | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 56% (280 of 500) |####################################################################################################################                                                                                            | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0044 secs
 56% (284 of 500) |######################################################################################################################                                                                                          | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0041 secs
 57% (287 of 500) |#######################################################################################################################                                                                                         | Elapsed Time: 0:00:11 ETA:   0:00:04Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0037 secs
 58% (291 of 500) |#########################################################################################################################                                                                                       | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0036 secs
 59% (295 of 500) |##########################################################################################################################                                                                                      | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
 59% (299 of 500) |############################################################################################################################                                                                                    | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 60% (302 of 500) |#############################################################################################################################                                                                                   | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
 61% (306 of 500) |###############################################################################################################################                                                                                 | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 62% (310 of 500) |################################################################################################################################                                                                                | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 62% (314 of 500) |##################################################################################################################################                                                                              | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0043 secs
 63% (319 of 500) |####################################################################################################################################                                                                            | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 64% (323 of 500) |######################################################################################################################################                                                                          | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0049 secs
 65% (327 of 500) |########################################################################################################################################                                                                        | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
 66% (331 of 500) |#########################################################################################################################################                                                                       | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0044 secs
 66% (334 of 500) |##########################################################################################################################################                                                                      | Elapsed Time: 0:00:12 ETA:   0:00:03Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
 67% (338 of 500) |############################################################################################################################################                                                                    | Elapsed Time: 0:00:12 ETA:   0:00:02Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 68% (342 of 500) |##############################################################################################################################################                                                                  | Elapsed Time: 0:00:12 ETA:   0:00:02Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 69% (346 of 500) |###############################################################################################################################################                                                                 | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
 70% (350 of 500) |#################################################################################################################################################                                                               | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
 70% (353 of 500) |##################################################################################################################################################                                                              | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
 71% (357 of 500) |####################################################################################################################################################                                                            | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
 72% (361 of 500) |######################################################################################################################################################                                                          | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 73% (365 of 500) |#######################################################################################################################################################                                                         | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 73% (368 of 500) |#########################################################################################################################################################                                                       | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0046 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 74% (372 of 500) |##########################################################################################################################################################                                                      | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0039 secs
 75% (376 of 500) |############################################################################################################################################################                                                    | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 76% (380 of 500) |##############################################################################################################################################################                                                  | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
 76% (384 of 500) |###############################################################################################################################################################                                                 | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
 77% (387 of 500) |################################################################################################################################################################                                                | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
 78% (391 of 500) |##################################################################################################################################################################                                              | Elapsed Time: 0:00:13 ETA:   0:00:02Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 79% (395 of 500) |####################################################################################################################################################################                                            | Elapsed Time: 0:00:13 ETA:   0:00:01Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 79% (399 of 500) |#####################################################################################################################################################################                                           | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0043 secs
 80% (402 of 500) |#######################################################################################################################################################################                                         | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0045 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0037 secs
 81% (406 of 500) |########################################################################################################################################################################                                        | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 82% (410 of 500) |##########################################################################################################################################################################                                      | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0037 secs
Finished 'execute' in 0.0079 secs
Finished 'execute' in 0.0036 secs
 82% (414 of 500) |############################################################################################################################################################################                                    | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
 83% (417 of 500) |#############################################################################################################################################################################                                   | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 84% (421 of 500) |###############################################################################################################################################################################                                 | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 85% (425 of 500) |################################################################################################################################################################################                                | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0037 secs
 85% (429 of 500) |##################################################################################################################################################################################                              | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
 86% (433 of 500) |####################################################################################################################################################################################                            | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
 87% (438 of 500) |######################################################################################################################################################################################                          | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
 88% (442 of 500) |#######################################################################################################################################################################################                         | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0037 secs
 89% (446 of 500) |#########################################################################################################################################################################################                       | Elapsed Time: 0:00:14 ETA:   0:00:01Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 90% (450 of 500) |###########################################################################################################################################################################################                     | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0041 secs
 91% (455 of 500) |#############################################################################################################################################################################################                   | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0045 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
 91% (459 of 500) |##############################################################################################################################################################################################                  | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
 92% (463 of 500) |################################################################################################################################################################################################                | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0041 secs
 93% (467 of 500) |##################################################################################################################################################################################################              | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
 94% (470 of 500) |###################################################################################################################################################################################################             | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
 94% (474 of 500) |#####################################################################################################################################################################################################           | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0038 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
 95% (478 of 500) |######################################################################################################################################################################################################          | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
 96% (482 of 500) |########################################################################################################################################################################################################        | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0042 secs
Finished 'execute' in 0.0039 secs
 97% (485 of 500) |#########################################################################################################################################################################################################       | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0041 secs
 97% (489 of 500) |###########################################################################################################################################################################################################     | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0039 secs
Finished 'execute' in 0.0038 secs
 98% (493 of 500) |#############################################################################################################################################################################################################   | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0041 secs
Finished 'execute' in 0.0039 secs
 99% (495 of 500) |#############################################################################################################################################################################################################   | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0038 secs
 99% (497 of 500) |##############################################################################################################################################################################################################  | Elapsed Time: 0:00:15 ETA:   0:00:00Finished 'execute' in 0.0040 secs
Finished 'execute' in 0.0046 secs
Finished 'execute' in 0.0043 secs
100% (500 of 500) |################################################################################################################################################################################################################| Elapsed Time: 0:00:15 Time:  0:00:15
top1_acc:0.694, top5_acc:0.906
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ 
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ 
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ ls work/
deploy_model.pb  fix_info.txt  quantize_eval_model.pb
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ ls
README.md    getModels.py                   inception_v1_baseline-pyfunc.pickle         inception_v1_baseline_partition_01-netcfg.json    inception_v1_baseline_partition_01.pb           inspect_tf_model.sh  run.py    work
__pycache__  inception_v1_baseline-fpga.pb  inception_v1_baseline_partition_01-data.h5  inception_v1_baseline_partition_01-quantcfg.json  inception_v1_baseline_partition_01.pb.cleanout  models               utils.py
(vitis-ai-tensorflow) yyan7@cci-carina:/workspace/alveo/examples/tensorflow$ 
```
