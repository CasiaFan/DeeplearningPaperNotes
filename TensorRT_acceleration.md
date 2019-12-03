## A. TensorRT Integration Speeds Up TensorFlow Inference

([Link](https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/)) ([Code Sample][https://github.com/NVIDIA-Jetson/tf_trt_models]) ([tf research tensorrt example](https://github.com/tensorflow/models/blob/master/research/tensorrt/tensorrt.py))

Inferrence Integration with TensorRT (**TF contrib tensorrt module**)

![TensorFlow RT Flow](https://devblogs.nvidia.com/wp-content/uploads/2018/03/TensorRT4_Graphics-modified-workflow-1-625x357.png)

Integration with **UINT8** for inference: NEED calibration first 

![TensorFlow calibration](https://devblogs.nvidia.com/wp-content/uploads/2018/03/calibrate_flow-625x246.png)



```python
import tensorflow as tf

trt = tf.contrib.tensorrt

trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph_def,
                outputs=output_node_name,
                max_batch_size=batch_size,
                max_workspace_size_bytes=workspace_size,
                precision_mode=precision)
```



**Graph Optimized (Left: initial, Right: optimized)**

![optimized graph](https://devblogs.nvidia.com/wp-content/uploads/2018/03/optimization_result_fig2.png)



**Effect:**

![TensorRT ResNet-50 Performance](https://devblogs.nvidia.com/wp-content/uploads/2018/06/perf_correct-1024x576.png)

## B. Install TensorRT

**Tar Install mode is Recommended** ([Tutorial Link](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar)) ([Download Path](https://developer.nvidia.com/tensorrt))

For **Ubuntu 16.04**, recommend version is **cuda 9.0 + tensorrt 4**. For Ubuntu 18.04, recommend version is **cuda 10.0 + tensorrt 5.0**, *BUT* this is not compatible for TensorFlow v1.11 now.

```shell
tar xzvf TensorRT-5.x.x.x.Ubuntu-1x.04.x.x86_64-gnu.cuda-x.x.cudnn7.3.tar.gz

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<eg:TensorRT-5.x.x.x/lib>

cd TensorRT-5.x.x.x/python
sudo pip3 install tensorrt-5.x.x.x-py2.py3-none-any.whl
cd TensorRT-5.x.x.x/uff
sudo pip3 install uff-0.5.1-py2.py3-none-any.whl
cd TensorRT-5.x.x.x/graphsurgeon
sudo pip3 install graphsurgeon-0.2.2-py2.py3-none-any.whl

```



## C. TensorRT SDK turtorial 

[Link](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_python)

UFF optimized model execution example:

```python
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import tensorflow as tf
import numpy as np
import pycuda.driver as cuda
import pycuda.autoint

G_LOGGER = trt.infer.ConsleLogger(trt.infer.LogSeverity.ERROR)

uff_model = uff.from_tensorflow_frozen_model("frozen_inference_graph.pb",
                                             output_filename="model.uff")
parser = uffparser.create_uff_parser()
parser.register_input("image_tensor", (3, 480, 480), 0)
parser.register_output("num_detections,detection_boxes,detection_scores,detection_classes")

engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                     uff_model,
                                     parser,
                                     1,
                                     1<<20)
parser.destroy()

# write engine
trt.utils.write_engine_to_file("model.engine", engine.serialize())
# load engine
engine = trt.utils.load_engine(G_LOGGER, "model.engine")

runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()

# allocate the size of input and expected output * batch size
MNIST_DATASETS = tf.contrib.learn.datasets.load_dataset("mnist")
img, label = MNIST_DATASETS.test.next_batch(1)
img = img[0]
img = img.astype(np.float32)
label = label[0]

output = np.empty(10, dtype=np.float32)
d_input = cuda.mem_alloc(1 * img.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)
bindings = [int(d_input), int(d_output)]
# create a cuda stream to run inference
stream = cuda.Stream()

# transfer data to GPU, run inference, transfer the results to the host
cuda.memcpy_htod_async(d_input, img, stream)
# execute the model
context.enqueue(1, bindings, stream.handle, None)
# transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
# syncronize threads
stream.synchronize()

# clean up
context.cleanup()
engine.destroy()
runtime.destroy()
```



### Important Errors

1. **AssertionError: UFF parsing failed on line 255 in statement assert(parser.parse(stream, network, model_datatype))**

   Happens when trying to use `trt.utils.uff_to_trt_engine` to create tensorrt engine. The reason is due to cycle in graph caused by `tf.map_fn` or `tf.while_loop`. ([post](https://devtalk.nvidia.com/default/topic/1037144/tensorrt/uffparser-error-graph-error-cycle-graph-detected-when-running-mobilenetv2/))