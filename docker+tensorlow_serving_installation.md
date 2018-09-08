# Install Tensorflow-serving in Docker
## 1. Install Docker
In your **local** directory.
See [official document](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-repository). <br>
1). install
```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce=17.02
```
2). test
```bash
$ sudo docker run hello-world
```

3). upgrade docker
```bash
$ sudo apt-get update
$ sudo apt-get install docker-ce=$new_version  
```

4). execute without `sudo `

```bash
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
```





## 2. Build Container

In your **local** directory, for tensorflow 1.4, this [dockerfile ](https://gist.github.com/jorgemf/c791841f769bff96718fd54bbdecfd4e) is recommended. <br>
a). Build container for Tensorflow-serving cpu version with following Dockerfile: [Dockerfile.devel](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel)
```bash
$ docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
```

b). build container for Tensorflow-serving gpu version with following Dockerfile: [Dockerfile.devel-gpu](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu)

Use **nvidia-docker** instead of default docker for better support of cuda:
```bash
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
# test
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
```

~~**For tensorflow-serving v1.4 (cuda8.0+cudnn6): [See this [reference](https://gist.github.com/jorgemf/c791841f769bff96718fd54bbdecfd4e)]**~~<br>
~~Dicarded due to lack of option to set gpu option~~

**For tensorflow-serving v1.5 (cuda9.0+cudnn6): (WORKS FINE)**<br>
To enable gpu option flag (prevent consume all gpu memory), r1.5 or commit [da24ed8](https://github.com/tensorflow/serving/commit/da24ed8e07d1e4e969e1ef10c2af39ed8d9ef8c1#diff-53fe26927596e47ad1110b4e1d166723) is required. So we modify upper command into following (go to [tensorflow](https://github.com/tensorflow/tensorflow/tree/r1.5) and [tensorflow serving](https://github.com/tensorflow/serving/tree/r1.5) commit history for detailed information): <br>
```
# @modify
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04  # seems to be a bug
# @to
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# @ replace
RUN git clone --recurse-submodules https://github.com/tensorflow/serving && \
  cd serving && \
  git checkout
# @ with
RUN git clone --recurse-submodules https://github.com/tensorflow/serving && \
  cd serving && \
  git checkout r1.5 && \
  cd tensorflow && \
  git checkout r1.5 && \
  cd ../tf_models && \
  git checkout r1.5

# @modify
ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so.6
# @to
ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7

# comment last line
# bazel clean --expunge
```

```bash
$ nvidia-docker build --pull -t $USER/tensorflow-serving-devel-gpu -f Dockerfile.devel-gpu .
```

**PS:** If need rebuild for changing configuration, use `--no-cache` tag.

### Issue Discussion:
- [\#8898](https://github.com/tensorflow/tensorflow/issues/8898): `Import Error Couldn't open CUDA library libcudnn.so.5.`
- [\#318](https://github.com/tensorflow/serving/issues/318#issuecomment-283498443)`no such target @org_tensorflow//third_party/gpus/crosstool:crosstool`
- [\#10766](https://github.com/tensorflow/tensorflow/issues/10776#issuecomment-309128975): `libcuda.so.1 cannot be found`
- [\#134](https://github.com/bazelbuild/bazel/issues/134): Running bazel inside docker build causes trouble, set up a bazelrc file forcing --batch: `RUN echo "startup --batch" >>/root/.bazelrc`
- [\#418](https://github.com/bazelbuild/bazel/issues/418): need to workaround sandboxing issues: `RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" >>/root/.bazelrc`

## 3. Run docker
In your **local** directory
a). **cpu-version**
```bash
$ docker run --name=tf_serving_cpu -it $USER/tensorflow-serving-devel
$ docker start -i tf_serving_cpu
```

b). **gpu-version**
**install cuda first** (See this [document](https://www.pyimagesearch.com/2017/09/27/setting-up-ubuntu-16-04-cuda-gpu-for-deep-learning-with-python/))

```bash
$ docker run --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --name=tf_serving_gpu -it $USER/tensorflow-serving-devel-gpu
$ docker start -i tf_serving_gpu
```
**PS1: For you cannot change default choice in docker build prompt session, so if you want to make your own choice, entering the container and run tensorflow/serving manually** <br>
configure and test tensorflow-gpu serving
```bash
$ cd serving/tensorflow
$ ./configure
    Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
    Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages]
    Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n
    Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
    Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
    Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
    Do you wish to build TensorFlow with XLA JIT support? [y/N]: y
    Do you wish to build TensorFlow with GDR support? [y/N]: n
    Do you wish to build TensorFlow with VERBS support? [y/N]: n
    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
    Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]:
    Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]:
    Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
    Do you want to use clang as CUDA compiler? [y/N]: n
    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
    Do you wish to build TensorFlow with MPI support? [y/N]: n
    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
$ bazel test tensorflow_serving/...
```

If you meet error like `CUDA_ERROR_NO_DEVICE inside docker`, here is the [solution](https://github.com/tensorflow/tensorflow/issues/808): add `--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl` when docker run. <br>

### 4. Install tensorflow_model_server (optional)
In your **container** <br>
a). use **cpu** version <br>
```bash
$ cd /serving
$ bazel build -c opt tensorflow_serving/...
# execute: bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```

b). **gpu** version <br>
In some case, downloading dependency packages will fail (for example, in China...), so you could run following command in your container (first **comment it** when docker building) instead of in docker build step. <br>
```bash
$ cd /serving
$ bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --verbose_failures  --crosstool_top=@local_config_cuda//crosstool:toolchain tensorflow_serving/model_servers:tensorflow_model_server
$ ln -s /usr/local/cuda /usr/local/nvidia
$ ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1
```

### 5. Copy model to docker container
In your **local** directory
```bash
$ cd /path/to/your/path
$ docker cp ./your_model tf_serving_cpu:/serving  # tf_serving_gpu is your container name; serving is your destination directory
```

### 6. Start serving
In your **docker container** <br>
```bash
$ bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=your_model_name --model_base_path=/your/model/base/path --per_process_gpu_memory_fraction=0.05& > model_log &
```

Using config file to start serving of multiple modles([example](https://github.com/tensorflow/serving/issues/45#issuecomment-311946608)): <br>
`modelf.config` file: <br>
```
model_config_list: {
config: {
name: "model1",
base_path: "/serving/models/model1",
model_platform: "tensorflow"
},
config: {
name: "model2",
base_path: "/serving/models/model2",
model_platform: "tensorflow"
}
}
```

### 7. Get container IP address
In your **local path** <br>
```bash
$ docker network inspect bridge | grep IPv4Address
# 172.17.0.2
```

### 8. create client and run inference (OPTIONAL)
**This will be introduced in another document describing how to save model for serving and client** <br>
Use python as example (see official [client example](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py) )<br>
```python
# Create stub
host, port = "172.17.0.2:9000"
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'your_model_name'

# Initalize prediction
# Specify signature name (should be the same as specified when exporting model)
request.model_spec.signature_name = "detection_signature"
request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto({FLAGS.input_image}))

# Call the prediction server
result = stub.Predict(request, 10.0)  # 10 secs timeout
```

```bash
$ python model_client.py --server=172.17.0.2:9000 --image=test.jpg
```
Something similar to following outputs will be posted:<br>
```
outputs {
  key: “scores”
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 10
      }
    }
    float_val: 8.630897802584857e-17
    float_val: 1.219293777054986e-09
    float_val: 6.613714575998131e-10
    float_val: 1.5203355241411032e-09
    float_val: 0.9999998807907104
    float_val: 9.070973139291283e-12
    float_val: 1.5690838628401593e-09
    float_val: 9.12262028080068e-17
    float_val: 1.0587883991775016e-07
    float_val: 1.0302327879685436e-08
  }
}
```

### 9. Save docker container image
Create a new image from a container’s changes: Use `docker commit` (see document [here](https://docs.docker.com/engine/reference/commandline/commit/#options)): <br>
```bash
$ sudo docker commit -m "face anti spoof" 56ea93ab243b arkenstone/tensorflow_gpu_r1.5:version1
```

If you want to export image into `tar` file, use `docker image save` (see document [here](https://docs.docker.com/engine/reference/commandline/image_save/)): <br>
```bash
$ sudo docker image save 32ac221218c3 -o face_live_docker.tar
```

### Reference
- https://hackernoon.com/docker-compose-gpu-tensorflow-%EF%B8%8F-a0e2011d36
- https://blog.deeppoint.ai/deployment-keras-model-into-tensorflow-serving-with-gpu-e3137b8c8bfb
- https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198
- https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7
- https://medium.freecodecamp.org/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9
- https://weiminwang.blog/2017/09/12/introductory-guide-to-tensorflow-serving/
- https://github.com/movchan74/serving/blob/master/tensorflow_serving/g3doc/serving_advanced.md
- https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md



## Update

Serving with **TensorFlow 1.8** using CPU only.
**Dockerfile**

```doc
FROM ubuntu:16.04

MAINTAINER Jeremiah Harmsen <jeremiah@google.com>

# change source mirror instead of default ubuntu repo
RUN rm -rf /etc/apt/sources.list
RUN echo "deb-src http://archive.ubuntu.com/ubuntu xenial main restricted" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted " >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted multiverse universe #Added by software-properties " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted " >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted multiverse universe #Added by software-properties " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial universe " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial multiverse " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-updates multiverse " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse " >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse #Added by software-properties " >> /etc/apt/sources.list
RUN echo "deb http://archive.canonical.com/ubuntu xenial partner " >> /etc/apt/sources.list
RUN echo "deb-src http://archive.canonical.com/ubuntu xenial partner " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted " >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted multiverse universe #Added by software-properties " >> /etc/apt/sources.list
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe" >> /etc/apt/sources.list

# use python2
RUN apt-get update && apt-get install -y \
    automake \
    libtool\
    build-essential \
    curl \
    git \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    mlocate \
    pkg-config \
    python-dev \
    python-numpy \
    python-pip \
    software-properties-common \
    swig \
    zip \
    zlib1g-dev \
    libcurl3-dev \
    openjdk-8-jdk\
    openjdk-8-jre-headless \
    wget \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up grpc
RUN pip install mock grpcio

# Set up Bazel.

ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.10.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# set up tensorflow serving python api (issue #700: https://github.com/tensorflow/serving/issues/700#issuecomment-363378196)
RUN pip install tensorflow-serving-api

# install tensorflow serving
RUN git clone --recurse-submodules https://github.com/tensorflow/serving

# set argument for tensorflow install
ENV CI_BUILD_PYTHON="/usr/bin/python"
ENV PYTHON_LIB_PATH="/usr/local/lib/python2.7/dist-packages"
ENV TF_NEED_CUDA=0
ENV CC_OPT_FLAGS="-march=native"
ENV TF_VERSION=1.8.0
ENV TF_ENABLE_XLA=0
ENV TF_NEED_OPENCL=0
ENV TF_NEED_HDFS=0
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_GCP=1

RUN cd /serving && \
    bazel build -c opt --copt=-march=native --copt=-msse4.1 --copt=-msse4.2 --copt=-O3 tensorflow_serving/...

# install Model server by apt-get
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
   curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
   apt-get update && apt-get install tensorflow-model-server

# RUN cd /serving && \
#   bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

EXPOSE 9000
VOLUME tmp/tensorflow-serving-persistence

# classification
# COPY ./classification_example/mobilenet_v2/2 /serving/build/2
# CMD ["/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server", "--port=9000", "--model_name=mobilenet_v2",  "--model_base_path=/serving/build"]

# detection
COPY ./model_deploy/0 /serving/build/0
CMD ["tensorflow_model_server", "--port=8500", "--model_name=detection", "--model_base_path=/serving/build"]

```

Command to build and run docker:

```bash
$ docker build --rm --pull -t tensorflow_serving_cpu -f Dockerfile .
$ docker run -p 9000:9000 -it tensorflow_serving_cpu
```

Client for requesting server:

```python
from tensorflow.python.framework.tensor_util import make_tensor_proto
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
from grpc.beta import implementations
from scipy import misc
import numpy as np
import time

server = "localhost:8500"

def serving_predict_detection(img):
    host, port = server.split(":")
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "detection"
    request.model_spec.signature_name = "serving_default"
    frame = np.expand_dims(misc.imread(img), 0)
    t0 = time.time()
    input_tensor_proto = make_tensor_proto(values=frame,
                                           shape=frame.shape,
                                           dtype=types_pb2.DT_UINT8)
    print("Cost time for input tensor preparation: ", time.time() - t0)
    request.inputs['inputs'].CopyFrom(input_tensor_proto)
    t1 = time.time()
    content = stub.Predict(request, timeout=3)
    print("Cost time for inference: ", time.time()-t1)
    return content.outputs["detection_boxes"].float_val


if __name__ == "__main__":
    image = "test_face.jpg"
    info = serving_predict_detection(image)
    print(info)
```





Solutions for some issues:

1. **Variable directory after frozen is empty**

   **DO NOT** export model for serving from a **frozen** model for all **variables** has been converted into **constants** (See [here](https://www.tensorflow.org/mobile/prepare_models#how_do_you_get_a_model_you_can_use_on_mobile)). Use **ckpt** model file instead. See issue: [# 1988](https://github.com/tensorflow/models/issues/1988) [#2045](https://github.com/tensorflow/models/issues/2045) 

2. **SaveModel from TensorFlow object detection API** 

   See [# 1988](https://github.com/tensorflow/models/issues/1988), [Deploying Object Detection Model with TensorFlow Serving — Part 1](https://medium.com/@KailaGaurav/deploying-object-detection-model-with-tensorflow-serving-7f12ee59b036)

3. **How to use Image Tensor (ndarray type) as input in serving client**

   ```python
   img = np.expand_dims(misc.imread(image), 0)
   
   request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img, shape=img.shape, dtype='uint8'))
   ```

   See [How to properly serve an object detection model from Tensorflow Object Detection API?](https://stackoverflow.com/questions/45362726/how-to-properly-serve-an-object-detection-model-from-tensorflow-object-detection) and [Introductory Tutorial to TensorFlow Serving](https://weiminwang.blog/2017/09/12/introductory-guide-to-tensorflow-serving/)

4. **TensorFlow Serving in docker bad performance: 10x slower than native execution**

   a). Optimization parameters must be used during bazel build tensorflow serving tools.

   ```bash
   bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 tensorflow_serving/...
   
   bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
   ```

   See [# 401](https://github.com/tensorflow/serving/issues/401); [# 456](https://github.com/tensorflow/serving/issues/456); 

   b). `    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(serialized, shape=[1]))` takes too much time when preparing input tensor for serving request (this step almost costs 1.8 s). 

   **Solution:**  replace `tf.contrib.util.make_tensor_proto()` with following code with TensorFlow core function:

   ```python
   from tensorflow.core.framework import tensor_shape_pb2, tensor_pb2, types_pb2
   
   dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
   tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
   tensor_proto = tensor_pb2.TensorProto(
       dtype=types_pb2.DT_STRING,
       tensor_shape=tensor_shape_proto,
       string_val=[data])
   ```

   Or just use `tensorflow.python.framework.tensor_util.make_tensor_proto` to make tensor protobuf directly (See `make_tensor_proto` [source code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_util.py)). 

   ```python
   from tensorflow.core.framework import types_pb2
   from tensorflow.python.framework.tensor_util import make_tensor_proto
   
   tensor_proto = make_tensor_proto(values=[data], shape=[1],dtype=type2_pb2.DT_STRING)
   ```

   **Reference Post: ** [TensorFlow Serving client: Making it slimmer and faster!](https://towardsdatascience.com/tensorflow-serving-client-make-it-slimmer-and-faster-b3e5f71208fb)

5.  **How to use TensorFlow object detection API trained model with docker**

   See post series: [Operationalizing TensorFlow Object Detection on Azure ](https://medium.com/@sozercan/tensorflow-object-detection-on-azure-part-2-using-kubernetes-to-run-distributed-tensorflow-ced5b9a6184a); and [Deploying Object Detection Model with TensorFlow Serving](https://medium.com/@KailaGaurav/deploying-object-detection-model-with-tensorflow-serving-part-3-6a3d59c1e7c0)

   

