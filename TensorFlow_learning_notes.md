## TensorFlow Learning Notes
### 1. Tensorflow basic concepts
1. [TensorFlow best practice series](https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3#.am2c248ex)

## Deep Learning Resources
1. [all fields reading list](https://github.com/handong1587/handong1587.github.io/tree/master/_posts/deep_learning)
2. [AI on embedded device](https://github.com/ysh329/awesome-embedded-ai)

## Install TensorFlow
#### 1. Install bazel
```bash
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
```

#### 2. Install docker & nvidia-docker
```bash
# docker
 sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce

# nvidia docker
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
```

#### 3. Install TensorRT
```bash
# download from TensorRT deb file this site https://developer.nvidia.com/nvidia-tensorrt-download
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
sudo apt-get update
sudo apt-get install tensorrt
sudo apt-get install python3-libnvinfer-doc
sudo apt-get install uff-converter-tf
```
#### 4. Install eigen
```bash
# goto git release to download https://github.com/eigenteam/eigen-git-mirror/releases/tag/3.3.4/
unzip eigen-git-mirror-3.3.4.zip
cd eigen-git-mirror
mkdir build && cd build
cmake ..
make
sudo make install
```
### 5. Install cmake 11
```bash
# Download image from https://cmake.org/download/
tar xcvf cmake-3.11.1-Linux-x86_64.tar.gz
cd cmake-3.11.1-Linux-x86_64
sudo apt-get purge cmake
sudo cp -r bin /usr/
sudo cp -r share /usr/
sudo cp -r doc /usr/share/
sudo cp -r man /usr/share/
```
#### 6. Install tensorflow
```bash
sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
sudo pip3 install six numpy wheel
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.8.0-py3-none-any.whl

bazel build //tensorflow:libtensorflow_cc.so
sudo mkdir /usr/local/include/tf
sudo mkdir /usr/local/include/tf/tensorflow
sudo cp -r bazel-genfiles/ /usr/local/include/tf
sudo cp -r tensorflow/cc /usr/local/include/tf/tensorflow
sudo cp -r tensorflow/core /usr/local/include/tf/tensorflow
sudo cp -r third_party /usr/local/include/tf
sudo cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
sudo cp bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib
```
