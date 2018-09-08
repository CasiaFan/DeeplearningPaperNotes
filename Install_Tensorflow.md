## 1. Install TensorFlow (CPU version) under CentOS 6.9

### Plan A. Install Tensorflow using Anaconda

1. Download Anaconda from this [site](https://www.anaconda.com/download/#linux)

2. Run `bash Anaconda-latest-Linux-x86_64.sh`

3. Use `conda list` to check install

4. Install Tensorflow with conda: `conda install tensorflow==1.7.0` (May suffer connection error and just retry)

   (For more info, see [conda doc](https://conda.io/docs/installation.html))

### Plan B. Install Tensorflow by pip

For TensorFlow need Python version 2.7.x or 3.x, but CentOS 6.9's default python version is 2.6. So First we need to upgrade python and here we upgrade to 3.5 version.

1. **Install Python 3.5.2 and pip**
```bash
# install prerequisites
yum groupinstall -y Development tools
yum install -y zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel
# install python3.5.2
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tar.xz
xz -d Python-3.5.2.tar.xz
tar xvf Python-3.5.2.tar
cd Python-3.5.2
./configure --prefix=/usr/local/python3.5
make && make install
ln -s /usr/local/python3.5/bin/python3.5 /usr/bin/python3.5
# add python bin into env path
export PATH=$PATH:/usr/local/python3.5/bin
# install python-pip
wget https://bootstrap.pypa.io/get-pip.py
python3.5 get-pip.py
```

2. **Upgrade gcc to 4.9.2**

This step is **required** for CentOS 6.9's default gcc is 4.4.7 which is not compatible with TensorFlow requirement. 

```bash
# download gcc 4.9.2
yum install libmpc-devel mpfr-devel gmp-devel

cd /usr/src/
curl http://ftp.gnu.org/gnu/gcc/gcc-4.9.2/gcc-4.9.2.tar.bz2 -O
tar xvfj gcc-4.9.2.tar.bz2
cd gcc-4.9.2
./configure --disable-multilib --enable-languages=c,c++
make -j `grep processor /proc/cpuinfo | wc -l`
make install
```



3. **Update GLIBC and CXXABI**

If version is too low, such error will be reported when import tensorflow. (issure[#5191](https://github.com/ContinuumIO/anaconda-issues/issues/5191))

`ImportError: /usr/lib64/libstdc++.so.6: version CXXABI_1.3.7 not found ` and `ImportError: /lib64/libc.so.6: version GLIBC_2.16not found`

**Solution**:  

```bash
# update glibc 
wget http://ftp.gnu.org/pub/gnu/glibc/glibc-2.17.tar.gz

tar -zxvf glibc-2.17.tar.gz

cd glibc-2.17

mkdir build

cd build

../configure --prefix=/usr --disable-profile--enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin 

make && make install

# create dynamic link repository to new gcc rather than default one
find / -name "libstdc++.so.*"
#output
#/usr/share/gdb/auto-load/usr/lib64/libstdc++.so.6.0.13-gdb.pyc
#/usr/share/gdb/auto-load/usr/lib64/libstdc++.so.6.0.13-gdb.pyo
#/usr/share/gdb/auto-load/usr/lib64/libstdc++.so.6.0.13-gdb.py
#/usr/share/gdb/auto-load/usr/lib/libstdc++.so.6.0.13-gdb.pyc
#/usr/share/gdb/auto-load/usr/lib/libstdc++.so.6.0.13-gdb.pyo
#/usr/share/gdb/auto-load/usr/lib/libstdc++.so.6.0.13-gdb.py
#/usr/lib64/libstdc++.so.6.0.20
#/usr/lib64/libstdc++.so.6.0.13
#/usr/lib64/libstdc++.so.6
#/usr/lib64/libstdc++.so.6.bak

mv libstdc++.so.6 libstdc++.so.6.bak
ln -s usr/lib64/libstdc++.so.6.0.20 libstdc++.so.6
# check 
strings /usr/lib64/libstdc++.so.6 | grep 'CXXABI'
#CXXABI_1.3
#CXXABI_1.3.1
#CXXABI_1.3.2
#CXXABI_1.3.3
#CXXABI_1.3.4
#CXXABI_1.3.5
#CXXABI_1.3.6
#CXXABI_1.3.7
#CXXABI_1.3.8
```

4. **Install TensorFlow**

```bash
pip3 install tensorflow
# check install 
python -c "import tensorflow as tf; print(tf.__version__)"
# 1.7.0
```

5. **Install libSM and libXext in case opencv report error under centos 7.5**

```bash
yum install libSM libXext
```

   

**Reference**: 

1. [centos6.5安装tensorflow](https://blog.csdn.net/tyutpanda/article/details/79109855)
2. [Building TensorFlow for CentOS 6](https://blog.abysm.org/2016/06/building-tensorflow-centos-6/)
3. [centos6.5升级gcc到4.9](http://blog.techbeta.me/2015/10/linux-centos6-5-upgrade-gcc/)
4. [“'CXXABI_1.3.8' not found” in tensorflow-gpu - install from source](https://stackoverflow.com/questions/39844772/cxxabi-1-3-8-not-found-in-tensorflow-gpu-install-from-source)
5. [Ubuntu../libstdc++.so.6: version `CXXABI_1.3.9' not found解决方法](https://blog.csdn.net/gaoprincess/article/details/78450587)
6. https://github.com/ContinuumIO/anaconda-issues/issues/5191
7. https://gist.github.com/WisdomFusion/a60285c8f7cd9faf06ef8c9244133219



## 2. Install TensorFlow from source under Ubuntu16.04

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
sudo pip3 install /tmp/tensorflow_pkg/tensorflow-1.8.0-py3-none-any.whl

# bazel build c++ api with opts: https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions

# if not build with cpu optimization opts, even slower than python

# use monolithic for conflict with opencv: https://github.com/tensorflow/tensorflow/issues/14267
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=monolithic //tensorflow:libtensorflow_cc.so
sudo mkdir /usr/local/include/tf
sudo mkdir /usr/local/include/tf/tensorflow
sudo cp -r bazel-genfiles/ /usr/local/include/tf
sudo cp -r tensorflow/cc /usr/local/include/tf/tensorflow
sudo cp -r tensorflow/core /usr/local/include/tf/tensorflow
sudo cp -r third_party /usr/local/include/tf
sudo cp bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib
sudo cp bazel-bin/tensorflow/libtensorflow_framework.so /usr/local/lib
```



```
cd /root/person_detect

source venv/bin/activate

./api_op restart

cd ../person_status_check_opencv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/person_status_check_opencv/hikvision_service_class/lib/HCNetSDKCom

./api_op restart
```





