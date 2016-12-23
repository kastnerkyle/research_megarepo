# Note – this is not a bash script (some of the steps require reboot)
# I named it .sh just so Github does correct syntax highlighting.
#
# This is also available as an AMI in us-east-1 (virginia): ami-cf5028a5
#
# The CUDA part is mostly based on this excellent blog post:
# http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/

# Install various packages
sudo apt-get update
sudo apt-get upgrade -y # choose “install package maintainers version”
sudo apt-get install -y build-essential python-pip python-dev git python-numpy swig python-dev default-jdk zip zlib1g-dev

# Blacklist Noveau which has some kind of conflict with the nvidia driver
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot # Reboot (annoying you have to do this in 2015!)

# Some other annoying thing we have to do
sudo apt-get install -y linux-image-extra-virtual
sudo reboot # Not sure why this is needed

# Install latest Linux headers
sudo apt-get install -y linux-source linux-headers-`uname -r` 

# Install CUDA 7.0 (note – don't use any other version)
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run
chmod +x cuda_7.0.28_linux.run
./cuda_7.0.28_linux.run -extract=`pwd`/nvidia_installers
cd nvidia_installers
sudo ./NVIDIA-Linux-x86_64-346.46.run 
sudo modprobe nvidia
sudo ./cuda-linux64-rel-7.0.28-19326674.run 
cd

# Install CUDNN 6.5 (note – don't use any other version)
# YOU NEED TO SCP THIS ONE FROM SOMEWHERE ELSE – it's not available online.
# You need to register and get approved to get a download link. Very annoying.
tar -xzf cudnn-6.5-linux-x64-v2.tgz 
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/

# At this point the root mount is getting a bit full
# I had a lot of issues where the disk would fill up and then Bazel would end up in this weird state complaining about random things
# Make sure you don't run out of disk space when building Tensorflow!
sudo mkdir /mnt/tmp
sudo chmod 777 /mnt/tmp
sudo rm -rf /tmp
sudo ln -s /mnt/tmp /tmp
# Note that /mnt is not saved when building an AMI, so don't put anything crucial on it

# Install Bazel
cd /mnt/tmp
git clone https://github.com/bazelbuild/bazel.git
cd bazel
git checkout tags/0.1.0
./compile.sh
sudo cp output/bazel /usr/bin

# Install TensorFlow
cd /mnt/tmp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
# Patch to support older K520 devices on AWS
# wget "https://gist.githubusercontent.com/infojunkie/cb6d1a4e8bf674c6e38e/raw/5e01e5b2b1f7afd3def83810f8373fbcf6e47e02/cuda_30.patch"
# git apply cuda_30.patch
# According to https://github.com/tensorflow/tensorflow/issues/25#issuecomment-156234658 this patch is no longer needed
# Instead, you need to run ./configure like below (not tested yet)
TF_UNOFFICIAL_SETTING=1 ./configure
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

# Build Python package
# Note: you have to specify --config=cuda here - this is not mentioned in the official docs
# https://github.com/tensorflow/tensorflow/issues/25#issuecomment-156173717
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# Test it!
cd tensorflow/models/image/cifar10/
python cifar10_multi_gpu_train.py 

# On a g2.2xlarge: step 100, loss = 4.50 (325.2 examples/sec; 0.394 sec/batch)
# On a g2.8xlarge: step 100, loss = 4.49 (337.9 examples/sec; 0.379 sec/batch)
# doesn't seem like it is able to use the 4 GPU cards unfortunately :(
