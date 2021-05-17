# convnet

__Task:__ Implement a convolutional neural network (CNN) for image classification.

Image classification using a CNN is probably the most famous application of deep learning -- image classification was a relatively difficult machine learning problem in the past, but CNNs have dramatically improved the state of the art on the task over the past 5-10 years.

There are many blog posts and tutorials that give background on CNNs applied to image classification:
 - [Fastai video](https://course.fast.ai/videos/?lesson=1)
 - [Simple Introduction to CNNs](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac)
 - [Keras tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
 - [Pytorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### How to Run Workload

- __Step1:__ Build [hb-pytorch](https://github.com/cornell-brg/hb-pytorch/tree/convnet) [Use Branch convnet].

       git clone -b convnet git@github.com:cornell-brg/hb-pytorch.git
       cd hb-pytorch
       python3.6 -m venv ./venv_pytorch
       source venv_pytorch/bin/activate
       pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing sklearn tqdm pytest ninja hypothesis thop pandas tabulate
       git submodule update --init --recursive
       source setup_cosim_build_env.sh [provide bladerunner dir in the setup_cosim_build_env.sh + change the machine if necessary]
       python setup.py install

- __Step2:__ Download CIFAR dataset with:

       sh prep.sh

- __Step3:__ Run training kernels profiling:

       python run-all-hb-training.py


- __Step4:__ Run inference kernels profiling:

       python run-all-hb-inference.py


- __Step5:__ Run Conv2d kernels profiling:

       python run-all-hb-conv2d.py

### How to Parse Results

Once all training kernels, inference kernels, and conv2d kernels are profiled, run following instruction to generate readable profiling results for each kernel:
       
    python parse_results.py --bladerunner-dir [BladeRunner Dir]
