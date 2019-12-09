# Experiments on Fashion Mnist



## Fashion MNIST
*'''` Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend `Fashion-MNIST` to serve as a direct **drop-in replacement** for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.'''*



Walkthrough of the trainings is in **[Fashion_Mnist_Experiments](/Notebooks/Fashion_Mnist_Experiments.ipynb)** notebook.

### Docker

Build Command:

> docker build --rm -f Dockerfile -t fashionimage.

To run on gpu:

Docker version **>19.03** is required.

Install nvidia-container-toolkit. Most cloud vendors(AWS,GCE etc.) has it alreay installed on their GPU servers.
```sh
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```


Then simply;

> docker run --gpus all -it --rm -p 8888:8888 fashionimage

If --gpus all flags are disabled, CPU will be used.
> docker run  -it --rm -p 8888:8888 fashionimage

Doesn't require jupyter tokens, jupyter-notebook can be accessed by
ip:8888 after running the containers




### Trained Models

- Mini VGG like Architecture

- WideResNet(16-4)

- A Basic CNN Architecture - With Dropout and Batch Normalization
- ShuffleNet
- ResNet18- with 0.25 dropout
- ResNet34 - with 0.25 dropout
- MobileNetV2
- ShakeShakeResNet34 - Extra Shake Shake Regularizer Added ResNet Architecture

### Training Methods:
- Cyclic Learning Rates: For faster convergence

- Stochastic Weight Averaging : For increased test performance.
- ![SWA](https://raw.githubusercontent.com/celbirlik/Fashion-Mnist-Experiments/master/Images/swa.png)
- Augmentation : With minimal amount since computation for searching augmentation parameters was limited.  {'horizontal_flip':0.1,'zoom_range':0.05}
- Ensembles were not used since variance between model predictions were low.


## Test Results

Results are with SWA,Cyclic learning rate and minimal Augmentation. Other combinations of these and other parameters such as different epoch numbers have not been tested due to long training times.
From the results we see that deeper models overfit the small images easily, resulting in worse results. models with strong regularizers(shake_Shake34) and shallower versions(resnet18 vs resnet34) perform better.




|   Model	| Accuracy  	|   	
|---:	|---	|
|  ShakeShake34 	| 95.71  	|   	
|  WideResnet16-4 	|  95.33 	|
|   Resnet18	|   94.74	|
|   ResNet34	| 94.71 	|
|   MiniVGG	|   94.59	|
|   BasicCNN	|   94.21	|
|   MobileNetV2	|   93.2	|
|   ShuffleNet	|   92.9	|



## Confusion Matrix,P-R and F-1 scores of ShakeShake34

As can be seen, visually similar classes like shirt and top gets confused most often. These classes often times are harder to distinguish with human eyes too.

![PR and F1 Scores of ShakeShake](/Images/pr-f1.png)
![Confusion matrix](/Images/ShakeShake34.png)

Some of the wrongly classified examples are below. 

![Incorrectly classified images](/Images/incorrect.png)







## Future Work

Instead of training a single multi classifier, 10 one vs all classifiers can be trained on each class. An Ensemble of classifiers directly focused on each class may improve accuracy further.