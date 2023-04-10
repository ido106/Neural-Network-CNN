# Neural Networks and CNNs
## Introduction
In this exercise we will touch the world of [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning).  
Machine learning is a branch of [Artificial Intelligence (AI)](https://en.wikipedia.org/wiki/Artificial_intelligence) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.  

In the first part, we will implement and train a  [Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)  (multi-layer perceptron) for handwriting recognition ([MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)), using  [numpy](https://numpy.org/)  only.  

In the second part, we will load and preprocess datasets using  [PyTorch](https://pytorch.org/).  
Then, we will implement and train a Neural Network (multi-layer perceptron) for handwriting recognition (MNIST dataset), using PyTorch.  
And finally we will implement and train a Convolutional Neural Network ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) on MNIST.

### Neural Networks
Neural networks, also known as artificial neural networks (ANNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another. Each neuron has a specific job, like recognizing patterns or making decisions. They communicate with each other and share information to come up with a solution.  

Before diving into the exercise, I recommend reading about neural networks [here](https://www.ibm.com/topics/neural-networks).  
<p align="center">
  <img 
    width="600"
    src="https://1.cms.s81c.com/sites/default/files/2021-04-15/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png"
  >
</p>

### Convolutional Neural Network
Convolutional Neural Networks (CNNs) are a type of artificial neural network that are specifically designed for analyzing images and other two-dimensional data.  
The idea of CNNs is breaking the input down into smaller, simpler features, which are represented by filters. These filters are applied to different regions of the input to extract the relevant information.  
By breaking down the image into smaller pieces and analyzing each piece, the CNN can recognize patterns and features that would be difficult to see otherwise. This is the reason why CNNs are often used for tasks like image recognition.  

Another important advantage of using CNNs is **computational**.  
CNNs often include **pooling layers** whose role is to reduce the resolution of the feature map but preserve the important features of the map.  
With the help of the pooling layers, we can significantly reduce the amount of features from a few millions to a few hundreds, which significantly reduces the difficulty and time to learn deep networks.  
The common types of pooling are [*max* and *average*](https://androidkt.com/explain-pooling-layers-max-pooling-average-pooling-global-average-pooling-and-global-max-pooling/).

Before diving into the exercise, I recommend reading about CNN [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).  
<p align="center">
  <img 
    width="800"
    src="https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1200%2C564&ssl=1"
  >
</p>

## Running Instructions
In the ML tasks, and in particular in this task, we will use [Google Colab](https://colab.research.google.com/) to run the code in an iterative and convenient way.  
To preview the notebook, you can click on `Neural_Network_and_CNN.ipynb` in this repository.  
In the notebook you can find a detailed explanation about the algorithms and the code step by step.  
To run the code, first clone the repository to your computer with `git clone https://github.com/ido106/Neural-Network-CNN.git`, then drag the notebook (`Neural_Network_and_CNN.ipynb`) and the MNIST dataset (`mnist_test.pth`) to your [Google Drive](https://www.google.com/drive/).  

**Enjoy !**
