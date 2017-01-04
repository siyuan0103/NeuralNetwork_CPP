# NeuralNetwork_CPP
a multi-layer neural network with backpropagation for handwritten digits recognization, written in C++

##Overview
---
In this project a multi-layer neural network is written in C++ for hand written digits recognization. Just the same with my earlier [Python version](https://github.com/siyuan0103/NN-Handwritten_Digits-Python), the digits database comes from [MNIST, Yann LeCun](http://yann.lecun.com/exdb/mnist/), and the backpropagation and gradient descent methods are used to train the neural network, so the two versions share the similar result. 

Different from the fixed 2-layer (1 hidden layer and 1 output layer) neural network in Python version, this time the number of hidden layers can be set to any positive integer (still default to be 1). 

The Eigen library provides matrix arithmetic APIs for this project, but its efficiency is much less than the Numpy library in Python. Its training speed  is only about one percent of the Python version.

The project includes three parts:

* `loadmnist.h` and `.cpp` contain the class `LoadMNIST` for the loading of MNIST database.
* `neuralnetwork.h` and `.cpp` contain the class `NeuralNetwork`.
* `DigitRecgnize_NN_CPP.cpp` contains the main(), give an example how to use the `LoadMNIST` and `NeuralNetwork` to train and test the neural network for digits recognition.

##How To Use
---
To use this code you need to install a C++ library [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page), clone all the source files and download the [MNIST database files](http://yann.lecun.com/exdb/mnist/) to the folder `MNIST`. This project was successfully complied in Microsoft Visual C++ 2015.

##Program Description
---
[The Python version](https://github.com/siyuan0103/NN-Handwritten_Digits-Python/blob/master/Documentation.md) for reference. A significant change in C++ version is that, data are mainly passed through `data members` between the `member functions`, since the C++ function can only return one variable.

##Licence
---
It's open source code using MIT licence.
