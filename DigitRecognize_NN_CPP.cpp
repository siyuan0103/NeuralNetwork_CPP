// DigitRecognize_NN_CPP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "loadmnist.h"
#include "neuralnetwork.h"
#include <iterator>
#include <time.h>

int main()
{
	LoadMNIST loader;
	MatrixXf train_images;
	VectorXf train_labels;
	loader.get_train(train_images, train_labels, 600);
	cout << "number of training samples: "<< train_images.rows() << endl;

	MatrixXf test_images;
	VectorXf test_labels;
	loader.get_test(test_images, test_labels, 1000);
	//MatrixXf test1 = test_images.row(0);
	//test1.resize(28, 28);
	//cout << test1.transpose() << endl;
	
	NeuralNetwork MyNN(784, 10, 1, 300, 0.1);
	MyNN.CheckGradient();
	
	MyNN.Train(train_images, train_labels, 50, 0.3);
	MyNN.Test(test_images, test_labels);
	
    return 0;
}