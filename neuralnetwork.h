#include "stdafx.h"
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class NeuralNetwork {
protected:
	int n_inputs;
	int n_outputs;
	int n_HLayers;
	int n_NeuronsPerHLayer;
	int n_samples;
	float rand_range;

	vector<MatrixXf> layers;  // vector of weight matrix each layer, initialized when create a instance of the class
	// following variables should be initialized with a new set of inputs;
	MatrixXf inputs;  // inputs with bias, n_samples*(n_inputs+1);
	MatrixXf labels;	// y in Matrix, n_samples*n_outputs;
	MatrixXf output_z; // sum of inputs for neurons in output layer, n_samples*n_outputs;
	MatrixXf output_a; // activitions of neurons in output layer, n_samples*n_outputs;
	vector<MatrixXf> hlayers_z; // sum of inputs for neurons in hidden layers, n_HLayers*n_samples*n_NeuronsPerHLayer
	vector<MatrixXf> hlayers_a; // activitions of neurons in hidden layers, n_HLayers*n_samples*(n_NeuronsPerHLayer+1)
	vector<MatrixXf> theta_grads;	// gradient of weights, inversed index with layers;
	vector<MatrixXf> theta_grads_num;	// gradient of weights using numerial methode, inversed index with layers;
	
	// forward propagation
	void Forward();
	// Backpropagation
	void Back(float lambda);
	//create layers and random initialize weights
	void CreateLayers();
	// cost function
	float CostFunction(float lambda);
	// gradient descent methode to minimize cost function
	void GradientDescent(float alpha);
	// sigmoid function
	MatrixXf Sigmoid(const MatrixXf &z);
	// add the bias items
	MatrixXf AddBias(const MatrixXf &a);
	// get inputs and n_samples;
	void GetInput(const MatrixXf &X);
	// convert label vector into Matrix
	void GetLabel(const VectorXf &y);
	// compute numerial gradient with +- epsilon
	void NumGradient(float lambda, float epsilon);

public:
	NeuralNetwork(int number_of_inputs = 784, int number_of_outputs = 10, int number_of_hidden_layers = 1,
		int number_of_neurons_per_hidden_layer = 300, float rand_range_of_weights = 0.1);

	vector<MatrixXf> Get_weights();

	void Init_weights(vector<MatrixXf> weights);

	void CheckGradient(float lambda = 0, float epsilon = 0.00001);

	VectorXf Predict(const MatrixXf &X);
	// Train the NeuralNetwork. alpha: learn rate; lambda: coefficient for regularization
	void Train(const MatrixXf &X, const VectorXf &y, int max_iter =50, float alpha =0.3, float lambda = 1.0);
	void Test(const MatrixXf &X, const VectorXf &y);
};
#endif
