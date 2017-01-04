#include "stdafx.h"
#include "neuralnetwork.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <iterator>


using namespace std;
using Eigen::ArrayXXf;

// Constructor with customized parameters
NeuralNetwork::NeuralNetwork(int number_of_inputs, int number_of_outputs, int number_of_hidden_layers, 
	int number_of_neurons_per_hidden_layer, float rand_range_of_weights) :
	n_inputs(number_of_inputs), n_outputs(number_of_outputs),
	n_HLayers(number_of_hidden_layers), n_NeuronsPerHLayer(number_of_neurons_per_hidden_layer), 
	rand_range(rand_range_of_weights) {
	CreateLayers();
}

MatrixXf NeuralNetwork::AddBias(const MatrixXf &a) {
	MatrixXf temp(a);
	VectorXf bias = VectorXf::Ones(temp.rows());
	temp.conservativeResize(Eigen::NoChange, temp.cols() + 1);
	temp.col(temp.cols() - 1) = bias;
	return temp;
}

void NeuralNetwork::Back(float lambda) {
	// allocate Delta for each layer
	clock_t t = clock();	//test
	vector<MatrixXf> Deltas;
	for (int i = 0; i <layers.size(); i++) {
		Deltas.push_back(MatrixXf::Zero(layers[i].rows(), layers[i].cols()));
	}
	vector<VectorXf> deltas;
	for (int i = 0; i < hlayers_a.size(); i++) {
		deltas.push_back(VectorXf::Zero(n_NeuronsPerHLayer));
	}
	deltas.push_back(VectorXf::Zero(n_outputs));
	// storage the weight matrix of each layer without the last row (for bias)
	MatrixXf temp;
	// storage Sigmoid(z)
	ArrayXXf tempvec;
	// iterate all the samples;
	for (int i = 0; i < n_samples; i++) {
		//output layer
		deltas[n_HLayers] = output_a.row(i) - labels.row(i);
		Deltas[n_HLayers] += (deltas[n_HLayers] * hlayers_a[n_HLayers - 1].row(i)).transpose();
		// hidden layers without first layer;
		for (int j = n_HLayers - 1; j > 0; j--) {
			temp = layers[j + 1].topRows(layers[j + 1].rows() - 1);
			tempvec = hlayers_a[j].row(i).leftCols(hlayers_a[j].cols()-1).transpose().array();
			deltas[j] = ((temp * deltas[j + 1]).array() * tempvec * (1 - tempvec)).matrix();
			Deltas[j] += (deltas[j] * hlayers_a[j - 1].row(i)).transpose();
		}
		// first hidden layer
		temp = layers[1].topRows(layers[1].rows() - 1);
		tempvec = hlayers_a[0].row(i).leftCols(hlayers_a[0].cols() - 1).transpose().array();
		deltas[0] = ((temp * deltas[1]).array() * tempvec * (1 - tempvec)).matrix();
		Deltas[0] += (deltas[0] * inputs.row(i)).transpose();
	}
	// calculate theta_grad
	theta_grads.clear();
	MatrixXf theta_grad;
	for (int i = 0; i < layers.size(); i++) {
		MatrixXf reg_item = lambda / (float)n_samples * layers[i];
		reg_item.row(reg_item.rows() - 1) = VectorXf::Zero(reg_item.cols());
		theta_grad = 1.0 / (float)n_samples * Deltas[i] + reg_item;
		theta_grads.push_back(theta_grad);
	}
	cout << "Backpropagation costs " << (float)(clock() - t) / CLOCKS_PER_SEC << " s" << endl;   //test
}
// create a smaller NeuralNetwork and check if the backpropagation works well
void NeuralNetwork::CheckGradient(float lambda, float epsilon) {
	cout << "Check gradient" << endl;
	// backup the size of NeuralNetwork
	int n_inputs_backup = n_inputs;
	int n_HLayers_backup = n_HLayers;
	int n_NeuronsPerHLayer_backup = n_NeuronsPerHLayer;
	int n_outputs_backup = n_outputs;
	// create a smaller NeuralNetwork
	n_inputs = 3;
	n_HLayers = 1;
	n_NeuronsPerHLayer = 5;
	n_outputs = 3;
	CreateLayers();
	// random initialize X and y
	int m = 5;	// number of samples
	srand((unsigned int)time(NULL));
	MatrixXf X = (ArrayXXf::Random(m, n_inputs) * 128 + 128).floor().matrix();
	VectorXf y(m*n_outputs);
	for (int i = 0; i < m*n_outputs; i++) {
		y(i) = i % n_outputs;
	}
	GetInput(X);
	GetLabel(y);
	// forward propagation
	Forward();
	// implement backpropagation and numerial gradient 
	Back(lambda);
	NumGradient(lambda, epsilon);
	// compare the difference
	float diff(0), count(0);
	cout << "Backpropagation:" << '\t' << "Numerial gradient" << endl;
	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			cout << *(theta_grads[i].data() + j) << '\t' << *(theta_grads_num[i].data() + j) << endl;
		}
		count += layers[i].size();
		diff += layers[i].size()*(theta_grads[i] - theta_grads_num[i]).norm()/
			(theta_grads[i] + theta_grads_num[i]).norm();
	}
	diff /= count;
	cout << "The normalized difference between them is " << diff << endl;
	// reset the size of NeuralNetwork
	n_inputs = n_inputs_backup;
	n_HLayers = n_HLayers_backup;
	n_NeuronsPerHLayer = n_NeuronsPerHLayer_backup;
	n_outputs = n_outputs_backup;
	CreateLayers();
}

float NeuralNetwork::CostFunction(float lambda) {
	float J = 0;
	ArrayXXf yi;
	ArrayXXf hxi;
	for (int i = 0; i < n_outputs; i++) {
		yi = labels.col(i).array();
		hxi = output_a.col(i).array();
		J -= 1.0 / (float)n_samples * (yi * hxi.log() + (1 - yi) * (1 - hxi).log()).sum();
	}
	// regularization
	float reg(0);
	ArrayXXf theta;
	for (int i = 0; i < layers.size(); i++) {
		theta = layers[i].topRows(layers[i].rows()-1).array();
		reg += theta.square().sum();
	}
	J += lambda / 2.0 / n_samples * reg;
	return J;
}
//create layers and random initialize weights
void NeuralNetwork::CreateLayers() {
	layers.clear();
	// initialize the hidden layer if n_HLayers > 0
	if (n_HLayers > 0) {
		srand((unsigned int)time(NULL));
		layers.push_back(rand_range * MatrixXf::Random(n_inputs + 1, n_NeuronsPerHLayer));   // first hidden layer
		for (int i = 1; i < n_HLayers; i++) {
			srand((unsigned int)time(NULL));
			layers.push_back(rand_range * MatrixXf::Random(n_NeuronsPerHLayer + 1, n_NeuronsPerHLayer));   // other hidden layer
		}
	}
	// initialize output layer
	srand((unsigned int)time(NULL));
	layers.push_back(rand_range * MatrixXf::Random(n_NeuronsPerHLayer + 1, n_outputs));
}
// forward propagation
void NeuralNetwork::Forward() {
	// first hidden layer when n_HLayers>0
	hlayers_z.resize(n_HLayers, MatrixXf(n_samples, n_NeuronsPerHLayer));
	hlayers_a.resize(n_HLayers, MatrixXf(n_samples, n_NeuronsPerHLayer+1));
	if (n_HLayers>0){
		hlayers_z[0] = inputs * layers[0];
		hlayers_a[0] = AddBias(Sigmoid(hlayers_z[0]));
	}
	else cout << "Error: number of hidden layer" << n_HLayers << "should be positive" << endl;
	// other hidden layers when n_HLayers>1
	for (int i = 1; i < n_HLayers; i++) {
		hlayers_z[i] = hlayers_a[i - 1] * layers[i];
		hlayers_a[i] = AddBias(Sigmoid(hlayers_z[i]));
	}
	// output
	output_z = hlayers_a[n_HLayers - 1] * layers[n_HLayers];
	output_a = Sigmoid(output_z);
}

void NeuralNetwork::GetInput(const MatrixXf &X) {
	inputs = X;
	inputs = AddBias(inputs);
	n_samples = X.rows();
}

void NeuralNetwork::GetLabel(const VectorXf &y) {
	labels = MatrixXf::Zero(n_samples, n_outputs);
	for (int i = 0; i < n_samples; i++) {
		labels(i, y(i)) = 1;
	}
}

vector<MatrixXf> NeuralNetwork::Get_weights() {
	return layers;
}

void NeuralNetwork::GradientDescent(float alpha) {
	for (int i = 0; i < layers.size(); i++) {
		layers[i] -= alpha * theta_grads[i];
	}
}

void NeuralNetwork::Init_weights(vector<MatrixXf> weights) {
	layers = weights;
}

void NeuralNetwork::NumGradient(float lambda, float epsilon) {
	cout << "Numerical gradient" << endl;
	vector<MatrixXf>original_layers (layers);
	// theta_grads_num allocate
	theta_grads_num.clear();
	for (int i = 0; i < layers.size(); i++) {
		theta_grads_num.push_back(MatrixXf::Zero(layers[i].rows(),layers[i].cols()));
	}
	// loop, approximate partial derivative
	float JPlus, JMinus;
	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].rows(); j++) {
			for (int k = 0; k < layers[i].cols(); k++) {
				// theta+epsilon
				layers[i](j, k) += epsilon;
				Forward();
				JPlus = CostFunction(lambda);
				// theta-epsilon
				layers[i](j, k) -= 2.0 * epsilon;
				Forward();
				JMinus = CostFunction(lambda);
				// calculate approximate partial derivative of theta(i,j,k)
				theta_grads_num[i](j, k) = (JPlus - JMinus) / 2.0 / epsilon;
				// restore
				layers[i](j, k) += epsilon;
			}
		}
	}
}

VectorXf NeuralNetwork::Predict(const MatrixXf &X) {
	GetInput(X);
	Forward();
	MatrixXf::Index maxidx;
	VectorXf h(n_samples);
	for (int i = 0; i < n_samples; i++) {
		output_a.row(i).maxCoeff(&maxidx);
		h(i) = maxidx;
	}
	return h;
}

MatrixXf NeuralNetwork::Sigmoid(const MatrixXf &z) {
	ArrayXXf temp1 = z.array();
	ArrayXXf temp2 = (1 + (-temp1).exp()).inverse();
	return temp2.matrix();
}

void NeuralNetwork::Test(const MatrixXf &X, const VectorXf &y) {
	cout << "Testing the trained NeuralNetwork" << endl;
	GetInput(X);
	VectorXf h = Predict(X);
	int count(0);
	for (int i = 0; i < n_samples; i++) {
		if (h(i) == y(i)) count++;
	}
	cout << "The accuracy of the test is " << (float)count / n_samples*100.0 << "%." << endl;
	//display
	cout << "For example:" << endl;
	for (int i = 0; i < 10; i++) {
		MatrixXf sample = X.row(i);
		sample.resize(28, 28);
		cout << "The " << i + 1 << ". sample is " << endl << sample.transpose() << endl <<
			"And it is predicted to be " << h(i) << endl << endl;
	}
}

void NeuralNetwork::Train(const MatrixXf &X, const VectorXf &y, int max_iter, float alpha, float lambda) {
	cout << "Training the NeuralNetwork, maximal iteration: " << max_iter << endl;
	clock_t t = clock();
	GetInput(X);
	GetLabel(y);
	float J, J_new, diff;
	cout << "Index: " << '\t' << "Cost:" <<'\t'<< "Difference:" << endl;
	for (int i = 0; i < max_iter; i++) {
		Forward();
		Back(lambda);
		J_new = CostFunction(lambda);
		GradientDescent(alpha);
		// check the difference
		if (i == 0) diff = NAN;
		else {
			diff = J- J_new;
			if (abs(diff) < 0.0001) {
				cout << i + 1 << '\t' << J_new << '\t' << diff << '\t' << 
					"Train process stops, because the improvement is small enough." << endl;
				//break;
			}
		}
		cout << i + 1 << '\t' << J_new << '\t' << diff << endl;
		J = J_new;
	}
	cout << "Training process success! It costs " << (clock() - t) / (float)CLOCKS_PER_SEC << " sec." << endl;
}