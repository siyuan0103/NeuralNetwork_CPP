#include "stdafx.h"
#ifndef LOADMNIST_H
#define LOADMNIST_H
#include <vector>
#include <string>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class LoadMNIST {
	// path of MNIST files
	string folder_path;
	string train_img_file;
	string train_lab_file;
	string test_img_file;
	string test_lab_file;
	// data
	MatrixXf train_img;
	VectorXf train_lab;
	MatrixXf test_img;
	VectorXf test_lab;
	// number of data
	int n_train;
	int n_test;
	int n_rows;
	int n_columns;

	int LoadMNIST::ReverseInt(int i);

	void load_image(const string &path, MatrixXf &img, int &n_images);

	void load_label(const string &path, VectorXf &lab, int &n_images);

public:
	LoadMNIST(string path_of_MNIST_folder = "./MNIST/");

	void get_train(MatrixXf &train_image,
		VectorXf &train_label, int number = 60000);

	void get_test(MatrixXf &test_image,
		VectorXf &test_label, int number = 10000);
};

#endif // !LOADMNIST_H