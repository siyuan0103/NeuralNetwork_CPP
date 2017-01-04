#include "stdafx.h"
#include "loadmnist.h"
#include <fstream>
#include <stdexcept>
#include <iostream>

// Constructor
LoadMNIST::LoadMNIST(string path_of_MNIST_folder): folder_path(path_of_MNIST_folder){
	// load train_image data
	train_img_file = folder_path + "train-images.idx3-ubyte";
	load_image(train_img_file, train_img, n_train);

	// load train_label data
	train_lab_file = folder_path + "train-labels.idx1-ubyte";
	load_label(train_lab_file, train_lab, n_train);

	// load test_image data
	test_img_file = folder_path + "t10k-images.idx3-ubyte";
	load_image(test_img_file, test_img, n_test);

	// load test_label data
	test_lab_file = folder_path + "t10k-labels.idx1-ubyte";
	load_label(test_lab_file, test_lab, n_test);


}
// image file loader
void LoadMNIST::load_image(const string &path, MatrixXf &img, int &n_images) {
	ifstream f (path, ios::binary);
	try {
		//check if the file is successfully opened
		if (f.is_open() == false) throw runtime_error("Could not open the file.");
		// check the magic number
		int magic = 0;
		f.read((char*)&magic, sizeof magic);
		magic = ReverseInt(magic);
		if (magic != 2051) cout << "Error: This is not a image data file under path: " << path << endl;
		// read parameters
		f.read((char*)&n_images, 4);
		n_images = ReverseInt(n_images);
		f.read((char*)&n_rows, 4);
		n_rows = ReverseInt(n_rows);
		f.read((char*)&n_columns, 4);
		n_columns = ReverseInt(n_columns);
		// load image data
		/*
		img.clear()
		vector<double> animage;
		animage.resize(n_rows*n_columns);
		char* buffer = new char[n_rows*n_columns];
		for (int i = 0; i < n_images; i++) {
			f.read(buffer, n_rows*n_columns);
			copy(buffer, buffer + n_rows*n_columns, animage.begin());
			img.push_back(animage);
		}
		delete[] buffer;
		*/
		unsigned char* buffer = new unsigned char[n_images*n_rows*n_columns];
		f.read((char*)buffer, n_images*n_rows*n_columns);
		Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> m = 
			Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(buffer, n_images, n_rows*n_columns);
		img = m.cast<float>();
		delete[] buffer;
		f.close();
	}
	catch (exception &e){
		cout << "Error: " << e.what() << "The path is " << path << endl;
		return;
	}
}

// label fiel loader
void LoadMNIST::load_label(const string &path, VectorXf &lab, int &n_images) {
	ifstream f(path, ios::binary);
	try {
		//check if the file is successfully opened
		if (f.is_open() == false) throw runtime_error("Could not open the file.");
		// check the magic number
		int magic = 0;
		f.read((char*)&magic, sizeof magic);
		magic = ReverseInt(magic);
		if (magic != 2049) cout << "Error: This is not a image data file under path: " << path << endl;
		// check the number of samples
		int n_labels = 0;
		f.read((char*)&n_labels, 4);
		n_labels = ReverseInt(n_labels);
		if (n_labels != n_images) cout << "Error: number of labels is " << n_labels << ", number of images is" 
			<< n_images << "They don't match each other. The path is "<< path << endl;
		// load image data
		/*
		lab.resize(n_labels);
		char* buffer = new char[n_labels];
		f.read(buffer, n_labels);
		copy(buffer, buffer + n_labels, lab.begin());
		delete[] buffer;
		*/
		unsigned char* buffer = new unsigned char[n_labels];
		f.read((char*)buffer, n_labels);
		Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> v =
			Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>>(buffer, n_labels, 1);
		lab = v.cast <float>();
		delete[] buffer;
		f.close();
	}
	catch (exception &e) {
		cout << "Error: " << e.what() << "The path is " << path << endl;
		return;
	}
}

// reverse data, 'cause MNIST-daten is MSB fisrt
int LoadMNIST::ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void LoadMNIST::get_train(MatrixXf &train_image,
							VectorXf &train_label, int number) {
	train_image = train_img.block(0, 0, number, n_rows*n_columns);
	train_label = train_lab.block(0, 0, number, 1);
}

void LoadMNIST::get_test(MatrixXf &test_image,
	VectorXf &test_label, int number) {
	test_image = test_img.block(0, 0, number, n_rows*n_columns);
	test_label = test_lab.block(0, 0, number, 1);
}