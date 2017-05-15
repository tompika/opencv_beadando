#include "opencv2/core.hpp"
#include <opencv2/face.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>

using namespace cv;
using namespace face;
using namespace std;



void trainFisherFace(vector<Mat> images, vector<int> labels) {

	Ptr<face::FaceRecognizer> model = face::createFisherFaceRecognizer();
	//Ptr<face::FaceRecognizer> model = createLBPHFaceRecognizer();

	cout << "--- FisherFaceRecognizer train ---" << endl;
	cout << "Train begin..." << endl;

	model->train(images, labels);
	model->save("FisherFaceRecognizer.yml");

	cout << "Train end!" << endl;

}

void trainLBPHF(vector<Mat> images, vector<int> labels) {
	
	Ptr<face::FaceRecognizer> model = createLBPHFaceRecognizer();

	cout << "--- LBPHFaceRecognizer train ---" << endl;
	cout << "Train begin..." << endl;

	model->train(images, labels);
	model->save("LBPHFaceRecognizer.yml");

	cout << "Train end!" << endl;

}


void trainEigen(vector<Mat> images, vector<int> labels) {

	Ptr<face::FaceRecognizer> model = createEigenFaceRecognizer();

	cout << "--- EigenFaceRecognizer train ---" << endl;
	cout << "Train begin..." << endl;

	model->train(images, labels);
	model->save("EigenFaceRecognizer.yml");

	cout << "Train end!" << endl;

}




void loadTrainData(int type, Ptr<FaceRecognizer>& model) {

	if (type == 1) {
		model = createFisherFaceRecognizer();
		model->load("FisherFaceRecognizer.yml");
		cout << "FisherFaceRecognizer.yml loaded!" << endl;
	}
	else if (type == 2) {
		model = createLBPHFaceRecognizer();
		model->load("LBPHFaceRecognizer.yml");
		cout << "LBPHFaceRecognizer.yml loaded!" << endl;
	}
	else if (type == 3) {
		model = createEigenFaceRecognizer();
		model->load("EigenFaceRecognizer.yml");
		cout << "EigenFaceRecognizer.yml loaded!" << endl;
	}
	else
		cout << "Hibas tipus lett megadva!" << endl;


}

