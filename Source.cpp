#include "opencv2/core.hpp"
#include <opencv2/face.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"


#include <Windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2\imgproc\imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

#include "methods.h"

using namespace cv;
using namespace std;
using namespace face;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "A megadott eleresi utvonalon nem talalhato fajl!.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	Mat temp;

	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			temp = imread(path, 0);
			resize(temp, temp, Size(600, 900));
			images.push_back(temp);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {

	string fn_haar = "data/haarcascade_frontalface_alt.xml"; // string(argv[1]);
	string fn_csv = "data/at.txt";// string(argv[2]);
	int deviceId = 0; // atoi(argv[3]);
	
	vector<Mat> images;
	vector<int> labels;

	////////////////
	bool DaniOpen = false;
	bool AdamOpen = false;
	bool PetiOpen = false;

	/****DÁTUM*****/
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%H_%M_%S", timeinfo);
	std::string current_date(buffer);

	stringstream ss;
	ss << "mkdir pillanatkep";
	system(ss.str().c_str());

	cout << "Pillanatkep konyvtar letrehozva!" << endl;

	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	//trainFisherFace(images, labels);
	//trainLBPHF(images, labels);
	//trainEigen(images, labels);

	//Típusok jelölése a beolvasásnál 
	//1 fisherface
	//2 BPHF
	//3 Eigen

	Ptr<FaceRecognizer> model;
	loadTrainData(1, model);



	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);

	cout << "Haar cascade loaded!" << endl;
	

	VideoCapture cap(0);
	
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}

	Mat frame;
	for (;;) {
		cap >> frame;
		
		Mat original = frame.clone();
		
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		
		vector< Rect_<int> > faces;

		haar_cascade.detectMultiScale(gray, faces);
		

		for (int i = 0; i < faces.size(); i++) {

			Rect face_i = faces[i];
			
			Mat face = gray(face_i);
			

			Mat face_resized;
			cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			
			int prediction = model->predict(face_resized);
			

			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);


			string name;
			if (prediction == 1) {
				name = "Adam";
				if (!AdamOpen) {
					//ShellExecute(0, 0, L"https://www.facebook.com/adam.kovacs.1000", 0, 0, SW_SHOW);
					AdamOpen = true;
					
				}
			}
			else if (prediction == 2) {
				name = "Dani";
				if (!DaniOpen) {
					//ShellExecute(0, 0, L"https://www.facebook.com/themightydesmond", 0, 0, SW_SHOW);
					DaniOpen = true;
				}
			}
			else if (prediction == 3) {
				name = "Peti";
				if (!PetiOpen) {
				//	ShellExecute(0, 0, L"https://www.facebook.com/peter.szilvacsku", 0, 0, SW_SHOW);
					PetiOpen = true;
				}
			}
			else {
				name = "Ismeretlen";
			}

			//string box_text = format("Prediction = %d", prediction);
			
			int pos_x = max(face_i.tl().x - 10, 0);
			int pos_y = max(face_i.tl().y - 10, 0);
			
			putText(original, name, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		}
		
		imshow("Face recognizer", original);
		

		int c = waitKey(10);

		switch (c)
		{
		case 'c':
			//EXIT
			break;
		case 27:
			//EXIT
			break;
		case 'p':
			ss.str("");
			ss.clear();
			ss << "pillanatkep/" << current_date << ".jpg";
			imwrite(ss.str(), original);
			cout << "Pillanatkep elmentve: " << ss.str() << endl;
			break;
		case 'k':
			
			//TODO 
			break;
		}

	}
	return 0;
}