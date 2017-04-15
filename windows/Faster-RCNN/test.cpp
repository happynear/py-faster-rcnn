#include <chrono>
#include <cstdlib>
#include <memory>
#include <Windows.h>

#include "Faster-RCNN.inc.h"

caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();

using namespace Feng;

int main(int argc, char* argv[])
{
  cout << "loading model...";
  string model_folder = ".\\";
  FasterRCNN faster_rcnn(model_folder + "faster_rcnn_test.pt", model_folder + "VGG16_faster_rcnn_final.caffemodel",0);
  cout << "done." << endl;
  cout << "warm up...";
  Mat warm_up_image = Mat::ones(375, 500, CV_8UC3);
  faster_rcnn.Detection(warm_up_image);
  cout << "done." << endl;

  string image_root = "D:\\deeplearning\\py-faster-rcnn\\data\\demo\\";
  vector<string> image_filename = { "000456.jpg", "000542.jpg","001150.jpg","001763.jpg","004545.jpg" };
  for (auto& f : image_filename) {
    Mat image = imread(image_root + f);
    cout << "image size:" << image.size() << endl;

    std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    auto result = faster_rcnn.Detection(image);
    std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
    cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
    for (int i = 0; i < result.size(); i++) {
      rectangle(image, result[i].rect, Scalar(255, 0, 0), 2);
    }
    while (image.cols > 1000) {
      resize(image, image, Size(0, 0), 0.75, 0.75);
    }
    imshow(f, image);
  }
  waitKey(0);
  //kCaffeBinding = nullptr;
  system("pause");
	return 0;
}

