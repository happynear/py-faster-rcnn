#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <boost/python.hpp>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;
#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))
ofstream test("/home/xyy/Desktop/doing/objectDetection/py-faster-rcnn/test_c++.txt");
class Detector {
public:
	Detector(const string& model_file, const string& trained_file);
	void Detection(const string& im_name, float img_scale);
	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
private:
	shared_ptr<Net<float> > net_;
	Detector(){}
};
Detector::Detector(const string& model_file, const string& trained_file)
{
	net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(trained_file);
}
struct myInfo
{
	float score;
	const float* head;
};
bool compare(const myInfo& myInfo1, const myInfo& myInfo2)
{
	return myInfo1.score > myInfo2.score;
}
void Detector::Detection(const string& im_name, float img_scale)
{
	float CONF_THRESH = 0.7;
	float NMS_THRESH = 0.2;
	cv::Mat cv_img = cv::imread(im_name);
	cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
	if(cv_img.empty())
    {
        return ;
    }
	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
	int num_out;
	cv::Mat cv_resized;

	float im_info[3];
	float data_buf[height*width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	const float* rois;
	const float* pred_cls;
	int num;
	for (int h = 0; h < cv_img.rows; ++h )
	{
		for (int w = 0; w < cv_img.cols; ++w)
		{	
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);

		}
	}	
	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;
	for (int h = 0; h < height; ++h )
	{
		for (int w = 0; w < width; ++w)
		{			
			data_buf[(0*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2*height+h)*width+w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}
	
	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	net_->blob_by_name("data")->set_cpu_data(data_buf);
	net_->blob_by_name("im_info")->set_cpu_data(im_info);
	net_->ForwardFrom(0);
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();
	
	
	
	rois = net_->blob_by_name("rois")->cpu_data();
	pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	boxes = new float[num*4];
	pred = new float[num*5*21];
	pred_per_class = new float[num*5];
	sorted_pred_cls = new float[num*5];
	keep = new int[num];
	for (int n = 0; n < num; n++)
	{
		for (int c = 0; c < 4; c++)
		{
			boxes[n*4+c] = rois[n*5+c+1] / img_scale;
      cout << boxes[n * 4 + c] << " ";
		}
    cout << endl;
	}
	
	bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, cv_img.rows, cv_img.cols);
	for (int i = 1; i < 21; i ++)
	{
		for (int j = 0; j< num; j++)
		{
			for (int k=0; k<5; k++)
				pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
		}
		boxes_sort(num, pred_per_class, sorted_pred_cls);
		_nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, 0);
		vis_detections(cv_img, keep, num_out, sorted_pred_cls, CONF_THRESH);
	}
	LOG(INFO)<<"Done.";
	cv::imshow("image", cv_img);
	cv::waitKey();
	delete []boxes;
	delete []pred;
	delete []pred_per_class;
	delete []keep;
	delete []sorted_pred_cls;

}
void Detector::vis_detections(cv::Mat image, int* keep, int num_out, float* sorted_pred_cls, float CONF_THRESH)
{
	int i=0;
	while(sorted_pred_cls[keep[i]*5+4]>CONF_THRESH && i < num_out)
	{
		if(i>=num_out)
			return;
		cv::rectangle(image,cv::Point(sorted_pred_cls[keep[i]*5+0], sorted_pred_cls[keep[i]*5+1]),cv::Point(sorted_pred_cls[keep[i]*5+2], sorted_pred_cls[keep[i]*5+3]),cv::Scalar(255,0,0));
		i++;  
	}
}
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<myInfo> my;
	myInfo tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i*5 + 4];
		tmp.head = pred + i*5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i=0; i<num; i++)
	{
		for (int j=0; j<5; j++)
			sorted_pred[i*5+j] = my[i].head[j];
	}
}
void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for(int i=0; i< num; i++)
	{
		width = boxes[i*4+2] - boxes[i*4+0] + 1.0;
		height = boxes[i*4+3] - boxes[i*4+1] + 1.0;
		ctr_x = boxes[i*4+0] + 0.5 * width;
		ctr_y = boxes[i*4+1] + 0.5 * height;
		for (int j=0; j< 21; j++)
		{

			dx = box_deltas[(i*21+j)*4+0];
			dy = box_deltas[(i*21+j)*4+1];
			dw = box_deltas[(i*21+j)*4+2];
			dh = box_deltas[(i*21+j)*4+3];
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);
			pred[(j*num+i)*5+0] = max(min(pred_ctr_x - 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+1] = max(min(pred_ctr_y - 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+2] = max(min(pred_ctr_x + 0.5* pred_w, img_width -1), 0);
			pred[(j*num+i)*5+3] = max(min(pred_ctr_y + 0.5* pred_h, img_height -1), 0);
			pred[(j*num+i)*5+4] = pred_cls[i*21+j];
		}
	}

}
int main()
{
	string model_file = "/home/xyy/Desktop/doing/objectDetection/py-faster-rcnn/models/ZF/faster_rcnn_alt_opt/faster_rcnn_test.pt";
	string trained_file = "/home/xyy/Desktop/doing/objectDetection/py-faster-rcnn/data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel";

	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);
	Detector det = Detector(model_file, trained_file);
	det.Detection("/home/xyy/Desktop/doing/objectDetection/py-faster-rcnn/data/demo/004545.jpg", 1.6);
	return 0;
}