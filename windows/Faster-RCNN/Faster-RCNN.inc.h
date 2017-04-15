#pragma once
#include <fstream>
#include <thread>
#include <opencv2\opencv.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <boost/shared_ptr.hpp>
#include "CaffeBinding.h"
#include "nms\gpu_nms.hpp"

extern caffe::CaffeBinding* kCaffeBinding;

using namespace cv;
using namespace std;

namespace Feng {

  struct myInfo {
    float score;
    const float* head;
  };
  bool compare(const myInfo& myInfo1, const myInfo& myInfo2) {
    return myInfo1.score > myInfo2.score;
  }

  struct ObjectInfo {
    Rect2d rect;
    int class_num;
    float score;
  };

  class FasterRCNN {
  public:
    FasterRCNN() {}
    FasterRCNN(string net_definition, string net_weights,
               int gpu_id = -1) : target_size_(600), max_size_(1000), confidence_threshold_(0.7), nms_threshold_(0.3) {
      net_ = kCaffeBinding->AddNet(net_definition, net_weights, gpu_id);
    }

    //return <bounding box, class number, confidence>
    vector<ObjectInfo> Detection(Mat& input_image) {
      Mat resized_image;
      int shape_min = min(input_image.rows, input_image.rows);
      int shape_max = max(input_image.rows, input_image.rows);
      float scale = float(target_size_) / float(shape_min);
      if (shape_max * scale > max_size_) scale = max_size_ / shape_max;
      resize(input_image, resized_image, Size(), scale, scale);
      float im_info[3];
      im_info[0] = resized_image.rows;
      im_info[1] = resized_image.cols;
      im_info[2] = scale;
      kCaffeBinding->SetBlobData("im_info", { 1, 3 }, im_info, net_);
      std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
      auto output = kCaffeBinding->Forward({ resized_image }, net_);
      std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
      cout << "forward time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
      
      auto rois = kCaffeBinding->GetBlobData("rois", net_);

      int num = rois.size[0];

      const float* bbox_delt = output["bbox_pred"].data;
      const float* pred_cls = output["cls_prob"].data;
      int total_class_num = output["cls_prob"].size[1];
      float* boxes = new float[num * 4];
      float* pred = new float[num * 5 * total_class_num];
      float* pred_per_class = new float[num * 5];
      float* sorted_pred_cls = new float[num * 5];
      int* keep = new int[num];
      int num_out;
      for (int n = 0; n < num; n++) {
        for (int c = 0; c < 4; c++) {
          boxes[n * 4 + c] = rois.data[n * 5 + c + 1] / scale;
        }
      }
      bbox_transform_inv(num, bbox_delt, pred_cls, boxes, pred, input_image.rows, input_image.cols);

      vector<ObjectInfo> results;

      for (int i = 1; i < total_class_num; i++) {
        for (int j = 0; j< num; j++) {
          for (int k = 0; k < 5; k++) {
            pred_per_class[j * 5 + k] = pred[(i*num + j) * 5 + k];
          }
        }
        boxes_sort(num, pred_per_class, sorted_pred_cls);
        _nms(keep, &num_out, sorted_pred_cls, num, 5, nms_threshold_, 0);
        for (int n = 0; n < num_out; n++) {
          if (sorted_pred_cls[keep[n] * 5 + 4] < confidence_threshold_) break;
          else {
            results.push_back({ Rect2d(sorted_pred_cls[keep[n] * 5 + 0], sorted_pred_cls[keep[n] * 5 + 1],
                                       sorted_pred_cls[keep[n] * 5 + 2] - sorted_pred_cls[keep[n] * 5 + 0],
                                       sorted_pred_cls[keep[n] * 5 + 3] - sorted_pred_cls[keep[n] * 5 + 1]),
                              i, sorted_pred_cls[keep[n] * 5 + 4] });
          }
        }
      }
      delete[]boxes;
      delete[]pred;
      delete[]pred_per_class;
      delete[]keep;
      delete[]sorted_pred_cls;
      return results;
    }

    float target_size_;
    float max_size_;
    double confidence_threshold_;
    double nms_threshold_;
  private:
    int net_;
    void boxes_sort(const int num, const float* pred, float* sorted_pred) {
      vector<myInfo> my;
      myInfo tmp;
      for (int i = 0; i < num; i++) {
        tmp.score = pred[i * 5 + 4];
        tmp.head = pred + i * 5;
        my.push_back(tmp);
      }
      std::sort(my.begin(), my.end(), compare);
      for (int i = 0; i < num; i++) {
        for (int j = 0; j < 5; j++)
          sorted_pred[i * 5 + j] = my[i].head[j];
      }
    }
    void bbox_transform_inv(int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width) {
      float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
      for (int i = 0; i < num; i++) {
        width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0;
        height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0;
        ctr_x = boxes[i * 4 + 0] + 0.5 * width;
        ctr_y = boxes[i * 4 + 1] + 0.5 * height;
        for (int j = 0; j < 21; j++) {

          dx = box_deltas[(i * 21 + j) * 4 + 0];
          dy = box_deltas[(i * 21 + j) * 4 + 1];
          dw = box_deltas[(i * 21 + j) * 4 + 2];
          dh = box_deltas[(i * 21 + j) * 4 + 3];
          pred_ctr_x = ctr_x + width*dx;
          pred_ctr_y = ctr_y + height*dy;
          pred_w = width * exp(dw);
          pred_h = height * exp(dh);
          pred[(j*num + i) * 5 + 0] = max(min<double>(pred_ctr_x - 0.5* pred_w, img_width - 1), 0.0);
          pred[(j*num + i) * 5 + 1] = max(min<double>(pred_ctr_y - 0.5* pred_h, img_height - 1), 0.0);
          pred[(j*num + i) * 5 + 2] = max(min<double>(pred_ctr_x + 0.5* pred_w, img_width - 1), 0.0);
          pred[(j*num + i) * 5 + 3] = max(min<double>(pred_ctr_y + 0.5* pred_h, img_height - 1), 0.0);
          pred[(j*num + i) * 5 + 4] = pred_cls[i * 21 + j];
        }
      }
    }
  };
}