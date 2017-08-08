# C++ Inference

Purely C++ inference module for Faster RCNN. The `proposal_layer` is copied from https://github.com/ihooercom/rpn/blob/master/caffe/src/caffe/layers/proposal_layer.cpp 
and some codes are copied from https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus.

# Requirements

My Caffe windows version https://github.com/happynear/caffe-windows/tree/hog (speed of ms branch may be very slow sometimes, the reason is unclear).

# Configuration

 - Build the DLL project `caffe.binding` in caffe-windows solution.
 - Change include and library folders in **this Faster-RCNN project** to your own folders.
 - Change the `model_folder` and `image_root` to your own. If you want to use your own prototxt file, please modify it referring to [faster_rcnn_test.pt](https://github.com/happynear/py-faster-rcnn/blob/master/windows/Faster-RCNN/faster_rcnn_test.pt).
 - Build.
