#include <sstream>
#include <vector>

#include "caffe/layers/binary_png_write_layer.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
void BinaryPngWriteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  iter_ = 0;
  prefix_ = this->layer_param_.binary_png_write_param().prefix();
  period_ = this->layer_param_.binary_png_write_param().period();
  CHECK_GT(period_, 0) << "period must be positive";
  if (this->layer_param_.binary_png_write_param().has_source()) {
    std::ifstream infile(this->layer_param_.binary_png_write_param().source().c_str());
    CHECK(infile.good()) << "Failed to open source file "
			 << this->layer_param_.binary_png_write_param().source();
    const int strip = this->layer_param_.binary_png_write_param().strip();
    CHECK_GE(strip, 0) << "Strip cannot be negative";
    string linestr;
    while (std::getline(infile, linestr)) {
      std::istringstream iss(linestr);
      string filename;
      iss >> filename;
      CHECK_GT(filename.size(), strip) << "Too much stripping";
      fnames_.push_back(filename.substr(0, filename.size() - strip));
    }
    LOG(INFO) << "BinaryPngWrite will save a maximum of " << fnames_.size() << " files.";
  }
}

template <typename Dtype>
void BinaryPngWriteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BinaryPngWriteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(0);
  if (iter_ % period_ == 0) {
    for (int i = 0; i < bottom.size(); ++i) {
      std::ostringstream oss;
      oss << prefix_;
      if (this->layer_param_.binary_png_write_param().has_source()) {
        CHECK_LT(iter_, fnames_.size()) << "Test has run for more iterations than it was supposed to";
        oss << fnames_[iter_];
      }
      else {
        oss << "iter_" << iter_;
      }
      oss << "_mask.png";

      Blob<Dtype>* visual_blob = bottom[i];
      cv::Mat img(visual_blob->height(), visual_blob->width(), CV_8UC1);
      for (int h = 0; h < img.rows; ++h) {
        for (int w = 0; w < img.cols; ++w) {
          // 2017.08.29 binary segmentation
          int value = 0;
          if (visual_blob->data_at(i, 1, h, w) > visual_blob->data_at(i, 0, h, w)) {
            value = 1;
          }
          img.at<uchar>(h, w) = cv::saturate_cast<uchar>(255*value);
        }
      }
      // CV_IMWRITE_PNG_COMPRESSION = 0, set to zero no compress
      cv::imwrite(oss.str().c_str(), img, compression_params);
    }
  }
  ++iter_;
}

template <typename Dtype>
void BinaryPngWriteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}


INSTANTIATE_CLASS(BinaryPngWriteLayer);
REGISTER_LAYER_CLASS(BinaryPngWrite);

}  // namespace caffe
