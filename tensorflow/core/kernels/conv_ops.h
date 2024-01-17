/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchConv2DOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64>& explicit_paddings, Tensor* output,
                  TensorFormat data_format);
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
struct LaunchConv2DOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64>& explicit_paddings, Tensor* output,
                  TensorFormat data_format);
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Used to keep track of persistent memory buffers used within the op.
// It uses malloc and free to avoid the time cost of initializing the memory.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(port::Malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { port::Free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() const { return "Im2ColBufferResource"; }
};

// Convolution parameters specified by Op attributes.
struct Conv2DParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
  std::vector<int64> explicit_paddings;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows_before;
  int64 pad_rows_after;
  int64 pad_cols_before;
  int64 pad_cols_after;
};

// Initializes and validates Conv2D parameters configured by OpKernel
// attributes.
Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params);

// Computes and validates convolutions dimensions from Conv2D parameters. If
// parameters are valid, dimensions will be updated with derived convolution
// dimensions, otherwise an error will be returned.
Status ComputeConv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DDimensions* dimensions);

using se::dnn::AlgorithmConfig;
class ConvAutoTuneMap : public AutoTuneMap<ConvParameters, se::dnn::AlgorithmConfig> {
 public:
  ConvAutoTuneMap(const string& name)
      : AutoTuneMap<ConvParameters, se::dnn::AlgorithmConfig>(name) {
    const char* import_folder = getenv("TF_AUTOTUNE_IMPORT_PREFIX");
    if (import_folder != nullptr) {
      string autotune_import_file(import_folder);
      string group_name = this->name_;
      std::transform(group_name.cbegin(), group_name.cend(), group_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      autotune_import_file = autotune_import_file + "/" + group_name;
      VLOG(1) << "Read conv autotune list from file: " << autotune_import_file;
      string conv_autotune_list_str;
      tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                   autotune_import_file,
                                   &conv_autotune_list_str);
      ConvAutoTuneList conv_autotune_list;
      conv_autotune_list.ParseFromString(conv_autotune_list_str);
      int autotune_size = conv_autotune_list.conv_params_size();
      for (int idx = 0; idx < autotune_size; ++idx) {
        this->params_config_map_.insert(std::make_pair(
            ConvParameters{conv_autotune_list.conv_params().Get(idx)},
            ValueType{conv_autotune_list.config().Get(idx),
                      conv_autotune_list.score().Get(idx),
                      conv_autotune_list.count().Get(idx)}));
      }
    }
  }

  ~ConvAutoTuneMap() {
    const char* export_folder = getenv("TF_AUTOTUNE_EXPORT_PREFIX");
    if (export_folder != nullptr) {
      ConvAutoTuneList conv_autotune_list;
      for (const auto& iter : this->params_config_map_) {
        ConvParamsProto* conv_params = conv_autotune_list.add_conv_params();
        *conv_params = iter.first.ToProto();
        stream_executor::dnn::AlgorithmConfigProto* conv_algo_cfg = conv_autotune_list.add_config();
        *conv_algo_cfg = iter.second.config.ToProto();
        conv_autotune_list.add_score(iter.second.score);
        conv_autotune_list.add_count(iter.second.count);
      }
      string conv_autotune_list_str;
      conv_autotune_list.SerializeToString(&conv_autotune_list_str);
      string autotune_export_file(export_folder);
      string group_name = this->name_;
      std::transform(group_name.cbegin(), group_name.cend(),
                     group_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      autotune_export_file = autotune_export_file + "/" + group_name;
      VLOG(1) << "Write conv autotune list to file: " << autotune_export_file;
      tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                    autotune_export_file,
                                    conv_autotune_list_str);
    }
  }
  
  template <class Group>
  friend class ConvAutoTuneSingleton;
};

template <class Group>
class ConvAutoTuneSingleton {
 public:
  typedef ConvAutoTuneMap ConvAutoTuneType;
  static ConvAutoTuneType* GetInstance() {
    // Have to do this to enforce destructor of AutoTuneType for caching
    struct EnableMakeShared : public ConvAutoTuneType {
      EnableMakeShared(const string& name)
          : ConvAutoTuneType(name) {}
    };
    static auto instance_ptr = std::static_pointer_cast<ConvAutoTuneType>(
        std::make_shared<EnableMakeShared>(Group::name()));
    return instance_ptr.get();
  }
};


}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_H_
