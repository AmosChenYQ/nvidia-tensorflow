/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/protobuf/matmul_autotuning.pb.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {
namespace functor {

// Helpers to define tensor<T> needed by MatMul op.
template <typename T>
struct MatMulTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned>
      in_type;
};

template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void MatMul(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
}

template <typename Device, typename T>
struct MatMulFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename MatMulTypes<T>::out_type out,
      typename MatMulTypes<T>::in_type in0,
      typename MatMulTypes<T>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
};

}  // end namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Encapsulate all the shape information that is used in matmul operations.
class MatmulParameters {
 public:
  MatmulParameters(bool transa, bool transb, uint64 m, uint64 n, uint64 k,
                   DataType dtype, int device_id)
      : transa_(transa),
        transb_(transb),
        m_(m),
        n_(n),
        k_(k),
        dtype_(dtype),
        device_id_(device_id) {
    UpdateHash();
  }

  MatmulParameters(const MatmulParamsProto& proto) {
    transa_ = proto.transa();
    transb_ = proto.transb();
    m_ = proto.m();
    n_ = proto.n();
    k_ = proto.k();
    dtype_ = proto.dtype();
    device_id_ = proto.device_id();
    UpdateHash();
  }

  MatmulParamsProto ToProto() const {
    MatmulParamsProto proto;
    proto.set_transa(transa_);
    proto.set_transb(transb_);
    proto.set_m(m_);
    proto.set_n(n_);
    proto.set_k(k_);
    proto.set_dtype(dtype_);
    proto.set_device_id(device_id_);
    return proto;
  }

  bool operator==(const MatmulParameters& other) const {
    return this->get_data_as_tuple() == other.get_data_as_tuple();
  }

  bool operator!=(const MatmulParameters& other) const {
    return !(*this == other);
  }
  uint64 hash() const { return hash_code_; }

  string ToString() const {
    // clang-format off
    return strings::StrCat(
        transa_, ", ", transb_, ", ",
        m_, ", ", n_, ", ", k_,
        dtype_, ", ", device_id_);
    // clang-format on
  }

 private:
  typedef std::tuple<bool, bool, int64, int64, int64, DataType, int>
      ParameterDataType;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(transa_, transb_, m_, n_, k_, dtype_, device_id_);
  }

  void UpdateHash() {
    hash_code_ = transa_;
    hash_code_ = Hash64Combine(hash_code_, transb_);
    hash_code_ = Hash64Combine(hash_code_, m_);
    hash_code_ = Hash64Combine(hash_code_, n_);
    hash_code_ = Hash64Combine(hash_code_, k_);
    hash_code_ = Hash64Combine(hash_code_, dtype_);
    hash_code_ = Hash64Combine(hash_code_, device_id_);
  }

  bool transa_;
  bool transb_;
  uint64 m_;
  uint64 n_;
  uint64 k_;
  DataType dtype_;
  int device_id_;
  uint64 hash_code_;
};

typedef Eigen::GpuDevice GPUDevice;

class MatmulAutoTuneMap
    : public AutoTuneMap<MatmulParameters, se::blas::AlgorithmConfig> {
 public:
  MatmulAutoTuneMap(const string& name)
      : AutoTuneMap<MatmulParameters, se::blas::AlgorithmConfig>(name) {
    const char* import_folder = getenv("TF_AUTOTUNE_IMPORT_PREFIX");
    if (import_folder != nullptr) {
      string autotune_import_file{import_folder};
      string group_name = this->name_;
      std::transform(group_name.cbegin(), group_name.cend(), group_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      autotune_import_file = autotune_import_file + "/" + group_name;
      VLOG(1) << "Read matmul autotune list from file: "
              << autotune_import_file;
      string matmul_autotune_list_str;
      tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                   autotune_import_file,
                                   &matmul_autotune_list_str);
      MatmulAutoTuneList matmul_autotune_list;
      matmul_autotune_list.ParseFromString(matmul_autotune_list_str);
      int autotune_size = matmul_autotune_list.matmul_params_size();
      for (int idx = 0; idx < autotune_size; ++idx) {
        this->params_config_map_.insert(std::make_pair(
            MatmulParameters{matmul_autotune_list.matmul_params().Get(idx)},
            ValueType{se::blas::AlgorithmConfig{
                          matmul_autotune_list.algorithm_type().Get(idx)},
                      matmul_autotune_list.score().Get(idx),
                      matmul_autotune_list.count().Get(idx)}));
      }
    }
  }

  ~MatmulAutoTuneMap() {
    const char* export_folder = getenv("TF_AUTOTUNE_EXPORT_PREFIX");
    if (export_folder != nullptr) {
      MatmulAutoTuneList matmul_autotune_list;
      for (const auto& iter : this->params_config_map_) {
        MatmulParamsProto* matmul_params =
            matmul_autotune_list.add_matmul_params();
        *matmul_params = iter.first.ToProto();
        matmul_autotune_list.add_algorithm_type(iter.second.config.algorithm());
        matmul_autotune_list.add_score(iter.second.score);
        matmul_autotune_list.add_count(iter.second.count);
      }
      string matmul_autotune_list_str;
      matmul_autotune_list.SerializeToString(&matmul_autotune_list_str);
      string autotune_export_file{export_folder};
      string group_name = this->name_;
      std::transform(group_name.cbegin(), group_name.cend(),
                     group_name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      autotune_export_file = autotune_export_file + "/" + group_name;
      VLOG(1) << "Write matmul autotune list to file: " << autotune_export_file;
      tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                    autotune_export_file,
                                    matmul_autotune_list_str);
    }
  }

  template <class Group>
  friend class MatmulAutoTuneSingleton;
};

template <class Group>
class MatmulAutoTuneSingleton {
 public:
  typedef MatmulAutoTuneMap MatmulAutoTuneType;
  static MatmulAutoTuneType* GetInstance() {
    // Have to do this to enforce destructor of AutoTuneType for caching
    struct EnableMakeShared : public MatmulAutoTuneType {
      EnableMakeShared(const string& name) : MatmulAutoTuneType(name) {}
    };
    static auto instance_ptr = std::static_pointer_cast<MatmulAutoTuneType>(
        std::make_shared<EnableMakeShared>(Group::name()));
    return instance_ptr.get();
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_H_
