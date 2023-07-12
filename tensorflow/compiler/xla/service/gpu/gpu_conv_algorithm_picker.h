/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace gpu {

using tensorflow::AutotuneResult;

class ConvAutotuneCache {
 public:
  static uint64 ConvAutotuneCacheKeyHasher(const se::StreamExecutor* stream_exec,
                                           const HloInstruction* instr);
  static ConvAutotuneCacheValue CreateConvAutotuneCacheValue(
      StatusOr<AutotuneResult> result_or, const se::StreamExecutor* stream_exec,
      const HloInstruction* instr);
  bool LookUpCache(uint64 key, ConvAutotuneCacheValue& cache_value);
  bool AddToCache(uint64 key, const ConvAutotuneCacheValue& cache_value);
  uint64 cache_hits;
  uint64 cache_misses;

 private:
  ConvAutotuneCache();
  ~ConvAutotuneCache();
  friend class ConvAutotuneCacheSingleton;
  std::string autotune_cache_filename_;
  bool in_use_;
  ConvAutotuneCacheProto conv_autotune_cache_proto_;
};

class ConvAutotuneCacheSingleton {
 public:
  static ConvAutotuneCache& GetInstance();
};

// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
class GpuConvAlgorithmPicker : public HloModulePass {
 public:
  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  GpuConvAlgorithmPicker(se::StreamExecutor* stream_exec,
                         se::DeviceMemoryAllocator* allocator)
      : stream_exec_(stream_exec), allocator_(allocator) {}

  absl::string_view name() const override {
    return "gpu-conv-algorithm-picker";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
  StatusOr<bool> RunOnInstruction(HloInstruction* instr);
  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithm(
      const HloCustomCallInstruction* instr);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithmNoCacheCuda(
      const HloCustomCallInstruction* instr,
      se::DeviceMemoryAllocator* allocator, se::Stream* stream);
#endif

  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithmNoCacheRocm(
      const HloCustomCallInstruction* instr,
      se::DeviceMemoryAllocator* allocator, se::Stream* stream);

  se::StreamExecutor* stream_exec_;       // never null
  se::DeviceMemoryAllocator* allocator_;  // may be null
};

}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_ALGORITHM_PICKER_H_
