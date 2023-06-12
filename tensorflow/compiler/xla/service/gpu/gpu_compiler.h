/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace gpu {

class OptimizedHloModuleCache {
  friend class OptimizedHloModuleCacheSingleton;

 public:
  using MapInMem =
      absl::flat_hash_map<std::string, OptimizedHloModuleCacheProto>;
  using MapIterInMem =
      typename absl::flat_hash_map<std::string,
                                   OptimizedHloModuleCacheProto>::iterator;

  enum class LoadStatus {
    kFoundInMem,
    kNotFoundInMem,
    kProtoNotInMem
  };

  // Responsiblle for update optimized_module_dir_.
  OptimizedHloModuleCache();
  // Responsible for writing the proto in the memory back to the corresponding
  // file when it is destructed.
  ~OptimizedHloModuleCache();
  // TryLoadFromFile is to load all of protos with the same module name but with
  // different shapes from file into memory. Return true iff the module's
  // name-proto pair is loaded into memoty.
  bool TryLoadFromFile(const HloModule* const module);
  // TryLoadFromMem is to load module with the required shape from flat_hash_map
  // in memory. Return KFoundInMem and fill optimized_module if proto collection
  // of the same name is in memory and the required module is in this proto.
  // Return kNotFoundInMem and left optimized_module to nullptr if proto
  // collection of the same name is in memory but the required module isn't in
  // this proto. Return KProtoNotInMem and left optimized_module to nullptr if
  // proto collection of the same name isn't in memory and this means a
  // TryLoadFromFile is needed in this case.
  LoadStatus
  TryLoadFromMem(const HloModule* const module,
                 std::unique_ptr<HloModule>* optimized_module);
  // MaybeLoadOptimizedModule will try to load optimized module proto from
  // memory firstly, if there is no required shape module in memory, try to load
  // optimized module proto from file to memory then try load required proto
  // from memory again.
  std::unique_ptr<HloModule> MaybeLoadOptimizedModule(
      const HloModule* const module);
  // FlushOptimizedModuleToMem will store the module optimized by pass
  // into memory. If the name doesn't exist in memory cache, create an empty
  // OptimizedHloModuleCacheProto and add module to it , otherwise update
  // OptimizedHloModuleCacheProto already existing in cache in memory.
  bool FlushOptimizedModuleToMem(const HloModule* const optimized_module,
                                 uint64 raw_module_hash);

  // Only when dir is valid it will be saved to optimized_module_dir_.
  bool has_dump_dir() const { return !optimized_module_dir_.empty(); }

 private:
  // cache_in_memory_ contains a map in memory whose key is module name and
  // value is all protos with the same name but with different shapes relying on
  // module hash to distinguish different shapes. OptimizedHloModuleCacheProto
  // contains a map whose key is module hash representing the shape and value is
  // HloModuleProto.
  absl::flat_hash_map<std::string, OptimizedHloModuleCacheProto>
      cache_in_memory_;
  std::string optimized_module_dir_;
};

class OptimizedHloModuleCacheSingleton {
 public:
  static OptimizedHloModuleCache* GetInstance();
};

// The GPU compiler generates efficient GPU executables.
class GpuCompiler : public LLVMCompiler {
 public:
  GpuCompiler(se::Platform::Id platform_id, const char* target_triple,
              const char* data_layout);
  ~GpuCompiler() override {}

  // Bring in
  // StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
  //     std::vector<std::unique_ptr<HloModule>> modules,
  //     std::vector<std::vector<se::StreamExecutor*>>
  //        stream_execs)
  using LLVMCompiler::Compile;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  Status OptimizeHloModule(HloModule* hlo_module,
                           se::StreamExecutor* stream_exec,
                           se::DeviceMemoryAllocator* device_allocator);

  virtual Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) = 0;

  virtual Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator);

  virtual HloDataflowAnalysis::CanShareBuffer GetCanShareBuffer() {
    return
        [](const HloInstruction*, const HloInstruction*,
           const ShapeIndex&) -> absl::optional<bool> { return absl::nullopt; };
  }

  virtual GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) = 0;

  virtual StatusOr<std::pair<std::string, std::vector<uint8>>>
  CompileTargetBinary(const HloModule* hlo_module, llvm::Module* llvm_module,
                      GpuVersion gpu_version,
                      se::StreamExecutor* stream_exec) = 0;

  Status PrepareHloModuleForIrEmitting(HloModule* hlo_module);

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     AotCompilationOptions const& options) override;

  se::Platform::Id PlatformId() const override { return platform_id_; }

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override {
    // Capture just the pointer size, not the entire GpuCompiler object.
    return [pointer_size = pointer_size_](const Shape& shape) {
      return GetSizeOfShape(shape, pointer_size);
    };
  }

  static int64 GetSizeOfShape(const Shape& shape, int pointer_size) {
    if (shape.is_static() || shape.IsTuple()) {
      return ShapeUtil::ByteSizeOf(shape, pointer_size);
    }
    // Each dynamic dimension size is represented as a S32.
    int64 metadata_size = sizeof(int32) * shape.dimensions_size();
    return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
  }

 private:
  se::Platform::Id platform_id_;

  // The triple that represents our target.
  const char* target_triple_;

  // The data layout of the emitted module.
  const char* data_layout_;

  // The size in bytes of a pointer. Used by ShapeSizeBytesFunction.
  const int64 pointer_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuCompiler);
};

// Compile `hlo_module` using XLA GPU and return the LLVM module thus generated.
// The GpuExecutable (and the Thunks that are part of it) are not returned.
StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, GpuDeviceInfo gpu_device_info,
    absl::optional<CudaComputeCapability> cuda_compute_capability,
    int pointer_size);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_COMPILER_H_
