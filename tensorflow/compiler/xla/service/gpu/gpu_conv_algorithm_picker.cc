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

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"

// TODO(AmosChenYQ): Check if container.h can be removed.
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_autotuning.pb.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_algorithm_blacklist.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/proto/proto_utils.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {
namespace {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
using tensorflow::AutotuneResult;
using tensorflow::Hash64;
using tensorflow::Hash64Combine;

class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal,
                   se::DeviceMemoryAllocator* memory_allocator)
      : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

  int64 GetMemoryLimitInBytes() override {
    return 1LL << 32;  // 4GB.  TODO(jlebar): Tune this?
  }
  int64 TotalAllocatedBytes() { return total_allocated_bytes_; }

  StatusOr<se::DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override;

  template <typename T>
  StatusOr<se::DeviceMemory<T>> Allocate(int64 num_elements) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemory<uint8> bytes,
                        AllocateBytes(num_elements * sizeof(T)));
    return se::DeviceMemory<T>(bytes);
  }

 private:
  const int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;
  std::vector<se::OwningDeviceMemory> allocated_buffers_;
  int64 total_allocated_bytes_ = 0;
};

StatusOr<se::DeviceMemory<uint8>> ScratchAllocator::AllocateBytes(
    int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8>(buffer_addr);
}

std::vector<AlgorithmDesc> GetAlgorithms(CudnnConvKind kind,
                                         se::StreamExecutor* stream_exec) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CudnnConvAlgorithmPicker::GetAlgorithms", 2);
  std::vector<AlgorithmDesc> algorithms;
  bool succ = false;
  switch (kind) {
    case CudnnConvKind::kBackwardFilter:
      succ =
          stream_exec->GetConvolveBackwardFilterAlgorithms(true, &algorithms);
      break;
    case CudnnConvKind::kBackwardInput:
      succ = stream_exec->GetConvolveBackwardDataAlgorithms(true, &algorithms);
      break;
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      succ = stream_exec->GetConvolveAlgorithms(true, &algorithms);
      break;
  }
  DCHECK(succ);

  // TODO(AmosChenYQ): Remove this log after debugging.
  VLOG(1) << "The number of algorithms for cudnn conv is " << algorithms.size();

  return algorithms;
}

StatusOr<std::vector<se::dnn::ProfileResult>> GetAlgorithms(
    const HloCustomCallInstruction* conv,
    absl::Span<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer, se::StreamExecutor* stream_exec,
    se::Stream* stream) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("RocmConvAlgorithmPicker::GetAlgorithms", 2);
  std::vector<se::dnn::ProfileResult> algorithms;

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind __attribute__((unused)),
                      GetDnnConvolutionKind(conv));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType dtype, GetDnnDataType(conv));

  TF_ASSIGN_OR_RETURN(GpuConvParams params,
                      GetGpuConvParams(conv, operand_buffers, result_buffer));

  CHECK(false) << "MIOpen not backported";
  // bool succ = stream_exec->GetMIOpenConvolveAlgorithms(
  //     kind, stream, dtype, params.input_descriptor, params.filter_descriptor,
  //     params.conv_desc, params.output_descriptor, &algorithms);
  // DCHECK(succ);

  return algorithms;
}

string AlgorithmToString(const AlgorithmDesc& algo) {
  if (algo.tensor_ops_enabled()) {
    return absl::StrCat(algo.algo_id(), "+TC");
  }
  return absl::StrCat(algo.algo_id());
}

string NumBytesToString(int64 bytes) {
  return absl::StrCat(tensorflow::strings::HumanReadableNumBytes(bytes), " (",
                      bytes, "B)");
}

tensorflow::CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
  tensorflow::CudnnVersion cudnn_version;
  if (auto* dnn = stream_executor->AsDnn()) {
    StatusOr<se::dnn::VersionInfo> version_or = dnn->GetVersion();
    if (version_or.ok()) {
      const auto& version = version_or.ValueOrDie();
      cudnn_version.set_major(version.major_version());
      cudnn_version.set_minor(version.minor_version());
      cudnn_version.set_patch(version.patch());
    }
  }
  return cudnn_version;
}

tensorflow::ComputeCapability GetComputeCapability(
    se::StreamExecutor* stream_executor) {
  tensorflow::ComputeCapability cc;
  int cc_major, cc_minor;
  stream_executor->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                  &cc_minor);
  cc.set_major(cc_major);
  cc.set_minor(cc_minor);
  return cc;
}

void PrintPlatformInfo(const se::Stream* stream) {
  auto* se = stream->parent();
  const auto& desc = se->GetDeviceDescription();
  LOG(ERROR) << "Device: " << desc.name();
  LOG(ERROR) << "Platform: " << desc.platform_version();
  LOG(ERROR) << "Driver: " << desc.driver_version();
  LOG(ERROR) << "Runtime: " << desc.runtime_version();

  auto* dnn = se->AsDnn();
  if (dnn) {
    auto dnn_version = dnn->GetVersion();
    if (dnn_version.ok()) {
      auto v = dnn_version.ValueOrDie();
      LOG(ERROR) << "cudnn version: " << v.major_version() << "."
                 << v.minor_version() << "." << v.patch();
    }
  }
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
// Returns true if the redzones in `allocator`'s allocations are unmodified.
//
// If the redzones are modified, logs an error, sets the appropriate failure
// bits on `result`, and returns false.
//
// Returns a status if an unexpected error has occurred, and the stream
// has been poisoned.
//
// `name` is a user-friendly name for the set of redzones being checked, e.g.
// "input/output" or "scratch".
StatusOr<bool> CheckRedzones(const se::RedzoneAllocator& allocator,
                             se::Stream* stream, absl::string_view name,
                             const HloInstruction* instr,
                             AutotuneResult* result) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CudnnConvAlgorithmPicker checking redzones",
                                 2);
  using RedzoneCheckStatus = se::RedzoneAllocator::RedzoneCheckStatus;
  TF_ASSIGN_OR_RETURN(RedzoneCheckStatus redzone_check,
                      allocator.CheckRedzones());
  if (redzone_check.ok()) {
    return true;
  }

  auto* fail = result->mutable_failure();
  fail->set_kind(AutotuneResult::REDZONE_MODIFIED);
  *fail->mutable_msg() = redzone_check.RedzoneFailureMsg();
  fail->set_buffer_address(
      reinterpret_cast<uint64>(redzone_check.user_buffer_address));

  LOG(ERROR) << absl::StreamFormat(
      "Detected cudnn out-of-bounds write in conv %s buffer! This is likely a "
      "cudnn bug. We will skip this algorithm in the future, but your GPU "
      "state may already be corrupted, leading to incorrect results. Within "
      "Google, no action is needed on your part. Outside of Google, please "
      "ensure you're running the latest version of cudnn. If that doesn't fix "
      "the problem, please file a bug with this full error message and we'll "
      "contact nvidia.",
      name);
  LOG(ERROR) << redzone_check.RedzoneFailureMsg();
  LOG(ERROR) << "HloInstruction " << instr->ToString();
  PrintPlatformInfo(stream);
  return false;
}
#endif

tensorflow::mutex conv_autotune_cache_mu(tensorflow::LINKER_INITIALIZED);

}  // anonymous namespace

StatusOr<AutotuneResult> GpuConvAlgorithmPicker::PickBestAlgorithm(
    const HloCustomCallInstruction* instr) {
  // Don't run this function concurrently on the same GPU.
  //
  // This is a bit of a hack and doesn't protect us against arbitrary concurrent
  // use of a GPU, but it's sufficient to let us compile two HLO modules
  // concurrently and then run them sequentially.
  //
  // Putting the lock in here rather than in PickBestAlgorithmNoCache lets us
  // avoid ever doing duplicate work.  If we have a cache miss, only one thread
  // will run PickBestAlgorithmImpl for a particular device.
  tensorflow::mutex_lock lock = LockGpu(stream_exec_);

  // We cache the autotuning results to avoid doing the duplicate work,
  // which can greatly improve both stability (deterministic numeric results
  // within a process for a given input) and performance (2x speedup on some
  // models).
  // Log conv instr and its operands message for debugging 
  auto log_conv_instr_and_operands_message =
      [](const HloCustomCallInstruction* instr) {
        auto options = HloPrintOptions::Canonical();
        options.set_print_backend_config(true);
        VLOG(1) << "Conv instruction in canonical string: ";
        VLOG(1) << instr->ToString(options);
        VLOG(1) << "Conv operands are: ";
        absl::c_for_each(instr->operands(),
                         [&](HloInstruction* const& operand) {
                           VLOG(1) << operand->ToString(options);
                         });
      };
  log_conv_instr_and_operands_message(instr);

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec_->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  // allocator either points to this->allocator_ or, if that's null, to a
  // se::StreamExecutorMemoryAllocator for stream_exec_.
  se::DeviceMemoryAllocator* allocator;
  optional<se::StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_);
    allocator = &*se_allocator;
  }

  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(stream_exec_->device_ordinal()));
  StatusOr<AutotuneResult> result_or;
  uint64 hashed_cache_key =
      ConvAutotuneCache::ConvAutotuneCacheKeyHasher(stream_exec_, instr);
  bool is_found_cache = false;
  uint64 conv_autotune_requests = 
      ConvAutotuneCacheSingleton::GetInstance().cache_hits +
          ConvAutotuneCacheSingleton::GetInstance().cache_misses + 1;
  ConvAutotuneCacheValue cache_value;
  {
    tensorflow::mutex_lock cache_lock(conv_autotune_cache_mu);
    is_found_cache = ConvAutotuneCacheSingleton::GetInstance().LookUpCache(
        hashed_cache_key, cache_value);
  }
  if (is_found_cache) {
   ConvAutotuneCacheSingleton::GetInstance().cache_hits++;
    VLOG(1) << "Autotuning cache hits/(hits + misses): "
            <<ConvAutotuneCacheSingleton::GetInstance().cache_hits << "/"
            << conv_autotune_requests;
    // return error status or conv autotune result
    if (cache_value.status_or_result_case() ==
        ConvAutotuneCacheValue::kStatus) {
      return se::port::Status(cache_value.status().code(),
                              cache_value.status().message());
    } else if (cache_value.status_or_result_case() ==
               ConvAutotuneCacheValue::kConvAutotuneResult) {
      return cache_value.conv_autotune_result();
    }
  }
 ConvAutotuneCacheSingleton::GetInstance().cache_misses++;
  VLOG(1) << "Autotuning cache misses/(hits + misses): "
          <<ConvAutotuneCacheSingleton::GetInstance().cache_misses << "/"
          << conv_autotune_requests;
  // Check StreamExecutor on which platform it is. ROCm and Cuda implementation
  // have diverged. Specifically, we need to make sure redzone allocator related
  // utilities are not used in ROCm routine
  if (stream_exec_->platform_kind() == se::PlatformKind::kROCm) {
    result_or = PickBestAlgorithmNoCacheRocm(instr, allocator, stream);
  } else if (stream_exec_->platform_kind() == se::PlatformKind::kCuda) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
    result_or = PickBestAlgorithmNoCacheCuda(instr, allocator, stream);
#endif
  } else {
    // If platform kind is neither Rocm nor Cuda, return error status of unknown
    // platform.
    return InternalError("Unknown platform.");
  }

  {
    tensorflow::mutex_lock cache_lock(conv_autotune_cache_mu);
    CHECK(ConvAutotuneCacheSingleton::GetInstance().AddToCache(
        hashed_cache_key, ConvAutotuneCache::CreateConvAutotuneCacheValue(
                              result_or, stream_exec_, instr)));
  }
  return result_or;
}

// The following function allows deterministic ops to be implemented relatively
// quickly using environment variables. It is intended to be temporary. The
// longer-term intention is to enable deterministic ops via tf.config and
// appropriate plumbing. See the discussion on PR 34951 for more information:
// https://github.com/tensorflow/tensorflow/pull/34951#discussion_r355682316
// This function and associated comment are replicated in the following three
// places:
//   1. tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.cc
//   2. tensorflow/core/kernels/gpu_utils.cc
//   3. tensorflow/stream_executor/cuda/cuda_dnn.cc
// When implementing the plumbing, you should also search for the use of
// TF_DETERMINISTIC_OPS on its own.
// TODO(duncanriach): move to an API that uses tf.config and implement the first
//                    phase of plumbing.
static bool RequireCudnnDeterminism() {
  static bool require_cudnn_determinism = [] {
    bool deterministic_ops = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                               /*default_val=*/false,
                                               &deterministic_ops));
    bool cudnn_deterministic = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_CUDNN_DETERMINISTIC",
                                               /*default_val=*/false,
                                               &cudnn_deterministic));
    return deterministic_ops || cudnn_deterministic;
  }();
  return require_cudnn_determinism;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
StatusOr<tensorflow::AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheCuda(
    const HloCustomCallInstruction* instr, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream) {
  // Right now Redzone allocator is available in Cuda target only
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr->ToString()));

  const Shape& result_shape = instr->shape().tuple_shapes(0);
  int64 rng_state = 0;

  const HloModuleConfig& hlo_module_config = instr->GetModule()->config();
  const int32 conv_autotune_level =
      hlo_module_config.debug_options().xla_gpu_autotune_level();
  const bool init_conv_data = conv_autotune_level > 1;
  const bool check_conv = conv_autotune_level > 3;
  const auto initialize_buffer = [init_conv_data, &stream, &rng_state](
                                     DeviceMemoryBase buffer,
                                     const Shape& buffer_shape) {
    if (init_conv_data) {
      InitializeBuffer(stream, buffer_shape.element_type(), &rng_state, buffer);
    }
  };

  // Allocate space for the input, filter, and output of the convolution.
  se::RedzoneAllocator input_output_allocator(
      stream, allocator, PtxOptsFromConfig(hlo_module_config));
  std::vector<se::DeviceMemoryBase> operand_buffers;
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer, operand->shape());
    operand_buffers.push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(auto result_buffer,
                      input_output_allocator.AllocateBytes(
                          ShapeUtil::ByteSizeOf(result_shape)));
  initialize_buffer(result_buffer, result_shape);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());

  optional<BufferComparator> comparator;
  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm suffices. It doesn't make
  // this algorithm considered correct, though.
  se::DeviceMemoryBase reference_result_buffer;
  AlgorithmDesc first_algorithm;

  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  std::vector<AutotuneResult> profile_results;

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();

  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  auto options = HloPrintOptions::Canonical();
  options.set_print_backend_config(true);
  const auto canonical_hlo = instr->ToString(options);

  string blas_version;
  if (auto* blas = stream_exec_->AsBlas()) {
    (void)blas->GetVersion(&blas_version);
  }

  absl::Span<const AlgorithmDesc> blacklisted_algos =
      GetBlacklistedConvAlgorithms(GetComputeCapability(stream_exec_),
                                   GetCudnnVersion(stream_exec_), blas_version,
                                   canonical_hlo);

  for (const AlgorithmDesc& alg : GetAlgorithms(kind, stream_exec_)) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL(
        absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                     AlgorithmToString(alg)),
        2);

    if (absl::c_linear_search(blacklisted_algos, alg)) {
      LOG(INFO) << "Omitted potentially buggy algorithm "
                << AlgorithmToString(alg) << " for conv " << instr->ToString();
      continue;
    }

    se::RedzoneAllocator scratch_allocator(
        stream, allocator, PtxOptsFromConfig(hlo_module_config));
    se::dnn::ProfileResult profile_result;
    VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
            << instr->ToString();

    // Use assignment instead of brace-list to make GCC 4.9 happy.
    RunConvOptions options;
    options.profile_result = &profile_result;
    options.algo_override = alg;

    // Warmup with first run.
    Status launch_status =
        RunGpuConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                   &scratch_allocator, stream, options);

    if (!launch_status.ok()) {
      // TODO(AmosChenYQ): Remove this after figuring out why getting fewer
      // results than need to profile
      VLOG(1) << "Algorithm " << AlgorithmToString(alg)
              << " warm up's launch is not ok";
      continue;
    }

    // Second run provides reliable timings.
    launch_status =
        RunGpuConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                   &scratch_allocator, stream, options);

    if (!launch_status.ok()) {
      // TODO(AmosChenYQ): Remove this after figuring out why getting fewer
      // results than need to profile
      VLOG(1) << "Algorithm " << AlgorithmToString(alg)
              << " profile's launch is not ok";
      continue;
    }

    if (!profile_result.is_valid()) {
      // TODO(AmosChenYQ): Remove this after figuring out why getting fewer
      // results than need to profile
      VLOG(1) << "Algorithm " << AlgorithmToString(alg)
              << " profile result is not valid";
      VLOG(1) << std::boolalpha << "Algorithm " << AlgorithmToString(alg)
              << " exceeds limits "
              << (profile_result.elapsed_time_in_ms() ==
                  std::numeric_limits<float>::max());
      continue;
    }

    profile_results.emplace_back();
    AutotuneResult& result = profile_results.back();
    result.mutable_conv()->set_algorithm(alg.algo_id());
    result.mutable_conv()->set_tensor_ops_enabled(alg.tensor_ops_enabled());

    int64 scratch_bytes_used =
        scratch_allocator.TotalAllocatedBytesExcludingRedzones();
    result.set_scratch_bytes(scratch_bytes_used);
    *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    if (!check_conv) {
      continue;
    }

    // Check for writes to redzones.
    TF_ASSIGN_OR_RETURN(bool input_output_allocator_redzone_clear,
                        CheckRedzones(input_output_allocator, stream,
                                      "input/output", instr, &result));

    TF_ASSIGN_OR_RETURN(
        bool scratch_allocator_redzone_clear,
        CheckRedzones(scratch_allocator, stream, "scratch", instr, &result));

    if (!input_output_allocator_redzone_clear ||
        !scratch_allocator_redzone_clear) {
      AlgorithmBlacklist proto;
      auto entry = proto.add_entries();
      entry->set_hlo(canonical_hlo);
      *entry->mutable_cc() = GetComputeCapability(stream_exec_);
      *entry->mutable_cudnn_version() = GetCudnnVersion(stream_exec_);
      entry->set_blas_version(blas_version);
      auto algo = entry->add_algos();
      algo->set_id(alg.algo_id());
      algo->set_tensor_ops(alg.tensor_ops_enabled());

      LOG(ERROR)
          << "To blacklist this algorithm for this convolution, "
             "copy-paste the following "
             "proto to the blacklist file pointed by XLA_FLAGS "
             "--xla_gpu_algorithm_blacklist_path="
          << GetDebugOptionsFromFlags().xla_gpu_algorithm_blacklist_path()
          << " : " << proto.ShortDebugString();
      continue;
    }

    if (comparator.has_value()) {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::CompareEqual", 2);
      StatusOr<bool> compare_result = comparator->CompareEqual(
          stream, reference_result_buffer, result_buffer);
      if (!compare_result.ok()) {
        LOG(ERROR) << "Unable to compare " << AlgorithmToString(first_algorithm)
                   << " against " << AlgorithmToString(alg) << " for "
                   << instr->ToString() << ": " << compare_result.status();
        if (compare_result.status().code() ==
            tensorflow::error::RESOURCE_EXHAUSTED) {
          // Possibly OOM. Propagate the error.
          return compare_result.status();
        }
        CHECK(!crash_on_checking_failure);
      } else if (!compare_result.ValueOrDie()) {
        LOG(ERROR)
            << "Results mismatch between different convolution algorithms. "
               "This is likely a bug/unexpected loss of precision in cudnn.\n"
            << instr->ToString() << " for "
            << AlgorithmToString(first_algorithm) << " vs "
            << AlgorithmToString(alg);
        PrintPlatformInfo(stream);
        VLOG(1) << "Full module on failure: \n"
                << instr->GetModule()->ToString();
        auto* fail = result.mutable_failure();
        fail->set_kind(AutotuneResult::WRONG_RESULT);
        fail->set_buffer_address(
            reinterpret_cast<uint64>(result_buffer.opaque()));
        auto* reference_conv = fail->mutable_reference_conv();
        reference_conv->set_algorithm(first_algorithm.algo_id());
        reference_conv->set_tensor_ops_enabled(
            first_algorithm.tensor_ops_enabled());
      }
    } else {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::Create", 2);
      comparator.emplace(result_shape, hlo_module_config);
      TF_ASSIGN_OR_RETURN(
          reference_result_buffer,
          input_output_allocator.AllocateBytes(result_buffer.size()));
      stream->ThenMemcpy(&reference_result_buffer, result_buffer,
                         result_buffer.size());
      first_algorithm = alg;
    }
  }

  // TODO(AmosChenYQ): Remove this after figuring out why getting fewer 
  // results than need to profile
  VLOG(1) << "The number of profile result is " << profile_results.size();

  // Log the autotuning result.
  {
    tensorflow::AutotuningLog log;
    {
      ConvInstructionLog instr_log;
      *instr_log.mutable_instruction() = instr->ToProto();
      for (int i = 0; i < instr->operand_count(); i++) {
        *instr_log.add_operand_shapes() = instr->operand(i)->shape().ToProto();
        instr_log.add_operand_addresses(
            reinterpret_cast<uint64>(operand_buffers[i].opaque()));
      }
      instr_log.set_result_address(
          reinterpret_cast<uint64>(result_buffer.opaque()));
      log.mutable_instr()->PackFrom(instr_log);
    }
    for (const auto& profile : profile_results) {
      *log.add_results() = profile;
    }
    *log.mutable_compute_capability() = GetComputeCapability(stream_exec_);
    *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec_);
    log.set_device_pci_bus_id(
        stream_exec_->GetDeviceDescription().pci_bus_id());
    log.set_blas_version(blas_version);
    VLOG(1) << "Autotuning result: " << log.ShortDebugString();
    // If we crash on checking failure, we are in a testing/benchmark mode, thus
    // omitting logging through the logger.
    if (!crash_on_checking_failure) {
      tensorflow::Logger::GetSingleton()->LogProto(log);
    }
  }

  // Crash on miscompares and redzone violations if desired.  Do this after
  // logging the autotuning results, otherwise we won't get any data!
  for (const auto& result : profile_results) {
    if (result.has_failure()) {
      CHECK(!crash_on_checking_failure);
    }
  }

  // Choose the fastest convolution that doesn't produce a REDZONE_MODIFIED
  // error.
  //
  // TODO(jlebar): We ought to be able to detect redzone reads by noticing NaNs
  // in the output of the conv and skip those.
  //
  // For now, we ignore WRONG_RESULT failures because false-positives are
  // possible (e.g. perhaps the reference algorithm is the one that's
  // incorrect!).  But we don't ignore REDZONE_MODIFIED failures because they're
  // quite severe and can be detected with high accuracy.
  std::vector<AutotuneResult> filtered_results;
  absl::c_copy_if(
      profile_results, std::back_inserter(filtered_results),
      [](const AutotuneResult& r) {
        return !(r.has_failure() &&
                 r.failure().kind() != AutotuneResult::WRONG_RESULT);
      });
  if (filtered_results.empty()) {
    return InternalError(
        "All algorithms tried for convolution %s failed. Falling back to "
        "default algorithm. ",
        instr->ToString());
  }

  auto selected_result = filtered_results.begin();
  if (!RequireCudnnDeterminism()) {
    VLOG(1) << "There is no need to guarantee the determinism of the cudnn "
               "algorithm used, the least time-consuming algorithm is used";
    selected_result = absl::c_min_element(
        filtered_results,
        [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
          return tensorflow::proto_utils::FromDurationProto(lhs.run_time()) <
                 tensorflow::proto_utils::FromDurationProto(rhs.run_time());
        });
  }

  return *selected_result;
}
#endif

StatusOr<tensorflow::AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheRocm(
    const HloCustomCallInstruction* instr, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr->ToString()));

  const auto device_ordinal = stream_exec_->device_ordinal();
  std::vector<se::DeviceMemoryBase> operand_buffers;

  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  const auto initialize_buffer = [stream](DeviceMemoryBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    stream->ThenMemZero(&buffer, buffer.size());
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers.push_back(buffer);
  }

  TF_ASSIGN_OR_RETURN(
      auto result_buffer,
      input_output_allocator.AllocateBytes(
          ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(0))));
  initialize_buffer(result_buffer);

  TF_ASSIGN_OR_RETURN(std::vector<se::dnn::ProfileResult> algorithms,
                      GetAlgorithms(instr, absl::MakeSpan(operand_buffers),
                                    result_buffer, stream_exec_, stream));

  std::vector<AutotuneResult> profile_results;

  if (algorithms.size() == 1) {
    auto profile_result = algorithms[0];
    profile_results.emplace_back();
    auto& result = profile_results.back();
    result.mutable_conv()->set_algorithm(profile_result.algorithm().algo_id());
    result.mutable_conv()->set_tensor_ops_enabled(
        profile_result.algorithm().tensor_ops_enabled());

    result.set_scratch_bytes(profile_result.scratch_size());
    *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));
  } else {
    for (const auto& miopen_alg : algorithms) {
      const auto& alg = miopen_alg.algorithm();
      XLA_SCOPED_LOGGING_TIMER_LEVEL(
          absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                       AlgorithmToString(alg)),
          2);

      ScratchAllocator scratch_allocator(device_ordinal, allocator);
      se::dnn::ProfileResult profile_result;
      VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
              << instr->ToString();

      // Use assignment instead of brace-list to make GCC 4.9 happy.
      RunConvOptions options;
      options.profile_result = &profile_result;
      options.algo_override = alg;
      Status launch_status =
          RunGpuConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                     &scratch_allocator, stream, options);

      if (!launch_status.ok()) {
        continue;
      }

      if (!profile_result.is_valid()) {
        continue;
      }

      profile_results.emplace_back();
      AutotuneResult& result = profile_results.back();
      result.mutable_conv()->set_algorithm(alg.algo_id());
      result.mutable_conv()->set_tensor_ops_enabled(alg.tensor_ops_enabled());

      int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
      result.set_scratch_bytes(scratch_bytes_used);
      *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    }
  }
  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return tensorflow::proto_utils::FromDurationProto(lhs.run_time()) <
               tensorflow::proto_utils::FromDurationProto(rhs.run_time());
      });

  if (best_result != profile_results.end()) {
    return *best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<bool> GpuConvAlgorithmPicker::RunOnInstruction(HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    LOG(WARNING) << "Failed to determine best cudnn convolution algorithm: "
                 << best_algo_or.status()
                 << "\n\nConvolution performance may be suboptimal.";
    return false;
  }

  auto best_algo = std::move(best_algo_or).ValueOrDie();
  VLOG(2) << "Setting cudnn conv to use algorithm "
          << best_algo.conv().algorithm() << " and "
          << NumBytesToString(best_algo.scratch_bytes())
          << " of scratch memory: " << instr->ToString()
          << " tensor_ops_enabled: " << best_algo.conv().tensor_ops_enabled();

  // Replace instr with a new CustomCall which has the correct algorithm, and
  // whose output shape has the appropriate amount of scratch memory.
  HloComputation* computation = instr->parent();
  Shape new_call_shape = ShapeUtil::MakeTupleShape(
      {instr->shape().tuple_shapes(0),
       ShapeUtil::MakeShape(U8, {best_algo.scratch_bytes()})});

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());
  backend_config.set_algorithm(best_algo.conv().algorithm());
  backend_config.set_tensor_ops_enabled(best_algo.conv().tensor_ops_enabled());

  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, instr->operands()));

  TF_RETURN_IF_ERROR(new_call->set_backend_config(backend_config));

  VLOG(2) << "Replacing convolution " << instr->ToString() << " with "
          << new_call->ToString();
  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(
          {computation->AddInstruction(HloInstruction::CreateGetTupleElement(
               new_call_shape.tuple_shapes(0), new_call, 0)),
           computation->AddInstruction(HloInstruction::CreateConstant(
               LiteralUtil::CreateR1<uint8>({})))}));

  TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, new_tuple));
  return true;
}

StatusOr<bool> GpuConvAlgorithmPicker::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* instr : computation->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(instr);
    }
  }

  bool changed = false;
  for (auto* instr : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr));
    changed |= result;
  }
  return changed;
}

StatusOr<bool> GpuConvAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("GpuConvAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "Convolution auto-tuning disabled, GpuConvAlgorithmPicker "
               "returning early.";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }

  return changed;
}

/* static */ ConvAutotuneCache& ConvAutotuneCacheSingleton::GetInstance() {
  // static std::shared_ptr<ConvAutotuneCache> conv_autotune_cache_ptr =
  //     std::make_shared<ConvAutotuneCache>();
  // return conv_autotune_cache_ptr.get();
  static ConvAutotuneCache conv_autotune_cache;
  return conv_autotune_cache;
}

/* static */ uint64 ConvAutotuneCache::ConvAutotuneCacheKeyHasher(
    const se::StreamExecutor* stream_exec, const HloInstruction* instr) {
  auto options = HloPrintOptions::Canonical();
  options.set_print_backend_config(true);
  uint64 hashed_instr = Hash64(instr->ToString(options));
  uint64 hashed_platform_name = Hash64(stream_exec->platform()->Name());
  return Hash64Combine(hashed_instr, hashed_platform_name);
}

/* static */ ConvAutotuneCacheValue
ConvAutotuneCache::CreateConvAutotuneCacheValue(
    StatusOr<AutotuneResult> result_or, const se::StreamExecutor* stream_exec,
    const HloInstruction* instr) {
  auto options = HloPrintOptions::Canonical();
  options.set_print_backend_config(true);
  ConvAutotuneCacheValue conv_autotune_cache;
  conv_autotune_cache.set_instr(instr->ToString(options));
  conv_autotune_cache.set_platform_name(stream_exec->platform()->Name());
  if (result_or.ok()) {
    (*conv_autotune_cache.mutable_conv_autotune_result()) =
        result_or.ValueOrDie();
  } else {
    Status s = result_or.status();
    conv_autotune_cache.mutable_status()->set_code(
        static_cast<tensorflow::error::Code>(s.code()));
    conv_autotune_cache.mutable_status()->set_message(s.error_message());
  }
  return std::move(conv_autotune_cache);
}

bool ConvAutotuneCache::LookUpCache(uint64 key,
                                    ConvAutotuneCacheValue& cache_value) {
  if (conv_autotune_cache_proto_.conv_autotune_cache_map().count(key) > 0) {
    cache_value = conv_autotune_cache_proto_.conv_autotune_cache_map().at(key); 
    return true;
  }
  return false;
}

bool ConvAutotuneCache::AddToCache(uint64 key,
                                   const ConvAutotuneCacheValue& cache_value) {
  if (conv_autotune_cache_proto_.conv_autotune_cache_map().count(key) > 0) {
    return false;
  }
  return conv_autotune_cache_proto_.mutable_conv_autotune_cache_map()
      ->insert({key, cache_value})
      .second;
}

ConvAutotuneCache::ConvAutotuneCache() {
  VLOG(1) << "ConvAutotune constructor";
  cache_hits = 0;
  cache_misses = 0;
  autotune_cache_filename_ =
      GetDebugOptionsFromFlags()
          .xla_gpu_conv_algorithm_autotune_cache_filename();
  in_use_ = !autotune_cache_filename_.empty();
  if (in_use_ &&
      tensorflow::Env::Default()->FileExists(autotune_cache_filename_).ok()) {
    VLOG(1) << "Loading conv autotune cache from " << autotune_cache_filename_;
    std::string serialized_proto_str;
    tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                 autotune_cache_filename_,
                                 &serialized_proto_str);
    conv_autotune_cache_proto_.ParseFromString(serialized_proto_str);
  }
}

ConvAutotuneCache::~ConvAutotuneCache() {
  VLOG(1) << "ConvAutotune destructor";
  if (in_use_ && !autotune_cache_filename_.empty()) {
    VLOG(1) << "Commiting conv autotune cache to " << autotune_cache_filename_;
    std::string serialized_proto_str;
    conv_autotune_cache_proto_.SerializeToString(&serialized_proto_str);
    tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                  autotune_cache_filename_,
                                  serialized_proto_str);
  }
}

}  // namespace gpu
}  // namespace xla
