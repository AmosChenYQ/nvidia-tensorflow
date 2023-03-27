/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"

#include <limits>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

namespace xla {
namespace gpu {

using tensorflow::Hash64;
using tensorflow::Hash64Combine;

using tensorflow::AutotuneResult;

static tensorflow::mutex gemm_autotune_cache_mu(tensorflow::LINKER_INITIALIZED);
// Experimentally tries to pick the best algorithm for the given gemm.
//
// This may fail under perfectly normal circumstances.  In particular, it will
// fail if the program was built with < CUDA 8 or if we're using a gpu older
// than sm_50 -- in both cases, cublas doesn't support gemm-with-algorithm at
// all.
static StatusOr<absl::optional<se::blas::AlgorithmType>> DoUncachedGemmAutotune(
    const HloInstruction* gemm, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase reference_result_buffer, se::Stream* stream,
    const se::RedzoneAllocator& allocator, const BufferComparator& comparator,
    bool crash_on_checking_failure) {
  if (!stream->parent()->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  GemmBackendConfig backend_config =
      gemm->backend_config<GemmBackendConfig>().ValueOrDie();
  const int32 cublas_autotune_level =
      gemm->GetModule()->config().debug_options().xla_gpu_autotune_level();
  const bool reinit_cublas_data = cublas_autotune_level > 2;
  const bool check_cublas = cublas_autotune_level > 3;

  VLOG(1) << "Starting autotune of GemmThunk " << gemm->ToString();

  std::vector<se::blas::AlgorithmType> algorithms;
  CHECK(stream->parent()->GetBlasGemmAlgorithms(&algorithms));

  absl::optional<se::blas::AlgorithmType> first_algorithm;
  std::vector<AutotuneResult> profile_results;

  for (se::blas::AlgorithmType algorithm : algorithms) {
    // Make sure the output buffer always has the same value if we use
    // the bias parameter.
    if (reinit_cublas_data && backend_config.beta() != 0) {
      int64 rng_state = 0;
      InitializeBuffer(stream, gemm->shape().element_type(), &rng_state,
                       output_buffer);
    }
    se::blas::ProfileResult profile_result;

    // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will fail
    // for all algorithms if we're targeting < sm_50.  But because we pass a
    // non-null ProfileResult, DoGemmWithAlgorithm should always return true,
    // and the actual success-ness is returned in ProfileResult::is_valid.
    CHECK(RunGemm(gemm, backend_config, lhs_buffer, rhs_buffer, output_buffer,
                  stream,
                  /*implements_whole_instruction=*/true,
                  /*profiler=*/nullptr,
                  /*profile_result=*/&profile_result, algorithm)
              .ok());

    if (!profile_result.is_valid()) {
      // Unsupported algorithm.
      continue;
    }

    profile_results.emplace_back();
    AutotuneResult& result = profile_results.back();
    result.mutable_gemm()->set_algorithm(algorithm);

    VLOG(1) << "cublas gemm algorithm " << algorithm << " took "
            << profile_result.elapsed_time_in_ms() << "ms" << std::endl;

    *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    if (!check_cublas) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
        allocator.CheckRedzones());
    if (!rz_check_status.ok()) {
      result.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
      *result.mutable_failure()->mutable_msg() =
          rz_check_status.RedzoneFailureMsg();
      LOG(ERROR) << "Detected cuBLAS out-of-bounds write in gemm buffer";
      CHECK(!crash_on_checking_failure);
      continue;
    }

    if (!first_algorithm) {
      // First run: set the reference result buffer.
      CHECK(reference_result_buffer.size() == output_buffer.size());
      stream->ThenMemcpy(&reference_result_buffer, output_buffer,
                         output_buffer.size());
      first_algorithm.emplace(algorithm);
    } else {
      // Perform the comparison.
      TF_ASSIGN_OR_RETURN(bool compare_result,
                          comparator.CompareEqual(stream, output_buffer,
                                                  reference_result_buffer));
      if (!compare_result) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision "
                   << "in cuBLAS.";
        CHECK(!crash_on_checking_failure);

        result.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
            *first_algorithm);
      }
    }
  }

  tensorflow::AutotuningLog log;
  for (const AutotuneResult& profile : profile_results) {
    *log.add_results() = profile;
  }
  if (!crash_on_checking_failure) {
    tensorflow::Logger::GetSingleton()->LogProto(log);
  }

  // Choose fastest correct GEMM, but allow for incorrect results (since the
  // reference result is chosen arbitrary).
  auto has_failure = [](const AutotuneResult& r) {
    return r.has_failure() &&
           r.failure().kind() != AutotuneResult::WRONG_RESULT;
  };

  auto result_comparison_key = [&has_failure](const AutotuneResult& r) {
    return std::make_tuple(
        has_failure(r),
        tensorflow::proto_utils::FromDurationProto(r.run_time()));
  };
  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return result_comparison_key(lhs) < result_comparison_key(rhs);
      });

  if (best_result != profile_results.end() && !has_failure(*best_result)) {
    return {best_result->gemm().algorithm()};
  }

  VLOG(1) << "Unable to autotune cuBLAS gemm on stream " << stream
          << " none of the " << algorithms.size() << " ran successfully";
  return {absl::nullopt};
}

static StatusOr<absl::optional<se::blas::AlgorithmType>> DoGemmAutotune(
    const HloInstruction* instr, const HloInstruction* lhs,
    const HloInstruction* rhs, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase reference_result_buffer, se::Stream* stream,
    bool crash_on_checking_failure, const se::RedzoneAllocator& allocator,
    const BufferComparator& comparator) {
  // Don't run autotuning concurrently on the same GPU.
  tensorflow::mutex_lock gpu_lock = LockGpu(stream->parent());

  GemmBackendConfig gemm_config =
      instr->backend_config<GemmBackendConfig>().ValueOrDie();

  auto log_gemm_instr_and_operands_msg = [](const HloInstruction* instr,
                                           const HloInstruction* lhs,
                                           const HloInstruction* rhs) {
    auto options = HloPrintOptions::Canonical();
    options.set_print_backend_config(true);
    VLOG(1) << "Gemm instr in canonical string:";
    VLOG(1) << instr->ToString();
    VLOG(1) << "Lhs instr in canonical string:";
    VLOG(1) << lhs->ToString();
    VLOG(1) << "Rhs instr in canonical string:";
    VLOG(1) << rhs->ToString();
  };
  log_gemm_instr_and_operands_msg(instr, lhs, rhs);
  auto hashed_cache_key = GemmAutotuneCache::GemmAutotuneCacheKeyHasher(
      stream->parent(), lhs->shape(), rhs->shape(), instr->shape(),
      gemm_config);
  VLOG(1) << "hashed_cache_key: " << hashed_cache_key;

  tensorflow::mutex_lock cache_lock(gemm_autotune_cache_mu);
  absl::optional<se::blas::AlgorithmType> result;
  bool is_found_cache = GemmAutotuneCacheSingleton::GetInstance()->LookupCache(
      hashed_cache_key, result);
  int64 gemm_autotune_requests =
      GemmAutotuneCacheSingleton::GetInstance()->cache_hits +
      GemmAutotuneCacheSingleton::GetInstance()->cache_misses;
  if (gemm_autotune_requests) {
    VLOG(1) << "Autotuning cache hits/(hits + misses): "
            << GemmAutotuneCacheSingleton::GetInstance()->cache_hits << "/"
            << gemm_autotune_requests;
  }

  if (is_found_cache) {
    GemmAutotuneCacheSingleton::GetInstance()->cache_hits++;
    VLOG(1) << "Autotuning cache hit, using algorithm: "
            << (result.has_value() ? absl::StrCat(result.value())
                                   : "<generic>");
    return result;
  }
  GemmAutotuneCacheSingleton::GetInstance()->cache_misses++;
  VLOG(1) << "Autotuning cache miss";

  int64 batch_size = gemm_config.batch_size();

  // TODO(AmosChenYQ): Test the correctness of autotune result.
  TF_ASSIGN_OR_RETURN(result, DoUncachedGemmAutotune(
                                  instr, lhs_buffer, rhs_buffer, output_buffer,
                                  reference_result_buffer, stream, allocator,
                                  comparator, crash_on_checking_failure));

  CHECK(GemmAutotuneCacheSingleton::GetInstance()->AddToCache(
      hashed_cache_key, GemmAutotuneCache::CreateGemmAutotuneCacheValue(
                            stream->parent(), lhs->shape(), rhs->shape(),
                            instr->shape(), gemm_config, result)));
  return result;
}

static StatusOr<bool> RunOnInstruction(HloInstruction* instr,
                                       se::StreamExecutor* executor,
                                       se::DeviceMemoryAllocator* allocator) {
  if (allocator == nullptr) {
    allocator = executor->GetAllocator();
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(executor->device_ordinal()));

  const HloModuleConfig& hlo_module_config = instr->GetModule()->config();
  const bool init_cublas_data =
      hlo_module_config.debug_options().xla_gpu_autotune_level() > 1;
  se::RedzoneAllocator input_output_allocator(
      stream, allocator, PtxOptsFromConfig(hlo_module_config),
      /*memory_limit=*/std::numeric_limits<int64>::max());

  BufferComparator comparator(instr->shape(), hlo_module_config);

  int64 rng_state = 0;
  auto get_initialized_buffer =
      [&](const HloInstruction* op) -> StatusOr<se::DeviceMemoryBase> {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(op->shape())));
    if (init_cublas_data) {
      InitializeBuffer(stream, op->shape().element_type(), &rng_state, buffer);
    }
    return buffer;
  };

  GemmBackendConfig gemm_config =
      instr->backend_config<GemmBackendConfig>().ValueOrDie();
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase lhs_buffer,
                      get_initialized_buffer(lhs));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase rhs_buffer,
                      get_initialized_buffer(rhs));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase output_buffer,
                      get_initialized_buffer(instr));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase reference_result_buffer,
                      get_initialized_buffer(instr));

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();

  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  TF_ASSIGN_OR_RETURN(
      absl::optional<se::blas::AlgorithmType> gemm_algorithm,
      DoGemmAutotune(instr, lhs, rhs, lhs_buffer, rhs_buffer, output_buffer,
                     reference_result_buffer, stream, crash_on_checking_failure,
                     input_output_allocator, comparator));

  // We update instruction->backend_config(); if no algorithms are supported,
  // a different API is used, which does not require specifying an algorithm.
  GemmBackendConfig updated_config = gemm_config;
  if (gemm_algorithm) {
    updated_config.set_selected_algorithm(*gemm_algorithm);
  }
  TF_RETURN_IF_ERROR(instr->set_backend_config(updated_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

static StatusOr<bool> RunOnComputation(HloComputation* computation,
                                       se::StreamExecutor* se,
                                       se::DeviceMemoryAllocator* allocator) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, se, allocator));
      changed |= result;
    }
  }
  return changed;
}

StatusOr<bool> GemmAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("GemmAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, stream_exec_, allocator_));
    changed |= result;
  }
  return changed;
}

/* static */ GemmAutotuneCache* GemmAutotuneCacheSingleton::GetInstance() {
  static std::shared_ptr<GemmAutotuneCache> gemm_autotune_cache_ptr =
      std::make_shared<GemmAutotuneCache>();
  return gemm_autotune_cache_ptr.get();
}

// TODO(AmosChenYQ): Use template variadic functions to replace this. 
/* static */ uint64 GemmAutotuneCache::GemmAutotuneCacheKeyHasher(
    se::StreamExecutor* stream_exec, Shape lhs_shape, Shape rhs_shape,
    Shape instr_shape, GemmBackendConfig gemm_config) {
  uint64 hashed_platform_name = Hash64(stream_exec->platform()->Name());
  uint64 hashed_lhs_shape = Hash64(lhs_shape.DebugString());
  uint64 hashed_rhs_shape = Hash64(rhs_shape.DebugString());
  uint64 hashed_instr_shape = Hash64(instr_shape.DebugString());
  uint64 hashed_gemm_config = Hash64(gemm_config.DebugString());
  return Hash64Combine(
      Hash64Combine(
          Hash64Combine(Hash64Combine(hashed_lhs_shape, hashed_rhs_shape),
                        hashed_instr_shape),
          hashed_gemm_config),
      hashed_platform_name);
}

/* static */ GemmAutotuneCacheValue
GemmAutotuneCache::CreateGemmAutotuneCacheValue(
    se::StreamExecutor* stream_exec, Shape lhs_shape, Shape rhs_shape,
    Shape instr_shape, GemmBackendConfig gemm_config,
    absl::optional<se::blas::AlgorithmType> result) {
  GemmAutotuneCacheValue cache_value_to_add;
  cache_value_to_add.set_platform_name(stream_exec->platform()->Name());
  cache_value_to_add.mutable_lhs_shape()->CopyFrom(lhs_shape.ToProto());
  cache_value_to_add.mutable_rhs_shape()->CopyFrom(rhs_shape.ToProto());
  cache_value_to_add.mutable_instr_shape()->CopyFrom(instr_shape.ToProto());
  cache_value_to_add.mutable_gemm_backend_config()->CopyFrom(gemm_config);
  if (result.has_value()) {
    cache_value_to_add.set_selected_algorithm(result.value());
  } else {
    cache_value_to_add.clear_selected_algorithm();
  }
  return cache_value_to_add;
}

bool GemmAutotuneCache::LookupCache(
    uint64 key, absl::optional<se::blas::AlgorithmType>& result) {
  if (gemm_autotune_cache_proto_.gemm_autotune_cache_map().count(key) > 0) {
    GemmAutotuneCacheValue cache_value =
        (*gemm_autotune_cache_proto_.mutable_gemm_autotune_cache_map())[key];
    if (cache_value.algorithm_case() ==
        GemmAutotuneCacheValue::ALGORITHM_NOT_SET) {
      result = absl::nullopt;
    } else {
      result = cache_value.selected_algorithm();
    }
    return true;
  }
  return false;
}

bool GemmAutotuneCache::AddToCache(uint64 key,
                                   const GemmAutotuneCacheValue& cache_value) {
  if (gemm_autotune_cache_proto_.gemm_autotune_cache_map().count(key) > 0) {
    return false;
  }
  return gemm_autotune_cache_proto_.mutable_gemm_autotune_cache_map()
      ->insert({key, cache_value})
      .second;
}

GemmAutotuneCache::GemmAutotuneCache() {
  VLOG(1) << "GemmAutotuneCache constructor";
  cache_hits = 0;
  cache_misses = 0;
  autotune_cache_filename_ =
      GetDebugOptionsFromFlags()
          .xla_gpu_gemm_algorithm_autotune_cache_filename();
  in_use_ = !autotune_cache_filename_.empty();
  if (in_use_ &&
      tensorflow::Env::Default()->FileExists(autotune_cache_filename_).ok()) {
    VLOG(1) << "Loading autotune cache from " << autotune_cache_filename_;
    std::string serialized_proto_str;
    tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                 autotune_cache_filename_,
                                 &serialized_proto_str);
    VLOG(1) << "Read file content:\n" << serialized_proto_str;
    gemm_autotune_cache_proto_.ParseFromString(serialized_proto_str);
    VLOG(1) << "Proto serialized result:\n"
            << gemm_autotune_cache_proto_.DebugString();
  }
}

GemmAutotuneCache::~GemmAutotuneCache() {
  VLOG(1) << "GemmAutotuneCache destructor";
  if (in_use_ && !autotune_cache_filename_.empty()) {
    VLOG(1) << "Commiting autotune cache to " << autotune_cache_filename_;
    std::string serialized_proto_str;
    gemm_autotune_cache_proto_.SerializeToString(&serialized_proto_str);
    tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                  autotune_cache_filename_,
                                  serialized_proto_str);
  }
}

}  // namespace gpu
}  // namespace xla
