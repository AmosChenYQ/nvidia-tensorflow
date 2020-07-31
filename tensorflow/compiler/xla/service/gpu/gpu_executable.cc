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

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"

#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_debug_info_manager.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/annotation.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/platform.h"
// TODO(benbarsdell): Plumb Graph APIs into stream_executor (and implement
// wrapper classes for Graph/GraphExec) instead of using this directly.
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"

namespace xla {
namespace gpu {
namespace {

using tensorflow::tracing::ScopedAnnotation;

// A helper function to decide whether to use GPU graph capture, which can
// reduce GPU launch latency overheads in some cases.
bool GpuGraphCaptureEnabled() {
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(
        tensorflow::ReadBoolFromEnvVar("TF_XLA_ENABLE_GPU_GRAPH_CAPTURE",
                                       /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
}

int64 GpuExecGraphCacheSize() {
  int64 cache_size = 0;
  TF_CHECK_OK(
      tensorflow::ReadInt64FromEnvVar("TF_XLA_GPU_EXEC_GRAPH_CACHE_SIZE",
                                      /*default_val=*/100, &cache_size));

  return cache_size;
}

bool IsThunkSafeForGpuGraphCapture(const Thunk* thunk) {
  // Thunks that synchronize with the host (i.e., call BlockHostUntilDone)
  // cannot be used with graph capture.
  static const absl::flat_hash_set<Thunk::Kind> thunk_kinds_safe_for_capture = {
      Thunk::kCholesky,
      Thunk::kCollectivePermute,
      Thunk::kConvolution,
      Thunk::kCopy,
      Thunk::kCudnnBatchNormBackward,
      Thunk::kCudnnBatchNormForwardInference,
      Thunk::kCudnnBatchNormForwardTraining,
      Thunk::kFft,
      Thunk::kGemm,
      Thunk::kKernel,
      Thunk::kMemset32BitValue,
      Thunk::kMemzero,
      Thunk::kNcclAllReduce,
      Thunk::kReplicaId,
      Thunk::kTriangularSolve,
      Thunk::kTuple,
  };
  if (thunk->kind() == Thunk::kSequential) {
    const auto* seq_thunk = static_cast<const SequentialThunk*>(thunk);
    for (const std::unique_ptr<Thunk>& sub_thunk : seq_thunk->thunks()) {
      if (!IsThunkSafeForGpuGraphCapture(sub_thunk.get())) return false;
    }
    return true;
  }
  if (dynamic_cast<const HostToDeviceCopyThunk*>(thunk)) {
    VLOG(1) << "HostToDeviceCopyThunk is not supported for a graph capture";
    return false;
  }

  if (!thunk_kinds_safe_for_capture.count(thunk->kind())) {
    VLOG(1) << ThunkKindToString(thunk->kind())
            << " is not supported for graph capture";
    return false;
  }

  return true;
}


}  // namespace

// Implementation note: HLO profiling is always enabled for GPU executables,
// since we can use timers around thunks.
GpuExecutable::GpuExecutable(
    const string& text, const std::vector<uint8>& binary,
    GpuVersion gpu_version, std::unique_ptr<const ThunkSchedule> thunk_schedule,
    std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      text_(text),
      binary_(binary),
      gpu_version_(gpu_version),
      thunk_schedule_(std::move(thunk_schedule)),
      assignment_(std::move(assignment)) {
  CHECK(has_module() && assignment_);
  GpuDebugInfoManager::Get()->RegisterModule(module().name(), shared_module(),
                                             assignment_);
  ComputeThunkAnnotations();
  can_use_gpu_graph_capture_ = CanUseGpuGraphCapture();
}

GpuExecutable::~GpuExecutable() {
  CHECK(has_module() && assignment_);
  GpuDebugInfoManager::Get()->UnregisterModule(module().name(), shared_module(),
                                               assignment_);

  {
    // We could have issued host->device mem copies in ResolveConstantGlobals.
    // Wait for those to finish so that we can safely deallocate the backing HLO
    // module.
    //
    // We need for the host->device memcpies to finish they are concurrently
    // reading memory (xla::Literal's) owned by the HLO module.
    tensorflow::mutex_lock lock(module_handle_mutex_);
    for (const auto& pair : module_globals_) {
      CHECK(pair.first->SynchronizeAllActivity());
    }
  }
  // float hit_rate = (graph_stats_.cache_hits / graph_stats_.times_called) *
  // 100;

  VLOG(1) << "For gpu_executable " << this
          << " Hits: " << graph_stats_.cache_hits
          << " Misses: " << graph_stats_.cache_miss
          << " Recent hash hits: " << graph_stats_.last_buf_key_hits
          << " called # " << graph_stats_.times_called;
  if (graph_stats_.times_called > 0) {
    VLOG(1) << "with hit rate of "
            << (graph_stats_.cache_hits * 100) / graph_stats_.times_called
            << "%";
  }
  VLOG(2) << "Most recent enqueued hash hits: "
          << graph_stats_.last_buf_key_hits;
  if (graph_stats_.cache_hits > 0) {
    VLOG(2) << " Most recent enqueued hash hit rate: "
            << (graph_stats_.last_buf_key_hits * 100) / graph_stats_.cache_hits;
  }
  /*while (!gpu_exec_graphs_.empty()) {
    auto* gpu_context = static_cast<stream_executor::gpu::GpuContext*>(
        gpu_exec_graphs_.begin()->first);
    VLOG(2) << "Cache size for gpu_executable " << this << " and gpu_context "
            << gpu_context << " is " << gpu_exec_graphs_[gpu_context].size();
    // TODO: Clean this up a bit.
    auto& map = gpu_exec_graphs_.begin()->second;
    while (!map.empty()) {
      auto* exec_graph =
          reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
              //&gpu_exec_graphs_.begin()->second);
              &map.begin()->second);
      using stream_executor::gpu::GpuDriver;
      GpuDriver::DestroyExecutableGraph(gpu_context, exec_graph);
      map.erase(map.begin());
    }
    gpu_exec_graphs_.erase(gpu_exec_graphs_.begin());
  }*/

  while (!gpu_exec_graphs_cache_.empty()) {
    auto* gpu_context = static_cast<stream_executor::gpu::GpuContext*>(
        gpu_exec_graphs_cache_.begin()->first);
    VLOG(2) << "Cache size for gpu_executable " << this << " and gpu_context "
            << gpu_context << " is "
            << gpu_exec_graphs_cache_[gpu_context].gpu_exec_graphs_.size();
    // TODO: Clean this up a bit.
    auto& cache = gpu_exec_graphs_cache_.begin()->second;
    auto& exec_graphs = cache.gpu_exec_graphs_;

    while (!exec_graphs.empty()) {
      auto* exec_graph =
          reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
              &exec_graphs.front());
      using stream_executor::gpu::GpuDriver;
      GpuDriver::DestroyExecutableGraph(gpu_context, exec_graph);
      exec_graphs.erase(exec_graphs.begin());
    }
    gpu_exec_graphs_cache_.erase(gpu_exec_graphs_cache_.begin());
  }
}

bool GpuExecutable::CanUseGpuGraphCapture() const {
  for (const Thunk* thunk : thunk_schedule_->TotalOrder()) {
    if (!IsThunkSafeForGpuGraphCapture(thunk)) return false;
  }
  return true;
}

void GpuExecutable::ComputeThunkAnnotations() {
  CanonicalNameMap canonical_name_map;
  for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
    const HloInstruction* hlo = thunk->hlo_instruction();
    CHECK(hlo);
    thunk_annotations_[thunk] =
        absl::StrFormat("Thunk#hlo_op=%s,hlo_module=%s#", hlo->name(),
                        hlo->GetModule()->name());
  }
}

Status GpuExecutable::CheckCompatibilityWithServiceExecutableRunOptions(
    const ServiceExecutableRunOptions* run_options) {
  se::Stream* main_stream = run_options->stream();

  stream_executor::PlatformKind platform_kind =
      main_stream->parent()->platform_kind();
  if (platform_kind == stream_executor::PlatformKind::kROCm) {
    int stream_isa_version;
    main_stream->parent()->GetDeviceDescription().rocm_amdgpu_isa_version(
        &stream_isa_version);
    GpuVersion amd_isa_version = stream_isa_version;
    TF_RET_CHECK(amd_isa_version == gpu_version_)
        << "AMDGPU GCN ISA version mismatch; expected {"
        << absl::get<int>(gpu_version_) << ", but was " << stream_isa_version;
  } else if (platform_kind == stream_executor::PlatformKind::kCuda) {
    std::pair<int, int> stream_compute_compatibility;
    main_stream->parent()->GetDeviceDescription().cuda_compute_capability(
        &stream_compute_compatibility.first,
        &stream_compute_compatibility.second);
    GpuVersion nvidia_compute_compatibility = stream_compute_compatibility;
    TF_RET_CHECK(nvidia_compute_compatibility == gpu_version_)
        << "Compute capability mismatch; expected {"
        << absl::get<std::pair<int, int>>(gpu_version_).first << ", "
        << absl::get<std::pair<int, int>>(gpu_version_).second << "}, but was {"
        << stream_compute_compatibility.first << ", "
        << stream_compute_compatibility.second << "}";
  } else {
    return InternalError("Unknown platform: %d", platform_kind);
  }

  return Status::OK();
}

Status GpuExecutable::ExecuteThunks(
    const ServiceExecutableRunOptions* run_options,
    const BufferAllocations& buffer_allocations, bool block_host_until_done,
    HloExecutionProfile* hlo_execution_profile) {
  TF_RETURN_IF_ERROR(
      CheckCompatibilityWithServiceExecutableRunOptions(run_options));
  GpuDebugInfoManager::Get()->OnModuleStart(module().name());
  auto cleanup = MakeCleanup(
      [&]() { GpuDebugInfoManager::Get()->OnModuleStop(module().name()); });

  se::Stream* main_stream = run_options->stream();
  se::StreamExecutor* executor = main_stream->parent();

  bool do_profile = hlo_execution_profile != nullptr;
  if (do_profile) {
    LOG(WARNING) << "PROFILING: profiling is enabled";
  }

  // TODO(benbarsdell): Enable for ROCm as well if it adds support for capture.
  bool use_gpu_graph_capture =
      GpuGraphCaptureEnabled() && can_use_gpu_graph_capture_ &&
      executor->platform_kind() == stream_executor::PlatformKind::kCuda;

  // Stream 0 indicates `main_stream` and substreams start from stream 1.
  std::vector<StreamPool::Ptr> sub_streams;
  sub_streams.reserve(thunk_schedule_->StreamCount() - 1);
  while (sub_streams.size() + 1 < thunk_schedule_->StreamCount()) {
    sub_streams.emplace_back();
    TF_ASSIGN_OR_RETURN(sub_streams.back(),
                        run_options->BorrowStream(executor->device_ordinal()));
  }

  se::Stream* capture_stream = main_stream;
  StreamPool::Ptr private_capture_stream;
  if (use_gpu_graph_capture) {
    // We need a private stream for capturing to avoid interference from other
    // threads.
    TF_ASSIGN_OR_RETURN(private_capture_stream,
                        run_options->BorrowStream(executor->device_ordinal()));
    capture_stream = private_capture_stream.get();
  }

  HloExecutionProfiler profiler(do_profile, hlo_execution_profile, main_stream,
                                sub_streams, hlo_module_->entry_computation());
  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  tensorflow::profiler::TraceMe hlo_module_activity(
      [&] { return absl::StrCat(hlo_module_->name(), ":XLA GPU module"); },
      tensorflow::profiler::TraceMeLevel::kInfo);

  auto bufs_key = buffer_allocations.Key();
  VLOG(3) << "******************* this, tmp buffer base: " << this
          << buffer_allocations.GetTempBufferBase().opaque()
          << " key hash: " << buffer_allocations.Key().hash();

  // TODO(benbarsdell): Clean this up once graph APIs are integrated into stream
  // executor.
  // TODO: Use this instead: auto gpu_stream =
  // se::gpu::AsGpuStreamValue(params.stream);
  CUstream main_cuda_stream = *reinterpret_cast<const CUstream*>(
      main_stream->implementation()->GpuStreamMemberHack());
  CUstream capture_cuda_stream = *reinterpret_cast<const CUstream*>(
      capture_stream->implementation()->GpuStreamMemberHack());
  stream_executor::gpu::GpuContext* gpu_context =
      static_cast<stream_executor::gpu::GpuExecutor*>(
          capture_stream->parent()->implementation())
          ->gpu_context();
  using stream_executor::gpu::GpuDriver;
  tensorflow::mutex_lock lock(module_handle_mutex_);
  /*auto& exec_graph =
      *reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
          &gpu_exec_graphs_[gpu_context][bufs_key]);
  if (gpu_exec_graphs_[gpu_context].size() > 2048) {
    LOG(WARNING) << "Too many exec graphs in cache! "
                 << gpu_exec_graphs_[gpu_context].size();
  }*/
  auto& graph_exec_cache = gpu_exec_graphs_cache_[gpu_context];
  graph_exec_cache.set_gpu_context(gpu_context);
  graph_exec_cache.set_cache_size(GpuExecGraphCacheSize());
  auto graph_exec = graph_exec_cache.get_exec_graph(bufs_key);
  auto& exec_graph =
      *reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(&graph_exec);

  tensorflow::mutex& mu = graph_stats_.graph_cache_mu;
  if (exec_graph && use_gpu_graph_capture) {
    mu.lock();
    graph_stats_.cache_hits++;
    graph_stats_.times_called++;
    if (bufs_key.hash() == graph_stats_.last_buf_key_hash) {
      graph_stats_.last_buf_key_hits++;
    }
    mu.unlock();
    VLOG(2) << "CACHE HIT -> Launching graph " << exec_graph << " blindly!"
            << " Hits: " << graph_stats_.cache_hits
            << " Misses: " << graph_stats_.cache_miss << " called # "
            << graph_stats_.times_called
            << " Cache size: " << graph_exec_cache.get_current_cache_size();
    VLOG(3) << "Most recent enqueued hash: " << bufs_key.hash()
            << "Most recent enqueued hash hits: "
            << graph_stats_.last_buf_key_hits;
    // main_stream->ThenWaitFor(&sub_streams); //(ToDo(amoitra): Check if this
    // is required)
    bool launch_success = GpuDriver::LaunchExecutableGraph(
        gpu_context, exec_graph, main_cuda_stream);
    if (!launch_success) {
      return InternalError("Failed to launch CUDA execution graph");
    }
    // GpuStream* cuda_stream = AsGpuStream(main_stream);
    // stream_executor::gpu::GpuStream* cuda_stream =
    //     static_cast<stream_executor::gpu::GpuStream*>(main_stream->implementation());
    // bool is_stream_idle = false;
    // while (!is_stream_idle){
    //   is_stream_idle = cuda_stream->IsIdle();
    //   LOG(INFO) << "Stream " << main_cuda_stream << " not idle yet";
    // }
  } else {
    if (use_gpu_graph_capture) {
      mu.lock();
      graph_stats_.cache_miss++;
      graph_stats_.times_called++;
      graph_stats_.last_buf_key_hash = bufs_key.hash();
      mu.unlock();
      VLOG(2) << "CACHE MISS"
              << " Hits: " << graph_stats_.cache_hits
              << " Misses: " << graph_stats_.cache_miss << " called # "
              << graph_stats_.times_called
              << " Cache size: " << graph_exec_cache.get_current_cache_size();
      VLOG(3) << "Most recently enqueued hash: " << bufs_key.hash()
              << "Most recent enqueued hash hits: "
              << graph_stats_.last_buf_key_hits;

      VLOG(2) << "Beginning GPU graph capture";
      // Note: Relaxed capture mode because we use a private stream and always
      // re-capture the graph.
      if (!GpuDriver::BeginGraphCaptureOnStream(
              gpu_context, capture_cuda_stream,
              CU_STREAM_CAPTURE_MODE_THREAD_LOCAL  // ToDo (amoitra): Check
                                                   // implications
                                                   // (including any
                                                   // performance related)
                                                   // of using this
              /*CU_STREAM_CAPTURE_MODE_RELAXED*/)) {
        return InternalError("Failed to begin GPU stream capture");
      }
    }

    std::map<const Thunk*, std::unique_ptr<se::Event>> thunk_to_finish_event;
    bool scoped_annotation_enabled = ScopedAnnotation::IsEnabled();
    for (Thunk* thunk : thunk_schedule_->TotalOrder()) {
      // Annotate execution of this op if tracing was enabled when we started
      // running this module.  If tracing is enabled *while* we're running the
      // module, we won't get any data, but that's probably an OK trade-off.
      absl::optional<ScopedAnnotation> op_annotation;
      CHECK(thunk->hlo_instruction());
      if (scoped_annotation_enabled) {
        op_annotation.emplace(FindOrDie(thunk_annotations_, thunk));
      }

      TF_RETURN_IF_ERROR(thunk->Initialize(*this, executor));
      int32 stream_no =
          thunk_schedule_->StreamNumberForHlo(*thunk->hlo_instruction());
      se::Stream* stream =
          (stream_no == 0 ? capture_stream : sub_streams[stream_no - 1].get());

      for (const Thunk* dependency : thunk_schedule_->DependsOn(thunk)) {
        VLOG(3) << ThunkKindToString(thunk->kind()) << " depends on "
                << ThunkKindToString(dependency->kind());
        stream->ThenWaitFor(FindOrDie(thunk_to_finish_event, dependency).get());
      }
      std::string copy_type = dynamic_cast<HostToDeviceCopyThunk*>(thunk)
                                  ? "HostToDeviceCopyThunk"
                                  : "";
      VLOG(3) << "Executing thunk of kind " << ThunkKindToString(thunk->kind())
              << copy_type;
      VLOG(4) << "Executing the thunk for "
              << thunk->hlo_instruction()->ToString() << " on stream "
              << stream_no;
      const GpuExecutableRunOptions* gpu_options =
          run_options->run_options().gpu_executable_run_options();
      Thunk::ExecuteParams thunk_params{
          &buffer_allocations,
          stream,
          run_options->run_options().run_id(),
          &profiler,
          run_options->run_options().device_assignment(),
          &deferred_host_callbacks,
          gpu_options && gpu_options->gpu_global_device_ids()
              ? &*gpu_options->gpu_global_device_ids()
              : nullptr,
          gpu_options && gpu_options->nccl_unique_id_callback()
              ? &gpu_options->nccl_unique_id_callback()
              : nullptr};
      TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(thunk_params));
      if (thunk_schedule_->Depended(thunk)) {
        VLOG(3) << " " << ThunkKindToString(thunk->kind())
                << " is depended by another thunk"
                << ". Hence pushing finish_event.";

        auto finish_event = absl::make_unique<se::Event>(main_stream->parent());
        finish_event->Init();
        stream->ThenRecordEvent(finish_event.get());
        thunk_to_finish_event[thunk] = std::move(finish_event);
      }
    }
    capture_stream->ThenWaitFor(&sub_streams);

    if (use_gpu_graph_capture) {
      stream_executor::gpu::GpuGraphHandle graph;
      if (!GpuDriver::EndGraphCaptureOnStream(gpu_context, capture_cuda_stream,
                                              &graph)) {
        return InternalError("GPU stream capture failed");
      }
      auto destroy_graph_fn = [&](stream_executor::gpu::GpuGraphHandle graph) {
        GpuDriver::DestroyGraph(gpu_context, &graph);
      };
      std::unique_ptr<
          std::remove_pointer<stream_executor::gpu::GpuGraphHandle>::type,
          decltype(destroy_graph_fn)>
          owned_graph(graph, destroy_graph_fn);
      // Use of gpu_exec_graphs_ must be made thread-safe.
      // TODO(benbarsdell): Should use a different mutex?
      // tensorflow::mutex_lock lock(module_handle_mutex_);
      // auto& exec_graph =
      //    *reinterpret_cast<stream_executor::gpu::GpuGraphExecHandle*>(
      //        &gpu_exec_graphs_[gpu_context][bufs_key]);
      if (exec_graph) {
        if (!GpuDriver::UpdateExecutableGraph(gpu_context, exec_graph, graph)) {
          LOG(WARNING) << "Failed to update GPU executable graph";
          GpuDriver::DestroyExecutableGraph(gpu_context, &exec_graph);
        }
      }
      if (!exec_graph) {
        bool instantiate_success = GpuDriver::InstantiateExecutableGraph(
            gpu_context, graph, &exec_graph);
        if (!instantiate_success) {
          return InternalError("Failed to instantiate GPU execution graph");
        }
        gpu_exec_graphs_cache_[gpu_context].update_cache(bufs_key, exec_graph);
      }
      owned_graph.reset();  // Can destroy the captured graph early
      // main_stream->ThenWaitFor(&sub_streams); //(ToDo(amoitra): Check if this
      // is required)
      // if (VLOG_IS_ON(3)) {
      //   stream_executor::gpu::GpuStream* main_gpu_stream =
      //       static_cast<stream_executor::gpu::GpuStream*>(
      //           main_stream->implementation());
      //   bool is_main_stream_idle = false;
      //   while (!is_main_stream_idle) {
      //     is_main_stream_idle = main_gpu_stream->IsIdle();
      //     LOG(INFO) << "Main stream " << main_cuda_stream << " not idle yet";
      //   }

      //   stream_executor::gpu::GpuStream* capture_gpu_stream =
      //       static_cast<stream_executor::gpu::GpuStream*>(
      //           capture_stream->implementation());
      //   bool is_capture_stream_idle = false;
      //   while (!is_capture_stream_idle) {
      //     is_capture_stream_idle = capture_gpu_stream->IsIdle();
      //     LOG(INFO) << "Capture stream " << capture_cuda_stream
      //               << " not idle yet (before graph launch)";
      //   }
      // }

      if (!GpuDriver::LaunchExecutableGraph(gpu_context, exec_graph,
                                            main_cuda_stream)) {
        return InternalError("Failed to launch CUDA execution graph");
      }
      // if (VLOG_IS_ON(3)) {
      //   stream_executor::gpu::GpuStream* capture_gpu_stream =
      //       static_cast<stream_executor::gpu::GpuStream*>(
      //           capture_stream->implementation());
      //   bool is_capture_stream_idle = false;
      //   while (!is_capture_stream_idle) {
      //     is_capture_stream_idle = capture_gpu_stream->IsIdle();
      //     LOG(INFO) << "Capture stream " << capture_cuda_stream
      //               << " not idle yet (after graph launch)";
      //   }

      //   // is_main_stream_idle = false;
      //   // while (!is_main_stream_idle){
      //   //   is_main_stream_idle = main_gpu_stream->IsIdle();
      //   //   VLOG(2) << "Stream " << main_cuda_stream << " not idle yet";
      //   // }
      // }
    }
    // capture_stream->ThenWaitFor(&sub_streams);

  }  // End of blind graph launch conditional. HACK

  if (!deferred_host_callbacks.empty()) {
    auto fn = [deferred_host_callbacks{std::move(deferred_host_callbacks)}]() {
      for (auto& callback : deferred_host_callbacks) {
        callback();
      }
    };
    if (run_options->run_options().then_execute_function()) {
      (*run_options->run_options().then_execute_function())(main_stream,
                                                            std::move(fn));
    } else {
      main_stream->ThenDoHostCallback(std::move(fn));
    }
  }

  // Make sure kernels are completed before deallocating temporary buffers or
  // the profiler state.
  // TODO(b/30100571): we could potentially postpone deallocating the temp
  // buffers until a different computation is executed.
  if (do_profile || block_host_until_done) {
    Status block_status = main_stream->BlockHostUntilDone();
    if (!block_status.ok()) {
      return InternalError(
          "Failed to complete all kernels launched on stream %p: %s",
          main_stream, block_status.error_message());
    }
  }

  // FinishExecution() blocks until main_stream has completed if profiling is
  // enabled; we therefore do not need to defer profile collection onto a
  // stream.
  profiler.FinishExecution();
  uint64 end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->run_options().execution_profile()) {
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));

    // If hlo profiling was disabled then the cycle count is left empty.
    if (do_profile) {
      profile->set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  return Status::OK();
}

StatusOr<const GpuExecutable::BufferAllocToDeviceMemoryMap*>
GpuExecutable::ResolveConstantGlobals(se::Stream* stream) {
  se::StreamExecutor* executor = stream->parent();

  tensorflow::mutex_lock lock(module_handle_mutex_);
  auto it = module_globals_.find(executor);
  if (it != module_globals_.end()) {
    return &it->second;
  }

  se::MultiModuleLoaderSpec module_spec;
  if (!binary().empty()) {
    module_spec.AddCudaCubinInMemory(binary());
  }
  module_spec.AddCudaPtxInMemory(text().c_str());

  absl::flat_hash_map<int64, se::DeviceMemoryBase> globals;
  if (executor->platform_kind() == se::PlatformKind::kCuda &&
      module_spec.cuda_ptx_in_memory() == nullptr) {
    // No custom PTX => no globals.
    return &module_globals_.emplace(executor, std::move(globals)).first->second;
  }

  se::ModuleHandle module_handle;
  TF_RETURN_IF_ERROR(executor->LoadModule(module_spec, &module_handle));

  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    if (allocation.is_constant()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase global,
          executor->GetUntypedSymbol(
              llvm_ir::ConstantBufferAllocationToGlobalName(allocation),
              module_handle));
      VLOG(4) << "Resolved global "
              << llvm_ir::ConstantBufferAllocationToGlobalName(allocation)
              << " to " << global.opaque();
      InsertOrDie(&globals, i, global);

      const Literal& literal =
          llvm_ir::LiteralForConstantAllocation(allocation);
      CHECK(literal.shape().IsArray());
      if (!ShouldEmitLiteralInLlvmIr(literal)) {
        VLOG(4) << "H2D memcpy for constant with shape "
                << ShapeUtil::HumanString(literal.shape());
        stream->ThenMemcpy(&global, literal.untyped_data(), allocation.size());
      }
    }
  }

  module_handles_.emplace(executor,
                          se::ScopedModuleHandle(executor, module_handle));
  return &module_globals_.emplace(executor, std::move(globals)).first->second;
}

StatusOr<ExecutionOutput> GpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ShapeTree<MaybeOwningDeviceMemory>> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat("GpuExecutable::ExecuteAsyncOnStream(",
                                        module().name(), ")"));
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();
  // Force synchronous execution if the allocator requires it.
  const bool block_host_until_done =
      !memory_allocator->AllowsAsynchronousDeallocation();

  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  BufferAllocations::Builder buffer_allocations_builder;
  const GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tensorflow::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(globals, ResolveConstantGlobals(run_options->stream()));
  }

  se::StreamExecutor* executor = run_options->stream()->parent();

  std::unique_ptr<BufferAllocations> buffer_allocations;

  {
    tensorflow::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Build buffer allocations"); },
        tensorflow::profiler::TraceMeLevel::kInfo);

    for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
         ++i) {
      const BufferAllocation& allocation = assignment_->GetAllocation(i);
      if (allocation.is_entry_computation_parameter()) {
        auto param_no = allocation.parameter_number();
        se::DeviceMemoryBase buffer =
            arguments[param_no]
                .element(allocation.param_shape_index())
                .AsDeviceMemoryBase();

        // All top-level buffers and sub-buffers must have an explicit, non-null
        // pointer, except for zero-sized buffers, which may be null.
        if (buffer.is_null() && buffer.size() > 0) {
          return FailedPrecondition(
              "Cannot run XLA computation because pointer to (sub-)buffer at "
              "index %s of parameter %d was null.  All pointers to "
              "(sub-)buffers must not be null, unless the (sub-)buffer has "
              "zero elements.",
              allocation.param_shape_index().ToString(), param_no);
        }

        buffer_allocations_builder.RegisterBuffer(i, buffer);
      }

      if (allocation.is_constant()) {
        buffer_allocations_builder.RegisterBuffer(i, FindOrDie(*globals, i));
      }
    }

    TF_ASSIGN_OR_RETURN(
        buffer_allocations,
        buffer_allocations_builder.Build(
            assignment_.get(), executor->device_ordinal(), memory_allocator));
  }

  TF_RETURN_IF_ERROR(ExecuteThunks(run_options, *buffer_allocations,
                                   block_host_until_done,
                                   hlo_execution_profile));

  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  auto device_ordinal = executor->device_ordinal();
  ScopedShapedBuffer shaped_buffer(root->shape(), root->shape(),
                                   memory_allocator, device_ordinal);

  // Copy DeviceMemoryBase values which contain the array(s) of the result into
  // the respective location in ShapedBuffer.
  std::set<se::DeviceMemoryBase> buffers_in_result;
  TF_RETURN_IF_ERROR(shaped_buffer.buffers().ForEachMutableElementWithStatus(
      [&buffer_allocations, &buffers_in_result, this](
          const ShapeIndex& index, se::DeviceMemoryBase* device_memory) {
        const auto& sources = this->GetRootValueSet().element(index);
        // The points-to set is unambiguous so the set should be a
        // singleton. That is, we know exactly which instruction
        // produced the array at this element.
        CHECK_EQ(1, sources.values().size());
        auto src_hlo = sources.values()[0]->instruction();

        VLOG(4) << "Looking at: " << sources.values()[0];

        // The source instruction should have a non-parameter buffer
        // assigned.
        TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                            this->assignment_->GetUniqueSlice(
                                src_hlo, sources.values()[0]->index()));

        se::DeviceMemoryBase src_base =
            buffer_allocations->GetDeviceAddress(slice.index());
        CHECK(!src_base.is_null() || src_base.size() == 0);
        if (!slice.allocation()->is_entry_computation_parameter()) {
          // If the buffer coming out of the result is from a parameter, it
          // means the caller aliased some parameter buffer to an output one
          // (via the HloInputOutputAliasConfig API). If that is the case, the
          // caller will receive a partially complete scoped shaped buffer,
          // which they will have to fill up on return.
          // Unfortunately the interface to the execute APIs are ShapedBuffer
          // pointer based, which assumes caller ownership, and hence a buffer
          // coming from there cannot be part of the new ScopedShapedBuffer we
          // create for the result (which assumes ownership).
          *device_memory = src_base;
        } else {
          const HloInputOutputAliasConfig& input_output_alias =
              module().input_output_alias_config();
          auto output_alias = input_output_alias.GetAliasedOutput(
              slice.allocation()->parameter_number(),
              slice.allocation()->param_shape_index());
          CHECK(output_alias)
              << "Output buffer is coming from parameter "
              << slice.allocation()->parameter_number() << " at index "
              << slice.allocation()->param_shape_index()
              << ", but no alias exists";
          CHECK_EQ(*output_alias, index);
        }
        buffers_in_result.insert(src_base);
        return Status::OK();
      }));
  // Also put all input operands of AsyncOutSend into buffers_in_result to
  // prevent XLA from releasing the buffers.
  for (auto* instr : hlo_module_->entry_computation()->instructions()) {
    if (instr->opcode() != HloOpcode::kAsyncOutSend) {
      continue;
    }
    const HloInstruction* opnd = instr->operand(0);
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        this->assignment_->GetUniqueSlice(opnd, {}));
    se::DeviceMemoryBase src_base =
        buffer_allocations->GetDeviceAddress(slice.index());
    CHECK(!src_base.is_null() || src_base.size() == 0);
    buffers_in_result.insert(src_base);
  }
  TF_RETURN_IF_ERROR(buffer_allocations->TearDown(buffers_in_result));

  std::vector<se::OwningDeviceMemory> buffers_to_free;
  for (ShapeTree<MaybeOwningDeviceMemory>& argument : arguments) {
    for (std::pair<ShapeIndex, MaybeOwningDeviceMemory>& buffer : argument) {
      auto maybe_owning_buffer = buffer.second.Release();
      if (maybe_owning_buffer) {
        buffers_to_free.push_back(std::move(*maybe_owning_buffer));
      }
    }
  }
  return ExecutionOutput(std::move(shaped_buffer), std::move(buffers_to_free),
                         {}, {});
}

const InstructionValueSet& GpuExecutable::GetRootValueSet() const {
  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64 GpuExecutable::SizeOfGeneratedCodeInBytes() {
  // Non-empty PTX but empty cubin: compilation must have failed, return
  // "unknown".
  if (binary().empty() && !text_.empty()) {
    return -1;
  }
  return binary().size();
}

}  // namespace gpu
}  // namespace xla
