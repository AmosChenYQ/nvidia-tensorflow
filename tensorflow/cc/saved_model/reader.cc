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

#include "tensorflow/cc/saved_model/reader.h"

#include <unordered_set>

#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace {

Status ReadSavedModel(const string& export_dir, SavedModel* saved_model_proto) {
  LOG(INFO) << "Reading SavedModel from: " << export_dir;

  const string saved_model_pb_path =
      io::JoinPath(export_dir, kSavedModelFilenamePb);
  if (Env::Default()->FileExists(saved_model_pb_path).ok()) {
    return ReadBinaryProto(Env::Default(), saved_model_pb_path,
                           saved_model_proto);
  }
  const string saved_model_pbtxt_path =
      io::JoinPath(export_dir, kSavedModelFilenamePbTxt);
  if (Env::Default()->FileExists(saved_model_pbtxt_path).ok()) {
    return ReadTextProto(Env::Default(), saved_model_pbtxt_path,
                         saved_model_proto);
  }
  return Status(error::Code::NOT_FOUND,
                "Could not find SavedModel .pb or .pbtxt at supplied export "
                "directory path: " +
                    export_dir);
}

Status FindMetaGraphDef(const SavedModel& saved_model_proto,
                        const std::unordered_set<string>& tags,
                        MetaGraphDef* meta_graph_def) {
  LOG(INFO) << "Reading meta graph with tags { " << absl::StrJoin(tags, " ")
            << " }";
  for (const MetaGraphDef& graph_def : saved_model_proto.meta_graphs()) {
    // Get tags from the graph_def.
    std::unordered_set<string> graph_tags;
    for (const string& tag : graph_def.meta_info_def().tags()) {
      graph_tags.insert(tag);
    }
    // Match with the set of tags provided.
    if (graph_tags == tags) {
      *meta_graph_def = graph_def;
      return Status::OK();
    }
  }
  return Status(
      error::Code::NOT_FOUND,
      strings::StrCat(
          "Could not find meta graph def matching supplied tags: { ",
          absl::StrJoin(tags, " "),
          " }. To inspect available tag-sets in the SavedModel, please "
          "use the SavedModel CLI: `saved_model_cli`"));
}

}  // namespace

Status ReadMetaGraphDefFromSavedModel(const string& export_dir,
                                      const std::unordered_set<string>& tags,
                                      MetaGraphDef* const meta_graph_def) {
  SavedModel saved_model_proto;
  const uint64 read_saved_model_start = Env::Default()->NowMicros();
  TF_RETURN_IF_ERROR(ReadSavedModel(export_dir, &saved_model_proto));
  const uint64 read_saved_model_end = Env::Default()->NowMicros();
  LOG(INFO) << "Reading saved model from proto file takes " << static_cast<float>(read_saved_model_end - read_saved_model_start) / 1000 << "ms.";;
  TF_RETURN_IF_ERROR(FindMetaGraphDef(saved_model_proto, tags, meta_graph_def));
  const uint64 find_meta_graph_def_end = Env::Default()->NowMicros();
  LOG(INFO) << "Find MetaGraphDef takes " << static_cast<float>(find_meta_graph_def_end - read_saved_model_end) / 1000 << "ms.";;
  return Status::OK();
}

}  // namespace tensorflow
