;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.clojure-cpp :refer [pointer]])
  (:import org.bytedeco.onnxruntime.global.onnxruntime))

(def ^:const onnx-data-type
  {:undef onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
   :float onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
   Float/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
   Float onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
   :u8 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
   :uint8 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
   :byte onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
   Byte/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
   Byte onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
   :u16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
   :uint16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
   :short onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
   :int16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
   Short/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
   Short onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
   :int onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
   :int32 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
   Integer/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
   Integer onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
   :long onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
   :int64 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
   Long/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
   Long onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
   String onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   Boolean/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   Boolean onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
   :half onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
   :f16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
   :float16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
   :double onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
   Double/TYPE onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
   Double onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
   :u32 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
   :uint32 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
   :u64 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
   :uint64 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
   :complex64 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
   :complex128 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
   :bf16 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
   :float8e4m3fn onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN
   :float8e4m3fnuz onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ
   :float8e5m2 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2
   :float8e5m2fnuz onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ
   :u4 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4
   :uint4 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4
   :int4 onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4})

(def ^:const dec-onnx-data-type
  {onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED :undef
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT :float
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 :uint8
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 :byte
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 :uint16
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 :short
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 :int
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 :long
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING String
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 :float16
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE :double
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 :uint32
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 :uint64
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 :complex64
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 :complex128
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 :bf16
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN :float8e4m3fn
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ :float8e4m3fnuz
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 :float8e5m2
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ :float8e5m2fnuz
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 :uint4
   onnxruntime/ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 :int4})

(def ^:const enc-onnx-type
  {:unknown onnxruntime/ONNX_TYPE_UNKNOWN
   :tensor onnxruntime/ONNX_TYPE_TENSOR
   :sequence onnxruntime/ONNX_TYPE_SEQUENCE
   :seq onnxruntime/ONNX_TYPE_SEQUENCE
   :map onnxruntime/ONNX_TYPE_MAP
   :opaque onnxruntime/ONNX_TYPE_OPAQUE
   :sparse onnxruntime/ONNX_TYPE_SPARSETENSOR
   :optional onnxruntime/ONNX_TYPE_OPTIONAL})

(defn dec-onnx-type [^long type]
  (case type
    0 :unknown
    1 :tensor
    2 :sequence
    3 :map
    4 :opaque
    5 :sparse
    6 :optional
    (dragan-says-ex "Unknown onnx type." {:type type})))

(def ^:const ort-type
  {:undefined onnxruntime/ORT_SPARSE_UNDEFINED
   :coo onnxruntime/ORT_SPARSE_COO
   :csrc onnxruntime/ORT_SPARSE_CSRC
   :block onnxruntime/ORT_SPARSE_BLOCK_SPARSE
   :block-sparse onnxruntime/ORT_SPARSE_BLOCK_SPARSE})

(def ^:const ort-logging-level
  {:verbose onnxruntime/ORT_LOGGING_LEVEL_VERBOSE
   :info onnxruntime/ORT_LOGGING_LEVEL_INFO
   :warning onnxruntime/ORT_LOGGING_LEVEL_WARNING
   :error onnxruntime/ORT_LOGGING_LEVEL_ERROR
   :fatal onnxruntime/ORT_LOGGING_LEVEL_FATAL})

(def ^:const ort-error-code
  {:ok onnxruntime/ORT_OK
   :fail onnxruntime/ORT_FAIL
   :invalid-argument onnxruntime/ORT_INVALID_ARGUMENT
   :no-such-file onnxruntime/ORT_NO_SUCHFILE
   :no-model onnxruntime/ORT_NO_MODEL
   :engine-error onnxruntime/ORT_ENGINE_ERROR
   :runtime-exception onnxruntime/ORT_RUNTIME_EXCEPTION
   :invalid-protobuf onnxruntime/ORT_INVALID_PROTOBUF
   :model-loaded onnxruntime/ORT_MODEL_LOADED
   :not-implemented onnxruntime/ORT_NOT_IMPLEMENTED
   :invalid-graph onnxruntime/ORT_INVALID_GRAPH
   :ep-fail onnxruntime/ORT_EP_FAIL})

(def ^:const ort-op-attr-type
  {:undefined onnxruntime/ORT_OP_ATTR_UNDEFINED
   :int onnxruntime/ORT_OP_ATTR_INT
   Integer/TYPE onnxruntime/ORT_OP_ATTR_INT
   Integer onnxruntime/ORT_OP_ATTR_INT
   :ints onnxruntime/ORT_OP_ATTR_INTS
   :float onnxruntime/ORT_OP_ATTR_FLOAT
   Float/TYPE onnxruntime/ORT_OP_ATTR_FLOAT
   Float onnxruntime/ORT_OP_ATTR_FLOAT
   :floats onnxruntime/ORT_OP_ATTR_FLOATS
   :string onnxruntime/ORT_OP_ATTR_STRING
   String onnxruntime/ORT_OP_ATTR_STRING
   :strings onnxruntime/ORT_OP_ATTR_STRINGS})

(def ^:const ort-graph-optimization
  {:disable onnxruntime/ORT_DISABLE_ALL
   :enable onnxruntime/ORT_ENABLE_ALL
   :all onnxruntime/ORT_ENABLE_ALL
   :basic onnxruntime/ORT_ENABLE_BASIC
   :extended onnxruntime/ORT_ENABLE_EXTENDED})

(def ^:const ort-execution-mode
  {:seq onnxruntime/ORT_SEQUENTIAL
   :sequential onnxruntime/ORT_SEQUENTIAL
   :parallel onnxruntime/ORT_PARALLEL})

(def ^:const ort-language-projection
  {:c onnxruntime/ORT_PROJECTION_C
   :cpp onnxruntime/ORT_PROJECTION_CPLUSPLUS
   :cs onnxruntime/ORT_PROJECTION_CSHARP
   :python onnxruntime/ORT_PROJECTION_PYTHON
   :java onnxruntime/ORT_PROJECTION_JAVA
   :winml onnxruntime/ORT_PROJECTION_WINML
   :nodejs onnxruntime/ORT_PROJECTION_NODEJS})

(def ^:const ort-allocator-type
  {:invalid onnxruntime/OrtInvalidAllocator
   :device onnxruntime/OrtDeviceAllocator
   :arena onnxruntime/OrtArenaAllocator})

(defn dec-ort-allocator-type [^long type]
  (case type
    -1 :invalid
    0 :device
    1 :arena
    (dragan-says-ex "Unknown allocator type." {:type type})))

(def ^:const ort-mem-type
  {:cpu onnxruntime/OrtMemTypeCPU
   :input onnxruntime/OrtMemTypeCPUInput
   :output onnxruntime/OrtMemTypeCPUOutput
   :cpu-input onnxruntime/OrtMemTypeCPUInput
   :cpu-output onnxruntime/OrtMemTypeCPUOutput
   :default onnxruntime/OrtMemTypeDefault})

(defn dec-ort-memory-type [^long type]
  (case type
    -2 :cpu-input
    -1 :cpu-output
    0 :default
    (dragan-says-ex "Unknown memory type." {:type type})))

(def ^:const ort-memory-info-device-type
  {:cpu onnxruntime/OrtMemoryInfoDeviceType_CPU
   :gpu onnxruntime/OrtMemoryInfoDeviceType_GPU
   :fpga onnxruntime/OrtMemoryInfoDeviceType_FPGA})

(defn dec-ort-memory-info-device-type [^long type]
  (case type
    0 :cpu
    1 :gpu
    2 :fpga
    (dragan-says-ex "Unknown device type." {:type type})))

(def ^:const ort-cudnn-conv
  {:exhaustive onnxruntime/OrtCudnnConvAlgoSearchExhaustive
   :heuristic onnxruntime/OrtCudnnConvAlgoSearchHeuristic
   :default onnxruntime/OrtCudnnConvAlgoSearchDefault})

(def ^:const ort-custom-op-output
  {:required onnxruntime/INPUT_OUTPUT_REQUIRED
   :optional onnxruntime/INPUT_OUTPUT_OPTIONAL
   :variadic onnxruntime/INPUT_OUTPUT_VARIADIC})

(def ^:const ort-allocator-name
  {:cpu "Cpu"
   :cuda "Cuda"
   :cuda-pinned "CudaPinned"
   :cann "Cann"
   :cann-pinned "CannPinned"
   :dnm "DML"
   :hip "Hip"
   :hip-pinned "HipPinned"
   :vino-cpu "OpenVINO_CPU"
   :openvino-cpu "OpenVINO_CPU"
   :openvino "OpenVINO_CPU"
   :openvino-gpu "OpenVINO_GPU"
   :openvino-rt "OpenVINO_RT"
   :openvino-npu "OpenVINO_NPU"
   :webgpu-buffer "WebGPU_Buffer"
   :webgpu "WebGPU_Buffer"
   :webnn "WebNN_Tensor"})

(def ^:const ort-allocator-keyword
  {"Cpu" :cpu
   "Cuda" :cuda
   "CudaPinned" :cuda-pinned
   "Cann" :cann
   "CannPinned" :cann-pinned
   "DML" :dnm
   "Hip" :hip
   "HipPinned" :hip-pinned
   "OpenVINO_CPU" :openvino-cpu
   "OpenVINO_GPU" :openvino-gpu
   "OpenVINO_RT" :openvino-rt
   "OpenVINO_NPU" :openvino-npu
   "WebGPU_Buffer" :webgpu-buffer
   "WebNN_Tensor" :webnn})

(def ^:const onnx-dimension-denotation
  {:data-batch "DATA_BATCH"
   :batch "DATA_BATCH"
   :data-channel "DATA_CHANNEL"
   :channel "DATA_CHANNEL"
   :data-time "DATA_TIME"
   :time "DATA_TIME"
   :data-feature "DATA_FEATURE"
   :feature "DATA_FEATURE"
   :filter-in-channel "FILTER_IN_CHANNEL"
   :in-channel "FILTER_IN_CHANNEL"
   :filter-out-channel "FILTER_OUT_CHANNEL"
   :out-channel "FILTER_OUT_CHANNEL"
   :filter-spatial "FILTER_SPATIAL"
   :spatial "FILTER_SPATIAL"})

(def ^:const ort-session-options-config-keys
  {:disable-prepacking "session.disable_prepacking"
   :prepacking "session.disable_prepacking"
   :use-env-allocators "session.use_env_allocators"
   :env-allocators "session.use_env_allocators"
   :load-format "session.load_model_format"
   :load-model-format "session.load_model_format"
   :save-format "session.save_model_format"
   :save-model-format "session.save_model_format"
   :denormal-as-zero "session.set_denormal_as_zero"
   :disable-quant-qdq "session.disable_quant_qdq"
   :quant-qdq "session.disable_quant_qdq"
   :disable-double-qdq-remover "session.disable_double_qdq_remover"
   :double-qdq-remover "session.disable_double_qdq_remover"
   :quant-qdq-cleanup "session.enable_quant_qdq_cleanup"
   :gelu-approximation "optimization.enable_gelu_approximation"
   :disable-aot-function-inlining "session.disable_aot_function_inlining"
   :disable-aot-inlining "session.disable_aot_function_inlining"
   :disable-function-inlining "session.disable_aot_function_inlining"
   :memory-optimizer-config "optimization.memory_optimizer_config"
   :memory-optimizer "optimization.memory_optimizer_config"
   :memory-probe-recompute-config "optimization.enable_memory_probe_recompute_config"
   :memory-probe-recompute "optimization.enable_memory_probe_recompute_config"
   :disable-specified-optimizers "optimization.disable_specified_optimizers"
   :use-device-allocator-for-initializers "session.use_device_allocator_for_initializers"
   :use-device-initializers "session.use_device_allocator_for_initializers"
   :use-device-allocator "session.use_device_allocator_for_initializers"
   :inter-op-allow-spinning "session.inter_op.allow_spinning"
   :inter-op-spinning "session.inter_op.allow_spinning"
   :intra-op-allow-spinning "session.intra_op.allow_spinning"
   :intra-op-spinning "session.intra_op.allow_spinning"
   :use-ort-model-bytes-directly  "session.use_ort_model_bytes_directly"
   :use-ort-model-bytes "session.use_ort_model_bytes_directly"
   :use-ort-model-bytes-for-initializers "session.use_ort_model_bytes_for_initializers"
   :use-ort-bytes-for-initializers "session.use_ort_model_bytes_for_initializers"
   :use-ort-for-initializers "session.use_ort_model_bytes_for_initializers"
   :use-ort-initializers "session.use_ort_model_bytes_for_initializers"
   :qdq-is-int8-allowed "session.qdqisint8allowed"
   :qdq-is-int8 "session.qdqisint8allowed"
   :x64quantprecision "session.x64quantprecision"
   :minimal-build-optimizations "optimization.minimal_build_optimizations"
   :minimal-optimizations "optimization.minimal_build_optimizations"
   :partitioning-stop-ops "ep.nnapi.partitioning_stop_ops"
   :dynamic-block-base "session.dynamic_block_base"
   :force-spinning-stop "session.force_spinning_stop"
   :strict-shape-type-inference "session.strict_shape_type_inference"
   :allow-released-opsets-only "session.allow_released_opsets_only"
   :released-opsets-only "session.allow_released_opsets_only"
   :node-partition-config-file "session.node_partition_config_file"
   :intra-op-thread-affinities "session.intra_op_thread_affinities"
   :debug-layout-transformation "session.debug_layout_transformation"
   :disable-cpu-ep-fallback "session.disable_cpu_ep_fallback"
   :cpu-ep-fallback "session.disable_cpu_ep_fallback"
   :optimized-model-external-initializers-file-name "session.optimized_model_external_initializers_file_name"
   :external-initializers-file-name "session.optimized_model_external_initializers_file_name"
   :optimized-modelexternal-initializers-min-size-in-bytes "session.optimized_model_external_initializers_min_size_in_bytes"
   :external-initializers-min-bytesize "session.optimized_model_external_initializers_min_size_in_bytes"
   :external-initializers-min-size-in-bytes "session.optimized_model_external_initializers_min_size_in_bytes"
   :model-external-initializers-file-folder-path "session.model_external_initializers_file_folder_path"
   :external-initializers-file-folder-path "session.model_external_initializers_file_folder_path"
   :save-external-prepacked-constant-initializers "session.save_external_prepacked_constant_initializers"
   :save-external-prepacked-initializers "session.save_external_prepacked_constant_initializers"
   :save-prepacked-initializers "session.save_external_prepacked_constant_initializers"
   :collect-node-memory-stats-to-file "session.collect_node_memory_stats_to_file"
   :resource-cuda-partitioning-settings "session.resource_cuda_partitioning_settings"
   :ep-context "ep.context_enable"
   :ep-context-file-path "ep.context_file_path"
   :ep-context-embed-mode "ep.context_embed_mode"
   :ep-context-node-name-prefix "ep.context_node_name_prefix"
   :ep-share-ep-contexts "ep.share_ep_contexts"
   :ep-stop-share-ep-contexts "ep.stop_share_ep_contexts"
   :ep-context-model-external-initializers-file-name "ep.context_model_external_initializers_file_name"
   :enable-gemm-fastmath-arm64-bloat16 "mlas.enable_gemm_fastmath_arm64_bfloat16"
   :qdq-matmultnbits-accuracy-level "session.qdq_matmulnbits_accuracy_level"
   :disable-model-compile "session.disable_model_compile"
   :model-compile "session.disable_model_compile"})

(defn true->one [entry]
  (case entry
    "0" "0"
    "1" "1"
    (if entry
      "1"
      "0")))

(defn true->zero [entry]
  (case entry
    "1" "0"
    "0" "1"
    (if entry
      "0"
      "1")))

(defn one->true [entry]
  (case entry
    "1" true
    "0" false
    entry))

(defn zero->true [entry]
  (case entry
    "1" false
    "0" true
    (not entry)))

(defn capitalized-name [obj]
  (clojure.string/capitalize (name obj)))

(defn lower-case-keyword [value]
  (clojure.string/lower-case (keyword (str value))))

(defn long->str [^long x]
  (str x))

(def ^:const ort-session-options-config-encoders
  {:disable-prepacking true->one
   :prepacking true->zero
   :use-env-allocators true->one
   :env-allocators true->one
   :use-session-allocators true->zero
   :session-allocators true->zero
   :load-model-format identity
   :load-format identity
   :save-model-format identity
   :save-format identity
   :denormal-as-zero true->one
   :disable-quant-qdq true->one
   :quant-qdq true->zero
   :disable-double-qdq-remover true->one
   :double-qdq-remover true->zero
   :quant-qdq-cleanup true->one
   :gelu-approximation true->one
   :disable-aot-function-inlining true->one
   :aot-function-inlining true->zero
   :disable-aot-inlining true->one
   :aot-inlining true->zero
   :disable-function-inlining true->one
   :function-inlining true->zero
   :memory-optimizer-config identity
   :memory-optimizer identity
   :memory-probe-recompute-config identity
   :memory-probe-recompute identity
   :disable-specified-optimizers identity
   :use-device-allocator-for-initializers true->one
   :use-device-initializers true->one
   :use-device-allocator true->one
   :inter-op-allow-spinning true->one
   :inter-op-spinning true->one
   :intra-op-allow-spinning true->one
   :intra-op-spinning true->one
   :use-ort-model-bytes-directly true->one
   :use-ort-model-bytes true->one
   :use-model-bytes-directly true->one
   :use-ort-model-bytes-for-initializers true->one
   :use-ort-bytes-for-initializers true->one
   :use-ort-for-initializers true->one
   :use-ort-initializers true->one
   :qdq-is-int8-allowed true->one
   :qdq-is-int8 true->one
   :x64quantprecision identity
   :minimal-build-optimizations name
   :minimal-optimizations name
   :partitioning-stop-ops identity
   :dynamic-block-base str
   :force-spinning-stop identity
   :strict-shape-type-inference true->one
   :allow-released-opsets-only true->one
   :released-opsets-only true->one
   :node-partition-config-file identity
   :intra-op-thread-affinities identity
   :debug-layout-transformation true->one
   :disable-cpu-ep-fallback true->one
   :cpu-ep-fallback true->zero
   :optimized-model-external-initializers-file-name identity
   :external-initializers-file-name identity
   :optimized-modelexternal-initializers-min-size-in-bytes str
   :external-initializers-min-bytesize str
   :external-initializers-min-size-in-bytes str
   :model-external-initializers-file-folder-path identity
   :external-initializers-file-folder-path identity
   :save-external-prepacked-constant-initializers true->one
   :save-external-prepacked-initializers true->one
   :save-prepacked-initializers true->one
   :collect-node-memory-stats-to-file identity
   :resource-cuda-partitioning-settings identity
   :ep-context true->one
   :ep-context-file-path identity
   :ep-context-embed-mode true->one
   :ep-context-node-name-prefix identity
   :ep-share-ep-contexts true->one
   :ep-stop-share-ep-contexts true->one
   :ep-context-model-external-initializers-file-name identity
   :enable-gemm-fastmath-arm64-bloat16 true->one
   :qdq-matmultnbits-accuracy-level str
   :disable-model-compile true->one
   :model-compile true->zero})

(def ^:const ort-session-options-config-decoders
  {:disable-prepacking one->true
   :prepacking zero->true
   :use-env-allocators one->true
   :env-allocators one->true
   :use-session-allocators zero->true
   :session-allocators zero->true
   :load-model-format identity
   :load-format identity
   :save-model-format identity
   :save-format identity
   :denormal-as-zero one->true
   :disable-quant-qdq one->true
   :quant-qdq zero->true
   :disable-double-qdq-remover one->true
   :double-qdq-remover zero->true
   :quant-qdq-cleanup one->true
   :gelu-approximation one->true
   :disable-aot-function-inlining one->true
   :aot-function-inlining zero->true
   :disable-aot-inlining one->true
   :aot-inlining zero->true
   :disable-function-inlining one->true
   :function-inlining zero->true
   :memory-optimizer-config identity
   :memory-optimizer identity
   :memory-probe-recompute-config identity
   :memory-probe-recompute identity
   :disable-specified-optimizers identity
   :use-device-allocator-for-initializers one->true
   :use-device-initializers one->true
   :use-device-allocator one->true
   :inter-op-allow-spinning one->true
   :inter-op-spinning one->true
   :intra-op-allow-spinning one->true
   :intra-op-spinning one->true
   :use-ort-model-bytes-directly one->true
   :use-ort-model-bytes one->true
   :use-model-bytes-directly one->true
   :use-ort-model-bytes-for-initializers one->true
   :use-ort-bytes-for-initializers one->true
   :use-ort-for-initializers one->true
   :use-ort-initializers one->true
   :qdq-is-int8-allowed one->true
   :qdq-is-int8 one->true
   :x64quantprecision identity
   :minimal-build-optimizations name
   :minimal-optimizations name
   :partitioning-stop-ops identity
   :dynamic-block-base read-string
   :force-spinning-stop identity
   :strict-shape-type-inference one->true
   :allow-released-opsets-only one->true
   :released-opsets-only one->true
   :node-partition-config-file identity
   :intra-op-thread-affinities identity
   :debug-layout-transformation one->true
   :disable-cpu-ep-fallback one->true
   :cpu-ep-fallback zero->true
   :optimized-model-external-initializers-file-name identity
   :external-initializers-file-name identity
   :optimized-modelexternal-initializers-min-size-in-bytes read-string
   :external-initializers-min-bytesize read-string
   :external-initializers-min-size-in-bytes read-string
   :model-external-initializers-file-folder-path identity
   :external-initializers-file-folder-path identity
   :save-external-prepacked-constant-initializers one->true
   :save-external-prepacked-initializers one->true
   :save-prepacked-initializers one->true
   :collect-node-memory-stats-to-file identity
   :resource-cuda-partitioning-settings identity
   :ep-context one->true
   :ep-context-file-path identity
   :ep-context-embed-mode one->true
   :ep-context-node-name-prefix identity
   :ep-share-ep-contexts one->true
   :ep-stop-share-ep-contexts one->true
   :ep-context-model-external-initializers-file-name identity
   :enable-gemm-fastmath-arm64-bloat16 one->true
   :qdq-matmultnbits-accuracy-level read-string
   :disable-model-compile one->true
   :model-compile zero->true})

(def ^:const ort-ep-dynamic-options-keys
  {:ep-dynamic-workload-type "ep.dynamic.workload_type"
   :ep-dynamic-workload "ep.dynamic.workload_type"})

(def ^:const ort-ep-dynamic-options-encoders
  {:ep-dynamic-workload-type capitalized-name
   :ep-dynamic-workload capitalized-name})

(def ^:const ort-cuda-provider-options-keys
  {:device-id "device_id"
   :copy-in-default-stream "do_copy_in_default_stream"
   ;; :cudnn-conv-algo-search "cudnn_conv_algo_search"
   ;; :conv-algo-search "cudnn_conv_algo_search"
   :gpu-mem-limit "gpu_mem_limit"
   :arena-extend-strategy "arena_extend_strategy"
   :default-memory-arena-cfg "default_memory_arena_cfg"
   :cudnn-conv-use-max-workspace "cudnn_conv_use_max_workspace"
   :conv-use-max-workspace "cudnn_conv_use_max_workspace"
   :enable-cuda-graph "enable_cuda_graph"
   :cudnn-conv1d-pad-to-nc1d "cudnn_conv1d_pad_to_nc1d"
   :conv1d-pad-to-nc1d "cudnn_conv1d_pad_to_nc1d"
   :tunable-op-enable "tunable_op_enable"
   :tunable-op-tuning-enable "tunable_op_tuning_enable"
   :tunable-op-max-tuning-duration-ms "tunable_op_max_tuning_duration_ms"
   :enable-skip-layer-norm-strict-mode "enable_skip_layer_norm_strict_mode"
   :skip-layer-norm-strict-mode "enable_skip_layer_norm_strict_mode"
   :prefer-nhwc "prefer_nhwc"
   :use-ep-level-unified-stream "use_ep_level_unified_stream"
   :ep-level-unified-stream "use_ep_level_unified_stream"
   :use-tf32 "use_tf32"
   :tf32 "use_tf32"
   :fuse-conv-bias "fuse_conv_bias"
   :sdpa-kernel "sdpa_kernel"})

(def ^:const ort-arena-extend-strategy
  {:default "-1"
   :next-pow2 "0"
   :requested "1"
   "-1" "-1"
   "0" "0"
   "1" "1"})

(def ^:const ort-cuda-provider-options-encoders
  {:device-id long->str
   :copy-in-default-stream true->one
   ;; :cudnn-conv-algo-search #(str (ort-cudnn-conv %))
   ;; :conv-algo-search #(str (ort-cudnn-conv %))
   :gpu-mem-limit long->str
   :arena-extend-strategy ort-arena-extend-strategy
   :default-memory-arena-cfg identity
   :cudnn-conv-use-max-workspace true->one
   :conv-use-max-workspace true->one
   :enable-cuda-graph true->one
   :cudnn-conv1d-pad-to-nc1d true->one
   :conv1d-pad-to-nc1d true->one
   :tunable-op-enable true->one
   :tunable-op-tuning-enable true->one
   :tunable-op-max-tuning-duration-ms long->str
   :enable-skip-layer-norm-strict-mode true->one
   :skip-layer-norm-strict-mode true->one
   :prefer-nhwc true->one
   :use-ep-level-unified-stream true->one
   :ep-level-unified-stream true->one
   :use-tf32 true->one
   :tf32 true->one
   :fuse-conv-bias true->one
   :sdpa-kernel true->one})
