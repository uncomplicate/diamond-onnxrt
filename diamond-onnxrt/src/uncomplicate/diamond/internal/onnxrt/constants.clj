;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
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
  {:exaustive onnxruntime/OrtCudnnConvAlgoSearchExhaustive
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
