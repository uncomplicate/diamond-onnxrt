;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.ort.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release release
                           ;;Info
                           ;; info Viewable view Bytes bytesize Entries sizeof* bytesize*
                           ;; sizeof size
                           ]]
             [utils :as utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer-pointer int-pointer long-pointer byte-pointer char-pointer size-t-pointer get-entry get-pointer get-string]])
  (:import [org.bytedeco.javacpp Pointer BytePointer Loader]
           org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime OrtApi OrtEnv OrtSession OrtSessionOptions OrtDnnlProviderOptions
            OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo

            StringVector LongVector OrtStatus SessionOptions Session
            AllocatorWithDefaultOptions OrtAllocator  Allocator
            TypeInfo TypeInfoImpl MapTypeInfoImpl ConstMapTypeInfo
            ConstTensorTypeAndShapeInfo TensorTypeAndShapeInfoImpl

            OptionalTypeInfoImpl MemoryInfoImpl MemoryInfo OrtMemoryInfo Value
            ConstValueImpl
            BaseAllocator BaseAllocatorWithDefaultOptions BaseArenaCfg BaseConstIoBinding
            BaseConstMapTypeInfo BaseConstSession BaseConstSessionOptions BaseConstTensorTypeAndShapeInfo
            BaseConstValue BaseCustomOpDomain BaseEnv BaseIoBinding BaseKernelInfo BaseMapTypeInfo
            BaseMemoryInfo BaseModelMetadata BaseOpAttr BaseOrtLoraAdapter BaseOrtOp BaseRunOptions
            BaseSequenceTypeInfo BaseSequenceTypeInfoImpl BaseSession BaseSessionOptions
            BaseStatus BaseTensorTypeAndShapeInfo BaseThreadingOptions BaseTypeInfo
            BaseValue]))

(def ^:dynamic *ort-api* (.call (.GetApi (onnxruntime/OrtGetApiBase)) onnxruntime/ORT_API_VERSION))

(def platform-pointer (if (.startsWith (Loader/getPlatform) "windows") char-pointer byte-pointer))

(defn ort-error
  ([^OrtApi ort-api ^OrtStatus ort-status]
   (let [err (.getString (.call (.GetErrorMessage ort-api) ort-status))]
     (ex-info (format "ONNX runtime error: %s." err)
              {:error err :type :ort-error})))
  ([ort-status]
   (ort-error *ort-api* ort-status)))

(defmacro with-check
  ([ort-api status form]
   `(with-release [status# ~status]
      (if (null? status#)
        ~form
        (throw (ort-error ~ort-api status#)))))
  ([ort-api status]
   `(with-check ~ort-api ~status ~ort-api)))

(defmacro extend-ort [t]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (locking this#
         (when-not (null? this#)
           (onnxruntime/OrtRelease this#)
           (.deallocate this#)
           (.setNull this#))
         true))))

(extend-ort OrtEnv)
(extend-ort OrtSession)
(extend-ort OrtStatus)
(extend-ort OrtSessionOptions)
(extend-ort OrtSession)
(extend-ort OrtAllocator)
(extend-ort OrtTypeInfo)
(extend-ort OrtTensorTypeAndShapeInfo)

(defn env* [^OrtApi ort-api ^long logging-level ^Pointer name]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CreateEnv ort-api logging-level name res)
      (.get res OrtEnv 0))))

(defn session-options* [^OrtApi ort-api]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CreateSessionOptions ort-api res)
      (.get res OrtSessionOptions 0))))

(defn graph-optimization* [^OrtApi ort-api ^OrtSessionOptions opt ^long level]
  (with-check ort-api
    (.SetGraphOptimizationLevel ort-api opt level)))

(defn execution-provider* [^OrtApi ort-api ^OrtSessionOptions opt provider ^long use-arena]
  (with-check ort-api
    (case provider
      :dnnl (onnxruntime/OrtSessionOptionsAppendExecutionProvider_Dnnl opt use-arena)
      :cuda (onnxruntime/OrtSessionOptionsAppendExecutionProvider_CUDA opt use-arena)
      :cpu (onnxruntime/OrtSessionOptionsAppendExecutionProvider_CPU opt use-arena))))

(defn session* [^OrtApi ort-api ^OrtEnv env ^Pointer model-path opt]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CreateSession ort-api env model-path opt res)
      (.get res OrtSession 0))))

(defn allocator* [^OrtApi ort-api]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.GetAllocatorWithDefaultOptions ort-api res)
      (.get res OrtAllocator 0))))

(def ^:dynamic *default-allocator* (allocator* *ort-api*))

(defn input-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (with-release [res (size-t-pointer 1)]
    (with-check ort-api
      (.SessionGetInputCount ort-api sess res)
      (get-entry res 0))))

(defn input-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.SessionGetInputName ort-api sess i allo res)
      (get-string (.get res BytePointer 0)))))

(defn output-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (with-release [res (size-t-pointer 1)]
    (with-check ort-api
      (.SessionGetOutputCount ort-api sess res)
      (get-entry res 0))))

(defn output-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.SessionGetOutputName ort-api sess i allo res)
      (get-string (.get res BytePointer 0)))))

(defn input-type-info* [^OrtApi ort-api ^OrtSession sess ^long i]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.SessionGetInputTypeInfo ort-api sess i res)
      (.get res OrtTypeInfo 0))))

(defn output-type-info* [^OrtApi ort-api ^OrtSession sess ^long i]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.SessionGetOutputTypeInfo ort-api sess i res)
      (.get res OrtTypeInfo 0))))

(defn tensor-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CastTypeInfoToTensorInfo ort-api info res)
      (.get res OrtTensorTypeAndShapeInfo 0))))

(defn sequence-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CastTypeInfoToSequenceTypeInfo ort-api info res)
      (.get res OrtSequenceTypeInfo 0))))

(defn map-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CastTypeInfoToMapTypeInfo ort-api info res)
      (.get res OrtMapTypeInfo 0))))

(defn optional-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.CastTypeInfoToOptionalTypeInfo ort-api info res)
      (.get res OrtOptionalTypeInfo 0))))

(defn type-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [t (int-pointer 1)]
    (with-check ort-api
      (.GetOnnxTypeFromTypeInfo ort-api info t)
      (case (get-entry t 0)
        1 (tensor-info* ort-api info)
        2 (sequence-info* ort-api info)
        3 (map-info* ort-api info)
        6 (optional-info* ort-api info)
        info))))

(defn tensor-type* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (with-release [res (int-pointer 1)]
    (with-check ort-api
      (.GetTensorElementType ort-api info res)
      (get-entry res 0))))

(defn sequence-type* [^OrtApi ort-api ^OrtSequenceTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.GetSequenceElementType ort-api info res)
      (.get res OrtTypeInfo 0))))

(defn key-type* ^long [^OrtApi ort-api ^OrtMapTypeInfo info]
  (with-release [res (int-pointer 1)]
    (with-check ort-api
      (.GetMapKeyType ort-api info res)
      (get-entry res 0))))

(defn value-type* [^OrtApi ort-api ^OrtMapTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (with-check ort-api
      (.GetMapValueType ort-api info res)
      (.get res OrtTypeInfo 0))))

(defn dimensions-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (with-release [res (size-t-pointer 1)]
    (with-check ort-api
      (.GetDimensionsCount ort-api info res)
      (get-entry res 0))))

(defn tensor-element-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (with-release [res (size-t-pointer 1)]
    (with-check ort-api
      (.GetTensorShapeElementCount ort-api info res)
      (get-entry res 0))))

(defn tensor-dimensions* [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (with-release [cnt (dimensions-count* ort-api info)]
    (let-release [res (long-pointer (max 1 cnt))]
      (with-check ort-api
        (.GetDimensions ort-api info res cnt)
        res))))
