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
            OrtAllocator OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo
            OrtStatus OrtArenaCfg OrtCustomOpDomain OrtIoBinding OrtKernelInfo
            OrtMemoryInfo OrtModelMetadata OrtOp OrtOpAttr OrtPrepackedWeightsContainer OrtRunOptions OrtValue]))

(def ^:dynamic *ort-api* (.. (onnxruntime/OrtGetApiBase)
                             (GetApi)
                             (call onnxruntime/ORT_API_VERSION)))

(def platform-pointer (if (.. (Loader/getPlatform) (startsWith "windows"))
                        char-pointer
                        byte-pointer))

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

(defmacro extend-ort [t release-function]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (locking this#
         (when-not (null? this#)
           (. *ort-api* (~release-function this#))
           (.deallocate this#)
           (.setNull this#))
         true))))

(extend-ort OrtEnv ReleaseEnv)
(extend-ort OrtStatus ReleaseStatus)
(extend-ort OrtSession ReleaseSession)
(extend-ort OrtSessionOptions ReleaseSessionOptions)
(extend-ort OrtAllocator ReleaseAllocator)
(extend-ort OrtTypeInfo ReleaseTypeInfo)
(extend-ort OrtTensorTypeAndShapeInfo ReleaseOrtTensorTypeAndShapeInfo)
(extend-ort OrtSequenceTypeInfo ReleaseOrtSequenceTypeInfo)
(extend-ort OrtMapTypeInfo ReleaseOrtMapTypeInfo)
(extend-ort OrtOptionalTypeInfo ReleaseOrtOptionalTypeInfo)
(extend-ort OrtStatus ReleaseOrtStatus)
(extend-ort OrtArenaCfg ReleaseOrtArenaCfg)
(extend-ort OrtCustomOpDomain ReleaseOrtCustomOpDomain)
(extend-ort OrtIoBinding ReleaseOrtIoBinding)
(extend-ort OrtKernelInfo ReleaseOrtKernelInfo)
(extend-ort OrtMemoryInfo ReleaseOrtMemoryInfo)
(extend-ort OrtModelMetadata ReleaseOrtModelMetadata)
(extend-ort OrtOp ReleaseOrtOp)
(extend-ort OrtOpAttr ReleaseOrtOpAttr)
(extend-ort OrtPrepackedWeightsContainer ReleaseOrtPrepackedWeightsContainer)
(extend-ort OrtRunOptions ReleaseOrtRunOptions)
(extend-ort OrtValue ReleaseOrtValue)

(defmacro call-pointer-pointer [ort-api type method & args]
  `(let [ort-api# ~ort-api]
     (with-release [res# (pointer-pointer 1)]
       (with-check ort-api#
         (. ort-api# (~method ~@args res#))
         (.get res# ~type 0)))))

(defmacro call-int [ort-api method & args]
  `(let [ort-api# ~ort-api]
     (with-release [res# (int-pointer 1)]
       (with-check ort-api#
         (. ort-api# (~method ~@args res#))
         (get-entry res# 0)))))

(defmacro call-size-t [ort-api method & args]
  `(let [ort-api# ~ort-api]
     (with-release [res# (size-t-pointer 1)]
       (with-check ort-api#
         (. ort-api# (~method ~@args res#))
         (get-entry res# 0)))))

(defn env* [^OrtApi ort-api ^long logging-level ^Pointer name]
  (call-pointer-pointer ort-api OrtEnv CreateEnv logging-level name))

(defn session-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtSessionOptions CreateSessionOptions))

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
  (call-pointer-pointer ort-api OrtSession CreateSession env model-path opt))

(defn allocator* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtAllocator GetAllocatorWithDefaultOptions))

(def ^:dynamic *default-allocator* (allocator* *ort-api*))

(defn input-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (call-size-t ort-api SessionGetInputCount sess))

(defn input-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (get-string (call-pointer-pointer ort-api BytePointer SessionGetInputName sess i allo)))

(defn output-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (call-size-t ort-api SessionGetOutputCount sess))

(defn output-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (get-string (call-pointer-pointer ort-api BytePointer SessionGetOutputName sess i allo)))

(defn input-type-info* [^OrtApi ort-api ^OrtSession sess ^long i]
  (call-pointer-pointer ort-api OrtTypeInfo SessionGetInputTypeInfo sess i))

(defn output-type-info* [^OrtApi ort-api ^OrtSession sess ^long i]
  (call-pointer-pointer ort-api OrtTypeInfo SessionGetOutputTypeInfo sess i))

(defn tensor-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api OrtTensorTypeAndShapeInfo CastTypeInfoToTensorInfo info))

(defn sequence-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api OrtSequenceTypeInfo CastTypeInfoToSequenceTypeInfo info))

(defn map-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api OrtMapTypeInfo CastTypeInfoToMapTypeInfo info))

(defn optional-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api OrtOptionalTypeInfo CastTypeInfoToOptionalTypeInfo info))

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
  (call-int ort-api GetTensorElementType info))

(defn sequence-type* [^OrtApi ort-api ^OrtSequenceTypeInfo info]
  (call-pointer-pointer ort-api OrtTypeInfo GetSequenceElementType info))

(defn key-type* ^long [^OrtApi ort-api ^OrtMapTypeInfo info]
  (call-int ort-api GetMapKeyType info))

(defn value-type* [^OrtApi ort-api ^OrtMapTypeInfo info]
  (call-pointer-pointer ort-api OrtTypeInfo GetMapValueType info))

(defn dimensions-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (call-size-t ort-api GetDimensionsCount info))

(defn tensor-element-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (call-size-t ort-api GetTensorShapeElementCount info))

(defn tensor-dimensions* [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (with-release [cnt (dimensions-count* ort-api info)]
    (let-release [res (long-pointer (max 1 cnt))]
      (with-check ort-api
        (.GetDimensions ort-api info res cnt)
        res))))
