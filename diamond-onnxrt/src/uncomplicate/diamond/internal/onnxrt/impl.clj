;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.onnxrt.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release size bytesize
                           ;;Info
                           ;; info Viewable view Bytes bytesize Entries sizeof* bytesize*
                           ;; sizeof size
                           ]]
             [utils :as utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer pointer-pointer int-pointer long-pointer byte-pointer char-pointer
                     size-t-pointer get-entry put-entry! get-string capacity! capacity get-pointer]])
  (:import [org.bytedeco.javacpp Loader Pointer BytePointer PointerPointer LongPointer]
           [org.bytedeco.onnxruntime OrtApiBase OrtApi OrtEnv OrtSession OrtSessionOptions
            OrtAllocator OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo
            OrtStatus OrtArenaCfg OrtCustomOpDomain OrtIoBinding OrtKernelInfo
            OrtMemoryInfo OrtModelMetadata OrtOp OrtOpAttr OrtPrepackedWeightsContainer OrtRunOptions OrtValue
            OrtDnnlProviderOptions OrtCUDAProviderOptionsV2

            OrtAllocator$Free_OrtAllocator_Pointer]))

(def ^:dynamic *ort-api*)
(def ^:dynamic *default-allocator*)

(defn api* [^OrtApiBase ort-api-base ^long version]
  (.call (.GetApi ort-api-base) version))

(def platform-pointer (if (.. (Loader/getPlatform) (startsWith "windows"))
                        char-pointer
                        byte-pointer))

(defn ort-error
  [^OrtApi ort-api ^OrtStatus ort-status]
  (let [err (.getString (.call (.GetErrorMessage ort-api) ort-status))]
    (ex-info (format "ONNX runtime error: %s." err)
             {:error err :type :ort-error})))

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
(extend-ort OrtTensorTypeAndShapeInfo ReleaseTensorTypeAndShapeInfo)
(extend-ort OrtSequenceTypeInfo ReleaseSequenceTypeInfo)
(extend-ort OrtMapTypeInfo ReleaseMapTypeInfo)
(extend-ort OrtOptionalTypeInfo ReleaseOptionalTypeInfo)
(extend-ort OrtArenaCfg ReleaseArenaCfg)
(extend-ort OrtCustomOpDomain ReleaseCustomOpDomain)
(extend-ort OrtIoBinding ReleaseIoBinding)
(extend-ort OrtKernelInfo ReleaseKernelInfo)
(extend-ort OrtMemoryInfo ReleaseMemoryInfo)
(extend-ort OrtModelMetadata ReleaseModelMetadata)
(extend-ort OrtOp ReleaseOp)
(extend-ort OrtOpAttr ReleaseOpAttr)
(extend-ort OrtPrepackedWeightsContainer ReleasePrepackedWeightsContainer)
(extend-ort OrtRunOptions ReleaseRunOptions)
(extend-ort OrtValue ReleaseValue)

(extend-type OrtTypeInfo
  Releaseable
  (release [this]
    (locking this
      (when-not (null? this)
        (.ReleaseTypeInfo *ort-api* this)
        (.deallocate this)
        (.setNull this))
      true)))

(defmacro extend-ort-call [t call]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (locking this#
         (when-not (null? this#)
           (with-check *ort-api*
             (.call (. *ort-api* (~call)) this#))
           (.deallocate this#)
           (.setNull this#))
         true))))
;;TODO add other provider options here
(extend-ort-call OrtDnnlProviderOptions ReleaseDnnlProviderOptions)
(extend-ort-call OrtCUDAProviderOptionsV2 ReleaseCUDAProviderOptions)

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

(defn version* [^OrtApiBase ort-api-base]
  (.call (.GetVersionString ort-api-base)))

(defn build-info* [^OrtApi ort-api]
  (.call (.GetBuildInfoString *ort-api*)))

(defn session-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtSessionOptions CreateSessionOptions))

(defn graph-optimization* [^OrtApi ort-api ^OrtSessionOptions opt ^long level]
  (with-check ort-api
    (.SetSessionGraphOptimizationLevel ort-api opt level)
    opt))

(defn available-providers* [^OrtApi ort-api]
  (with-release [cnt (int-pointer 1)]
    (let [res (pointer-pointer nil)]
      (with-check ort-api
        (.GetAvailableProviders ort-api res cnt)
        (capacity! res (get-entry cnt 0))))))

(defn release-available-providers* [^OrtApi ort-api ^PointerPointer providers]
  (with-check ort-api
    (.ReleaseAvailableProviders ort-api providers (capacity providers))))

(defn dnnl-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtDnnlProviderOptions CreateDnnlProviderOptions))

(defn append-dnnl* [^OrtApi ort-api ^OrtSessionOptions opt ^OrtDnnlProviderOptions dnnl-opt]
  (with-check ort-api
    (.SessionOptionsAppendExecutionProvider_Dnnl ort-api opt dnnl-opt)
    opt))

(defn cuda-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtCUDAProviderOptionsV2 CreateCUDAProviderOptions))

(defn append-cuda* [^OrtApi ort-api ^OrtSessionOptions opt ^OrtCUDAProviderOptionsV2 cuda-opt]
  (with-check ort-api
    (.SessionOptionsAppendExecutionProvider_CUDA_V2 ort-api opt cuda-opt)
    opt))

(defn free-dimension-override-by-name* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer name ^long value]
  (with-check ort-api
    (.AddFreeDimensionOverrideByName ort-api opt name value)
    opt))

(defn free-dimension-override-by-denotation* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer denotation ^long value]
  (with-check ort-api
    (.AddFreeDimensionOverride ort-api opt denotation value)
    opt))

(defn session* [^OrtApi ort-api ^OrtEnv env ^Pointer model-path opt]
  (call-pointer-pointer ort-api OrtSession CreateSession env model-path opt))

(defn default-allocator* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtAllocator GetAllocatorWithDefaultOptions))

(defn free*
  ([^OrtAllocator allo ^OrtAllocator$Free_OrtAllocator_Pointer free ^Pointer ptr]
   (.call (.Free allo) allo ptr))
  ([^OrtAllocator allo ^Pointer ptr]
   (.call (.Free allo) allo ptr))
  ([^OrtAllocator allo]
   (.Free allo)))

(defn get-string*
  ([ptr]
   (get-string (get-pointer ptr BytePointer 0)))
  ([allo ptr]
   (try
     (get-string (get-pointer ptr BytePointer 0))
     (finally (free* allo ptr))))
  ([allo free ptr]
   (try
     (get-string (get-pointer ptr BytePointer 0))
     (finally (free* allo free ptr)))))

(defn input-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (call-size-t ort-api SessionGetInputCount sess))

(defn input-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (call-pointer-pointer ort-api BytePointer SessionGetInputName sess i allo))

(defn input-names* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo]
  (let-release [cnt (input-count* ort-api sess)
                res (pointer-pointer cnt)]
    (dotimes [i cnt]
      (put-entry! res i (input-name* ort-api sess allo i)))
    res))

(defn output-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (call-size-t ort-api SessionGetOutputCount sess))

(defn output-name* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo ^long i]
  (call-pointer-pointer ort-api BytePointer SessionGetOutputName sess i allo))

(defn output-names* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo]
  (let-release [cnt (output-count* ort-api sess)
                res (pointer-pointer cnt)]
    (dotimes [i cnt]
      (put-entry! res i (output-name* ort-api sess allo i)))
    res))

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

(defn type-info-type* ^long [^OrtApi ort-api ^OrtTypeInfo info]
  (call-int ort-api GetOnnxTypeFromTypeInfo info))

(defn tensor-type* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (call-int ort-api GetTensorElementType info))

(defn sequence-type* [^OrtApi ort-api ^OrtSequenceTypeInfo info]
  (call-pointer-pointer ort-api OrtTypeInfo GetSequenceElementType info))

(defn key-type* ^long [^OrtApi ort-api ^OrtMapTypeInfo info]
  (call-int ort-api GetMapKeyType info))

(defn val-type* [^OrtApi ort-api ^OrtMapTypeInfo info]
  (call-pointer-pointer ort-api OrtTypeInfo GetMapValueType info))

(defn dimensions-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (call-size-t ort-api GetDimensionsCount info))

(defn tensor-element-count* ^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
  (call-size-t ort-api GetTensorShapeElementCount info))

(defn tensor-dimensions*
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
   (with-release [cnt (dimensions-count* ort-api info)]
     (let-release [res (long-pointer (max 1 cnt))]
       (with-check ort-api
         (.GetDimensions ort-api info res cnt)
         res))))
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info ^LongPointer values]
   (with-check ort-api
     (.SetDimensions ort-api info values (size values))
     info)))

(defn symbolic-dimensions*
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
   (let-release [cnt (dimensions-count* ort-api info)
                 res (pointer-pointer (max 1 cnt))]
     (with-check ort-api
       (.GetSymbolicDimensions ort-api info res cnt)
       res)))
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info ppnames]
   (with-check ort-api
     (.SetSymbolicDimensions ort-api info ppnames (size ppnames))
     info)))

(defn memory-info* [^OrtApi ort-api ^BytePointer name type id mem-type]
  (call-pointer-pointer ort-api OrtMemoryInfo CreateMemoryInfo name (int type) (int id) (int mem-type)))

(defn device-type*
  ([^OrtApi ort-api]
   (.MemoryInfoGetDeviceType ort-api))
  (^long [call ^OrtMemoryInfo mem-info]
   (with-release [res (int-pointer 1)]
     (.call call mem-info res)
     (get-entry res 0))))

(defn device-id* ^long [^OrtApi ort-api ^OrtMemoryInfo mem-info]
  (call-int ort-api MemoryInfoGetId mem-info))

(defn memory-type* ^long [^OrtApi ort-api ^OrtMemoryInfo mem-info]
  (call-int ort-api MemoryInfoGetMemType mem-info))

(defn device-name* [^OrtApi ort-api ^OrtMemoryInfo mem-info]
  (call-pointer-pointer ort-api BytePointer MemoryInfoGetName mem-info))

(defn allocator-type* ^long [^OrtApi ort-api ^OrtMemoryInfo mem-info]
  (call-int ort-api MemoryInfoGetType mem-info))

(defn create-tensor* [^OrtApi ort-api ^OrtMemoryInfo mem-info ^Pointer data shape type]
  (call-pointer-pointer ort-api OrtValue CreateTensorWithDataAsOrtValue mem-info
                        data (bytesize data)
                        shape (size shape)
                        (int type)))

(defn allocate-tensor* [^OrtApi ort-api ^OrtAllocator alloc shape type]
  (call-pointer-pointer ort-api OrtValue CreateTensorWithOrtValue alloc
                        shape (size shape) (int type)))

(defn value-info* [^OrtApi ort-api ^OrtValue value]
  (call-pointer-pointer ort-api OrtTypeInfo GetTypeInfo value))

(defn value-type* ^long [^OrtApi ort-api ^OrtValue value]
  (call-int ort-api GetValueType value))

(defn value-count* [^OrtApi ort-api ^OrtValue value]
  (call-size-t ort-api GetValueCount value))

(defn is-tensor* [^OrtApi ort-api ^OrtValue value]
  (= 1 (call-int ort-api IsTensor value)))

;; TODO new in 1.23.
(defn tensor-size-in-bytes* [^OrtApi ort-api ^OrtValue value]
  (call-size-t ort-api GetTensoriSizeInBytes value))

(defn tensor-mutable-data* [^OrtApi ort-api ^OrtValue value]
  (call-pointer-pointer ort-api Pointer GetTensorMutableData value))

(defn value-value* [^OrtApi ort-api ^OrtAllocator allo ^OrtValue value ^long i]
  (call-pointer-pointer ort-api OrtValue GetValue value i allo))

(defn create-value* [^OrtApi ort-api ^long type ^PointerPointer in]
  (call-pointer-pointer ort-api OrtValue CreateValue in (size in) type))

(defn run* [^OrtApi ort-api ^OrtSession sess ^OrtRunOptions opt
            ^PointerPointer input-names ^PointerPointer inputs
            ^PointerPointer output-names ^PointerPointer outputs]
  (with-check ort-api
    (.Run ort-api sess opt input-names inputs (size inputs) output-names (size output-names) outputs)
    outputs))
