;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release size bytesize]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer pointer-pointer int-pointer long-pointer byte-pointer char-pointer
                     size-t-pointer get-entry put-entry! capacity! capacity get-pointer
                     limit!]])
  (:import [org.bytedeco.javacpp Loader Pointer BytePointer PointerPointer LongPointer SizeTPointer]
           [org.bytedeco.onnxruntime OrtApiBase OrtApi OrtEnv OrtSession OrtSessionOptions
            OrtAllocator OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo
            OrtOptionalTypeInfo OrtStatus OrtArenaCfg OrtCustomOpDomain OrtIoBinding OrtKernelInfo
            OrtMemoryInfo OrtModelMetadata OrtOp OrtOpAttr OrtPrepackedWeightsContainer OrtRunOptions
            OrtValue OrtDnnlProviderOptions OrtCUDAProviderOptions OrtCUDAProviderOptionsV2 OrtLoggingFunction
            OrtThreadingOptions OrtGraph OrtKeyValuePairs OrtLoraAdapter OrtModel OrtNode
            OrtCustomCreateThreadFn OrtCustomJoinThreadFn ;;TODO 1.23+ OrtSyncStream
            OrtAllocator$Free_OrtAllocator_Pointer OrtApi$MemoryInfoGetDeviceType_OrtMemoryInfo_IntPointer
            OrtApi$ClearBoundInputs_OrtIoBinding OrtApi$ClearBoundOutputs_OrtIoBinding]))

(def ^{:dynamic true :tag OrtApi} *ort-api*)
(def ^{:dynamic true :tag OrtAllocator} *default-allocator*)
(def ^{:dynamic true
       :tag OrtApi$ClearBoundInputs_OrtIoBinding}
  *clear-bound-inputs*)

(def ^{:dynamic true
       :tag OrtApi$ClearBoundOutputs_OrtIoBinding}
  *clear-bound-outputs*)

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

(extend-ort OrtAllocator ReleaseAllocator)
(extend-ort OrtArenaCfg ReleaseArenaCfg)
(extend-ort OrtCustomOpDomain ReleaseCustomOpDomain)
(extend-ort OrtEnv ReleaseEnv)
(extend-ort OrtGraph ReleaseGraph)
(extend-ort OrtIoBinding ReleaseIoBinding)
(extend-ort OrtKernelInfo ReleaseKernelInfo)
(extend-ort OrtKeyValuePairs ReleaseKeyValuePairs)
(extend-ort OrtLoraAdapter ReleaseLoraAdapter)
(extend-ort OrtMapTypeInfo ReleaseMapTypeInfo)
(extend-ort OrtMemoryInfo ReleaseMemoryInfo)
(extend-ort OrtModel ReleaseModel)
(extend-ort OrtModelMetadata ReleaseModelMetadata)
(extend-ort OrtNode ReleaseNode)
(extend-ort OrtOp ReleaseOp)
(extend-ort OrtOpAttr ReleaseOpAttr)
(extend-ort OrtPrepackedWeightsContainer ReleasePrepackedWeightsContainer)
(extend-ort OrtRunOptions ReleaseRunOptions)
(extend-ort OrtSequenceTypeInfo ReleaseSequenceTypeInfo)
(extend-ort OrtSession ReleaseSession)
(extend-ort OrtSessionOptions ReleaseSessionOptions)
(extend-ort OrtStatus ReleaseStatus)
(extend-ort OrtTensorTypeAndShapeInfo ReleaseTensorTypeAndShapeInfo)
(extend-ort OrtThreadingOptions ReleaseThreadingOptions)
(extend-ort OrtTypeInfo ReleaseTypeInfo)
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

(extend-ort-call OrtDnnlProviderOptions ReleaseDnnlProviderOptions)
(extend-ort-call OrtCUDAProviderOptionsV2 ReleaseCUDAProviderOptions)

(defmacro call-pointer-pointer [ort-api type method & args]
  `(let [ort-api# ~ort-api]
     (with-release [res# (pointer-pointer 1)]
       (with-check ort-api#
         (. ort-api# (~method ~@args res#))
         (.get res# ~type 0)))))

(defmacro call-int [ort-api method & args]
  `(int (let [ort-api# ~ort-api]
          (with-release [res# (int-pointer 1)]
            (with-check ort-api#
              (. ort-api# (~method ~@args res#))
              (get-entry res# 0))))))

(defmacro call-long [ort-api method & args]
  `(long (let [ort-api# ~ort-api]
          (with-release [res# (long-pointer 1)]
            (with-check ort-api#
              (. ort-api# (~method ~@args res#))
              (get-entry res# 0))))))

(defmacro call-size-t [ort-api method & args]
  `(long (let [ort-api# ~ort-api]
          (with-release [res# (size-t-pointer 1)]
            (with-check ort-api#
              (. ort-api# (~method ~@args res#))
              (get-entry res# 0))))))

;; ================= OrtApi ========================================================================

(defn version*
  ([^OrtApiBase ort-api-base]
   (.call (.GetVersionString ort-api-base)))
  ([^OrtApi ort-api ^OrtModelMetadata metadata]
   (call-long ort-api ModelMetadataGetVersion metadata)))

(defn build-info* [^OrtApi ort-api]
  (.call (.GetBuildInfoString *ort-api*)))

;; ============ Misc ===============================================================================

(defn available-providers* [^OrtApi ort-api]
  (let-release [res (pointer-pointer nil)]
    (capacity! res (call-int ort-api GetAvailableProviders res))))

(defn release-available-providers* [^OrtApi ort-api ^PointerPointer providers]
  (with-check ort-api
    (.ReleaseAvailableProviders ort-api providers (size providers))))

(defn current-gpu-device-id*
  (^long [^OrtApi ort-api]
   (call-int ort-api GetCurrentGpuDeviceId))
  ([^OrtApi ort-api ^long id]
   (with-check ort-api
     (.SetCurrentGpuDeviceId ort-api (int-pointer [id]))
     ort-api)))

;; ================= Allocators ================================================

(defn default-allocator* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtAllocator GetAllocatorWithDefaultOptions))

;;TODO 1.23+ allocator-stats*
;;TODO 1.23+ shared-allocator*

(defn free*
  ([^OrtAllocator allo ^OrtAllocator$Free_OrtAllocator_Pointer free ^Pointer ptr]
   (when-not (null? ptr)
     (.call free allo ptr))
   allo)
  ([^OrtAllocator allo ^Pointer ptr]
   (free* allo (.Free allo) ptr))
  ([^OrtAllocator allo]
   (let [free (.Free allo)]
     (fn [^Pointer ptr]
       (when-not (null? ptr)
         (free* allo free ptr))
       free))))

;; ===================== OrtEnv ====================================================================

(defn env*
  ([^OrtApi ort-api ^long logging-level ^BytePointer name]
   (call-pointer-pointer ort-api OrtEnv CreateEnv logging-level name))
  ([^OrtApi ort-api ^long logging-level ^BytePointer name ^OrtThreadingOptions opts]
   (call-pointer-pointer ort-api OrtEnv CreateEnvWithGlobalThreadPools logging-level name opts)))

(defn enable-telemetry* [^OrtApi ort-api ^OrtEnv env]
  (with-check ort-api
    (.EnableTelemetryEvents ort-api env)
    env))

(defn disable-telemetry* [^OrtApi ort-api ^OrtEnv env]
  (with-check ort-api
    (.DisableTelemetryEvents ort-api env)
    env))

(defn language-projection* [^OrtApi ort-api ^OrtEnv env ^long projection]
  (with-check ort-api
    (.SetLanguageProjection ort-api env projection)
    env))

;; ===================== OrtThreadingOptions =======================================================

(defn threading-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtThreadingOptions CreateThreadingOptions))

(defn global-intra-op-threads* [^OrtApi ort-api ^OrtThreadingOptions threading-opt
                                ^long num-threads]
  (with-check ort-api
    (.SetGlobalIntraOpNumThreads ort-api threading-opt num-threads)
    threading-opt))

(defn global-inter-op-threads* [^OrtApi ort-api ^OrtThreadingOptions threading-opt
                                ^long num-threads]
  (with-check ort-api
    (.SetGlobalInterOpNumThreads ort-api threading-opt num-threads)
    threading-opt))

(defn global-spin-control* [^OrtApi ort-api ^OrtThreadingOptions threading-opt
                            ^long allow-spinning]
  (with-check ort-api
    (.SetGlobalSpinControl ort-api threading-opt allow-spinning)
    threading-opt))

(defn global-denormal-as-zero* [^OrtApi ort-api ^OrtThreadingOptions threading-opt]
  (with-check ort-api
    (.SetGlobalDenormalAsZero ort-api threading-opt)
    threading-opt))

(defn global-custom-thread-creation* [^OrtApi ort-api ^OrtThreadingOptions threading-opt ^Pointer custom-options]
  (with-check ort-api
    (.SetGlobalCustomThreadCreationOptions ort-api threading-opt custom-options)
    threading-opt))

(defn global-custom-create-thread* [^OrtApi ort-api ^OrtThreadingOptions threading-opt
                                    ^OrtCustomCreateThreadFn fn]
  (with-check ort-api
    (.SetGlobalCustomCreateThreadFn ort-api threading-opt fn)
    threading-opt))

(defn global-custom-join-thread* [^OrtApi ort-api ^OrtThreadingOptions threading-opt
                                    ^OrtCustomCreateThreadFn fn]
  (with-check ort-api
    (.SetGlobalCustomJoinThreadFn ort-api threading-opt fn)
    threading-opt))

;; ===================== OrtSessionOptions =========================================================

(defn session-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtSessionOptions CreateSessionOptions))

(defn clone-session-options*
  ([^OrtApi ort-api ^OrtSessionOptions opt]
   (call-pointer-pointer ort-api OrtSessionOptions CloneSessionOptions opt)))

(defn execution-mode* [^OrtApi ort-api ^OrtSessionOptions opt ^long mode]
  (with-check ort-api
    (.SetSessionExecutionMode ort-api opt mode)
    opt))

(defn enable-profiling* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer path]
  (with-check ort-api
    (.EnableProfiling ort-api opt path)
    opt))

(defn disable-profiling* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.DisableProfiling ort-api opt)
    opt))

(defn enable-mem-pattern* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.EnableMemPattern ort-api opt)
    opt))

(defn disable-mem-pattern* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.DisableMemPattern ort-api opt)
    opt))

(defn enable-cpu-mem-arena* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.EnableCpuMemArena ort-api opt)
    opt))

(defn disable-cpu-mem-arena* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.DisableCpuMemArena ort-api opt)
    opt))

(defn session-log-id* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer log-id]
  (with-check ort-api
    (.SetSessionLogId ort-api opt log-id)
    opt))

(defn session-severity* [^OrtApi ort-api ^OrtSessionOptions opt ^long level]
  (with-check ort-api
    (.SetSessionLogSeverityLevel ort-api opt level)
    opt))

(defn session-verbosity* [^OrtApi ort-api ^OrtSessionOptions opt ^long level]
  (with-check ort-api
    (.SetSessionLogVerbosityLevel ort-api opt level)
    opt))

(defn intra-op-threads* [^OrtApi ort-api ^OrtSessionOptions opt ^long num-threads]
  (with-check ort-api
    (.SetIntraOpNumThreads ort-api opt num-threads)
    opt))

(defn inter-op-threads* [^OrtApi ort-api ^OrtSessionOptions opt ^long num-threads]
  (with-check ort-api
    (.SetInterOpNumThreads ort-api opt num-threads)
    opt))

(defn graph-optimization* [^OrtApi ort-api ^OrtSessionOptions opt ^long level]
  (with-check ort-api
    (.SetSessionGraphOptimizationLevel ort-api opt level)
    opt))

(defn user-logging-function* [^OrtApi ort-api ^OrtSessionOptions opt
                              ^OrtLoggingFunction user-logging-fn ^Pointer param]
  (with-check ort-api
    (.SetUserLoggingFunction ort-api opt user-logging-fn param)
    opt))

(defn dnnl-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtDnnlProviderOptions CreateDnnlProviderOptions))

(defn append-dnnl* [^OrtApi ort-api ^OrtSessionOptions opt ^OrtDnnlProviderOptions dnnl-opt]
  (with-check ort-api
    (.SessionOptionsAppendExecutionProvider_Dnnl ort-api opt dnnl-opt)
    opt))

(defn cuda-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtCUDAProviderOptionsV2 CreateCUDAProviderOptions))

(defn update-cuda-options* [^OrtApi ort-api ^OrtCUDAProviderOptionsV2 opt ^PointerPointer keys ^PointerPointer values]
  (with-check ort-api
    (.UpdateCUDAProviderOptions ort-api opt keys values (size keys))
    opt))

(defn update-cuda-options-with-value* [^OrtApi ort-api ^OrtCUDAProviderOptionsV2 opt ^BytePointer key ^Pointer value]
  (with-check ort-api
    (.UpdateCUDAProviderOptionsWithValue ort-api opt key value)
    opt))

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

(defn disable-per-session-threads* [^OrtApi ort-api ^OrtSessionOptions opt]
  (with-check ort-api
    (.DisablePerSessionThreads ort-api opt)
    opt))

(defn add-session-config-entry* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer key ^BytePointer value]
  (with-check ort-api
    (.AddSessionConfigEntry ort-api opt key value)
    opt))

(defn has-session-config-entry* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer key]
  (= 1 (call-int ort-api HasSessionConfigEntry opt key)))

(defn get-session-config-entry*
  ([^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer key]
   (let [none nil
         actual-size (call-size-t ort-api GetSessionConfigEntry opt key ^BytePointer none)]
     (let-release [res (byte-pointer actual-size)]
       (limit! (get-session-config-entry* ort-api opt key res) (dec actual-size)))))
  ([^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer key ^BytePointer res]
   (with-release [actual-size (size-t-pointer [(capacity res)])]
     (with-check ort-api
       (.GetSessionConfigEntry ort-api opt key res actual-size)
       (limit! res (dec (long (get-entry actual-size 0))))))))

(defn initializer* [^OrtApi ort-api ^OrtSessionOptions opt ^BytePointer name ^OrtValue val]
  (with-check ort-api
    (.AddInitializer ort-api opt name val)
    opt))

;; ================================== Session ==================================

(defn session*
  ([^OrtApi ort-api ^OrtEnv env
    ^Pointer model-path ^OrtSessionOptions opt]
   (call-pointer-pointer ort-api OrtSession CreateSession env model-path opt))
  ([^OrtApi ort-api ^OrtEnv env
    ^Pointer model-path ^OrtSessionOptions opt
    ^OrtPrepackedWeightsContainer prepacked-weihgts-container]
   (call-pointer-pointer ort-api OrtSession CreateSessionWithPrepackedWeightsContainer env
                         model-path opt prepacked-weihgts-container)))

(defn session-from-array*
  ([^OrtApi ort-api ^OrtEnv env
    ^Pointer model-data ^OrtSessionOptions opt]
   (call-pointer-pointer ort-api OrtSession CreateSessionFromArray env
                         model-data (size model-data) opt))
  ([^OrtApi ort-api ^OrtEnv env
    ^Pointer model-data ^OrtSessionOptions opt
    ^OrtPrepackedWeightsContainer prepacked-weihgts-container]
   (call-pointer-pointer ort-api OrtSession CreateSessionFromArrayWithPrepackedWeightsContainer env
                         model-data (size model-data) opt prepacked-weihgts-container)))

(defn run*
  ([^OrtApi ort-api ^OrtSession sess ^OrtRunOptions run-opt
    ^PointerPointer input-names ^PointerPointer inputs
    ^PointerPointer output-names ^PointerPointer outputs]
   (with-check ort-api
     (.Run ort-api sess run-opt input-names inputs (size inputs) output-names (size output-names) outputs)
     outputs))
  ([^OrtApi ort-api ^OrtSession sess ^OrtRunOptions run-opt ^OrtIoBinding binding]
   (with-check ort-api
     (.RunWithBinding ort-api sess run-opt binding)
     binding)))

(defn prepackaged-weights* [^OrtApi ort-api]
  (call-pointer-pointer ort-api
      OrtPrepackedWeightsContainer CreatePrepackedWeightsContainer))

(defn overridable-initializer-count* ^long [^OrtApi ort-api ^OrtSession sess]
  (call-size-t ort-api SessionGetOverridableInitializerCount sess))

(defn overridable-initializer-name* [^OrtApi ort-api ^OrtSession sess
                                     ^OrtAllocator allo ^long i]
  (call-pointer-pointer ort-api
      BytePointer SessionGetOverridableInitializerName sess i allo))

(defn overridable-initializer-type-info* [^OrtApi ort-api ^OrtSession sess ^long i]
  (call-pointer-pointer ort-api
      OrtTypeInfo SessionGetOverridableInitializerTypeInfo sess i))

(defn profiling-start-time* [^OrtApi ort-api ^OrtSession sess]
  (call-long ort-api SessionGetProfilingStartTimeNs sess))

(defn end-profiling* [^OrtApi ort-api ^OrtSession sess ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer SessionEndProfiling sess allo))

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

(defn ep-dynamic-options* [^OrtApi ort-api ^OrtSession sess
                           ^PointerPointer keys ^PointerPointer values]
  (with-check ort-api
    (.SetEpDynamicOptions ort-api sess keys values (size keys))
    sess))

;; ==================== Model Metadata =============================================================

(defn session-model-metadata* [^OrtApi ort-api ^OrtSession sess]
  (call-pointer-pointer ort-api OrtModelMetadata SessionGetModelMetadata sess))

(defn producer-name* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer ModelMetadataGetProducerName metadata allo))

(defn graph-name* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer ModelMetadataGetGraphName metadata allo))

(defn domain* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer ModelMetadataGetDomain metadata allo))

(defn description* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer ModelMetadataGetDescription metadata allo))

(defn graph-description* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (call-pointer-pointer ort-api BytePointer ModelMetadataGetDescription metadata allo))

(defn custom-map-keys* [^OrtApi ort-api ^OrtModelMetadata metadata ^OrtAllocator allo]
  (with-release [res (pointer-pointer nil)
                 cnt (long-pointer 1)]
    (with-check ort-api
      (.ModelMetadataGetCustomMetadataMapKeys ort-api metadata allo res cnt)
      (capacity! res (get-entry cnt 0)))))

;; ==================== IO Binding =================================================================

(defn io-binding* [^OrtApi ort-api ^OrtSession sess]
  (call-pointer-pointer ort-api OrtIoBinding CreateIoBinding sess))

(defn bind-input* [^OrtApi ort-api ^OrtIoBinding binding ^BytePointer name ^OrtValue value]
  (with-check ort-api
    (.BindInput ort-api binding name value)
    binding))

(defn bind-output* [^OrtApi ort-api ^OrtIoBinding binding ^BytePointer name ^OrtValue value]
  (with-check ort-api
    (.BindOutput ort-api binding name value)
    binding))

(defn bind-output-to-device* [^OrtApi ort-api ^OrtIoBinding binding
                              ^BytePointer name ^OrtMemoryInfo mem-info]
  (with-check ort-api
    (.BindOutputToDevice ort-api binding name mem-info)
    binding))

(defn bound-names* [^OrtApi ort-api ^OrtIoBinding binding ^OrtAllocator allo]
  (let [res-cnt (with-release [pp (pointer-pointer nil)]
                  (call-size-t ort-api GetBoundOutputValues binding allo pp))]
    (let-release [res (pointer-pointer res-cnt)
                  lengths (pointer-pointer res-cnt)]
      (let [cnt (call-size-t ort-api GetBoundOutputNames binding allo res lengths)]
        (capacity! res cnt)
        (capacity! lengths cnt)
        [res lengths]))))

(defn bound-values* [^OrtApi ort-api ^OrtIoBinding binding ^OrtAllocator allo]
  (let-release [res (pointer-pointer nil)]
    (capacity! res (call-size-t ort-api GetBoundOutputValues binding allo res))))

(defn clear-bound-inputs* [^OrtApi ort-api]
  (.ClearBoundInputs ort-api))

(defn clear-bound-outputs* [^OrtApi ort-api]
  (.ClearBoundOutputs ort-api))

;; =================== Run Options =================================================================

(defn run-options* [^OrtApi ort-api]
  (call-pointer-pointer ort-api OrtRunOptions CreateRunOptions))

(defn run-severity*
  ([^OrtApi ort-api ^OrtRunOptions run-opt ^long level]
   (with-check ort-api
     (.RunOptionsSetRunLogSeverityLevel ort-api run-opt level)
     run-opt))
  (^long [^OrtApi ort-api ^OrtRunOptions run-opt]
   (call-int ort-api RunOptionsGetRunLogSeverityLevel run-opt)))

(defn run-verbosity*
  ([^OrtApi ort-api ^OrtRunOptions run-opt ^long level]
   (with-check ort-api
     (.RunOptionsSetRunLogVerbosityLevel ort-api run-opt level)
     run-opt))
  (^long [^OrtApi ort-api ^OrtRunOptions run-opt]
   (call-int ort-api RunOptionsGetRunLogVerbosityLevel run-opt)))

(defn run-tag*
  ([^OrtApi ort-api ^OrtRunOptions run-opt ^BytePointer tag]
   (with-check ort-api
     (.RunOptionsSetRunTag ort-api run-opt tag)
     run-opt))
  ([^OrtApi ort-api ^OrtRunOptions run-opt]
   (call-pointer-pointer ort-api BytePointer RunOptionsGetRunTag run-opt)))

(defn set-terminate* [^OrtApi ort-api ^OrtRunOptions run-opt]
  (with-check ort-api
    (.RunOptionsSetTerminate ort-api run-opt)))

(defn unset-terminate* [^OrtApi ort-api ^OrtRunOptions run-opt]
  (with-check ort-api
    (.RunOptionsUnsetTerminate ort-api run-opt)))

(defn add-run-config-entry* [^OrtApi ort-api ^OrtRunOptions run-opt ^BytePointer key ^BytePointer value]
  (with-check ort-api
    (.AddRunConfigEntry ort-api run-opt key value)
    run-opt))

;; TODO 1.23+ get-run-config-entry*

;; ==================== OrtTypeInfo ================================================================

(defn tensor-info*
  ([^OrtApi ort-api ^OrtTypeInfo info]
   (call-pointer-pointer ort-api
       OrtTensorTypeAndShapeInfo CastTypeInfoToTensorInfo info))
  ([^OrtApi ort-api]
    (call-pointer-pointer ort-api OrtTensorTypeAndShapeInfo CreateTensorTypeAndShapeInfo)))

(defn sequence-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api
      OrtSequenceTypeInfo CastTypeInfoToSequenceTypeInfo info))

(defn map-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api OrtMapTypeInfo CastTypeInfoToMapTypeInfo info))

(defn optional-info* [^OrtApi ort-api ^OrtTypeInfo info]
  (call-pointer-pointer ort-api
      OrtOptionalTypeInfo CastTypeInfoToOptionalTypeInfo info))

(defn type-info-type* ^long [^OrtApi ort-api ^OrtTypeInfo info]
  (call-int ort-api GetOnnxTypeFromTypeInfo info))

(defn denotation* [^OrtApi ort-api ^OrtTypeInfo info]
  (with-release [res (pointer-pointer 1)]
    (let [actual-size (max 0 (dec (call-size-t ort-api GetDenotationFromTypeInfo info res)))]
      (limit! (.get res BytePointer 0) actual-size))))

;; ==================== OrtTensorTypeAndShapeinfo ==================================================

(defn tensor-type*
  (^long [^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info]
   (call-int ort-api GetTensorElementType info))
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info ^long type]
   (with-check ort-api
     (.SetTensorElementType ort-api info type)
     info)))

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
                 res (pointer-pointer cnt)]
     (with-check ort-api
       (.GetSymbolicDimensions ort-api info res cnt)
       res)))
  ([^OrtApi ort-api ^OrtTensorTypeAndShapeInfo info ^PointerPointer ppnames]
   (with-check ort-api
     (.SetSymbolicDimensions ort-api info ppnames (size ppnames))
     info)))

;; =================== Memory Info =================================================================

;;TODO use v2 from 1.23+
(defn memory-info* [^OrtApi ort-api ^BytePointer name type id mem-type]
  (call-pointer-pointer ort-api
      OrtMemoryInfo CreateMemoryInfo name (int type) (int id) (int mem-type)))

(defn compare-memory-info* [^OrtApi ort-api ^OrtMemoryInfo info1 ^OrtMemoryInfo info2]
  (= 0 (call-int ort-api CompareMemoryInfo info1 info2)))

(defn device-type*
  ([^OrtApi ort-api]
   (.MemoryInfoGetDeviceType ort-api))
  (^long [^OrtApi$MemoryInfoGetDeviceType_OrtMemoryInfo_IntPointer call ^OrtMemoryInfo mem-info]
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

;; =================== OrtValue ====================================================================
;;TODO 1.23+
(defn create-tensor* [^OrtApi ort-api ^OrtMemoryInfo mem-info ^Pointer data ^LongPointer shape type]
  (call-pointer-pointer ort-api OrtValue CreateTensorWithDataAsOrtValue mem-info
                        data (bytesize data)
                        shape (size shape)
                        (int type)))

(defn allocate-tensor* [^OrtApi ort-api ^OrtAllocator alloc ^LongPointer shape type]
  (call-pointer-pointer ort-api OrtValue CreateTensorAsOrtValue alloc
                        shape (size shape) (int type)))

(defn value-info* [^OrtApi ort-api ^OrtValue value]
  (call-pointer-pointer ort-api OrtTypeInfo GetTypeInfo value))

(defn value-tensor-info* [^OrtApi ort-api ^OrtValue value]
  (call-pointer-pointer ort-api OrtTensorTypeAndShapeInfo GetTensorTypeAndShape value))

(defn value-type* ^long [^OrtApi ort-api ^OrtValue value]
  (call-int ort-api GetValueType value))

(defn value-count* [^OrtApi ort-api ^OrtValue value]
  (call-size-t ort-api GetValueCount value))

(defn is-tensor* [^OrtApi ort-api ^OrtValue value]
  (= 1 (call-int ort-api IsTensor value)))

(defn has-value* ^long [^OrtApi ort-api ^OrtValue value]
  (call-int ort-api HasValue value))

;; TODO new in 1.23.
#_(defn tensor-size-in-bytes* [^OrtApi ort-api ^OrtValue value]
  (call-size-t ort-api GetTensoriSizeInBytes value))

(defn tensor-mutable-data* [^OrtApi ort-api ^OrtValue value]
  (call-pointer-pointer ort-api Pointer GetTensorMutableData value))

(defn value-value* [^OrtApi ort-api ^OrtAllocator allo ^OrtValue value ^long i]
  (call-pointer-pointer ort-api OrtValue GetValue value i allo))

(defn create-value* [^OrtApi ort-api ^long type ^PointerPointer in]
  (call-pointer-pointer ort-api OrtValue CreateValue in (size in) type))
