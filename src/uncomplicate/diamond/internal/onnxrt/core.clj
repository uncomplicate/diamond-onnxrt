;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.core
  (:require [clojure.string :as st :refer [lower-case split]]
            [uncomplicate.commons
             [core :refer [let-release with-release Releaseable release Info info bytesize size]]
             [utils :refer [enc-keyword dragan-says-ex mask]]]
            [uncomplicate.clojure-cpp
             :refer [get-string get-entry byte-pointer long-pointer null? pointer pointer-pointer
                     pointer-type pointer-vec safe safe2 get-pointer capacity! limit size-t-pointer]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer :all]
             [impl :refer :all]])
  (:import [clojure.lang Seqable IFn AFn]
           [org.bytedeco.javacpp Pointer PointerPointer]
           org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime OrtDnnlProviderOptions OrtTypeInfo OrtTensorTypeAndShapeInfo
            OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo OrtMemoryInfo OrtValue
            OrtThreadingOptions OrtModelMetadata OrtIoBinding OrtSessionOptions OrtRunOptions
            OrtSession OrtCUDAProviderOptionsV2]))

(defprotocol OnnxType
  (onnx-type [this]))

(defprotocol Options
  (verbosity! [this level])
  (severity! [this level])
  (config! [this config])
  (config [this] [this key]))

(defn check-index [^long i ^long cnt object]
  (when-not (< -1 i cnt)
    (throw (IndexOutOfBoundsException. (format "The requested %s name is out of bounds of this %s pointer." object object)))))
;; ================= API ===========================================================================

(defn init-ort-api!
  ([^long ort-api-version]
   (alter-var-root (var *ort-api*)
                   (constantly (api* (onnxruntime/OrtGetApiBase) ort-api-version)))
   (alter-var-root (var *default-allocator*)
                   (constantly (default-allocator* (safe *ort-api*))))
   (alter-var-root (var *clear-bound-inputs*)
                   (constantly (clear-bound-inputs* (safe *ort-api*))))
   (alter-var-root (var *clear-bound-outputs*)
                   (constantly (clear-bound-outputs* (safe *ort-api*)))))
  ([]
   (init-ort-api! onnxruntime/ORT_API_VERSION)))

(init-ort-api!)

(defn version []
  (with-release [p (version* (onnxruntime/OrtGetApiBase))]
    (let [v (mapv parse-long (split (get-string p) #"\."))]
      {:major (v 0)
       :minor (v 1)
       :update (v 2)})))

(defn build-info []
  (with-release [p (build-info* *ort-api*)]
    (get-string p)))

;; =================== Allocator ===================================================================

(def free (free* (safe *default-allocator*)))

(defn get-string*
  ([ptr]
   (get-string* free ptr))
  ([free ptr]
   (if-not (null? ptr)
     (try
       (get-string (byte-pointer ptr))
       (finally (free ptr)))
     nil)))

;; =================== Misc ========================================================================

(defn available-providers []
  (with-release [pprov (available-providers* *ort-api*)]
    (try
      (doall (mapv #(-> (byte-pointer %)
                        (get-string)
                        (st/replace "ExecutionProvider" "")
                        (lower-case)
                        (keyword))
                   (pointer-vec pprov)))
      (finally
        (release-available-providers* *ort-api* pprov)))))

(defn current-gpu-device-id ^long []
  (current-gpu-device-id* *ort-api*))

(defn current-gpu-device-id! [^long id]
  (current-gpu-device-id* *ort-api* id))

;; ===================== Threading Options =========================================================

(defn threading-options
  ([]
   (threading-options* *ort-api*))
  ([options]
   (let [ort-api *ort-api*]
     (let-release [res (threading-options* ort-api)]
       (when-let [num-threads (:intra-op-threads options)]
         (global-intra-op-threads* ort-api res num-threads))
       (when-let [num-threads (:inter-op-threads options)]
         (global-inter-op-threads* ort-api res num-threads))
       (when (contains? options :spin)
         (global-spin-control* ort-api res (if (:spin options) 1 0)))
       (when (contains? options :denormal-as-zero)
         (global-denormal-as-zero* ort-api res))
       res))))

(defn intra-op-threads! [opt! ^long num-threads]
  (if (instance? OrtThreadingOptions opt!)
    (global-intra-op-threads* *ort-api* (safe opt!) (max 0 num-threads))
    (intra-op-threads* *ort-api* (safe opt!) (max 0 num-threads))))

(defn inter-op-threads! [opt! ^long num-threads]
  (if (instance? OrtThreadingOptions opt!)
    (global-inter-op-threads* *ort-api* (safe opt!) num-threads)
    (inter-op-threads* *ort-api* (safe opt!) num-threads)))

(defn spin-control! [opt! allow-spinning]
  (global-spin-control* *ort-api* (safe opt!) (if allow-spinning 1 0)))

(defn denormal-as-zero! [opt!]
  (global-denormal-as-zero* *ort-api* (safe opt!)))

(defn custom-thread-creation! [opt! custom-options] ;;TODO options helper
  (global-custom-thread-creation* *ort-api* (safe opt!) (safe custom-options)))

(defn custom-create-thread! [opt! fn] ;;TODO function helper
  (global-custom-create-thread* *ort-api* (safe opt!) (safe fn)))

(defn custom-join-thread! [opt! fn] ;;TODO function helper
  (global-custom-join-thread* *ort-api* (safe opt!) (safe fn)))

;; ===================== Environment  ==============================================================

(defn environment
  ([logging-level log-name options]
   (with-release [log-name (byte-pointer (if (seq log-name) log-name "default"))]
     (if (instance? OrtThreadingOptions options)
       (env* *ort-api* (enc-keyword ort-logging-level logging-level) log-name (safe options))
       (with-release [threading-opt (threading-options options)]
         (env* *ort-api* (enc-keyword ort-logging-level logging-level) log-name (safe threading-opt))))))
  ([logging-level log-name]
   (with-release [log-name (byte-pointer (if (seq log-name) log-name "default"))]
     (env* *ort-api* (enc-keyword ort-logging-level logging-level) log-name)))
  ([options]
   (environment :warning "default" options))
  ([]
   (environment :warning "default")))

(defn telemetry!
  ([env! enable?]
   (if enable?
     (enable-telemetry* *ort-api* (safe env!))
     (disable-telemetry* *ort-api* (safe env!))))
  ([env!]
   (enable-telemetry* *ort-api* (safe env!))))

(defn telemetry-language! [env! projection]
  (language-projection* *ort-api* (safe env!) (enc-keyword ort-language-projection projection)))

;; ===================== Session Options ==========================================================

(defn options
  ([]
   (session-options* *ort-api*))
  ([clonee]
   (clone-session-options* *ort-api* (safe clonee))))

(defn execution-mode! [opt! mode]
  (execution-mode* *ort-api* (safe opt!) (enc-keyword ort-execution-mode mode)))

(defn profiling!
  ([opt! ^String path]
   (if path
     (with-release [ppath (byte-pointer path)]
       (enable-profiling* *ort-api* (safe opt!) ppath))
     (disable-profiling* *ort-api* (safe opt!))))
  ([opt!]
   (enable-profiling* *ort-api* (safe opt!))))

(defn mem-pattern!
  ([opt! enable?]
   (if enable?
     (enable-mem-pattern* *ort-api* (safe opt!))
     (disable-mem-pattern* *ort-api* (safe opt!))))
  ([opt!]
   (enable-mem-pattern* *ort-api* (safe opt!))))

(defn cpu-mem-arena!
  ([opt! enable?]
   (if enable?
     (enable-cpu-mem-arena* *ort-api* (safe opt!))
     (disable-cpu-mem-arena* *ort-api* (safe opt!))))
  ([opt!]
   (enable-cpu-mem-arena* *ort-api* (safe opt!))))

(defn log-id! [opt! id]
  (with-release [id (byte-pointer (name id))]
    (session-log-id* *ort-api* (safe opt!) id)
    opt!))

(defn graph-optimization! [opt! level]
  (graph-optimization* *ort-api* (safe opt!) (enc-keyword ort-graph-optimization level))
  opt!)

(defn override-dimension! [opt! name ^long value]
  (let [ort-api *ort-api*]
    (if (keyword? name)
      (with-release [name (byte-pointer (enc-keyword onnx-dimension-denotation name))]
        (free-dimension-override-by-denotation* ort-api (safe opt!) name value))
      (with-release [name (byte-pointer (str name))]
        (free-dimension-override-by-name* ort-api (safe opt!) name value)))
    opt!))

(defn disable-per-session-threads! [opt!]
  (disable-per-session-threads* *ort-api* (safe opt!))
  opt!)

(extend-type OrtSessionOptions
  Options
  (config! [opt! config]
    (let [ort-api *ort-api*]
      (doseq [[key value] (seq config)]
        (with-release [config-key (byte-pointer (get ort-session-options-config-keys key (name key)))
                       config-value (byte-pointer ((get ort-session-options-config-encoders key identity) value))]
          (add-session-config-entry* ort-api (safe opt!) config-key config-value)))
      opt!))
  (config
    ([opt key]
     (let [ort-api (safe *ort-api*)
           opt (safe opt)]
       (with-release [config-key (byte-pointer (get ort-session-options-config-keys key (name key)))]
         (if (has-session-config-entry* ort-api opt config-key)
           (with-release [value (get-session-config-entry* ort-api opt config-key)]
             ((get ort-session-options-config-decoders key identity) (get-string value)))
           nil))))
    ([opt]
     (let [ort-api (safe *ort-api*)
           opt (safe opt)]
       (reduce (fn [kv [k v]]
                 (if-let [value (config opt k)]
                   (assoc kv k value)
                   kv))
               {}
               ort-session-options-config-keys))))
  (severity! [opt! ^long level]
    (session-severity* *ort-api* (safe opt!) level)
    opt!)
  (verbosity! [opt! level]
    (session-verbosity* *ort-api* (safe opt!) (enc-keyword ort-logging-level level))
    opt!))

;;TODO 1.23+ SessionGetMemoryInfoForInputs(), SessionGetEpDeviceForInputs() etc.

(defn initializer! [opt! init-name val]
  (with-release [init-name (byte-pointer (name init-name))]
    (initializer* *ort-api* (safe opt!) init-name)))

(defn append-dnnl! [opt! opt-map]
  (with-release [dnnl ^OrtDnnlProviderOptions (dnnl-options* *ort-api*)]
    (.use_arena dnnl (if (:arena opt-map) 1 0))
    (append-dnnl* *ort-api* (safe opt!) dnnl)
    opt!))

(defn config-keys [provider-options-keys opt-map]
  (map #(byte-pointer (or (provider-options-keys %)
                          (dragan-says-ex "Unknown provider option."
                                          {:requested %
                                           :available (keys provider-options-keys)})))
       (keys opt-map)))

(defn config-vals [provider-options-encoders opt-map]
  (map (fn [[k v]]
         (byte-pointer ((get provider-options-encoders k identity) v)))
       opt-map))

(defn append-cuda! [opt! opt-map]
  (let [ort-api (safe *ort-api*)
        stream (:stream opt-map)
        opt-map (dissoc opt-map :stream)]
    (with-release [cuda (safe (cuda-options* ort-api))]
      (with-release [config-keys (config-keys ort-cuda-provider-options-keys opt-map)
                     config-values (config-vals ort-cuda-provider-options-encoders opt-map)
                     ppkeys (safe (pointer-pointer config-keys))
                     ppvalues (safe (pointer-pointer config-values))]
        (update-cuda-options* ort-api cuda ppkeys ppvalues)
        (when stream
          (let [key (byte-pointer "user_compute_stream")]
            (update-cuda-options-with-value* ort-api cuda key stream))))
      (append-cuda* ort-api opt! cuda)
      opt!)))

(defn append-ep! [opt! ep-name opt-map]
  (let [ep-name (enc-keyword ort-execution-provider ep-name)]
    (with-release [config-keys (config-keys ort-coreml-provider-options-keys opt-map)
                   config-values (config-vals ort-coreml-provider-options-encoders opt-map)
                   pname (safe (byte-pointer ep-name))
                   ppkeys (pointer-pointer config-keys)
                   ppvals (pointer-pointer config-values)]
      (append-ep* (safe *ort-api*) (safe opt!) pname ppkeys ppvals)))
  opt!)

(defn append-provider!
  ([opt! provider opt-map]
   (case provider
     :dnnl (append-dnnl! opt! opt-map)
     :cuda (append-cuda! opt! opt-map)
     (append-ep! opt! provider opt-map))
   opt!)
  ([opt! provider]
   (append-provider! opt! provider nil))
  ([opt!]
   (append-provider! opt! :dnnl nil)))

;;TODO create a mechanism to wrap any clojure function into OrtLoggingFunction
(defn user-logging-fn! [opt! logging-fn param]
  (user-logging-function* *ort-api* (safe opt!) (safe logging-fn) (safe2 param))
  opt!)

;; ========================= Session ===============================================================

(defn session
  ([env model-path-or-data options]
   (if (string? model-path-or-data)
     (with-release [model-path (safe (platform-pointer model-path-or-data))]
       (session* *ort-api* (safe env) model-path (safe options)))
     (session-from-array* *ort-api* (safe env) (safe model-path-or-data) (safe options))))
  ([env model-path-or-data options prepackaged-weights]
   (if (string? model-path-or-data)
     (with-release [model-path (safe (platform-pointer model-path-or-data))]
       (session* *ort-api* (safe env) model-path (safe options)
                 (safe prepackaged-weights)))
     (session-from-array* *ort-api* (safe env) (safe model-path-or-data) (safe options)
                          (safe prepackaged-weights)))))

(defn dynamic-options! [sess! config]
  (let [ort-api *ort-api*]
    (with-release [config-keys (map #(-> (get ort-ep-dynamic-options-keys % (name %))
                                         (byte-pointer))
                                    (keys config))
                   config-values (map (fn [[k v]]
                                        (-> ((get ort-ep-dynamic-options-encoders k identity) v)
                                            (byte-pointer)))
                                      config)
                   ppkeys (pointer-pointer config-keys)
                   ppvalues (pointer-pointer config-values)]
      (ep-dynamic-options* ort-api (safe sess!) (safe ppkeys) (safe ppvalues)))
    sess!))

(defn initializer-count ^long [sess]
  (overridable-initializer-count* *ort-api* (safe sess)))

(defn initializer-name
  ([sess ^long i]
   (check-index i (initializer-count sess) "input")
   (get-string* (overridable-initializer-name* *ort-api* (safe sess) (safe *default-allocator*) i)))
  ([sess]
   (let [ort-api *ort-api*
         allo (safe *default-allocator*)]
     (doall (mapv #(get-string* (overridable-initializer-name* ort-api (safe sess) allo %))
                  (range (initializer-count sess)))))))

(defn initializer-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (check-index i (overridable-initializer-count* ort-api sess) "initializer")
     (overridable-initializer-type-info* ort-api sess i)))
  ([sess]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (map #(overridable-initializer-type-info* ort-api sess %)
          (range (overridable-initializer-count* ort-api sess))))))

(defn profiling-start-time ^long [sess]
  (profiling-start-time* *ort-api* (safe sess)))

(defn end-profiling [sess]
  (get-string* (end-profiling* *ort-api* (safe sess) (safe *default-allocator*))))

(defn input-count ^long [sess]
  (input-count* *ort-api* (safe sess)))

(defn input-name
  ([sess ^long i]
   (check-index i (input-count sess) "input")
   (get-string* (input-name* *ort-api* (safe sess) (safe *default-allocator*) i)))
  ([sess]
   (let [ort-api (safe *ort-api*)
         allo (safe *default-allocator*)
         sess (safe sess)]
     (doall (mapv #(get-string* (input-name* ort-api sess allo %))
                  (range (input-count sess)))))))

(defn output-count ^long [sess]
  (output-count* *ort-api* (safe sess)))

(defn output-name
  ([sess ^long i]
   (check-index i (output-count sess) "output")
   (get-string* (output-name* (safe *ort-api*) (safe sess) (safe *default-allocator*) i)))
  ([sess]
   (let [ort-api *ort-api*
         allo (safe *default-allocator*)
         sess (safe sess)]
     (doall (mapv #(get-string* (output-name* ort-api sess allo %))
                  (range (output-count sess)))))))

(defn input-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (check-index i (input-count* ort-api sess) "input")
     (input-type-info* ort-api sess i)))
  ([sess]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (map #(input-type-info* ort-api sess %)
          (range (input-count* ort-api sess))))))

(defn output-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (check-index i (output-count* ort-api sess) "output")
     (output-type-info* ort-api sess i)))
  ([sess]
   (let [ort-api *ort-api*
         sess (safe sess)]
     (map #(output-type-info* ort-api sess %)
          (range (output-count* ort-api sess))))))

(extend-type OrtSession
  Info
  (info
    ([this]
     (let [res {:input (apply hash-map (interleave (input-name this) (info (input-type-info this))))
                :output (apply hash-map (interleave (output-name this) (info (output-type-info this))))}]
       (let [init (apply hash-map (interleave (initializer-name this)
                                              (info (initializer-type-info this))))]
         (if (seq init)
           (assoc res :initializer init)
           res))))
    ([this type-info]
     (case type-info
       :input (apply hash-map (interleave (input-name this) (info (input-type-info this))))
       :output (apply hash-map (interleave (output-name this) (info (output-type-info this))))
       :initializer (apply hash-map (interleave (initializer-name this) (info (initializer-type-info this))))
       nil))))

;; ==================== Model Metadata =============================================================

(defn session-model-metadata [sess]
  (session-model-metadata* *ort-api* (safe sess)))

(defn producer-name [metadata]
  (get-string* (producer-name* *ort-api* (safe metadata) (safe *default-allocator*))))

(defn graph-name [metadata]
  (get-string* (graph-name* *ort-api* (safe metadata) (safe *default-allocator*))))

(defn domain [metadata]
  (get-string* (domain* *ort-api* (safe metadata) (safe *default-allocator*))))

(defn description [metadata]
  (get-string* (description* *ort-api* (safe metadata) (safe *default-allocator*))))

(defn graph-description [metadata]
  (get-string* (graph-description* *ort-api* (safe metadata) (safe *default-allocator*))))

(defn custom-map-keys [metadata]
  (let [allo (safe *default-allocator*)]
    (with-release [map-keys (custom-map-keys* *ort-api* (safe metadata) allo)]
      (doall (mapv get-string* (pointer-vec map-keys))))))

(extend-type OrtModelMetadata
  Info
  (info
    ([this]
     {:producer (producer-name this)
      :description (description this)
      :graph (graph-name this)
      :graph-description (graph-description this)
      :domain (domain this)
      :custom-map-keys (custom-map-keys this)})
    ([this type-info]
     (case type-info
       :producer (producer-name this)
       :description (description this)
       :graph (graph-name this)
       :graph-description (graph-description this)
       :domain (domain this)
       :custom-map-keys (custom-map-keys this)
       nil))))

;; ==================== Initializer Info ===========================================================

;; TODO 1.23+

;; ==================== IO Binding =================================================================

(defn bind-input [binding name value]
  (with-release [name (byte-pointer name)]
    (bind-input* *ort-api* (safe binding) name (safe value))))

(defn bind-output [binding name value-or-mem-info]
  (with-release [name (byte-pointer name)]
    (if (instance? OrtValue value-or-mem-info)
      (bind-output* *ort-api* (safe binding) name (safe value-or-mem-info));;TODO this can be a protocol.
      (bind-output-to-device* *ort-api* (safe binding) name (safe value-or-mem-info)))))

(defn bound-names [binding]
  (let [allo (safe *default-allocator*)]
    (with-release [ppnames (bound-names* (safe *ort-api*) (safe binding) allo)
                   lengths (mapv #(capacity! (size-t-pointer %) 1) (pointer-vec (second ppnames)))]
      (mapv (fn [pname plength]
              (if-not (null? pname)
                (get-string* (capacity! pname (get-entry plength 0)))
                nil))
            (pointer-vec (first ppnames)) lengths))))

(defn bound-values [binding]
  (with-release [ppvalues (bound-values* *ort-api* (safe binding) *default-allocator*)]
    (mapv #(get-pointer % OrtValue 0) (pointer-vec ppvalues))))

(defn clear-bound-inputs
  ([]
   (clear-bound-inputs* (safe *ort-api*)))
  ([binding]
   (clear-bound-inputs* (safe *clear-bound-inputs*) (safe binding))))

(defn clear-bound-outputs
  ([]
   (clear-bound-outputs* (safe *ort-api*)))
  ([binding]
   (clear-bound-outputs* (safe *clear-bound-outputs*) (safe binding))))

(defn io-binding
  ([sess]
   (io-binding* (safe *ort-api*) (safe sess)))
  ([sess bindings]
   (io-binding bindings bindings))
  ([sess inputs outputs]
   (let [ort-api (safe *ort-api*)
         allo (safe *default-allocator*)
         input-cnt (input-count sess)
         output-cnt (output-count sess)]
     (letfn [(get-value [values in-name]
               (let [name-string (get-string* in-name)]
                 (or (inputs name-string)
                     (dragan-says-ex "You have to provide names that match session model's specification."
                                     {:requested (keys values) :expected name-string}))))]
       (let-release [res (safe (io-binding* ort-api (safe sess)))]
         (if (= 1 input-cnt)
           (with-release [in-name (safe (input-name* ort-api sess allo 0))]
             (bind-input res in-name
                         (safe (cond (map? inputs) (get-value inputs in-name)
                                     (sequential? inputs) (first inputs)
                                     :default inputs))))
           (cond (map? inputs)
                 (with-release [in-names (input-names* ort-api sess allo)]
                   (doseq [in-name (pointer-vec in-names)]
                     (bind-input res in-name (get-value inputs in-name))))
                 (sequential? inputs)
                 (let [inputs (vec inputs)]
                   (dotimes [i (count inputs)]
                     (with-release [in-name (input-name* ort-api sess allo i)]
                       (bind-input res in-name (inputs i)))))
                 :default (dragan-says-ex "Unsupported input typeu." {:inputs inputs})))
         (if (= 1 output-cnt)
           (with-release [out-name (safe (output-name* ort-api sess allo 0))]
             (bind-output res out-name
                          (safe (cond (map? outputs) (get-value outputs out-name)
                                      (sequential? outputs) (first outputs)
                                      :default outputs))))
           (cond (map? outputs)
                 (with-release [out-names (output-names* ort-api sess allo)]
                   (doseq [out-name (pointer-vec out-names)]
                     (let [output (get-value outputs out-name)]
                       (if (instance? OrtMemoryInfo output)
                         (bind-output-to-device* ort-api res out-name output)
                         (bind-output* ort-api res out-name output)))))
                 (sequential? outputs)
                 (let [outputs (vec outputs)]
                   (dotimes [i (count outputs)]
                     (with-release [out-name (output-name* ort-api sess allo i)]
                       (let [output (outputs i)]
                         (if (instance? OrtMemoryInfo output)
                           (bind-output-to-device* ort-api res out-name output)
                           (bind-output* ort-api res out-name output))))))
                 :default (dragan-says-ex "Unsupported output type." {:outputs outputs})))
         res)))))

(extend-type OrtIoBinding
  Info
  (info
    ([this]
     {:names (bound-names this)})
    ([this type-info]
     (case type-info
       :names (bound-names this)
       nil))))

(defn binding? [this]
  (instance? OrtIoBinding this))

;; ==================== OrtTypeInfo ================================================================

(defn cast-type [^OrtTypeInfo info]
  (let [ort-api *ort-api*
        info (safe info)]
    (case (type-info-type* ort-api info)
      1 (tensor-info* ort-api info)
      2 (sequence-info* ort-api info)
      3 (map-info* ort-api info)
      6 (optional-info* ort-api info)
      info)))

(defn denotation [^OrtTypeInfo info]
  (let [den (denotation* *ort-api* (safe info))]
    (if (< 0 (size den))
      (keyword (get-string den))
      nil)))

(extend-type OrtTypeInfo
  Info
  (info
    ([this]
     (info (cast-type this)))
    ([this info-type]
     (info (cast-type this) info-type)))
  OnnxType
  (onnx-type [this]
    (dec-onnx-type (type-info-type* *ort-api* (safe this)))))

;; ==================== OrtTensorTypeAndShapeinfo ==================================================

(defn tensor-info [shape type]
  (let [ort-api (safe *ort-api*)]
    (let-release [res (safe (tensor-info* ort-api))]
      (with-release [shape (safe (long-pointer (seq shape)))]
        (tensor-dimensions* ort-api res shape)
        (tensor-type* ort-api res (enc-keyword onnx-data-type type)))
      res)))

(defn scalar? [tensor-info]
  (= 0 (dimensions-count* *ort-api* (safe tensor-info))))

(defn shape [tensor-info]
  (with-release [dims (safe (tensor-dimensions* *ort-api* (safe tensor-info)))]
    (doall (pointer-vec dims))))

(defn shape! [tensor-info! values]
  (let [ort-api *ort-api*
        cnt (dimensions-count* ort-api (safe tensor-info!))]
    (if (<= 0 (count values) cnt)
      (with-release [values (long-pointer (seq values))]
        (tensor-dimensions* ort-api tensor-info! values))
      (dragan-says-ex "You have to provide value for each dimension."
                      {:required cnt :provided (cnt values)}))))

(defn symbolic-shape [tensor-info]
  (let [allo (safe *default-allocator*)]
    (with-release [symbolic-dims (symbolic-dimensions* *ort-api* (safe tensor-info))]
      (doall (mapv #(get-string (byte-pointer %)) (pointer-vec symbolic-dims))))))

(defn symbolic-shape! [tensor-info names]
  (let [ort-api *ort-api*
        cnt (dimensions-count* ort-api tensor-info)]
    (if (= (count names) cnt)
      (with-release [ppnames (pointer-pointer (seq names))]
        (symbolic-dimensions* *ort-api* (safe tensor-info) ppnames))
      (dragan-says-ex "You have to provide name for each dimension."
                      {:required cnt :provided (cnt names)}))))

(defn tensor-type [tensor-info]
  (dec-onnx-data-type (tensor-type* *ort-api* (safe tensor-info))))

(defn tensor-type! [tensor-info type]
  (tensor-type* *ort-api* (safe tensor-info) (enc-keyword onnx-data-type type)))

(defn tensor-count [tensor-info]
  (tensor-element-count* *ort-api* (safe tensor-info)))

(extend-type OrtTensorTypeAndShapeInfo
  Info
  (info
    ([this]
     (if (scalar? this)
       (tensor-type this)
       {:data-type (tensor-type this)
        :shape (shape this)}))
    ([this info-type]
     (case info-type
       :data-type (tensor-type this)
       :shape (shape this)
       :count (tensor-count this)
       :type :tensor
       nil))))

;; ==================== OrtSequenceTypeInfo ========================================================

(defn sequence-type [seq-info]
  (sequence-type* *ort-api* (safe seq-info)))

(extend-type OrtSequenceTypeInfo
  Info
  (info
    ([this]
     {:structure [(with-release [sti (sequence-type this)] (info sti :structure))]})
    ([this type-info]
     (case type-info
       :type :sequence
       :structure [(with-release [sti (sequence-type this)] (info sti :structure))]
       nil))))

;; ==================== OrtOptionalTypeInfo ========================================================

(extend-type OrtOptionalTypeInfo
  Info
  (info
    ([this]
     {:type :optional})
    ([this type-info]
     (case type-info
       :type :optional
       nil))))

;; ==================== OrtMapTypeInfo =============================================================

(defn key-type [map-info]
  (dec-onnx-data-type (key-type* *ort-api* (safe map-info))))

(defn val-type [map-info]
  (val-type* *ort-api* (safe map-info)))

(extend-type OrtMapTypeInfo
  Info
  (info
    ([this]
     {:structure [(key-type this) (with-release [vi (val-type this)] (info vi))]})
    ([this info-type]
     (case info-type
       :type :map
       :structure [(key-type this) (with-release [vi (val-type this)] (info vi))]
       nil))))

;; ======================= OrtMemoryInfo ===========================================================

(defn memory-info
  ([alloc-key alloc-type device-id mem-type]
   (with-release [name (safe (byte-pointer (get ort-allocator-name alloc-key alloc-key)))]
     (memory-info* *ort-api* name
                   (enc-keyword ort-allocator-type alloc-type)
                   device-id
                   (enc-keyword ort-mem-type mem-type))))
  ([alloc-key alloc-type mem-type]
   (memory-info alloc-key alloc-type 0 mem-type))
  ([alloc-key alloc-type]
   (memory-info alloc-key alloc-type 0 :default))
  ([alloc-key]
   (memory-info alloc-key :arena 0 :default))
  ([]
   (memory-info :cpu)))

(defn device-type
  ([]
   (device-type* *ort-api*))
  ([mem-info]
   (dec-ort-memory-info-device-type
    (device-type* (safe (device-type* (safe *ort-api*))) (safe mem-info)))))

(defn device-id [mem-info]
  (device-id* *ort-api* (safe mem-info)))

(defn memory-type [mem-info]
  (dec-ort-memory-type (memory-type* *ort-api* (safe mem-info))))

(defn allocator-key [mem-info]
  (ort-allocator-keyword (get-string (device-name* *ort-api* (safe mem-info)))))

(defn allocator-type [mem-info]
  (dec-ort-allocator-type (allocator-type* *ort-api* (safe mem-info))))

(defn equal-memory-info? [info1 info2]
  (if (or (identical? info1 info2)
          (and info1 info2
               (compare-memory-info* *ort-api* (safe info1) (safe info2))))
    true
    false))

(extend-type OrtMemoryInfo
  Info
  (info
    ([this]
     {:device-id (device-id this)
      :device-type (device-type this)
      :memory-type (memory-type this)
      :allocator-key (allocator-key this)
      :allocator-type (allocator-type this)})
    ([this type-info]
     (case type-info
       :device-id (device-id this)
       :device-type (device-type this)
       :memory-type (memory-type this)
       :allocator-key (allocator-key this)
       :allocator-type (allocator-type this)
       nil))))

;; =================== OrtValue ====================================================================

(defn onnx-tensor
  ([mem-info shape data data-type]
   (let [data (pointer data)]
     (create-tensor* *ort-api* (safe mem-info) (safe data)
                     (safe (long-pointer (seq shape)))
                     (enc-keyword onnx-data-type data-type))))
  ([mem-info-or-alloc shape data-or-type]
   (if (or (keyword data-or-type) (number? data-or-type))
     (allocate-tensor* *ort-api* (safe mem-info-or-alloc)
                       (safe (long-pointer (seq shape)))
                       (enc-keyword pointer-type data-or-type))
     (onnx-tensor mem-info-or-alloc shape data-or-type
                  (enc-keyword pointer-type (type (pointer data-or-type))))))
  ([shape data-type]
   (onnx-tensor *default-allocator* shape data-type)))

(defn value
  ([ptr]
   (get-pointer ptr OrtValue 0))
  ([^PointerPointer pptr ^long i]
   (.get pptr OrtValue i)))

(defn value-info [value]
  (value-info* *ort-api* (safe value)))

(defn value-tensor-info [value]
  (value-tensor-info* *ort-api* (safe value)))

(defn value-count ^long [value]
  (let [ort-api *ort-api*]
    (if (<= 2 (value-type* ort-api (safe value)) 3)
      (value-count* ort-api value)
      1)))

(defn tensor? [value]
  (is-tensor* *ort-api* (safe value)))

(defn value? [value]
  (= 1 (has-value* *ort-api* (safe value))))

(defn none? [value]
  (= 0 (has-value* *ort-api* (safe value))))

;;TODO new in 1.23
#_(defn mutable-data [value]
    (limit! (tensor-mutable-data* *ort-api* value)
            (tensor-size-in-bytes* *ort-api* value)))

(defn mutable-data [value]
  (tensor-mutable-data* *ort-api* (safe value)))

(defn value-value
  ([value ^long i]
   (let [ort-api *ort-api*]
     (check-index i (value-count* ort-api value) "value")
     (value-value* ort-api *default-allocator* value i)))
  ([value]
   (let [ort-api *ort-api*
         allo *default-allocator*]
     (doall (mapv #(value-value* ort-api allo value %)
                  (range (value-count* ort-api (safe value))))))))

(defn map-keys [value]
  (value-value value 0))

(defn map-vals [value]
  (value-value value 1))

(defn onnx-sequence [values]
  (with-release [values (pointer-pointer (seq values))]
    (create-value* *ort-api* onnxruntime/ONNX_TYPE_SEQUENCE (safe values))))

(defn onnx-map [keys values]
  (with-release [kvs (pointer-pointer [keys values])]
    (create-value* *ort-api* onnxruntime/ONNX_TYPE_MAP (safe kvs))))

(extend-type OrtValue
  Info
  (info
    ([this]
     {:value (with-release [vti (value-info this)] (info vti))})
    ([this type-info]
     (case type-info
       :count (value-count this)
       :type :value
       :value (with-release [vti (value-info this)] (info vti))
       nil)))
  OnnxType
  (onnx-type [this]
    (dec-onnx-type (value-type* *ort-api* (safe this)))))

;; ======================== Run Options ============================================================

(defn run-options []
  (run-options* *ort-api*))

(defn run-tag! [run-opt! tag]
  (with-release [tag-name (byte-pointer tag)]
    (run-tag* *ort-api* (safe run-opt!) (safe tag-name))))

(defn run-tag [run-opt tag]
  (get-string (run-tag* *ort-api* (safe run-opt))))

(defn terminate!
  ([run-opt]
   (set-terminate* *ort-api* (safe run-opt)))
  ([run-opt terminate?]
   (if terminate?
     (set-terminate* *ort-api* (safe run-opt))
     (unset-terminate* *ort-api* (safe run-opt)))))

(extend-type OrtRunOptions
  Options
  (config! [run-opt! config]
    (let [ort-api *ort-api*]
      (doseq [[key value] (seq config)]
        (with-release [config-key (byte-pointer (get ort-run-options-config-keys key (name key)))
                       config-value (byte-pointer ((get ort-run-options-config-encoders key identity) value))]
          (add-run-config-entry* ort-api (safe run-opt!) config-key config-value)))
      run-opt!))
  ;; (config ;;TODO 1.23+
  ;;   ([opt key]
  ;;    (let [ort-api (safe *ort-api*)
  ;;          opt (safe opt)]
  ;;      (with-release [config-key (byte-pointer (get ort-run-options-config-keys key (name key)))]
  ;;        (if (has-run-config-entry* ort-api opt config-key)
  ;;          (with-release [value (get-run-config-entry* ort-api opt config-key)]
  ;;            ((get ort-run-options-config-decoders key identity) (get-string value)))
  ;;          nil))))
  ;;   ([opt]
  ;;    (let [ort-api (safe *ort-api*)
  ;;          opt (safe opt)]
  ;;      (reduce (fn [kv [k v]]
  ;;                (if-let [value (config opt k)]
  ;;                  (assoc kv k value)
  ;;                  kv))
  ;;              {}
  ;;              ort-run-options-config-keys))))
  (severity! [opt! ^long level]
    (run-severity* *ort-api* (safe opt!) level)
    opt!)
  (verbosity! [opt! level]
    (run-verbosity* *ort-api* (safe opt!) (enc-keyword ort-logging-level level))
    opt!))

;; ========================== Runner ===============================================================

(deftype Runner [ort-api free sess run-opt in-cnt out-cnt in-names out-names]
  Releaseable
  (release [_]
    (dotimes [i in-cnt]
      (free (get-entry in-names i)))
    (release in-names)
    (dotimes [i out-cnt]
      (free (get-entry out-names i)))
    (release out-names)
    true)
  IFn
  (invoke [this in out]
    (let-release [out (or out (pointer-pointer (repeat out-cnt nil)))]
      (run* (safe ort-api) (safe sess) (safe2 run-opt) in-names (safe in) out-names (safe out))
      out))
  (invoke [this binding]
    (run* (safe ort-api) (safe sess) (safe2 run-opt) (safe binding)))
  (invoke [this]
    run-opt)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn runner*
  ([sess run-opt]
   (let [ort-api (safe *ort-api*)
         allo (safe *default-allocator*)
         sess (safe sess)
         run-opt (safe2 run-opt)
         in-cnt (input-count* ort-api sess)
         out-cnt (output-count* ort-api sess)]
     (->Runner ort-api (free* allo) sess run-opt in-cnt out-cnt
               (input-names* ort-api sess allo)
               (output-names* ort-api sess allo))))
  ([sess]
   (runner* sess nil)))
