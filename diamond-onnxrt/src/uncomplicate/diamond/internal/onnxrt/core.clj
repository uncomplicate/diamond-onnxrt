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
             :refer [get-string get-entry byte-pointer long-pointer null? pointer pointer-pointer pointer-type
                     pointer-vec safe safe2 get-pointer capacity!]]
            [uncomplicate.diamond.internal.onnxrt
             [constants :refer :all]
             [impl :refer :all]])
  (:import [clojure.lang Seqable IFn AFn]
           [org.bytedeco.javacpp Pointer PointerPointer]
           org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime OrtDnnlProviderOptions
            OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo
            OrtMemoryInfo OrtValue OrtThreadingOptions]))

(defprotocol OnnxType
  (onnx-type [this]))

(defn check-index [^long i ^long cnt object]
  (when-not (< -1 i cnt)
    (throw (IndexOutOfBoundsException. (format "The requested %s name is out of bounds of this %s pointer." object object)))))

;; ================= API ===========================================================================

(defn init-ort-api!
  ([^long ort-api-version]
   (alter-var-root (var *ort-api*)
                   (constantly (api* (onnxruntime/OrtGetApiBase) ort-api-version)))
   (alter-var-root (var *default-allocator*)
                   (constantly (default-allocator* *ort-api*))))
  ([]
   (init-ort-api! onnxruntime/ORT_API_VERSION)))

(defn version []
  (with-release [p (version* (onnxruntime/OrtGetApiBase))]
    (let [v (mapv parse-long (split (get-string p) #"\."))]
      {:major (v 0)
       :minor (v 1)
       :update (v 2)})))

(defn build-info []
  (with-release [p (build-info* *ort-api*)]
    (get-string p)))

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

;; ===================== Threading Options ==========================================================

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

(defn execution-mode [opt! mode]
  (execution-mode* *ort-api* (safe opt!) (enc-keyword ort-execution-mode mode)))

(defn profiling!
  ([opt! enable?]
   (if enable?
     (enable-profiling* *ort-api* (safe opt!))
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

(defn severity! [opt! ^long level]
  (session-severity* *ort-api* (safe opt!) level)
  opt!)

(defn verbosity! [opt! level]
  (session-verbosity* *ort-api* (safe opt!) (enc-keyword ort-logging-level level))
  opt!)

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

(defn config! [opt! config]
  (let [ort-api *ort-api*]
    (doseq [[key value] (seq config)]
      (with-release [config-key (byte-pointer (get ort-session-options-config-keys key (name key)))
                     config-value (byte-pointer ((get ort-session-options-config-encoders key identity) value))]
        (add-session-config-entry* ort-api opt! config-key config-value)))
    opt!))

(defn config
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

;;TODO 1.23+ SessionGetMemoryInfoForInputs(), SessionGetEpDeviceForInputs() etc.

(defn initializer! [opt! init-name val]
  (with-release [init-name (byte-pointer (name init-name))]
    (initializer* *ort-api* (safe opt!) init-name)))

;;TODO EnableOrtCustomOps for now, we do not support loading ort custom extensions. When we come to it, we will implement all related to this.

(defn append-dnnl! [opt! opt-map]
  (with-release [dnnl (dnnl-options* *ort-api*)]
    (.use_arena dnnl (get :arena opt-map 0))
    (append-dnnl* *ort-api* (safe opt!) dnnl)
    opt!))

(defn append-cuda! [opt! opt-map]
  (with-release [cuda (cuda-options* *ort-api*)]
    (.use_arena cuda (get :arena opt-map 0))
    (append-cuda* *ort-api* opt! cuda)
    opt!))

(defn append-provider!
  ([opt! provider opt-map]
   (case provider
     :dnnl (append-dnnl! opt! opt-map)
     :cuda (append-cuda! opt! opt-map)
     (dragan-says-ex "Unknown provider. Please use DNNL, CUDA, or one of supported execution providers."
                     {:requested provider :available [:dnnl :cuda]}))
   opt!)
  ([opt! provider]
   (append-provider! opt! provider nil))
  ([opt!]
   (append-provider! opt! :dnnl nil)))

(defn user-logging-fn! [opt! logging-fn param] ;;TODO create a mechanism to wrap any clojure function into OrtLoggingFunction
  (user-logging-function* *ort-api* (safe opt!) (safe logging-fn) (safe2 param))
  opt!)

;; ========================= Session ===============================================================

(defn session
  ([env model-path-or-data options]
   (if (string? model-path-or-data)
     (with-release [model-path (platform-pointer model-path-or-data)]
       (session* *ort-api* (safe env)
                 (safe (platform-pointer model-path))
                 (safe options)))
     (session-from-array* *ort-api* (safe env)
                          (safe model-path-or-data)
                          (safe options))))
  ([env model-path options prepackaged-weights]
   (with-release [model-path (platform-pointer model-path)]
     (session-from-prepackaged-weights* *ort-api* (safe env)
                                        (safe model-path) (safe options)
                                        (safe prepackaged-weights)))))

(defn initializer-count ^long [sess]
  (overridable-initializer-count* *ort-api* (safe sess)))

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

(defn initializer-name
  ([sess ^long i]
   (let [allo (safe *default-allocator*)]
     (check-index i (initializer-count sess) "input")
     (get-string* allo (overridable-initializer-name*
                        *ort-api* (safe sess) allo i))))
  ([sess]
   (let [allo (safe *default-allocator*)
         free (free* allo)]
     (doall (mapv #(get-string* allo free (overridable-initializer-name*
                                           *ort-api* (safe sess) allo %))
                  (range (initializer-count sess)))))))

(defn profiling-start-time ^long [sess]
  (profiling-start-time* *ort-api* (safe sess)))

(defn end-profiling [sess]
  (let [allo (safe *default-allocator*)]
    (get-string* allo (end-profiling* *ort-api* (safe sess) allo))))

(defn input-count ^long [sess]
  (input-count* *ort-api* (safe sess)))

(defn input-name
  ([sess ^long i]
   (let [allo (safe *default-allocator*)]
     (check-index i (input-count sess) "input")
     (get-string* allo (input-name* *ort-api* (safe sess) allo i))))
  ([sess]
   (let [allo (safe *default-allocator*)
         free (free* allo)]
     (doall (mapv #(get-string* allo free (input-name* *ort-api* (safe sess) allo %))
                  (range (input-count sess)))))))

(defn output-count ^long [sess]
  (output-count* *ort-api* (safe sess)))

(defn output-name
  ([sess ^long i]
   (let [allo (safe *default-allocator*)]
     (check-index i (output-count sess) "output")
     (get-string* allo (output-name* *ort-api* (safe sess) allo i))))
  ([sess]
   (let [allo (safe *default-allocator*)
         free (free* allo)]
     (doall (mapv #(get-string* allo free (output-name* *ort-api* (safe sess) allo %))
                  (range (output-count sess)))))))

(defn scalar? [info]
  (= 0 (dimensions-count* *ort-api* (safe info))))

(defn shape [info]
  (with-release [dims (safe (tensor-dimensions* *ort-api* (safe info)))]
    (doall (pointer-vec dims))))

(defn shape! [info! values]
  (let [ort-api *ort-api*
        cnt (dimensions-count* ort-api (safe info!))]
    (if (<= 0 (count values) cnt)
      (with-release [values (long-pointer (seq values))]
        (tensor-dimensions* ort-api info! values))
      (dragan-says-ex "You have to provide value for each dimension."
                      {:required cnt :provided (cnt values)}))))

(defn symbolic-shape [info]
  (let [allo (safe *default-allocator*)
        free (free* allo)]
    (with-release [symbolic-dims (symbolic-dimensions* *ort-api* (safe info))]
      (doall (mapv #(get-string* %) (pointer-vec symbolic-dims))))))

(defn symbolic-shape! [info names]
  (let [ort-api *ort-api*
        cnt (dimensions-count* ort-api info)]
    (if (= (count names) cnt)
      (with-release [ppnames (pointer-pointer (seq names))]
        (symbolic-dimensions* *ort-api* (safe info) ppnames))
      (dragan-says-ex "You have to provide name for each dimension."
                      {:required cnt :provided (cnt names)}))))

(defn cast-type [^OrtTypeInfo info]
  (let [ort-api *ort-api*
        info (safe info)]
    (case (type-info-type* ort-api info)
      1 (tensor-info* ort-api info)
      2 (sequence-info* ort-api info)
      3 (map-info* ort-api info)
      6 (optional-info* ort-api info)
      info)))

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

(defn tensor-type [info]
  (dec-onnx-data-type (tensor-type* *ort-api* (safe info))))

(defn tensor-count [info]
  (tensor-element-count* *ort-api* (safe info)))

(extend-type OrtTensorTypeAndShapeInfo
  Info
  (info
    ([this]
     (if (scalar? this)
       (tensor-type this)
       {:data-type (tensor-type this)
        :shape (shape this)
        :count (tensor-count this)
        :type :tensor}))
    ([this info-type]
     (case info-type
       :data-type (tensor-type this)
       :shape (shape this)
       :count (tensor-count this)
       :type :tensor
       nil))))

(defn sequence-type [info]
  (sequence-type* *ort-api* (safe info)))

(extend-type OrtSequenceTypeInfo
  Info
  (info
    ([this]
     {:type :sequence
      :element (with-release [sti (sequence-type this)] (info sti))})
    ([this type-info]
     (case type-info
       :type :sequence
       :element (with-release [sti (sequence-type this)] (info sti))
       nil))))

(extend-type OrtOptionalTypeInfo
  Info
  (info
    ([this]
     {:type :optional})
    ([this type-info]
     (case type-info
       :type :optional
       nil))))

(defn key-type [info]
  (dec-onnx-data-type (key-type* *ort-api* (safe info))))

(defn val-type [info]
  (val-type* *ort-api* (safe info)))

(extend-type OrtMapTypeInfo
  Info
  (info
    ([this]
     {:type :map
      :key (key-type this)
      :val (with-release [vi (val-type this)] (info vi))})
    ([this info-type]
     (case info-type
       :type :map
       :key (key-type this)
       :val (with-release [vi (val-type this)] (info vi))
       nil))))

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
   (device-type (device-type* *ort-api*) mem-info))
  ([call mem-info]
   (dec-ort-memory-info-device-type (device-type* (safe call) (safe mem-info)))))

(defn device-id [mem-info]
  (device-id* *ort-api* (safe mem-info)))

(defn memory-type [mem-info]
  (dec-ort-memory-type (memory-type* *ort-api* (safe mem-info))))

(defn allocator-key [mem-info]
  (ort-allocator-keyword (get-string (device-name* *ort-api* (safe mem-info)))))

(defn allocator-type [mem-info]
  (dec-ort-allocator-type (allocator-type* *ort-api* (safe mem-info))))

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

(extend-type OrtValue
  OnnxType
  (onnx-type [this]
    (dec-onnx-type (value-type* *ort-api* (safe this)))))

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
  (capacity! (tensor-mutable-data* *ort-api* value)
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
     {:count (value-count this)
      :type :value
      :val (with-release [vti (value-info this)] (info vti))})
    ([this type-info]
     (case type-info
       :count (value-count this)
       :type :value
       :val (with-release [vti (value-info this)] (info vti))
       nil))))

(deftype Runner [ort-api sess opt allo free in-cnt out-cnt in-names out-names]
  Releaseable
  (release [_]
    (dotimes [i in-cnt]
      (free* (safe allo) free (get-entry in-names i)))
    (release in-names)
    (dotimes [i out-cnt]
      (free* (safe allo) free (get-entry out-names i)))
    (release out-names)
    true)
  IFn
  (invoke [this in out]
    (run* ort-api (safe sess) (safe2 opt) in-names in out-names out)
    out)
  (invoke [this in]
    (let-release [out (pointer-pointer (repeat out-cnt nil))]
      (.invoke this in out)
      out))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn runner*
  ([sess opt]
   (let [ort-api (safe *ort-api*)
         allo (safe *default-allocator*)
         free (free* allo)
         sess (safe sess)
         opt (safe2 opt)
         in-cnt (input-count* ort-api sess)
         out-cnt (output-count* ort-api sess)]
     (->Runner ort-api sess opt allo free in-cnt out-cnt
               (input-names* ort-api sess allo)
               (output-names* ort-api sess allo))))
  ([sess]
   (runner* sess nil)))
