;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.onnxrt.core
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
            OrtMemoryInfo OrtValue]))

(defprotocol OnnxType
  (onnx-type [this]))

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

(defn options []
  (session-options* *ort-api*))

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

(defn graph-optimization! [opt! level]
  (graph-optimization* *ort-api* opt! (enc-keyword ort-graph-optimization level))
  opt!)

(defn override-dimension! [opt! name ^long value]
  (let [ort-api *ort-api*]
    (if (keyword? name)
      (with-release [name (byte-pointer (enc-keyword onnx-dimension-denotation name))]
        (free-dimension-override-by-denotation* ort-api (safe opt!) name value))
      (with-release [name (byte-pointer (str name))]
        (free-dimension-override-by-name* ort-api (safe opt!) name value)))
    opt!))

(defn environment
  ([logging-level log-name]
   (with-release [log-name (byte-pointer (if (seq log-name) log-name "default"))]
     (env* *ort-api* (enc-keyword ort-logging-level logging-level) log-name)))
  ([]
   (environment :warning "default")))

(defn session [env ^String model-path options]
  (with-release [model-path (platform-pointer model-path)]
    (session* *ort-api* (safe env) (safe (platform-pointer model-path)) (safe options))))

(defn input-count ^long [sess]
  (input-count* *ort-api* (safe sess)))

(defn check-index [^long i ^long cnt object]
  (when-not (< -1 i cnt)
    (throw (IndexOutOfBoundsException. (format "The requested %s name is out of bounds of this %s pointer." object object)))))

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
      (dragan-says-ex "You have to provide value for each dimension." {:required cnt :provided (cnt values)}))))

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
      (dragan-says-ex "You have to provide name for each dimension." {:required cnt :provided (cnt names)}))))

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

(defn create-tensor
  ([mem-info shape data data-type]
   (let [data (pointer data)]
     (create-tensor* *ort-api* (safe mem-info) (safe data) (safe (long-pointer (seq shape)))
                     (enc-keyword onnx-data-type data-type))))
  ([mem-info-or-alloc shape data-or-type]
   (if (or (keyword data-or-type) (number? data-or-type))
     (allocate-tensor* *ort-api* (safe mem-info-or-alloc) (safe (long-pointer (seq shape)))
                       (enc-keyword pointer-type data-or-type))
     (create-tensor mem-info-or-alloc shape data-or-type
                    (enc-keyword pointer-type (type (pointer data-or-type))))))
  ([shape data-type]
   (create-tensor *default-allocator* shape data-type)))

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

(defn create-sequence [values]
  (with-release [values (pointer-pointer (seq values))]
    (create-value* *ort-api* onnxruntime/ONNX_TYPE_SEQUENCE (safe values))))

(defn create-map [keys values]
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
