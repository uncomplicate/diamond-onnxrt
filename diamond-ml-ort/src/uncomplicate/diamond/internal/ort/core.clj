;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.ort.core
  (:require [clojure.string :as st :refer [lower-case split]]
            [uncomplicate.commons
             [core :refer [let-release with-release Releaseable view Info info bytesize size]]
             [utils :refer [enc-keyword dragan-says-ex mask]]]
            [uncomplicate.clojure-cpp :refer [get-string byte-pointer null? pointer pointer-type pointer-seq safe]]
            [uncomplicate.diamond.internal.ort
             [constants :refer :all]
             [impl :refer :all]])
  (:import org.bytedeco.javacpp.Pointer
           org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime OrtDnnlProviderOptions
            OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo]))

(defn init-ort-api!
  ([^long ort-api-version]
   (alter-var-root (var *ort-api*)
                   (constantly (api* (onnxruntime/OrtGetApiBase) ort-api-version)))
   (alter-var-root (var *default-allocator*)
                   (constantly (allocator* *ort-api*))))
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
  (let [pprov (available-providers* *ort-api*)]
    (try
      (vec (doall (mapv #(-> (byte-pointer %)
                             (get-string)
                             (st/replace "ExecutionProvider" "")
                             (lower-case)
                             (keyword))
                        (pointer-seq pprov))))
      (finally
        (release-available-providers* *ort-api* pprov)))))

(defn options []
  (session-options* *ort-api*))

(defn append-dnnl! [opt! opt-map]
  (with-release [dnnl (dnnl-options* *ort-api*)]
    (.use_arena dnnl (get :arena opt-map 0))
    (append-dnnl* *ort-api* opt! dnnl)
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

(defn graph-optimization [opt! level]
  (graph-optimization* *ort-api* opt! (enc-keyword ort-graph-optimization level)))

(defn environment
  ([logging-level name]
   (with-release [name (platform-pointer name)]
     (env* *ort-api* (enc-keyword ort-logging-level logging-level) name)))
  ([]
   (environment :warning "default")))

(defn session [env ^String model-path options]
  (with-release [model-path (platform-pointer model-path)]
    (session* *ort-api* env (platform-pointer model-path) options)))

(defn input-count ^long [sess]
  (input-count* *ort-api* sess))

(defn check-index [^long i ^long cnt object]
  (when-not (< -1 i cnt)
    (throw (IndexOutOfBoundsException. (format "The requested %s name is out of bounds of this session %s pointer." object object)))))

(defn input-name
  ([sess ^long i]
   (check-index i (input-count sess) "input")
   (get-string* (input-name* *ort-api* sess i *default-allocator*)))
  ([sess]
   (let [allo *default-allocator*
         free (free* allo)]
     (doall (map #(get-string* allo free (input-name* *ort-api* sess allo %))
                 (range (input-count sess)))))))

(defn output-count ^long [sess]
  (output-count* *ort-api* sess))

(defn output-name
  ([sess ^long i]
   (check-index i (output-count sess) "output")
   (get-string* (output-name* *ort-api* sess i *default-allocator*)))
  ([sess]
   (let [allo *default-allocator*
         free (free* allo)]
     (doall (map #(get-string* allo free (output-name* *ort-api* sess allo %))
                 (range (output-count sess)))))))

(defn scalar? [info]
  (= 0 (dimensions-count* *ort-api* info)))

(defn shape [info]
  (with-release [dims (safe (tensor-dimensions* *ort-api* info))]
    (vec (doall (pointer-seq dims)))))

(defn cast-type [info]
  (cast-type* *ort-api* info))

(extend-type OrtTypeInfo
  Info
  (info
    ([this]
     (info (cast-type this)))
    ([this info-type]
     (info (cast-type this) info-type))))

(defn tensor-type [info]
  (dec-onnx-data-type (tensor-type* *ort-api* info)))

(extend-type OrtTensorTypeAndShapeInfo
  Info
  (info
    ([this]
     (if (scalar? this)
       (tensor-type this)
       {:data-type (tensor-type this)
        :shape (shape this)
        :count (tensor-element-count* *ort-api* this)
        :type :tensor}))
    ([this info-type]
     (case info-type
       :data-type (tensor-type this)
       :shape (shape this)
       :count (tensor-element-count* *ort-api* this)
       :type :tensor
       nil))))

(defn sequence-type [info]
  (sequence-type* *ort-api* info))

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
  (dec-onnx-data-type (key-type* *ort-api* info)))

(defn val-type [info]
  (value-type* *ort-api* info))

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
   (let [ort-api *ort-api*]
     (check-index i (input-count* ort-api sess) "input")
     (input-type-info* ort-api sess i)))
  ([sess]
   (let [ort-api *ort-api*]
     (map #(input-type-info* ort-api sess %)
          (range (input-count* ort-api sess))))))

(defn output-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*]
     (check-index i (output-count* ort-api sess) "output")
     (output-type-info* ort-api sess i)))
  ([sess]
   (let [ort-api *ort-api*]
     (map #(output-type-info* ort-api sess %)
          (range (output-count* ort-api sess))))))

;; (defn memory-info* [^String name ^long allocator ^long device-id ^long mem-type]
;;   (MemoryInfo. name allocator device-id mem-type ))

;; (defn memory-info
;;   ([name allocator device-id mem-type]
;;    (memory-info* (get ort-allocator-name name name)
;;                  (enc-keyword ort-allocator-type allocator)
;;                  device-id
;;                  (enc-keyword ort-mem-type mem-type)))
;;   ([name allocator mem-type]
;;    (memory-info name allocator 0 mem-type))
;;   ([name allocator]
;;    (memory-info name allocator 0 :default))
;;   ([name]
;;    (memory-info* "Cpu" onnxruntime/OrtArenaAllocator
;;                  0 onnxruntime/OrtMemTypeDefault))
;;   ([]
;;    (memory-info* "Cpu" onnxruntime/OrtArenaAllocator 0 onnxruntime/OrtMemTypeDefault)))

;; (defn allocator-name [^MemoryInfoImpl mem-info]
;;   (with-release [all-name (.GetAllocatorName mem-info)]
;;     (let [name (get-string all-name)]
;;       (get ort-allocator-keyword name name))))

;; (defn allocator-type [^MemoryInfoImpl mem-info]
;;   (dec-ort-allocator-type (.GetAllocatorType mem-info)))

;; (defn device-id ^long [^MemoryInfoImpl mem-info]
;;   (.GetDeviceId mem-info))

;; (defn device-type [^MemoryInfoImpl mem-info]
;;   (dec-ort-memory-info-device-type (.GetDeviceType mem-info)))

;; (defn memory-type [^MemoryInfoImpl mem-info]
;;   (dec-ort-memory-type (.GetMemoryType mem-info)))

;; (defn create-tensor*
;;   ([^OrtAllocator allocator ^longs shape ^long type]
;;    (Value/CreateTensor allocator shape (alength shape) type))
;;   ([^OrtMemoryInfo mem-info ^longs shape ^long type ^Pointer data]
;;    (Value/CreateTensor mem-info data (bytesize data) shape (alength shape) type)))

;; (defprotocol TensorCreator
;;   (create-tensor [this shape source] [this shape type data]))

;; (extend-type BaseMemoryInfo
;;   TensorCreator
;;   (create-tensor
;;     ([this shape data]
;;      (let [data (pointer data)]
;;        (create-tensor this shape (enc-keyword pointer-type (type data)) data)))
;;     ([this shape type data]
;;      (create-tensor* (.asOrtMemoryInfo this) (long-array (seq shape)) (enc-keyword onnx-data-type type) (pointer data)))))

;; (extend-type BaseAllocator
;;   TensorCreator
;;   (create-tensor
;;     ([this shape type]
;;      (create-tensor* (.asOrtAllocator this) (long-array (seq shape)) (enc-keyword onnx-data-type type)))
;;     ([this shape type data]
;;      (dragan-says-ex "Allocators can't accept data. They should be the ones creating the data pointer."))))

;; (defn value-type-info [^ConstValueImpl value]
;;   (type-info (.GetTypeInfo value)))
