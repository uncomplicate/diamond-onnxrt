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
           [org.bytedeco.onnxruntime
            OrtTypeInfo OrtTensorTypeAndShapeInfo OrtSequenceTypeInfo OrtMapTypeInfo OrtOptionalTypeInfo]))

(defn version []
  (with-release [p (onnxruntime/GetVersionString)]
    (let [v (mapv parse-long (split (get-string p) #"\."))]
      {:major (v 0)
       :minor (v 1)
       :update (v 2)})))

(defn build-info []
  (with-release [p (onnxruntime/GetBuildInfoString)]
    (get-string p)))

(defn available-providers []
  (with-release [sv (onnxruntime/GetAvailableProviders)
                 p (.get sv)]
    (map #(keyword (lower-case (st/replace (get-string %) "ExecutionProvider" ""))) p)))

(defn options []
  (session-options* *ort-api*))

(defn execution-provider
  ([opt provider ^long use-arena]
   (execution-provider* *ort-api* opt provider use-arena)
   opt)
  ([opt provider]
   (execution-provider* *ort-api* opt provider 0))
  ([opt]
   (execution-provider* *ort-api* opt :cpu 0)))

(defn graph-optimization [opt level]
  (graph-optimization* *ort-api* opt (enc-keyword ort-graph-optimization level)))

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
   (input-name* *ort-api* sess i *default-allocator*))
  ([sess]
   (doall (map #(input-name* *ort-api* sess *default-allocator* %)
               (range (input-count sess))))))

(defn output-count ^long [sess]
  (output-count* *ort-api* sess))

(defn output-name
  ([sess ^long i]
   (check-index i (output-count sess) "output")
   (output-name* *ort-api* sess i *default-allocator*))
  ([sess]
   (doall (map #(output-name* *ort-api* sess *default-allocator* %)
               (range (output-count sess))))))

(defn scalar? [info]
  (= 0 (dimensions-count* *ort-api* info)))

(defn shape [info]
  (with-release [dims (safe (tensor-dimensions* *ort-api* info))]
    (vec (doall (pointer-seq dims)))))

(defprotocol ElementInfo
  (element-type [this]))

(defn type-info [info]
  (type-info* *ort-api* info))

(extend-type OrtTypeInfo
  Info
  (info [this]
    (info (type-info this)))
  (info [this info-type]
    (info (type-info this) info-type)))

(extend-type OrtTensorTypeAndShapeInfo
  Info
  (info
    ([this]
     (if (scalar? this)
       (element-type this)
       {:data-type (element-type this)
        :shape (shape this)
        :count (tensor-element-count* *ort-api* this)
        :type :tensor}))
    ([this info-type]
     (case info-type
       :data-type (element-type this)
       :shape (shape this)
       :count (tensor-element-count* *ort-api* this)
       :type :tensor
       nil)))
  ElementInfo
  (element-type [this]
    (dec-onnx-data-type (tensor-type* *ort-api* this))))

(extend-type OrtSequenceTypeInfo
  Info
  (info
    ([this]
     {:type :sequence
      :element [(info (element-type this))]})
    ([this type-info]
     (case type-info
       :type :sequence
       :element (info (element-type this))
       nil)))
  ElementInfo
  (element-type [this]
    (type-info (sequence-type* *ort-api* this))))

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
  (type-info* *ort-api* (value-type* *ort-api* info)))

(extend-type OrtMapTypeInfo
  Info
  (info
    ([this]
     {:type :map
      :key (key-type this)
      :val (info (val-type this))})
    ([this info-type]
     (case info-type
       :tyte :map
       :key (key-type this)
       :val (info (val-type this))
       nil)))
  ElementInfo
  (element-type [this]
    [(key-type this) (val-type this)]))

(defn input-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*]
     (check-index i (input-count* ort-api sess) "input")
     (type-info* ort-api (input-type-info* ort-api sess i))))
  ([sess]
   (let [ort-api *ort-api*]
     (map #(type-info* ort-api (input-type-info* ort-api sess %))
          (range (input-count* ort-api sess))))))

(defn output-type-info
  ([sess ^long i]
   (let [ort-api *ort-api*]
     (check-index i (output-count* ort-api sess) "output")
     (type-info* ort-api (output-type-info* ort-api sess i))))
  ([sess]
   (let [ort-api *ort-api*]
     (map #(type-info* ort-api (output-type-info* ort-api sess %))
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
