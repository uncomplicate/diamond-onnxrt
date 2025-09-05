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
            [uncomplicate.clojure-cpp :refer [get-string byte-pointer null?]]
            [uncomplicate.diamond.internal.ort.constants :refer :all])
  (:import org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime Env StringVector LongVector OrtStatus SessionOptions Session
            AllocatorWithDefaultOptions OrtAllocator TypeInfo TypeInfoImpl MapTypeInfoImpl ConstMapTypeInfo
            ConstTensorTypeAndShapeInfo TensorTypeAndShapeInfoImpl BaseSequenceTypeInfoImpl
            OptionalTypeInfoImpl]))

(defmacro extend-ort [t]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (locking this#
         (when-not (null? this#)
           (let [ort# (.release this#)]
             (when-not (null? ort#)
               (onnxruntime/OrtRelease ort#)
               (.deallocate ort#)
               (.setNull ort#)))
           (.deallocate this#)
           (.setNull this#))
         true))))

(extend-ort Env)
(extend-ort SessionOptions)
(extend-ort Session)
(extend-ort TypeInfoImpl)
(extend-ort TensorTypeAndShapeInfoImpl)
(extend-ort ConstTensorTypeAndShapeInfo)
(extend-ort ConstMapTypeInfo)
(extend-ort MapTypeInfoImpl)
(extend-ort BaseSequenceTypeInfoImpl)
(extend-ort OptionalTypeInfoImpl)

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
  (SessionOptions.))

(defn execution-provider
  ([^SessionOptions opt provider ^long use-arena]
   (let [ort-opt (.asOrtSessionOptions opt)
         status ^OrtStatus;;TODO check status
                  (case provider
                    :cpu (onnxruntime/OrtSessionOptionsAppendExecutionProvider_CPU ort-opt use-arena)
                    :dnnl (onnxruntime/OrtSessionOptionsAppendExecutionProvider_Dnnl ort-opt use-arena)
                    :cuda (onnxruntime/OrtSessionOptionsAppendExecutionProvider_CUDA ort-opt use-arena))]
     opt))

  ([opt provider]
   (execution-provider opt provider 0))
  ([opt]
   (execution-provider opt :cpu 0)))

(defn graph-optimization [^SessionOptions opt level]
  (.SetGraphOptimizationLevel opt (enc-keyword ort-graph-optimization level)))

(defn environment
  ([logging-level name]
   (Env. (enc-keyword ort-logging-level logging-level) name))
  ([]
   (Env. onnxruntime/ORT_LOGGING_LEVEL_WARNING "default")));

(declare type-info element-count shape)

(defprotocol ElementInfo
  (element-type* [this])
  (element-type [this]))

(defprotocol TensorInfo
  (element-count* [this])
  (dimensions-count* [this])
  (shape* [this]))

(defprotocol MapInfo
  (key-type* [this])
  (key-type [this])
  (value-type [this]))

(defn element-count ^long [info]
  (element-count info))

(defn scalar? [info]
  (= 0 (dimensions-count* info)))

(defn shape [info]
  (vec (.get ^LongVector (shape* info))))

(defmacro extend-tensor-info [t]
  `(extend-type ~t
     Info
     (info
       ([this#]
        (if (scalar? this#)
          (element-type this#)
          {:data-type (element-type this#)
           :shape (shape this#)
           :count (.GetElementCount this#)
           :type :tensor}))
       ([this# info-type#]
        (case info-type#
          :data-type (element-type this#)
          :shape (shape this#)
          :count (.GetElementCount this#)
          :type :tensor
          nil)))
     ElementInfo
     (element-type* [this#]
       (.GetElementType this#))
     (element-type [this#]
       (dec-onnx-data-type (.GetElementType this#)))
     TensorInfo
     (dimensions-count* [this#]
       (.GetDimensionsCount this#))
     (element-count* [this#]
       (.GetElementCount this#))
     (shape* [this#]
       (.GetShape this#))))

(extend-tensor-info TensorTypeAndShapeInfoImpl)
(extend-tensor-info ConstTensorTypeAndShapeInfo)

(defn type-info [^TypeInfo type-info]
  (case (.GetONNXType type-info)
    1 (.GetTensorTypeAndShapeInfo type-info)
    2 (.GetSequenceTypeInfo type-info)
    3 (.GetMapTypeInfo type-info)
    6 (.GetOptionalTypeInfo type-info)
    (dragan-says-ex "TODO")))

(extend-type TypeInfoImpl
  Info
  (info [this]
    (info (type-info this)))
  (info [this info-type]
    (info (type-info this) info-type)))

(extend-type BaseSequenceTypeInfoImpl
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
  (element-type* [this#]
    (.GetSequenceElementType this#))
  (element-type [this#]
    (type-info (.GetSequenceElementType this#))))

(extend-type OptionalTypeInfoImpl
  Info
  (info
    ([this]
     {:type :optional})
    ([this type-info]
     (case type-info
       :type :optional
       nil))))

(defmacro extend-map-info [t]
  `(extend-type ~t
     Info
     (info
       ([this#]
        {:type :map
         :key (key-type this#)
         :value (info (value-type this#))})
       ([this# info-type#]
        (case info-type#
          :tyte :map
          :key (key-type this#)
          :value (info (value-type this#))
          nil)))
     ElementInfo
     (element-type* [this#]
       [(key-type* this#) (value-type this#)])
     (element-type [this#]
       [(key-type this#) (value-type this#)])
     MapInfo
     (key-type* [this#]
       (.GetMapKeyType this#))
     (key-type [this#]
       (dec-onnx-data-type (.GetMapKeyType this#)))
     (value-type [this#]
       (type-info (.GetMapValueType this#)))))

(extend-map-info ConstMapTypeInfo)
(extend-map-info MapTypeInfoImpl)

(defn session [^Env env ^String model-path ^SessionOptions options]
  (Session. env (byte-pointer model-path) options))

(defn input-count ^long [^Session sess]
  (.GetInputCount sess))

(def default-allocator (.asUnownedAllocator (AllocatorWithDefaultOptions.)))

(defn check-index [^long i ^long cnt object]
  (when-not (< -1 i cnt)
    (throw (IndexOutOfBoundsException. (format "The requested %s name is out of bounds of this session %s pointer." object object)))))

(defn input-name
  ([^Session sess ^long i]
   (check-index i (input-count sess) "input")
   (get-string (.GetInputNameAllocated sess i (OrtAllocator. default-allocator))))
  ([^Session sess]
    (let [cnt (input-count sess)
          alloc (OrtAllocator. default-allocator)]
      (map #(get-string (.GetInputNameAllocated sess % alloc)) (range cnt)))))

(defn input-type-info
  ([^Session sess ^long i]
   (check-index i (input-count sess) "input")
   (type-info (.GetInputTypeInfo sess i)))
  ([^Session sess]
   (map #(type-info (.GetInputTypeInfo sess %)) (range (input-count sess)))))

(defn output-count ^long [^Session sess]
  (.GetOutputCount sess))

(defn output-name
  ([^Session sess ^long i]
   (check-index i (output-count sess) "output")
   (get-string (.GetOutputNameAllocated sess i (OrtAllocator. default-allocator))))
  ([^Session sess]
    (let [cnt (output-count sess)
          alloc (OrtAllocator. default-allocator)]
      (map #(get-string (.GetOutputNameAllocated sess % alloc)) (range cnt)))))

(defn output-type-info
  ([^Session sess ^long i]
   (check-index i (output-count sess) "output")
   (type-info (.GetOutputTypeInfo sess i)))
  ([^Session sess]
   (map #(type-info (.GetOutputTypeInfo sess %)) (range (output-count sess)))))
