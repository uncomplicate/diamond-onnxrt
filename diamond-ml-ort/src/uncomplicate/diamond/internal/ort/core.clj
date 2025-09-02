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
             [core :refer [let-release with-release view Info bytesize size]]
             [utils :refer [enc-keyword dragan-says-ex mask]]]
            [uncomplicate.clojure-cpp :refer [get-string byte-pointer]]
            [uncomplicate.diamond.internal.ort.constants :refer :all])
  (:import org.bytedeco.onnxruntime.global.onnxruntime
           [org.bytedeco.onnxruntime Env StringVector LongVector OrtStatus SessionOptions Session
            AllocatorWithDefaultOptions OrtAllocator TypeInfo]))

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

(defn session [^Env env ^String model-path ^SessionOptions options]
  (Session. env (byte-pointer model-path) options))

(defn input-count ^long [^Session sess]
  (.GetInputCount sess))

(def default-allocator (.asUnownedAllocator (AllocatorWithDefaultOptions.)))

(defn tensor-type-info [^TypeInfo type-info]
  (let [tensor-info (.GetTensorTypeAndShapeInfo type-info)]
    {:type (.GetElementType tensor-info)
     :shape (vec (.get (.GetShape tensor-info)))}))

(defn input-name
  ([^Session sess ^long i]
   (let [cnt (input-count sess)]
     (if (< -1 i cnt)
       (get-string (.GetInputNameAllocated sess i (OrtAllocator. default-allocator)))
       (throw (IndexOutOfBoundsException. "The requested input name is out of bounds of this session inputs pointer.")))))
  ([^Session sess]
    (let [cnt (input-count sess)
          alloc (OrtAllocator. default-allocator)]
      (map #(get-string (.GetInputNameAllocated sess % alloc)) (range cnt)))))

(defn input-type-info
  ([^Session sess ^long i]
   (if (< -1 i (input-count sess))
     (tensor-type-info (.GetInputTypeInfo sess i))
     (throw (IndexOutOfBoundsException. "The requested input type info is out of bounds of this session inputs pointer."))))
  ([^Session sess]
   (map #(tensor-type-info (.GetInputTypeInfo sess %)) (range (input-count sess)))))

(defn output-count ^long [^Session sess]
  (.GetOutputCount sess))

(defn output-name
  ([^Session sess ^long i]
   (let [cnt (output-count sess)]
     (if (< -1 i cnt)
       (get-string (.GetOutputNameAllocated sess i (OrtAllocator. default-allocator)))
       (throw (IndexOutOfBoundsException. "The requested output name is out of bounds of this session outputs pointer.")))))
  ([^Session sess]
    (let [cnt (output-count sess)
          alloc (OrtAllocator. default-allocator)]
      (map #(get-string (.GetOutputNameAllocated sess % alloc)) (range cnt)))))

(defn output-type-info
  ([^Session sess ^long i]
   (if (< -1 i (output-count sess))
     (tensor-type-info (.GetOutputTypeInfo sess i))
     (throw (IndexOutOfBoundsException. "The requested output type info is out of bounds of this session outputs pointer."))))
  ([^Session sess]
   (map #(tensor-type-info (.GetOutputTypeInfo sess %)) (range (output-count sess)))))
