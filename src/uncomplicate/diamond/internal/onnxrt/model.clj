;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.model
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release Info info view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.clojure-cpp :refer [pointer-pointer pointer-vec]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.diamond.tensor
             :refer [default-desc Transfer input output connector revert shape
                     data-type layout TensorDescriptor view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc diff-desc Initializable init batch-index create-tensor
                      create-tensor-desc]]
             [utils :refer [default-strides transfer-weights-bias! concat-strides
                            concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.onnxrt.core :as onnx
             :refer [onnx-tensor runner* cast-type input-type-info output-type-info tensor-type
                     io-binding options config]])
  (:import [clojure.lang IFn AFn]))

;; ================================ One input, one output ==========================================

(deftype SingleIOInference [fact bluep src-conn dst-tz infer! bindings ins outs]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release bindings)
    (release ins)
    (release outs)
    (release infer!))
  Info
  (info [this]
    {:onnx (info bluep :onnx)
     :src (info src-conn)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :onnx (info bluep :onnx)
      :src (info src-conn)
      :dst (info dst-tz)
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (src-conn)
    (infer! bindings)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype SingleIOInferenceBlueprint [fact sess opt run-opt mem-info src-desc dst-desc
                                     onnx-in-shape onnx-out-shape]
  Releaseable
  (release [_]
    (release sess)
    (release opt)
    (release mem-info)
    (release src-desc)
    (release dst-desc))
  Info
  (info [this]
    (into (info sess)
          {:src (info src-desc)
           :dst (info dst-desc)
           :in-shape (info onnx-in-shape)
           :out-shape (info onnx-out-shape)
           :run-options (info run-opt)}))
  (info [this info-type]
    (case info-type
      :src (info src-desc)
      :dst (info dst-desc)
      :run-options (info run-opt)
      (info sess info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    dst-desc)
  (train-desc [_]
    dst-desc)
  (diff-desc [_]
    dst-desc)
  TensorDescriptor
  (shape [this]
    (shape dst-desc))
  (data-type [this]
    (data-type dst-desc))
  (layout [this]
    (layout dst-desc))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz src-desc)
                  dst-tz (create-tensor fact dst-desc (batch-index src-tz) false)
                  infer! (runner* sess run-opt)
                  in-onnx (onnx-tensor mem-info onnx-in-shape (buffer (output src-conn)))
                  out-onnx (onnx-tensor mem-info onnx-out-shape (buffer (output dst-tz)))
                  binding (io-binding sess [in-onnx] [out-onnx])]
      (->SingleIOInference fact this src-conn dst-tz infer! binding [in-onnx] [out-onnx])))
  (invoke [this _ _]
    (dragan-says-ex "ONNX Runtime doesn't support training. (yet!)"))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn onnx-single-io-model
  ([fact sess opt run-opt mem-info]
   (let [in-info (cast-type (input-type-info sess 0))
         in-shape (onnx/shape in-info)
         in-type (tensor-type in-info)
         out-info (cast-type (output-type-info sess 0))
         out-shape (onnx/shape out-info)
         out-type (tensor-type out-info)]
     (let-release [src-desc (create-tensor-desc fact in-shape in-type (default-strides in-shape))
                   dst-desc (create-tensor-desc fact out-shape out-type (default-strides out-shape))]
       (->SingleIOInferenceBlueprint fact sess opt run-opt mem-info src-desc dst-desc in-shape out-shape))))
  ([fact sess mem-info]
   (let-release [opt (options)]
     (onnx-single-io-model fact sess opt nil mem-info))))

;; ================================ Multiple inputs, Multiple outputs ==========================================

(deftype MultiIOInference [fact bluep src-conns dst-tzs infer! bindings ins outs]
  Releaseable
  (release [_]
    (doseq [sc src-conns] (release sc))
    (doseq [dt dst-tzs] (release dt))
    (release bindings)
    (release ins)
    (release outs)
    (release infer!))
  Info
  (info [this]
    {:onnx (info bluep :onnx)
     :src (fmap info src-conns)
     :dst (fmap info dst-tzs)})
  (info [this info-type]
    (case info-type
      :onnx (info bluep :onnx)
      :src (fmap info src-conns)
      :dst (fmap info dst-tzs)
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (fmap input src-conns))
  (output [_]
    dst-tzs)
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (doseq [sc src-conns] (sc))
    (infer! bindings)
    dst-tzs)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype MultiIOInferenceBlueprint [fact sess opt run-opt mem-info src-descs dst-descs
                                    onnx-in-shapes onnx-out-shapes]
  Releaseable
  (release [_]
    (release sess)
    (release opt)
    (release mem-info)
    (doseq [sd src-descs] (release sd))
    (doseq [dd dst-descs] (release dd)))
  Info
  (info [this]
    (into (info sess)
          {:src (fmap info src-descs)
           :dst (fmap info dst-descs)
           :in-shapes (fmap info onnx-in-shapes)
           :out-shapes (fmap info onnx-out-shapes)
           :run-options (info run-opt)}))
  (info [this info-type]
    (case info-type
      :src (fmap info src-descs)
      :dst (fmap info dst-descs)
      :in-shapes (fmap info onnx-in-shapes)
      :out-shapes (fmap info onnx-out-shapes)
      :run-options (info run-opt)
      (info sess info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    dst-descs)
  (train-desc [_]
    dst-descs)
  (diff-desc [_]
    dst-descs)
  TensorDescriptor
  (shape [this]
    (fmap shape dst-descs))
  (data-type [this]
    (fmap data-type dst-descs))
  (layout [this]
    (fmap layout dst-descs))
  IFn
  (invoke [this prev-layer]
    (let [src-tzs (fmap (comp view output) prev-layer)]
      (let-release [src-conns (fmap connector src-tzs src-descs)
                    dst-tzs (fmap #(create-tensor fact % false) dst-descs)
                    infer! (runner* sess run-opt)
                    in-onnx-s (fmap (fn [onnx-in-shape src-conn]
                                      (onnx-tensor mem-info onnx-in-shape (buffer (output src-conn))))
                                    onnx-in-shapes src-conns)
                    out-onnx-s (fmap (fn [onnx-out-shape dst-tz]
                                       (onnx-tensor mem-info onnx-out-shape (buffer (output dst-tz))))
                                     onnx-out-shapes dst-tzs)
                    binding (io-binding sess in-onnx-s out-onnx-s)]
        (->MultiIOInference fact this src-conns dst-tzs infer! binding in-onnx-s out-onnx-s))))
  (invoke [this _ _]
    (dragan-says-ex "ONNX Runtime doesn't support training. (yet!)"))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn onnx-multi-io-model
  ([fact sess opt run-opt mem-info]
   (let [ins-info (mapv cast-type (input-type-info sess))
         ins-shape (mapv onnx/shape ins-info)
         ins-type (mapv tensor-type ins-info)
         outs-info (mapv cast-type (output-type-info sess))
         outs-shape (mapv onnx/shape outs-info)
         outs-type (mapv tensor-type outs-info)]
     (let-release [src-descs (mapv (fn [in-shape in-type]
                                     (create-tensor-desc fact in-shape in-type (default-strides in-shape)))
                                   ins-shape ins-type)
                   dst-descs (mapv (fn [out-shape out-type]
                                     (create-tensor-desc fact out-shape out-type (default-strides out-shape)))
                                   outs-shape outs-type)]
       (->MultiIOInferenceBlueprint fact sess opt run-opt mem-info src-descs dst-descs ins-shape outs-shape))))
  ([fact sess mem-info]
   (let-release [opt (options)]
     (onnx-multi-io-model fact sess opt nil mem-info))))
