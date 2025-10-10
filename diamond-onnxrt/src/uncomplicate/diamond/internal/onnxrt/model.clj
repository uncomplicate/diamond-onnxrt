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
             [core :refer [Releaseable release let-release Info info]]
             [utils :refer [dragan-says-ex]]]
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
             :refer [onnx-tensor runner* cast-type input-type-info output-type-info tensor-type]])
  (:import [clojure.lang IFn AFn]))

;; ================================ Activation =============================================

(deftype StraightInference [fact bluep src-conn dst-tz infer! in-pp out-pp]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release (pointer-vec in-pp))
    (release (pointer-vec out-pp))
    (release in-pp)
    (release out-pp)
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
    (infer! in-pp out-pp)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype StraightInferenceBlueprint [fact sess run-opt mem-info src-desc dst-desc]
  Releaseable
  (release [_]
    (release sess)
    (release mem-info)
    (release src-desc)
    (release dst-desc))
  Info
  (info [this]
    {:session (info sess)
     :options (info run-opt)})
  (info [this info-type]
    (case info-type
      :session (info sess)
      :options (info run-opt)
      nil))
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
                  in-onnx (onnx-tensor mem-info (shape src-desc) (buffer (output src-conn)))
                  in-pp (pointer-pointer [in-onnx])
                  out-onnx (onnx-tensor mem-info (shape dst-desc) (buffer (output dst-tz)))
                  out-pp (pointer-pointer [out-onnx])]
      (->StraightInference fact this src-conn dst-tz infer! in-pp out-pp)))
  (invoke [this src-tz diff-tz]
    (dragan-says-ex "ONNX Runtime doesn't support training. (yet!)"))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn onnx-straight-model
  ([fact sess run-opt mem-info]
   (let [in-info (cast-type (input-type-info sess 0))
         in-shape (onnx/shape in-info)
         in-type (tensor-type in-info)
         out-info (cast-type (output-type-info sess 0))
         out-shape (onnx/shape out-info)
         out-type (tensor-type out-info)]
     (let-release [src-desc (create-tensor-desc fact in-shape in-type (default-strides in-shape))
                   dst-desc (create-tensor-desc fact out-shape out-type (default-strides out-shape))]
       (->StraightInferenceBlueprint fact sess run-opt mem-info src-desc dst-desc))))
  ([fact sess mem-info]
   (onnx-straight-model fact sess nil mem-info)))
