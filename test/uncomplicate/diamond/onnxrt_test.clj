;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.onnxrt-test
  (:require [midje.sweet :refer [facts => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.fluokitten.core :refer [foldmap]]
            [uncomplicate.neanderthal.core :refer [iamax transfer! native view-vctr entry!]]
            [uncomplicate.diamond
             [tensor :refer [tensor output]]
             [dnn :refer [network activation]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer [options override-dimension!]]
             [core-test :refer [test-image-0]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defn test-onnx-layer-single-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 onnx-bp (onnx fact "data/mnist-12.onnx" nil)
                 mnist-infer! (onnx-bp src-tz)]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      (iamax (native (mnist-infer!))) => 7)))


(defn test-onnx-layer-multi-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 onnx-bp (onnx fact "data/mnist-12.onnx" {:multi-io true})
                 mnist-infer! (onnx-bp [src-tz])]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      (iamax (native (first (mnist-infer!)))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-onnx-layer-single-io fact)
  (test-onnx-layer-multi-io fact))

(defn test-onnx-network-single-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-bp (network fact src-tz
                                   [(onnx "data/mnist-12.onnx")
                                    (activation :relu)])
                 mnist-infer! (mnist-bp src-tz)]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      (iamax (native (mnist-infer!))) => 7)))

(defn test-onnx-network-multi-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-bp (network fact [src-tz]
                                   [(onnx "data/mnist-12.onnx")])
                 mnist-infer! (mnist-bp [src-tz])]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      ;;TODO it seems cuda tensor engine can also support this.
      (iamax (native (first (mnist-infer!)))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-onnx-network-single-io fact)
  (test-onnx-network-multi-io fact))

(with-release [fact (cudnn-factory)]
  (test-onnx-network-single-io fact)
  (test-onnx-network-multi-io fact))

(defn test-onnx-layer-smollm [fact]
  (let [neand-fact (neanderthal-factory fact)]
    (with-release [opt (-> (options)
                           (override-dimension! "batch_size" 1)
                           (override-dimension! "sequence_length" 1)
                           (override-dimension! "past_sequence_length" 1)
                           (override-dimension! "past_sequence_length + 1" 2));;TODO technically, 1 works too!
                   src-tz (tensor fact [1 1 28 28] :float :nchw)
                   onnx-bp (onnx fact "data/SmolLM-135M/onnx/model.onnx" {:options opt})
                   input-ids (tensor neand-fact [1 1] :long :nc)
                   position-ids (tensor neand-fact [1 1] :long :nc)
                   attention-mask (tensor neand-fact [1 2] :long :nc)
                   past-key-values (repeatedly 60 #(tensor fact [1 3 1 64] :float :nchw))
                   smollm-next! (onnx-bp (into [input-ids attention-mask position-ids] past-key-values))]
      (transfer! [2] input-ids)
      (transfer! [0] position-ids)
      (transfer! [1 1] attention-mask)
      (doseq [pkv past-key-values]
        (transfer! (repeat 0) pkv))
      (facts
        "ONNX SmolLM blueprint inference test."
        (foldmap + 0 -
                 (take 10 (view-vctr (native (first (smollm-next!)))))
                 [4.141319274902344 -4.2067766189575195 -4.31782341003418 -5.135868072509766 -4.436248779296875
                  -4.15079402923584 2.627662181854248 -4.15079402923584 9.071796417236328 -0.8716740608215332])
        => (roughly 0.0 0.001)))))

(with-release [fact (dnnl-factory)]
  (test-onnx-layer-smollm fact))

(with-release [fact (cudnn-factory)]
  (test-onnx-layer-smollm fact))
