;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.onnxrt-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons [core :refer [with-release release]]]
            [uncomplicate.neanderthal.core :refer [iamax transfer! native]]
            [uncomplicate.diamond
             [tensor :refer [tensor output]]
             [dnn :refer [network activation]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.internal.onnxrt.core-test :refer [test-image-0]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defn test-onnx-layer-single-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 onnx-bp (onnx fact "data/mnist-12.onnx" nil)
                 mnist-infer! (onnx-bp src-tz)]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      ;;TODO it seems cuda tensor engine can also support this.
      (iamax (native (mnist-infer!))) => 7)))


(defn test-onnx-layer-multi-io [fact]
  (with-release [src-tz (tensor fact [1 1 28 28] :float :nchw)
                 onnx-bp (onnx fact "data/mnist-12.onnx" {:multi-io true})
                 mnist-infer! (onnx-bp [src-tz])]

    (transfer! test-image-0 src-tz)
    (facts
      "ONNX MNIST network inference test."
      ;;TODO it seems cuda tensor engine can also support this.
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
      ;;TODO it seems cuda tensor engine can also support this.
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
