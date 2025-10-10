;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.model-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! asum scal!]]
             [vect-math :refer [exp!]]]
            [uncomplicate.diamond.tensor :refer [tensor]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer :all]
             [core-test :refer [test-image-0]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]))

(defn softmax! [xs]
  (scal! (/ 1.0 (asum (exp! xs))) xs))

(defn test-onnx-model [fact]
  (with-release [env (environment :warning "test")
                 opt (-> (options)
                         (append-provider! :dnnl)
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt)
                 mem-info (memory-info :cpu :arena 0 :default)
                 mnist-bluep (onnx-straight-model fact sess mem-info)
                 src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-infer! (mnist-bluep src-tz)]

    (transfer! test-image-0 src-tz)

    (facts
      "ONNX MNIST inference test."
      (iamax (softmax! (mnist-infer!))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-onnx-model fact))

;; TODO timings: env: 30 microseconds
;; TODO timings: opts with settings: 30 microseconds
;; TODO timings: session loading: 2 milliseconds
;; TODO timings: mem-info: 25 microseconds
