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
            [uncomplicate.commons [core :refer [with-release release]]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! asum scal!]]
             [vect-math :refer [exp!]]]
            [uncomplicate.diamond
             [tensor :refer [tensor]]
             [dnn :refer [network activation]]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer :all]
             [core-test :refer [test-image-0]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]))

(defn softmax! [xs]
  (scal! (/ 1.0 (asum (exp! xs))) xs))

(defn test-onnx-model [fact]
  (with-release [env (environment :warning "test");;TODO I shuld just test how long does it take to create this. Anyway, it's a part of the context
                 opt (-> (options)
                         (append-provider! :dnnl);;TODO this is also a kind of factory-ish job
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt);;TODO this can even be a part of a (proxied) factory...
                 mem-info (memory-info :cpu :arena 0 :default);;TODO gpu I think that this can be provided by the factory (extend-type DnnlFactory even)
                 mnist-bluep (onnx-model fact sess nil mem-info)
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

(defn test-onnx-network [fact]
  (with-release [env (environment :warning "test");;TODO I shuld just test how long does it take to create this. Anyway, it's a part of the context
                 opt (-> (options)
                         (append-provider! :dnnl);;TODO this is also a kind of factory-ish job
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt);;TODO this can even be a part of a (proxied) factory...
                 mem-info (memory-info :cpu :arena 0 :default);;TODO gpu I think that this can be provided by the factory (extend-type DnnlFactory even)
                 src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-bp (network fact src-tz
                                   [(onnx sess mem-info)
                                    #_(activation [10] :softmax)])
                 mnist-infer! (mnist-bp src-tz)]

    (transfer! test-image-0 src-tz)

    (facts
      "ONNX MNIST network inference test."
      (iamax (softmax! (mnist-infer!))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-onnx-network fact))
