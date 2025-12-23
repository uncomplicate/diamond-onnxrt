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
            [uncomplicate.commons [core :refer [with-release info]]]
            [uncomplicate.neanderthal
             [core :refer [iamax transfer! asum scal! native view-vctr]]
             [vect-math :refer [exp!]]]
            [uncomplicate.neanderthal.internal.api :refer [device]]
            [uncomplicate.diamond.tensor :refer [tensor]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer :all]
             [model :refer :all]
             [core-test :refer [test-image-0]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]))

(defn softmax! [xs]
  (scal! (/ 1.0 (asum (exp! xs))) xs))

(defn test-single-io-onnx-model [fact]
  (with-release [env (environment :warning "test" nil)
                 opt (-> (options)
                         (append-provider! :dnnl)
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt)
                 mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
                 mnist-bp (onnx-single-io-model fact sess mem-info)
                 src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-infer! (mnist-bp src-tz)]

    (transfer! test-image-0 src-tz)

    (facts
      "ONNX MNIST inference test."
      (info mnist-bp) => {:dst {:class "uncomplicate.diamond.internal.dnnl.impl.MemoryDescImpl"
                                :data-type :float
                                :device :cpu
                                :shape [1 10]
                                :strides [10 1]}
                          :in-shape [1 1 28 28]
                          :input {"Input3" {:data-type :float :shape [1 1 28 28]}}
                          :out-shape [1 10]
                          :output {"Plus214_Output_0" {:data-type :float :shape [1 10]}}
                          :run-options nil
                          :src {:class "uncomplicate.diamond.internal.dnnl.impl.MemoryDescImpl"
                                :data-type :float
                                :device :cpu
                                :shape [1 1 28 28]
                                :strides [784 784 28 1]}}
      (iamax (softmax! (mnist-infer!))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-single-io-onnx-model fact))

;; TODO timings: env: 30 microseconds
;; TODO timings: opts with settings: 30 microseconds
;; TODO timings: session loading: 2 milliseconds
;; TODO timings: mem-info: 25 microseconds

(defn test-multi-io-onnx-model [fact]
  (with-release [env (environment :warning "test" nil)
                 opt (options)
                 sess (session env "data/mnist-12.onnx" opt)
                 mem-info (memory-info :cpu :device 0 :default)
                 mnist-bp (onnx-multi-io-model fact sess opt nil mem-info)
                 src-tz (tensor fact [1 1 28 28] :float :nchw)
                 mnist-infer! (mnist-bp [src-tz])]

    (transfer! test-image-0 src-tz)

    (facts
      "ONNX MNIST inference test."
      (info mnist-bp) => {:dst [{:class "uncomplicate.diamond.internal.dnnl.impl.MemoryDescImpl"
                                 :data-type :float
                                 :device :cpu
                                 :shape [1 10]
                                 :strides [10 1]}]
                          :in-shapes [[1 1 28 28]]
                          :input {"Input3" {:data-type :float :shape [1 1 28 28]}}
                          :out-shapes [[1 10]]
                          :output {"Plus214_Output_0" {:data-type :float :shape [1 10]}}
                          :run-options nil
                          :src [{:class "uncomplicate.diamond.internal.dnnl.impl.MemoryDescImpl"
                                 :data-type :float
                                 :device :cpu
                                 :shape [1 1 28 28]
                                 :strides [784 784 28 1]}]}
      (iamax (softmax! (first (mnist-infer!)))) => 7)))

(with-release [fact (dnnl-factory)]
  (test-multi-io-onnx-model fact))

(with-release [fact (dnnl-factory)
               vect-fact (neanderthal-factory fact)
               env (environment :warning "test" nil)
               opt (-> (options)
                       (append-provider! :dnnl)
                       (override-dimension! "batch_size" 1)
                       (override-dimension! "sequence_length" 1)
                       (override-dimension! "past_sequence_length" 0)
                       (override-dimension! "past_sequence_length + 1" 1)
                       (graph-optimization! :extended))
               ;; huggingface model HuggingFaceTB/SmolLM-135M at revision 1d461723eec654e65efdc40cf49301c89c0c92f4
               sess (session env "data/SmolLM-135M/onnx/model.onnx" opt)
               mem-info (memory-info (device (neanderthal-factory fact :float)) :device 0 :default)
               smollm-bp (onnx-multi-io-model fact sess opt nil mem-info)
               src-tz (tensor fact [1 1 28 28] :float :nchw)
               input-ids (tensor vect-fact [1 1] :long :nc)
               position-ids (tensor vect-fact [1 1] :long :nc)
               attention-mask (tensor vect-fact [1 1] :long :nc)
               past-key-values (repeatedly 60 #(tensor fact [1 3 0 64] :float :nchw))
               smollm-next! (smollm-bp (into [input-ids attention-mask position-ids] past-key-values))]

  (facts
    "ONNX SmolLM inference test."
    (transfer! [2] input-ids)
    (transfer! [0] position-ids)
    (transfer! [1] attention-mask)
    (doseq [pkv past-key-values]
      (transfer! (repeat 0) pkv))
    (take 16 (view-vctr (native (first (smollm-next!)))))
    => (map float [13.046633 -1.2745271 -1.2023203 -2.2959335 -1.5224829 -1.2160451 1.2734042 -1.2160451
                   -5.103885 9.137959 -1.2160451 -1.2160451 -1.2160451 -1.2160766 -1.2160451 -1.2160451])))
