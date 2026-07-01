;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.cuda-test
  (:require [midje.sweet :refer [facts throws => =not=> roughly truthy just]]
            [uncomplicate.commons.core :refer [with-release info bytesize size release]]
            [uncomplicate.fluokitten.core :refer [fold fmap!]]
            [uncomplicate.clojure-cpp
             :refer [null? float-pointer long-pointer pointer-vec capacity! put-entry! fill! get-entry
                     pointer-pointer pointer zero!]]
            [uncomplicate.clojurecuda.core
             :refer [with-context context device cuda-malloc mem-alloc-runtime
                     memcpy-host! init stream memset! synchronize!]]
            [uncomplicate.neanderthal.math :refer [exp]]
            [uncomplicate.diamond.onnxrt :refer [ort-cuda-context]]
            [uncomplicate.diamond.internal.onnxrt.core :refer :all]
            [uncomplicate.diamond.internal.onnxrt.core-test :refer [test-image-0 softmax]])
  (:import clojure.lang.ExceptionInfo))

(init)
(init-ort-api!)

(facts
  "Test system."
  (filter #{:cuda :dnnl :cpu} (available-providers)) => [:cuda :dnnl :cpu]
  (type (build-info)) => String)

(facts
  "Test CUDA  execution provider."
  (with-release [env (environment nil)
                 opt (options)]
    (info opt) => truthy
    (append-provider! opt :random nil) => (throws ExceptionInfo)
    (append-provider! opt :dnnl) => opt
    (append-provider! opt :cuda) => opt))

(facts
  "Test session and session options."
  (let [env (environment nil)
        opt (-> (options) (append-provider! :cuda))
        sess (session env "data/logreg_iris.onnx" opt)
        metamodel (session-model-metadata sess)]
    (initializer-count sess) => 0
    (initializer-name sess) => []
    (initializer-type-info sess) => []
    (config opt) => {}
    (config! opt {:env-allocators true}) => opt
    (config opt :env-allocators) => true
    (config opt) => {:env-allocators true :use-env-allocators true}
    (dynamic-options! sess {:ep-dynamic-workload :default}) => sess
    (producer-name metamodel) => "OnnxMLTools"
    (graph-name metamodel) => "3c59201b940f410fa29dc71ea9d5767d"
    (domain metamodel) => "onnxml"
    (description metamodel) => ""
    (graph-description metamodel) => ""
    (custom-map-keys metamodel) => []
    (info metamodel) => {:custom-map-keys []
                         :description ""
                         :domain "onnxml"
                         :graph "3c59201b940f410fa29dc71ea9d5767d"
                         :graph-description ""
                         :producer "OnnxMLTools"}
    (info sess) => {:input {"float_input" {:data-type :float :shape [3 2]}}
                    :output {"label" {:data-type :long :shape [3]}
                             "probabilities" {:structure [[:long :float]]}}}
    (input-count sess) => 1
    (output-count sess) => 2
    (input-name sess) => ["float_input"]
    (output-name sess) => ["label" "probabilities"]))

(facts
  "Test cuda memory-info."
  (with-release [env (environment nil)
                 opt (-> (options)
                         (append-provider! :cuda)
                         (config! {:log-severity-level 1}))
                 mem-info (memory-info :cuda :arena 0 :default)
                 mem-info1 (memory-info :cuda :arena 0 :default)
                 mem-info2 (memory-info :cuda :device 0 :default)
                 mem-info3 (memory-info :cuda :device 0 :default)]
    (allocator-key mem-info) => :cuda
    (allocator-key mem-info1) => :cuda
    (allocator-key mem-info2) => :cuda
    (allocator-type mem-info) => :arena
    (allocator-type mem-info2) => :device
    (device-id mem-info) => 0
    (device-type mem-info) => :gpu
    (memory-type mem-info) => :default
    (equal-memory-info? mem-info nil) => false
    (equal-memory-info? mem-info mem-info) => true
    (equal-memory-info? mem-info mem-info1) => true
    (equal-memory-info? mem-info mem-info2) => false
    (equal-memory-info? mem-info2 mem-info3) => true))

(with-release [dev (device 0)]
  (with-context (context dev :map-host)
    (facts
     "Test CUDA tensor values."
     (with-release [env (environment nil)
                    opt (-> (options) (append-provider! :cuda))
                    mem-info (memory-info :cuda :device 0 :default)
                    data (mem-alloc-runtime (* 5 Float/BYTES) :float)
                    val (onnx-tensor mem-info [2 2] data)
                    empty-val (onnx-tensor mem-info [0 1] data)
                    empty-val-zero-size (onnx-tensor [0 1] :float)
                    val-type-info (value-info val)
                    val-tensor-type-info (value-tensor-info val)
                    tz-info (tensor-info [1 2] :double)]
       (info val) => {:value {:data-type :float :shape [2 2]}}
       (info empty-val) => {:value {:data-type :float :shape [0 1]}}
       (info empty-val-zero-size) => {:value {:data-type :float :shape [0 1]}}
       (size (mutable-data val)) => 16
       (size (mutable-data empty-val)) => 0
       (mutable-data empty-val) => (mutable-data val)
       (mutable-data empty-val-zero-size) => nil
       (info tz-info) => {:data-type :double :shape [1 2]}
       (tensor-type! tz-info :float) => tz-info
       (info tz-info) => {:data-type :float :shape [1 2]}
       (value? val) => true
       (none? val) => false
       (tensor? val) => true
       (info val-type-info) => (:value (info val))
       (info (cast-type val-type-info)) => (info val-tensor-type-info)
       (onnx-type val) => :tensor
       (value-count val) => 1
       (release val-type-info) => true
       (info val-type-info) => (throws RuntimeException)
       (onnx-tensor mem-info [0 -1] data) => (throws RuntimeException)
       (onnx-tensor mem-info [0 0] nil) => (throws RuntimeException)
       (onnx-tensor mem-info 3 data) => (throws RuntimeException)))))

(with-context (ort-cuda-context)
  (facts
    "Simple MNIST inference test. The context is set up by (append-provider! :cuda) in previous tests!"
    (with-release [env (environment :warning "test" nil)
                   hstream (stream)
                   opt (-> (options)
                           (append-provider! :cuda {:stream hstream})
                           (graph-optimization! :extended))
                   sess (session env "data/mnist-12.onnx" opt)
                   mem-info (memory-info :cuda :device 0 :default)
                   x-data (mem-alloc-runtime (* 784 Float/BYTES) :float)
                   x (onnx-tensor mem-info [1 1 28 28] x-data)
                   y-data! (mem-alloc-runtime (* 10 Float/BYTES) :float)
                   y! (onnx-tensor mem-info [1 10] y-data!)
                   classify! (runner* sess)
                   data-binding (io-binding sess [x] [y!])]
      (memcpy-host! (float-pointer test-image-0) x-data) => x-data
      (classify! data-binding) => data-binding
      (let [res (pointer-vec (softmax (memcpy-host! y-data! (float-pointer 10))))
            seven (res 7)]
        seven => 1.0
        (apply max res) => seven))))

(with-context (ort-cuda-context)
 (facts
   "GPT inference test."
   (with-release [env (environment :warning "test" nil)
                  hstream (stream)
                  opt (-> (options)
                          (append-provider! :cuda {:stream hstream})
                          (override-dimension! "batch_size" 1) ;;optional
                          (override-dimension! "seq_len" 3) ;;optional
                          (graph-optimization! :extended))
                  sess (session env "data/gpt2-lm-head-bs-12.onnx" opt)
                  cpu-mem-info (memory-info :cpu :arena 0 :default)
                  cuda-mem-info (memory-info :cuda :device 0 :default)
                  input-ids-data (mem-alloc-runtime (* 3 Long/BYTES) :long)
                  input-ids (onnx-tensor cuda-mem-info [1 3] input-ids-data) ;; Grass is
                  attention-mask-data (memset! (mem-alloc-runtime (* 3 Float/BYTES) :float) (float 1.0))
                  attention-mask (onnx-tensor cuda-mem-info [1 3] attention-mask-data)
                  out-token-num-data (mem-alloc-runtime Long/BYTES :long)
                  out-token-num (onnx-tensor cuda-mem-info [1] out-token-num-data)
                  lp-v0-2866-data (mem-alloc-runtime (* 4 36 Long/BYTES) :long)
                  lp-v0-2866 (onnx-tensor cpu-mem-info [4 36] lp-v0-2866-data);; The ORT itself complains about the shape whatever I put. Only mem-info works
                  data-binding (io-binding sess [input-ids attention-mask out-token-num] [cpu-mem-info])
                  answer! (runner* sess)]
     (memcpy-host! (long-pointer [5]) out-token-num-data)
     (memcpy-host! (float-pointer [1 1 1]) attention-mask-data)
     (memcpy-host! (long-pointer [8642, 562, 318]) input-ids-data)
     (synchronize! hstream)
     (answer! data-binding) => data-binding
     (synchronize! hstream)
     (pointer-vec (capacity! (long-pointer (mutable-data (first (bound-values data-binding)))) 14))
     => [8642 562 318 407 262 691 835 284 8642 562 318 407 262 691]
     ))) ;; Grass is not the only way to Grass is not the only way to

(with-context (ort-cuda-context)
  (facts
    "SmolLM inference test."
    (let [batch-size 1
          seq-len 1
          past-seq-len 0
          total-seq-len 1]
      (with-release [env (environment :warning "test" nil)
                     hstream (stream)
                     opt (-> (options)
                             (append-provider! :cuda {:stream hstream})
                             (override-dimension! "batch_size" batch-size)
                             (override-dimension! "sequence_length" seq-len)
                             (override-dimension! "past_sequence_length" past-seq-len)
                             (override-dimension! "past_sequence_length + 1" total-seq-len)
                             (graph-optimization! :extended))
                     sess (session env "data/SmolLM-135M/onnx/model.onnx" opt)
                     mem-info (memory-info :cuda :device 0 :default)
                     input-info (input-type-info sess)
                     output-info (output-type-info sess)
                     input-ids-data (mem-alloc-runtime (* batch-size seq-len Long/BYTES) :long)
                     input-ids (onnx-tensor mem-info [batch-size seq-len] input-ids-data)
                     position-ids-data (mem-alloc-runtime (* batch-size seq-len Long/BYTES) :long)
                     position-ids (onnx-tensor mem-info [batch-size seq-len] position-ids-data)
                     attention-mask-data (mem-alloc-runtime (* batch-size total-seq-len Long/BYTES) :long)
                     attention-mask (onnx-tensor mem-info [batch-size total-seq-len] attention-mask-data)
                     past-key-values-data (vec (repeatedly 60 #(mem-alloc-runtime (* batch-size 3 (max past-seq-len 1) 64 Float/BYTES) :float)))
                     past-key-values (mapv #(onnx-tensor mem-info [batch-size 3 past-seq-len 64] %) past-key-values-data)
                     present-key-values-data (vec (repeatedly 60 #(mem-alloc-runtime (* batch-size 3 total-seq-len 64 Float/BYTES) :float)))
                     present-key-values (mapv #(onnx-tensor mem-info [batch-size 3 total-seq-len 64] %) present-key-values-data)
                     logits-data (mem-alloc-runtime (* batch-size seq-len 49152 Float/BYTES) :float)
                     logits-data-host (zero! (float-pointer (* batch-size seq-len 49152)))
                     logits (onnx-tensor mem-info [batch-size seq-len 49152] logits-data)
                     data-binding (io-binding sess
                                              {"attention_mask" attention-mask
                                               "input_ids" input-ids
                                               "past_key_values.0.key" (past-key-values 0)
                                               "past_key_values.0.value" (past-key-values 1)
                                               "past_key_values.1.key" (past-key-values 2)
                                               "past_key_values.1.value" (past-key-values 3)
                                               "past_key_values.10.key" (past-key-values 20)
                                               "past_key_values.10.value" (past-key-values 21)
                                               "past_key_values.11.key" (past-key-values 22)
                                               "past_key_values.11.value" (past-key-values 23)
                                               "past_key_values.12.key" (past-key-values 24)
                                               "past_key_values.12.value" (past-key-values 25)
                                               "past_key_values.13.key" (past-key-values 26)
                                               "past_key_values.13.value" (past-key-values 27)
                                               "past_key_values.14.key" (past-key-values 28)
                                               "past_key_values.14.value" (past-key-values 29)
                                               "past_key_values.15.key" (past-key-values 30)
                                               "past_key_values.15.value" (past-key-values 31)
                                               "past_key_values.16.key" (past-key-values 32)
                                               "past_key_values.16.value" (past-key-values 33)
                                               "past_key_values.17.key" (past-key-values 34)
                                               "past_key_values.17.value" (past-key-values 35)
                                               "past_key_values.18.key" (past-key-values 36)
                                               "past_key_values.18.value" (past-key-values 37)
                                               "past_key_values.19.key" (past-key-values 38)
                                               "past_key_values.19.value" (past-key-values 39)
                                               "past_key_values.2.key" (past-key-values 4)
                                               "past_key_values.2.value" (past-key-values 5)
                                               "past_key_values.20.key" (past-key-values 40)
                                               "past_key_values.20.value" (past-key-values 41)
                                               "past_key_values.21.key" (past-key-values 42)
                                               "past_key_values.21.value" (past-key-values 43)
                                               "past_key_values.22.key" (past-key-values 44)
                                               "past_key_values.22.value" (past-key-values 45)
                                               "past_key_values.23.key" (past-key-values 46)
                                               "past_key_values.23.value" (past-key-values 47)
                                               "past_key_values.24.key" (past-key-values 48)
                                               "past_key_values.24.value" (past-key-values 49)
                                               "past_key_values.25.key" (past-key-values 50)
                                               "past_key_values.25.value" (past-key-values 51)
                                               "past_key_values.26.key" (past-key-values 52)
                                               "past_key_values.26.value" (past-key-values 53)
                                               "past_key_values.27.key" (past-key-values 54)
                                               "past_key_values.27.value" (past-key-values 55)
                                               "past_key_values.28.key" (past-key-values 56)
                                               "past_key_values.28.value" (past-key-values 57)
                                               "past_key_values.29.key" (past-key-values 58)
                                               "past_key_values.29.value" (past-key-values 59)
                                               "past_key_values.3.key" (past-key-values 6)
                                               "past_key_values.3.value" (past-key-values 7)
                                               "past_key_values.4.key" (past-key-values 8)
                                               "past_key_values.4.value" (past-key-values 9)
                                               "past_key_values.5.key" (past-key-values 10)
                                               "past_key_values.5.value" (past-key-values 11)
                                               "past_key_values.6.key" (past-key-values 12)
                                               "past_key_values.6.value" (past-key-values 13)
                                               "past_key_values.7.key" (past-key-values 14)
                                               "past_key_values.7.value" (past-key-values 15)
                                               "past_key_values.8.key" (past-key-values 16)
                                               "past_key_values.8.value" (past-key-values 17)
                                               "past_key_values.9.key" (past-key-values 18)
                                               "past_key_values.9.value" (past-key-values 19)
                                               "position_ids" position-ids}
                                              {"logits" logits
                                               "present.0.key" (present-key-values 0)
                                               "present.0.value" (present-key-values 1)
                                               "present.1.key" (present-key-values 2)
                                               "present.1.value" (present-key-values 3)
                                               "present.10.key" (present-key-values 20)
                                               "present.10.value" (present-key-values 21)
                                               "present.11.key" (present-key-values 22)
                                               "present.11.value" (present-key-values 23)
                                               "present.12.key" (present-key-values 24)
                                               "present.12.value" (present-key-values 25)
                                               "present.13.key" (present-key-values 26)
                                               "present.13.value" (present-key-values 27)
                                               "present.14.key" (present-key-values 28)
                                               "present.14.value" (present-key-values 29)
                                               "present.15.key" (present-key-values 30)
                                               "present.15.value" (present-key-values 31)
                                               "present.16.key" (present-key-values 32)
                                               "present.16.value" (present-key-values 33)
                                               "present.17.key" (present-key-values 34)
                                               "present.17.value" (present-key-values 35)
                                               "present.18.key" (present-key-values 36)
                                               "present.18.value" (present-key-values 37)
                                               "present.19.key" (present-key-values 38)
                                               "present.19.value" (present-key-values 39)
                                               "present.2.key" (present-key-values 4)
                                               "present.2.value" (present-key-values 5)
                                               "present.20.key" (present-key-values 40)
                                               "present.20.value" (present-key-values 41)
                                               "present.21.key" (present-key-values 42)
                                               "present.21.value" (present-key-values 43)
                                               "present.22.key" (present-key-values 44)
                                               "present.22.value" (present-key-values 45)
                                               "present.23.key" (present-key-values 46)
                                               "present.23.value" (present-key-values 47)
                                               "present.24.key" (present-key-values 48)
                                               "present.24.value" (present-key-values 49)
                                               "present.25.key" (present-key-values 50)
                                               "present.25.value" (present-key-values 51)
                                               "present.26.key" (present-key-values 52)
                                               "present.26.value" (present-key-values 53)
                                               "present.27.key" (present-key-values 54)
                                               "present.27.value" (present-key-values 55)
                                               "present.28.key" (present-key-values 56)
                                               "present.28.value" (present-key-values 57)
                                               "present.29.key" (present-key-values 58)
                                               "present.29.value" (present-key-values 59)
                                               "present.3.key" (present-key-values 6)
                                               "present.3.value" (present-key-values 7)
                                               "present.4.key" (present-key-values 8)
                                               "present.4.value" (present-key-values 9)
                                               "present.5.key" (present-key-values 10)
                                               "present.5.value" (present-key-values 11)
                                               "present.6.key" (present-key-values 12)
                                               "present.6.value" (present-key-values 13)
                                               "present.7.key" (present-key-values 14)
                                               "present.7.value" (present-key-values 15)
                                               "present.8.key" (present-key-values 16)
                                               "present.8.value" (present-key-values 17)
                                               "present.9.key" (present-key-values 18)
                                               "present.9.value" (present-key-values 19)})
                     next! (runner* sess)]
        (memcpy-host! (long-pointer [2]) input-ids-data)
        (memcpy-host! (long-pointer [0]) position-ids-data)
        (memcpy-host! (long-pointer [1]) attention-mask-data)
        (doseq [present present-key-values]
          (memcpy-host! (zero! (float-pointer 384)) present))
        (synchronize! hstream)
        (next! data-binding) => data-binding
        (synchronize! hstream)
        (memcpy-host! logits-data logits-data-host)
        (take 8 (pointer-vec logits-data-host))
        => (map float [13.046758 -1.2744303 -1.2022223 -2.295833 -1.5223873 -1.2159472 1.27348 -1.2159472])))))
