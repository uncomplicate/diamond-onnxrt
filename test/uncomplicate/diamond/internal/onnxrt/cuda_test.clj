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
                     pointer-pointer pointer]]
            [uncomplicate.clojurecuda.core
             :refer [with-context context device cuda-malloc memcpy-to-device! memcpy-to-host! init
                     stream]]
            [uncomplicate.neanderthal.math :refer [exp]]
            [uncomplicate.diamond.internal.onnxrt.core :refer :all]
            [uncomplicate.diamond.internal.onnxrt.core-test :refer [test-image-0 softmax]])
  (:import clojure.lang.ExceptionInfo))

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
                 mem-info2 (memory-info :cuda :device 0 :default)]
    (allocator-key mem-info) => nil
    (allocator-key mem-info1) => nil
    (allocator-key mem-info2) => nil
    (allocator-type mem-info) => :arena
    (allocator-type mem-info2) => :device
    (device-id mem-info) => 0
    (device-type mem-info) => :gpu
    (memory-type mem-info) => :default
    (equal-memory-info? mem-info nil) => false
    (equal-memory-info? mem-info mem-info) => true
    (equal-memory-info? mem-info mem-info1) => false
    (equal-memory-info? mem-info mem-info2) => false))

(with-release [dev (device 0)]
  (with-context (context dev :map-host)
    (facts
     "Test CUDA tensor values."
     (with-release [env (environment nil)
                    opt (-> (options) (append-provider! :cuda))
                    mem-info (memory-info :cuda :device 0 :default)
                    data (cuda-malloc (* 5 Float/BYTES) :float)
                    val (onnx-tensor mem-info [2 2] data)
                    val-type-info (value-info val)
                    val-tensor-type-info (value-tensor-info val)
                    tz-info (tensor-info [1 2] :double)]
       (info val) => {:value {:data-type :float :shape [2 2]}}
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

(init)

(facts
  "Simple MNIST inference test."
  (with-release [env (environment :warning "test" nil)
                 hstream (stream)
                 opt (-> (options)
                         (append-provider! :cuda {:stream hstream})
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt)
                 mem-info (memory-info :cuda :device 0 :default)
                 x-data (cuda-malloc (* 784 Float/BYTES) :float)
                 x (onnx-tensor mem-info [1 1 28 28] x-data)
                 y-data! (cuda-malloc (* 10 Float/BYTES) :float)
                 y! (onnx-tensor mem-info [1 10] y-data!)
                 classify! (runner* sess)
                 data-binding (io-binding sess [x] [y!])]
    (memcpy-to-device! (float-pointer test-image-0) x-data) => x-data
    (classify! data-binding) => data-binding
    (let [res (pointer-vec (softmax (memcpy-to-host! y-data! (float-pointer 10))))
          seven (res 7)]
      seven => 1.0
      (apply max res) => seven)))
