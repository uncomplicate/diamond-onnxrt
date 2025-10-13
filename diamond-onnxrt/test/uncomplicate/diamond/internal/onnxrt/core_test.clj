;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.onnxrt.core-test
  (:require [midje.sweet :refer [facts throws => =not=> roughly truthy just]]
            [uncomplicate.commons.core :refer [with-release info bytesize size release]]
            [uncomplicate.fluokitten.core :refer [fold fmap!]]
            [uncomplicate.clojure-cpp
             :refer [null? float-pointer long-pointer pointer-vec capacity! put-entry! fill! get-entry
                     pointer-pointer]]
            [uncomplicate.neanderthal.math :refer [exp]]
            [uncomplicate.diamond.internal.onnxrt.core :refer :all])
  (:import clojure.lang.ExceptionInfo))

(init-ort-api!)

(facts
  "Test system."
  (version) => {:major 1 :minor 22 :update 2}
  (filter #{:dnnl :cpu} (available-providers)) => [:dnnl :cpu]
  (type (build-info)) => String)

(facts
  "Test DNNL execution provider."
  (with-release [env (environment nil)
                 opt (options)]
    (info opt) => truthy
    (append-provider! opt :random nil) => (throws ExceptionInfo)
    (append-provider! opt :dnnl) => opt
    (append-provider! opt :dnnl) => opt))

(facts
  "Test session releasing."
  (let [env (environment nil)
        opt (options)
        sess (session env "data/logreg_iris.onnx" opt)]
    (null? opt) => false
    (null? env) => false
    (null? sess) => false
    (release opt) => true
    opt => nil
    (null? env) => false
    (null? sess) => false))

(facts
  "Test session and session options."
  (let [env (environment nil)
        opt (options)
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
  "Test memory-info."
  (with-release [env (environment nil)
                 opt (options)
                 mem-info (memory-info :cpu :arena 0 :default)
                 mem-info1 (memory-info :cpu :arena 0 :default)
                 mem-info2 (memory-info :cpu :device 0 :default)]
    (allocator-key mem-info) => :cpu
    (allocator-type mem-info) => :arena
    (device-id mem-info) => 0
    (device-type mem-info) => :cpu
    (memory-type mem-info) => :default
    (equal-memory-info? mem-info nil) => false
    (equal-memory-info? mem-info mem-info) => true
    (equal-memory-info? mem-info mem-info1) => true
    (equal-memory-info? mem-info mem-info2) => false
    (info mem-info) => {:allocator-key :cpu
                        :allocator-type :arena
                        :device-id 0
                        :device-type :cpu
                        :memory-type :default}))

(facts
  "Test tensor values."
  (with-release [env (environment nil)
                 opt (options)
                 mem-info (memory-info :cpu :arena 0 :default)
                 data (float-array 5)
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
    (onnx-tensor mem-info 3 data) => (throws RuntimeException)))

(facts
  "Hello world example test."
  ;; This uses a mysterious format of logreg_iris that I am not sure is even sensible.
  ;; The example comes from an older onnxruntime, though, and the shape [3 2] is indeed correct
  ;; That shape just dont' match the original [-1 4], but it doesn't matter here,
  ;; because we only test whether the api works as intended, not whether the model or data makes any sense.
  (with-release [env (environment nil)
                 opt (options)
                 sess (session env "data/logreg_iris.onnx" opt)
                 input-info (input-type-info sess 0)
                 x-info (info input-info)
                 output-info-0 (output-type-info sess 0)
                 output-info-1 (output-type-info sess 1)
                 inputs-info (input-type-info sess)
                 output-1-element (sequence-type (cast-type output-info-1))
                 output-1-val (val-type (cast-type output-1-element))
                 mem-info (memory-info :cpu :arena 0 :default)
                 x-data (float-pointer (range (info input-info :count)))
                 x (onnx-tensor mem-info (:shape x-info) x-data)
                 infer! (runner* sess)
                 labels-data (long-pointer [0 1 2])
                 labels (onnx-tensor mem-info [3] labels-data)
                 probabilities-data (repeatedly 3 (partial float-pointer 3))
                 probabilities (mapv #(onnx-tensor mem-info [3] %) probabilities-data)
                 outputs! (onnx-sequence (map #(onnx-map labels %) probabilities))]
    sess =not=> nil
    (onnx-type input-info) => :tensor
    (denotation input-info) => nil
    (info input-info) => {:data-type :float :shape [3 2]}
    (input-type-info sess 1) => (throws IndexOutOfBoundsException)
    (map info inputs-info) => [{:data-type :float :shape [3 2]}]
    (info output-info-0) => {:data-type :long :shape [3]}
    (scalar? (cast-type output-1-val)) => true
    (info output-info-1) => {:structure [[:long :float]]}
    (output-type-info sess 2) => (throws IndexOutOfBoundsException)
    (symbolic-shape (cast-type input-info)) => ["" ""]
    (info x) => {:value {:data-type :float :shape [3 2]}}

    (with-release [outputs (infer! (pointer-pointer [x]))]
      (tensor? (value outputs 0)) => true
      (pointer-vec (capacity! (long-pointer (mutable-data (value outputs 0))) 3)) => [0 0 0]
      (value-count (value outputs 1)) => 3

      (map #(vector (pointer-vec (capacity! (long-pointer (mutable-data (value-value % 0))) 3))
                    (pointer-vec (capacity! (float-pointer (mutable-data (value-value % 1))) 3)))
           (value-value (value outputs 1)))
      => [[[0 1 2] (mapv float [0.64399236 0.3070779 0.04892978])]
          [[0 1 2] (mapv float [0.99137473 0.0012765623 0.0073487093])]
          [[0 1 2] (mapv float [0.9991861 2.4719932E-6 8.1140944E-4])]]

      (map #(vector (pointer-vec (capacity! (long-pointer (mutable-data (value-value % 0))) 3))
                    (pointer-vec (capacity! (float-pointer (mutable-data (value-value % 1))) 3)))
           (value-value (value (infer! (pointer-pointer [x]) (pointer-pointer [labels outputs!])) 1)))
      => [[[0 1 2] (mapv float [0.64399236 0.3070779 0.04892978])]
          [[0 1 2] (mapv float [0.99137473 0.0012765623 0.0073487093])]
          [[0 1 2] (mapv float [0.9991861 2.4719932E-6 8.1140944E-4])]])))

(facts
  "The correct logreg iris test."
  ;; This uses a logreg_iris.onnx model that matches the iris data as described in literature
  (with-release [env (environment nil)
                 opt (doto (options)
                       (append-provider! :dnnl)
                       (graph-optimization! :extended)
                       (override-dimension! :batch 2))
                 sess (session env "data/logreg_iris_correct.onnx" opt)
                 input-info (input-type-info sess 0)
                 output-info (output-type-info sess)
                 mem-info (memory-info :cpu :arena 0 :default)
                 x-data (float-pointer [5.1 3.5 1.4 0.2
                                        4.9 3.0	1.4 0.2])
                 x (onnx-tensor mem-info [2 4] x-data)
                 infer! (runner* sess)]
    (let [x-info (cast-type input-info)]
      (shape x-info) => [-1 4]
      (symbolic-shape x-info) => ["" ""]
      (symbolic-shape! x-info ["free_dimension" "2"])
      (symbolic-shape x-info) => ["free_dimension" "2"]
      (shape! x-info [2 4]) => x-info
      (tensor-count x-info) => 8)

    (with-release [outputs (infer! (pointer-pointer [x]))]
      (map #(vector (pointer-vec (capacity! (long-pointer (mutable-data (value-value % 0))) 3))
                    (pointer-vec (capacity! (float-pointer (mutable-data (value-value % 1))) 3)))
           (value-value (value outputs 1))) => [[[0 1 2] (mapv float [0.9794105 0.020589434 4.5429704E-8])]
                                                [[0 1 2] (mapv float [0.9692533 0.030746665 6.886014E-8])]])))

(defonce test-image-0 (map #(float (/ % 255)) [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 84.0 185.0 159.0 151.0 60.0 36.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 222.0 254.0 254.0 254.0 254.0 241.0 198.0 198.0 198.0 198.0 198.0 198.0 198.0 198.0 170.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 67.0 114.0 72.0 114.0 163.0 227.0 254.0 225.0 254.0 254.0 254.0 250.0 229.0 254.0 254.0 140.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 17.0 66.0 14.0 67.0 67.0 67.0 59.0 21.0 236.0 254.0 106.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 83.0 253.0 209.0 18.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 22.0 233.0 255.0 83.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 129.0 254.0 238.0 44.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 59.0 249.0 254.0 62.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 133.0 254.0 187.0 5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 9.0 205.0 248.0 58.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 126.0 254.0 182.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 75.0 251.0 240.0 57.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 19.0 221.0 254.0 166.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 203.0 254.0 219.0 35.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 38.0 254.0 254.0 77.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 31.0 224.0 254.0 115.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 133.0 254.0 254.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 61.0 242.0 254.0 254.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 121.0 254.0 254.0 219.0 40.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 121.0 254.0 207.0 18.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]))

(defn softmax [xs]
  (let [sum (fold + (fmap! exp xs))]
    (fmap! #(/ % sum) xs)))

(facts
  "Simple MNIST inference test."
  (with-release [env (environment :warning "test" nil)
                 opt (-> (options)
                         (append-provider! :dnnl)
                         (graph-optimization! :extended))
                 sess (session env "data/mnist-12.onnx" opt)
                 mem-info (memory-info :cpu :arena 0 :default)
                 input-info (input-type-info sess 0)
                 output-info (output-type-info sess 0)
                 x-data (float-pointer test-image-0)
                 x (onnx-tensor mem-info [1 1 28 28] x-data)
                 y-data! (fill! (float-pointer 10) 0)
                 y! (onnx-tensor mem-info [1 10] y-data!)
                 classify! (runner* sess)]
    (info input-info) => {:data-type :float :shape [1 1 28 28]}
    (info output-info) => {:data-type :float :shape [1 10]}
    (input-name sess 0) => "Input3"
    (output-name sess 0) => "Plus214_Output_0"
    (info sess) => {:input {"Input3" {:data-type :float :shape [1 1 28 28]}}
                    :output {"Plus214_Output_0" {:data-type :float :shape [1 10]}}}
    ;; This takes, more or less, 40 microseconds per one call (measured after init, with 100000 iterations)
    (get-entry (classify! (pointer-pointer [x]) (pointer-pointer [y!])))
    => (get-entry (pointer-pointer [y!]))
    (let [res (pointer-vec (softmax y-data!))
          seven (res 7)]
      seven => 1.0
      (apply max res) => seven)))
