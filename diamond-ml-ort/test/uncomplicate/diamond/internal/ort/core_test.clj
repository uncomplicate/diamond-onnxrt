;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.ort.core-test
  (:require [midje.sweet :refer [facts throws => =not=> roughly truthy just]]
            [uncomplicate.commons.core :refer [with-release info bytesize size release]]
            [uncomplicate.clojure-cpp :refer [null?]]
            [uncomplicate.diamond.internal.ort.core :refer :all])
  (:import clojure.lang.ExceptionInfo))

(facts
 "Tests release."
 (let [opt (options)
       env (environment)
       sess (session env "data/logreg_iris.onnx" opt)]
   (null? opt) => false
   (null? env) => false
   (null? sess) => false
   (release opt) => true
   opt => nil
   (null? env) => false
   (null? sess) => false))

(facts
  "Hello world example test"
  (with-release [opt (options)
                 env (environment)
                 sess (session env "data/logreg_iris.onnx" opt)]
    sess =not=> nil
    (input-count sess) => 1
    (output-count sess) => 2
    (input-name sess) => ["float_input"]
    (output-name sess) => ["label" "probabilities"]
    (info (input-type-info sess 0)) => {:count 6 :shape [3 2] :data-type :float :type :tensor}
    (input-type-info sess 1) => (throws IndexOutOfBoundsException)
    (map info (input-type-info sess)) => [{:count 6 :shape [3 2] :data-type :float :type :tensor}]
    (info (output-type-info sess 0)) => {:count 3 :shape [3] :data-type :long :type :tensor}
    (scalar? (value-type (element-type (output-type-info sess 1)))) => true
    (info (output-type-info sess 1)) => {:element [{:key :long :type :map :value :float}] :type :sequence}
    (output-type-info sess 2) => (throws IndexOutOfBoundsException)))
