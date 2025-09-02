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
            [uncomplicate.commons.core :refer [with-release bytesize size release]]
            [uncomplicate.diamond.internal.ort.core :refer :all])
  (:import clojure.lang.ExceptionInfo))

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
    (input-type-info sess 0) => {:shape [3 2] :type 1}
    (input-type-info sess 1) => (throws IndexOutOfBoundsException)
    (input-type-info sess) => [{:shape [3 2] :type 1}]
    (output-type-info sess 0) => {:shape [3] :type 7}
    (.GetElementType (.GetTensorTypeAndShapeInfo (output-type-info sess 1))) => :a
    (output-type-info sess 2) => (throws IndexOutOfBoundsException)))
