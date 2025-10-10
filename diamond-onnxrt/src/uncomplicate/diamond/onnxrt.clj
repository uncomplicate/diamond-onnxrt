;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.onnxrt
  (:require [uncomplicate.commons
             [core :refer [with-release let-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.internal.api :refer [device]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory*]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer [environment options session  memory-info
                           graph-optimization! available-providers append-provider!]]
             [model :refer [onnx-straight-model]]]))

(def ^:dynamic *session-options*
  {:logging-level :warning
   :log-name (name (gensym "diamond_onnxrt_"))
   :graph-optimization :extended
   :ep nil
   :dnnl nil
   :cuda nil
   :coreml nil
   :run-options nil})

(defn onnx
  ([model-path args]
   (let [args (into *session-options* args)
         available-ep (set (available-providers))]
     (let-release [env (environment (:logging-level args) (:log-name args))]
       (fn onnx-fn
         ([fact src-desc]
          (let [dev (device (neanderthal-factory fact :float))
                eproviders (get args :ep (filter available-ep (if (= :cuda dev)
                                                                [:cuda]
                                                                [:coreml :dnnl])))
                uses-device (some #{:cuda} eproviders)
                alloc-type (if (or uses-device (= :cuda dev))
                             :device
                             :arena)
                mem-type (if (and (= :device alloc-type) (= :cpu dev))
                           :cpu
                           :default)]
            (with-release [opt (-> (options)
                                   (graph-optimization! (:graph-optimization args)))]
              (let-release [sess (session env model-path opt)
                            mem-info (memory-info dev alloc-type mem-type)]
                (doseq [ep (reverse eproviders)]
                  (append-provider! opt
                                    (or (available-ep ep)
                                        (dragan-says-ex (format "Execution provider %s is not available." ep)
                                                        {:requested ep :available available-ep}))
                                    (args ep)))
                (onnx-straight-model fact sess (:run-options args) mem-info)))))
         ([src-desc]
          (onnx-fn *diamond-factory*))))))
  ([model-path]
   (onnx model-path nil)))
