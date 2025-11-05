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
            [uncomplicate.neanderthal.internal.api :refer [device flow]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory*]]
            [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory diamond-factory]]
            [uncomplicate.diamond.internal.onnxrt
             [core :refer [environment options session  memory-info threading-options
                           graph-optimization! available-providers append-provider!
                           disable-per-session-threads! run-options config! input-count output-count]]
             [model :refer [onnx-single-io-model onnx-multi-io-model]]]))

(def ^:dynamic *onnx-options*
  {:env nil
   :env-options nil
   :logging-level :warning
   :log-name (name (gensym "diamond_onnxrt_"))
   :graph-optimization :extended
   :options nil
   :ep nil
   :dnnl {:arena true}
   :cuda {:device-id 0
          :copy-in-default-stream true
          ;;:conv-algo-search :exhaustive ;;TODO
          :conv-use-max-workspace true
          :enable-cuda-graph false
          :conv1d-pad-to-nc1d false
          :tunable-op-enable false
          :tunable-op-tuning-enable false
          :tunable-op-max-tuning-duration-ms 0
          :skip-layer-norm-strict-mode false
          :prefer-nhwc false
          :use-ep-level-unified-stream false
          :ep-level-unified-stream false
          :tf32 true
          :fuse-conv-bias false
          :sdpa-kernel false}
   :coreml {:compute-units :all
            :model-format :ml-program
            :static-input true
            :enable-on-subgraphs true
            :specialization-strategy :fast-prediction
            :profile-compute-plan false
            :low-precision-accumulation false}
   :run-options nil})

;; TODO use spec for detailed args validation.
(defn onnx
  ([fact model-path args]
   (doseq [s [args (:run-options args) (:dnnl args) (:cuda args) (:coreml args)]]
     (when-not (or (nil? s) (map? s))
       (dragan-says-ex "This configuration must be either nil or a map."
                       :config s)))
   (let [fact (diamond-factory fact)
         dev (device (neanderthal-factory fact :float))
         merged-args (into *onnx-options* args)]
     (with-release [env-options (threading-options (:env-options merged-args))]
       (let-release [env (or (:env merged-args)
                             (environment (:logging-level merged-args)
                                          (:log-name merged-args)
                                          env-options))]
         (let [available-ep (set (available-providers))
               eproviders (or (:ep merged-args)
                              (filter available-ep (if (= :cuda dev) [:cuda] [:coreml :dnnl])))
               uses-device (some #{:cuda} eproviders)
               alloc-type (if (or uses-device (= :cuda dev));;TODO I have to check what's the case on MacOS.
                            :device
                            :arena)
               mem-type (if (and (= :device alloc-type) (= :cpu dev))
                          :cpu
                          :default)
               merged-args (if uses-device
                             (assoc-in merged-args [:cuda :stream] (flow fact))
                             merged-args)]
           (let-release [opt (-> (if-let [opt (:options merged-args)]
                                   (options opt)
                                   (options))
                                 (disable-per-session-threads!)
                                 (graph-optimization! (:graph-optimization merged-args)))
                         run-opt (if-let [run-opts (:run-options merged-args)]
                                   (config! (run-options) run-opts)
                                   nil)
                         mem-info (memory-info dev alloc-type mem-type)]
             (doseq [ep eproviders]
               (append-provider! opt
                                 (or (available-ep ep)
                                     (dragan-says-ex (format "Execution provider %s is not available." ep)
                                                     {:requested ep :available available-ep}))
                                 (into (*onnx-options* ep) (merged-args ep))))
             (let-release [sess (session env model-path opt)]
               (if (and (not (:multi-io merged-args)) (= 1 (input-count sess) (output-count sess)))
                 (onnx-single-io-model fact sess opt run-opt mem-info)
                 (onnx-multi-io-model fact sess opt run-opt mem-info)))))))))
  ([model-path args]
   (fn onnx-fn
     ([fact src-desc]
      (onnx fact model-path (if (sequential? src-desc) (assoc args :multi-io true))))
     ([src-desc]
      (onnx-fn *diamond-factory* src-desc))))
  ([model-path]
   (onnx model-path nil)))
