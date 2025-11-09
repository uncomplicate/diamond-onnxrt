;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.cuda
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [transfer! iamax native]]
            [uncomplicate.diamond
             [tensor :refer [tensor with-diamond]]
             [dnn :refer [network]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [hello-world.native :refer [input-desc input-tz mnist-onnx]]))

;; There are a few flavors of how you can run this, these are just two of many! Explore :)
;; We can even reuse most of the general parts from the native example.

(with-diamond cudnn-factory []
  (with-release [cuda-input-tz (tensor input-desc)
                 mnist (network cuda-input-tz [mnist-onnx])
                 classify! (mnist cuda-input-tz)]
    (transfer! input-tz cuda-input-tz)
    (iamax (native (classify!)))))

;; If you see the number 7, you're ready to go.

;; If you see an exception, you should probably check your GPU drivers
;; and CUDA versions compatibility whit the CUDA binaries you use.
;; If that happens, first check your general CUDA setup by trying
;; either Neanderthal or Deep Diamond hello world. Make them work first,
;; since they are generally not that picky.
;; ONNX Runtime can be brittle regarding CUDA binaries, so deal with it
;; only when you know your system is well set.
