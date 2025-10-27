;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.native
  (:require [uncomplicate.neanderthal.core :refer [transfer! iamax]]
            [uncomplicate.diamond
             [tensor :refer [tensor desc]]
             [dnn :refer [network activation]]
             [onnxrt :refer [onnx]]]
            [uncomplicate.diamond.native]))

;; Creating an ONNX runtime model in Deep Diamond is almost trivial: just point it to
;; a location that contains the model data!

(def mnist-onnx (onnx "../../data/mnist-12.onnx"))

;; What to do with it now? Well, it's a function, one that can create Deep Diamond
;; layers and fit in with the rest of the Tensor-y stuff that DD already provides,
;; and this is just what we need.

(def input-desc (desc [1 1 28 28] :float :nchw))

;; We create an abstract blueprint that can create networks that classify MNIST images.
;; The whole ONNX model has already been trained, we now load it and use it in production.
;; Note that the ReLU activation layer is not relevant for inference, so it can be left out
;; (that's why it had been left out from the ONNX model in the first place). I include it here
;; just to show you how you can mix ONNX models with deep-diamond DNN layers, in cases when
;; you do need that. You can also use ONNX at different layers of abstraction; explore DD!

(def mnist (network input-desc [mnist-onnx (activation :relu)]))

;; Or, alternatively, without the redundant activation:
;; (def mnist (network input-desc [mnist-onnx]))

;; We create a tensor in main memory that will be the input to the real network. It should
;; be filled with each image (28x28 pixels) that we want to classify, one at a time.
;; (It's just a hello-world, after all. Typically, you'll provide many images at once)...

(def input-tz (tensor input-desc))

;; A blueprint (mnist in this case) is a function that can create networks optimized for inference
;; with concrete tensors, with adequate internal tensors, and parameter tensors.
;; The following line is the moment when the network is actually created from the abstract descriptors
;; contained in its blueprint, to the actual engines, operation primitives, and tensors in memory.
;; (mnist input-tz) produces classify!, which is a function!

(def classify! (mnist input-tz))

;; This is how you would typically use it:
;; 1. classify! is now a typical Clojure function! Evaluate it:

(classify!)

;; You'll get an output tensor with the classification, but the content is gibberish; we didn't put an actual image data in that input.
;; => {:shape [1 10], :data-type :float, :layout [10 1]} (0.0 0.007791661191731691 0.06810081750154495 0.02999374084174633 0.0 0.14021874964237213 0.0 0.0 0.08432205021381378 0.0)

;; 2. But we have to place the image data in network's input somehow. This could be done in
;; a few different ways (for example, by memory-mapping the image data on disk), but we'll
;; keep it simple, and we'll just transfer it naively from an in-place Clojure sequence. (It's a hello-world :)
;; 28 by 28, times 1 channel is 756 pixels, so the following sequence is a little ugly,
;; but it's pedagogical!

(transfer! (map #(float (/ % 255)) [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 84.0 185.0 159.0 151.0 60.0 36.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 222.0 254.0 254.0 254.0 254.0 241.0 198.0 198.0 198.0 198.0 198.0 198.0 198.0 198.0 170.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 67.0 114.0 72.0 114.0 163.0 227.0 254.0 225.0 254.0 254.0 254.0 250.0 229.0 254.0 254.0 140.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 17.0 66.0 14.0 67.0 67.0 67.0 59.0 21.0 236.0 254.0 106.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 83.0 253.0 209.0 18.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 22.0 233.0 255.0 83.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 129.0 254.0 238.0 44.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 59.0 249.0 254.0 62.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 133.0 254.0 187.0 5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 9.0 205.0 248.0 58.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 126.0 254.0 182.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 75.0 251.0 240.0 57.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 19.0 221.0 254.0 166.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 203.0 254.0 219.0 35.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 38.0 254.0 254.0 77.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 31.0 224.0 254.0 115.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 133.0 254.0 254.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 61.0 242.0 254.0 254.0 52.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 121.0 254.0 254.0 219.0 40.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 121.0 254.0 207.0 18.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
           input-tz)

;; To know which category the image belongs, we can evaluate the network, and then see
;; which index contains the larges number in its output.
(iamax (classify!))

;; If you see the number 7, this means that this image displays the number 7.
;; (It does, this is the first image from the traditional MNIST dataset, which we know to be 7.)

;; Now, mind that this example doesn't release any data and models that we used.
;; See the cuda example for a more careful approach towards garbage disposal :)
