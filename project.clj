;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/diamond-onnxrt "0.10.0"
  :description "Fast Clojure Machine Learning Model Integration"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.3"]
                 [org.uncomplicate/deep-diamond-base "0.39.0"]
                 [org.uncomplicate/deep-diamond-dnnl "0.39.0"]
                 [org.bytedeco/onnxruntime-platform "1.22.2-1.5.13-20250919.193005-2"]]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]]
                       :resource-paths ["data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]]
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.bytedeco/onnxruntime-platform-gpu "1.22.2-1.5.13-20250919.192912-2"]
                                    [org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.uncomplicate/deep-diamond-cuda "0.39.1"]
                                    #_[org.bytedeco/cuda-redist "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    #_[org.bytedeco/cuda-redist-cublas "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    #_[org.bytedeco/cuda-redist-cudnn "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    ]}
             :windows {:dependencies [[org.bytedeco/onnxruntime-platform-gpu "1.22.2-1.5.13-20250919.192912-2"]
                                      [org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.uncomplicate/deep-diamond-cuda "0.39.1"]
                                      #_[org.bytedeco/cuda-redist "12.9-9.10-1.5.12" :classifier "windows-x86_64"]
                                      #_[org.bytedeco/cuda-redist-cublas "12.9-9.10-1.5.12" :classifier "windows-x86_64"]
                                      #_[org.bytedeco/cuda-redist-cudnn "12.9-9.10-1.5.12" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.57.0"]
                                     [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]}}

  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
