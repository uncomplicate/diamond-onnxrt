;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/diamond-onnxrt "0.25.0-SNAPSHOT"
  :description "Fast Clojure Machine Learning Model Integration"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [com.cnuernber/charred "1.038"]
                 [uncomplicate/commons "0.21.0"]
                 [org.uncomplicate/deep-diamond-base "0.45.0-SNAPSHOT"]
                 [org.uncomplicate/deep-diamond-dnnl "0.45.0-SNAPSHOT"]
                 [org.bytedeco/onnxruntime-platform "1.26.0-1.5.14-SNAPSHOT"]
                 [org.bytedeco/cuda-platform "13.2-9.21-1.5.14-SNAPSHOT"]];;TODO remove

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
             :linux {:dependencies [[org.bytedeco/onnxruntime-platform-gpu "1.26.0-1.5.14-SNAPSHOT"]
                                    [org.bytedeco/openvino-platform "2026.1.0-1.5.14-SNAPSHOT"]
                                    [org.uncomplicate/neanderthal-mkl "0.63.0-SNAPSHOT"]
                                    [org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]
                                    [org.uncomplicate/deep-diamond-cuda "0.45.0-SNAPSHOT"]
                                    [org.bytedeco/cuda-redist "13.2-9.21-1.5.14-SNAPSHOT" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "13.2-9.21-1.5.14-SNAPSHOT" :classifier "linux-x86_64"]
                                    #_[org.bytedeco/cuda-redist-cudnn "13.2-9.21-1.5.14-SNAPSHOT" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-nccl "13.2-9.21-1.5.14-SNAPSHOT" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.bytedeco/onnxruntime-platform-gpu "1.26.0-1.5.14-SNAPSHOT"]
                                      [org.bytedeco/openvino-platform "2026.1.0-1.5.14-SNAPSHOT"]
                                      [org.uncomplicate/neanderthal-mkl "0.63.0-SNAPSHOT"]
                                      [org.bytedeco/mkl "2025.3-1.5.13" :classifier "windows-x86_64-redist"]
                                      [org.uncomplicate/deep-diamond-cuda "0.45.0-SNAPSHOT"]
                                      [org.bytedeco/cuda-redist "13.1-9.19-1.5.13" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "13.1-9.19-1.5.13" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cudnn "13.1-9.19-1.5.13" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-nccl "13.1-9.19-1.5.13" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.63.0-SNAPSHOT"]
                                     [org.bytedeco/openblas "0.3.31-1.5.13" :classifier "macosx-arm64"]]}}

  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
