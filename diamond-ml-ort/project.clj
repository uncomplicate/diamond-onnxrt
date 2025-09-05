;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/diamond-ml-ort "0.1.0-SNAPSHOT"
  :description "Fast Clojure Machine Learning Model Integration"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [org.uncomplicate/deep-diamond-base "0.35.2"]
                 [org.bytedeco/onnxruntime-platform "1.20.0-1.5.11"]]


  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]]
                       :resource-paths ["data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [org.uncomplicate/deep-diamond-test "0.35.2"]]
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.uncomplicate/deep-diamond-dnnl "0.35.2"]]}
             :windows {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.uncomplicate/deep-diamond-dnnl "0.35.2"]]}
             :macosx {:dependencies [[org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]
                                     [org.uncomplicate/deep-diamond-bnns "0.35.2"]
                                     [org.bytedeco/onnxruntime-platform "1.22.2-1.5.13-SNAPSHOT"]]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]
                 ["maven-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
