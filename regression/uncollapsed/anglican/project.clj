(defproject anglican-regression "0.1.0-SNAPSHOT"
  :jvm-opts ["-Xss50M"] ; See `deps.edn` for an explanation of this setting]
  :dependencies [[org.clojure/clojure "1.8.0"]
		 [anglican "1.0.0"]
		 [org.clojure/data.csv "0.1.4"]]
  :main anglican-regression.core/experiment)
