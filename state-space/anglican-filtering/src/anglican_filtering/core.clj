(ns anglican-filtering.core
  (:use [anglican emit runtime core]))

(defquery template
  "query template"
  (let [x (sample (bernoulli 0.5))]
    (observe (normal x 1.) 0.8)
    x))

(defn filtering-experiment
  "Run the filtering experiment; print the log marginal likelihood estimate"
  [& args]
  (println "Hello, World!")
  (println (take 1 (doquery :smc template [] :number-of-particles 10000))))
