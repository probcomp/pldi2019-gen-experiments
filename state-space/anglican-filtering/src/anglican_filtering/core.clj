(ns anglican-filtering.core
  (:use clojure.repl
        [anglican core runtime emit])
  (:require [clojure.data.json :as json]))

(def prior [0.2 0.3 0.5])

(def emission-dists [[0.1 0.2 0.7] [0.2 0.7 0.1] [0.7 0.2 0.1]])

(def transition-dists
    [[0.4 0.4 0.2]
     [0.2 0.3 0.5]
     [0.9 0.05 0.05]])

(def observations [0 0 1 2])

(defquery hmm
  "A hidden Markov model"
  []
    (let
        [init-state (sample (discrete prior))]
        (observe (discrete (get emission-dists init-state)) (get observations 0))
        (reduce 
            (fn [states obs]
                (let [state (sample (discrete (get transition-dists (peek states))))]
                (observe (discrete (get emission-dists state)) obs)
                (conj states state)))
            [init-state]
            (rest observations))))

(defn run-particle-filter
  "Run a particle filter and return the marginal likelihood estimate"
  [number-of-particles]
    (let [start (System/nanoTime)
          lml ((first (doquery :smc hmm [] :number-of-particles number-of-particles))
               :log-weight)
          elapsed (double (/ (- (System/nanoTime) start) 1e9))]
         {:lml lml :elapsed elapsed}))

(defn filtering-experiment
    []
    (let [num-particles-list [1 2 3 5 7 10 15 20 25 30 35 40 45 50 60 70 80 90 100 200 300]
          num-reps 50]
        (println (spit "anglican-results.json" (json/write-str (zipmap num-particles-list (mapv
            (fn [num-particles]
                (let 
                    ; results is a vector of {:lml lml :elapsed elapsed}
                    [results (mapv (fn [i] (run-particle-filter num-particles))
                                   (range num-reps))]
                    {:lmls (mapv (fn [res] (res :lml))
                                 results)
                     :elapsed (mapv (fn [res] (res :elapsed))
                                    results)}))
            num-particles-list)))))))

;; see:
;; https://bitbucket.org/probprog/anglican-user
;; https://github.com/probprog/anglican-examples/blob/master/worksheets/aistats/hmm-aistats.clj
