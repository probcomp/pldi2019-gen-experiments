(ns anglican-filtering.core
  (:use clojure.repl
        [anglican core runtime emit]))

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
    (first (doquery :smc hmm-unrolled [] :number-of-particles number-of-particles)))

(defn filtering-experiment
    []
    (println (run-particle-filter 100000)))

;; see:
;; https://bitbucket.org/probprog/anglican-user
;; https://github.com/probprog/anglican-examples/blob/master/worksheets/aistats/hmm-aistats.clj
