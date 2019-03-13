(ns anglican-filtering.core
  (:use clojure.repl
        [anglican core runtime emit]))

(defquery tiny-model
    "A tiny model"
    []
    (let [x (sample (bernoulli 0.5))]
        (observe (normal x 1.) 0.8)
        x))

(def prior [0.2 0.3 0.5])

(def emission-dists [[0.1 0.2 0.7] [0.2 0.7 0.1] [0.7 0.2 0.1]])

(def transition-dists
    [[0.4 0.4 0.2]
     [0.2 0.3 0.5]
     [0.9 0.05 0.05]])

(def observations [1 1 2 3])

(defquery hmm-unrolled
  "An unrolled hidden Markov model"
   []
  (let
    [z0 (sample (discrete prior))
     x0 (observe (discrete (get emission-dists z0)) (get observations 0))
     z1 (sample (discrete (get transition-dists z0)))
     x1 (observe (discrete (get emission-dists z1)) (get observations 1))
     z2 (sample (discrete (get transition-dists z1)))
     x2 (observe (discrete (get emission-dists z2)) (get observations 2))
     z3 (sample (discrete (get transition-dists z2)))
     x3 (observe (discrete (get emission-dists z3)) (get observations 3))]
    [z0 z1 z2 z3]))

(defquery hmm
  "A hidden Markov model"
  []
    (reduce 
      (fn [states obs]
        (let [state (sample (discrete (get transition-dists (peek states))))]
          (observe (discrete (get emission-dists state)) obs)
          (conj states state)))
      [(sample (discrete prior))]
      observations))

(defn run-particle-filter
  "Run a particle filter and return the marginal likelihood estimate"
  [number-of-particles]
  ((first (doquery :smc tiny-model [] :number-of-particles number-of-particles)) :log-weight))
  ;;((first (doquery :smc hmm-unrolled [] :number-of-particles number-of-particles)) :log-weight))
  ;;((first (doquery :smc hmm [] :number-of-particles number-of-particles)) :log-weight))

(defn filtering-experiment
    []
    (println (run-particle-filter 10000)))

;; see:
;; https://bitbucket.org/probprog/anglican-user
;; https://github.com/probprog/anglican-examples/blob/master/worksheets/aistats/hmm-aistats.clj
