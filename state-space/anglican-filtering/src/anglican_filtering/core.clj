(ns anglican-filtering.core
  (:use clojure.repl
        [anglican core runtime emit])
  (:require [clojure.data.json :as json]))

;; simple discrete hmm, for testing
(def prior [0.2 0.3 0.5])
(def emission-dists [[0.1 0.2 0.7] [0.2 0.7 0.1] [0.7 0.2 0.1]])
(def transition-dists
    [[0.4 0.4 0.2]
     [0.2 0.3 0.5]
     [0.9 0.05 0.05]])
(def hmm-observations [0 0 1 2])
(defquery hmm
  "A hidden Markov model"
  []
    (let
        [init-state (sample (discrete prior))]
        (observe (discrete (get emission-dists init-state)) (get hmm-observations 0))
        (reduce 
            (fn [states obs]
                (let [state (sample (discrete (get transition-dists (peek states))))]
                (observe (discrete (get emission-dists state)) obs)
                (conj states state)))
            [init-state]
            (rest hmm-observations))))

;; actual nonlinear state space model
(def times [
    0.0
    0.05263157894736842
    0.10526315789473684
    0.15789473684210525
    0.21052631578947367
    0.2631578947368421
    0.3157894736842105
    0.3684210526315789
    0.42105263157894735
    0.47368421052631576
    0.5263157894736842
    0.5789473684210527
    0.631578947368421
    0.6842105263157895
    0.7368421052631579
    0.7894736842105263
    0.8421052631578947
    0.8947368421052632
    0.9473684210526315
    1.0])

(def start_x 0.1)

(def start_y 0.1)

(def stop_x 0.5)

(def stop_y 0.5)

(def speed 0.5)

(def noise 0.02)

(def dist_slack 0.2)

(def start [start_x start_y])

(def stop [stop_x stop_y])

(def path [
    [0.1 0.1]
    [0.077362 0.146073]
    [0.167036 0.655448]
    [0.168662 0.649074]
    [0.156116 0.752046]
    [0.104823 0.838075]
    [0.196407 0.873581]
    [0.390309 0.988468]
    [0.408272 0.91336]
    [0.5 0.5]])

(def distances-from-start [0.0, 0.0513339, 0.568542, 0.57512, 0.678854, 0.779013, 0.877239, 1.10262, 1.17985, 1.60326])

(println "count of distances-from-start:" (count distances-from-start))
    
(def measured_xs [0.0896684, 0.148145, 0.123211, 0.11035, 0.148417, 0.185746, 0.175872, 0.178704, 0.150475, 0.175573, 0.150151, 0.172628, 0.121426, 0.222041, 0.155273, 0.164001, 0.136586, 0.0687045, 0.146904, 0.163813])

(def measured_ys [0.217256, 0.416599, 0.376985, 0.383586, 0.500322, 0.608227, 0.632844, 0.653351, 0.532425, 0.881112, 0.771766, 0.653384, 0.756946, 0.870473, 0.8697, 0.808217, 0.598147, 0.163257, 0.611928, 0.657514])

(def observations (apply vector (map vector measured_xs measured_ys)))

(def dt 0.05263157894736842)

(defn walk-path
    [path distances-from-start dist]
    (cond
        (<= dist 0.0) (first path)
        (>= dist (last distances-from-start)) (last path)
        :else (let [
        path_point_index (loop [i 0]
            (if (> (distances-from-start (+ i 1)) dist)
                i
                (recur (+ i 1))))
        dist_from_path_point (- dist (distances-from-start path_point_index))
        dist_between_points (- (distances-from-start (+ path_point_index 1))
                               (distances-from-start path_point_index))
        fraction_next (/ dist_from_path_point dist_between_points)
        x (+ (* fraction_next (first (path (+ path_point_index 1))))
             (* (- 1.0 fraction_next) (first (path path_point_index))))
        y (+ (* fraction_next (second (path (+ path_point_index 1))))
             (* (- 1.0 fraction_next) (second (path path_point_index))))]
        [x y])))

(with-primitive-procedures [walk-path]
(defquery model
  "A nonlinear state space model"
  []
    (let
        [init-dist (sample (normal (* speed dt) dist_slack))
         init-loc (walk-path path distances-from-start init-dist)
        ]
        (observe (normal (first init-loc) noise) (first (get observations 0)))
        (observe (normal (second init-loc) noise) (second (get observations 0)))
        (reduce 
            (fn [dists obs]
                (let [dist (sample (normal (+ (peek dists) (* speed dt)) dist_slack))
                      loc (walk-path path distances-from-start dist)
                     ]
                    (observe (normal (first loc) noise) (first obs))
                    (observe (normal (second loc) noise) (second obs))
                    (conj dists dist)))
            [init-dist]
            (rest observations)))))

(defn run-particle-filter
  "Run a particle filter and return the marginal likelihood estimate"
  [number-of-particles]
    (let [start (System/nanoTime)
          lml ((first (doquery :smc model [] :number-of-particles number-of-particles))
               :log-weight)
          elapsed (double (/ (- (System/nanoTime) start) 1e9))]
         {:lml lml :elapsed elapsed}))

(defn filtering-experiment
    []
    (let [num-particles-list [1 2 3 5 7 10 15 20 25 30 35 40 45 50 60 70 80 90 100 200 300]
          num-reps 50]
        (println (spit "anglican-results.json" (json/write-str (zipmap num-particles-list (mapv
            (fn [num-particles]
                (println num-particles)
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
