(ns r.core
  (:gen-class)
  (:require
    [anglican.core :refer :all]
    [anglican.runtime :refer :all]
    [anglican.emit :refer :all]
    [clojure.data.csv :as csv]
    [clojure.pprint :refer [pprint]]
    [clojure.java.io :as io]))


;; Load xs and ys from train.csv
(def data
  (with-open [in-file (io/reader "../../train.csv")]
    (doall
      (csv/read-csv in-file))))

(def xs (vec (map (comp read-string first) (rest data))))
(def ys (vec (map (comp read-string second) (rest data))))


;; Uncollapsed model
(defquery regress-uncollapsed
  [xs ys]
  (let [N  (count ys)
        slope (sample (normal 0 2))
        intercept (sample (normal 0 2))
        inlier-std-choice (sample (normal 0 2))
        inlier-std (sqrt (exp inlier-std-choice))
        outlier-std-choice (sample (normal 0 2))
        outlier-std (sqrt (exp outlier-std-choice))]
        (loop [n 0, outliers []]
          (if (= n N)
            {:slope slope, :intercept intercept, :inlier-std-choice inlier-std-choice, :outlier-std-choice outlier-std-choice, :outliers outliers}
            (let [outlier? (sample (bernoulli 0.5))
                  line-at-x (+ intercept (* (nth xs n) slope))
                  std (if (= outlier? 1) outlier-std inlier-std)]
               (observe (normal line-at-x std) (nth ys n))
               (recur (inc n) (conj outliers outlier?)))))))

(with-primitive-procedures [pprint]
(defquery regress-uncollapsed-score
  [xs ys trace]
  (let [N (count ys)
        {:keys [slope intercept inlier-std-choice outlier-std-choice outliers]} trace
        _ (observe (normal 0 2) slope)
        _ (observe (normal 0 2) intercept)
        _ (observe (normal 0 2) inlier-std-choice)
        _ (observe (normal 0 2) outlier-std-choice)
        inlier-std (sqrt (exp inlier-std-choice))
        outlier-std (sqrt (exp outlier-std-choice))]
     (loop [n 0]
       (if (= n N)
         nil
         (let [outlier? (observe (bernoulli 0.5) (nth outliers n))
               std (if (= outlier? 1) outlier-std inlier-std)
               line-at-x (+ intercept (* slope (nth xs n)))]
           (observe (normal line-at-x std) (nth ys n))
           (recur (inc n))))))))

;; Collapsed model
(defdist two-normal
    "Mixture of two normals"
    [mu sig1 sig2]
    []
    (sample* [this] (+ mu (if (= (sample* (bernoulli 0.5)) 1)
                              (sample* (normal mu sig1))
                              (sample* (normal mu sig2)))))
    (observe* [this value]
      (if (or (< sig1 0) (< sig2 0))
          (- (/ 1.0 0.0))
          (let [l1 (+ (log 0.5) (observe* (normal mu sig1) value))
                l2 (+ (log 0.5) (observe* (normal mu sig2) value))
                m (max l1 l2)]
              (+ m (log (+ (exp (- l1 m)) (exp (- l2 m)))))))))

(with-primitive-procedures [two-normal]
  (defquery regress-collapsed [xs ys]
    (let [N (count ys)
          slope (sample (normal 0 2))
          intercept (sample (normal 0 2))
          inlier-std-choice (sample (normal 0 2))
          outlier-std-choice (sample (normal 0 2))
          inlier-std (sqrt (exp inlier-std-choice))
          outlier-std (sqrt (exp outlier-std-choice))]
        (loop [n 0]
          (if (= n N)
              nil
              (let [x (nth xs n)
                    y (nth ys n)
                    fx (+ intercept (* x slope))]
                (observe (two-normal (+ intercept (* x slope)) inlier-std outlier-std) y)
                (recur (inc n)))))
        {:slope slope
         :intercept intercept
         :inlier-std-choice inlier-std-choice
         :outlier-std-choice outlier-std-choice})))

(with-primitive-procedures [two-normal pprint]
  (defquery regress-collapsed-score
    [xs ys trace]
    (let [N (count ys)
          {:keys [slope intercept inlier-std-choice outlier-std-choice]} trace
          _ (observe (normal 0 2) slope)
          _ (observe (normal 0 2) intercept)
          _ (observe (normal 0 2) inlier-std-choice)
          _ (observe (normal 0 2) outlier-std-choice)
          inlier-std (sqrt (exp inlier-std-choice))
          outlier-std (sqrt (exp outlier-std-choice))]
        (loop [n 0]
          (if (= n N)
              nil
              (let [x (nth xs n)
                    y (nth ys n)
                    fx (+ intercept (* x slope))]
                (observe (two-normal fx inlier-std outlier-std) y)
                (recur (inc n))))))))

(defn score-trace [scorer trace i]
  (dissoc (assoc trace :step i, :log-weight (:log-weight (first (doquery :importance scorer [xs ys trace])))) :outliers))

(defn run-lmh [model scorer steps]
  (let [start (System/nanoTime)
        results (doall (take steps (map :result (doquery :lmh model [xs ys]))))
        elapsed (double (/ (- (System/nanoTime) start) 1e6))
        ]
    {:elapsed elapsed
     :per-step (/ elapsed steps)
     :traces (map-indexed #(score-trace scorer %2 %1) results)}))

(defn process-lmh-to-csv-rows [steps result]
  (doseq [i (range steps)]
    (let [trace (nth (:traces result) i)]
      (println (str i "," (* (inc i) (:per-step result)) "," (:slope trace) "," (:intercept trace) "," (:inlier-std-choice trace) ","
        (:outlier-std-choice trace))))))

(def num-steps 200)
(def num-experiments 1)
(defn experiment []
  (doseq [i (range num-experiments)]
    ;(println (str "Lightweight MH for " n-steps " steps (collapsed) - iterate " i))
    (println "num_steps,runtime,score,slope,intercept,inlier_log_var,outlier_log_var")
    (process-lmh-to-csv-rows num-steps
      (run-lmh regress-collapsed regress-collapsed-score num-steps))))
  ; (println "Lightweight MH for 1000 steps (collapsed)")
  ; (pprint (run-lmh regress-collapsed regress-collapsed-score 1000)))
