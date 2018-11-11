data {
    int<lower=0> N;
    real x[N];
    real y[N];
}

parameters {
    real slope;
    real intercept;
    real inlier_log_var;
    real outlier_log_var;
}

model {
    slope ~ normal(0, 2);
    intercept ~ normal(0, 2);
    inlier_log_var ~ normal(0, 1);
    outlier_log_var ~ normal(0, 1);
    for (i in 1:N) {
        target += log_sum_exp(
            log(0.5) + normal_lpdf(y[i] | slope * x[i] + intercept, sqrt(exp(inlier_log_var))),
            log(0.5) + normal_lpdf(y[i] | slope * x[i] + intercept, sqrt(exp(outlier_log_var))));
    }
}
