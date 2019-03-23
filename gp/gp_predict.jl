import Statistics

using Gen

"""Obtain conditional multivariate normal predictive distribution."""
function get_conditional_mu_cov(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    return conditional_mu, conditional_cov_matrix
end

"""Return predictive log likelihood of new input/output values."""
function compute_log_likelihood_predictive(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64},
        new_ys::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return Gen.logpdf(mvnormal, new_ys, mu, cov)
end

"""Return predictive samples of output values for new inputs."""
function gp_predictive_samples(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return mvnormal(mu, cov)
end

function gp_predictive_samples(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64},
        N::Int)
    return [
        gp_predictive_samples(covariance_fn, noise, xs, ys, new_xs)
        for _i in 1:N
    ]
end

"""Return mean of predictive distribution on output values for new inputs."""
function gp_predictive_mean(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return mu
end
