"""
Bayesian Inference with PyMC: NUTS Sampling, Hierarchical Models, Posterior Analysis

Demonstrates prior specification, likelihood, NUTS (No-U-Turn Sampler),
posterior predictive checks, and ArviZ summaries.
"""

import pymc as pm
import numpy as np
import arviz as az
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")


def simple_bayesian_inference() -> Tuple[pm.Model, az.InferenceData]:
    """
    Simple model: estimate mean and std of normally distributed data.
    Data: y ~ N(μ, σ²)
    Priors: μ ~ N(0, 10), σ ~ Exp(0.5)
    """
    # Generate synthetic data
    np.random.seed(42)
    y_data = np.random.normal(loc=2.5, scale=1.3, size=100)

    model = pm.Model()
    with model:
        # Priors
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.Exponential("sigma", lam=0.5)

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

        # Sample posterior
        trace = pm.sample(
            2000,  # draws
            tune=1000,  # warmup/burn-in
            chains=4,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True,
        )

    return model, trace


def hierarchical_bayesian_model() -> Tuple[pm.Model, az.InferenceData]:
    """
    Hierarchical model for multi-group data.

    Schools example (classic): estimate effect sizes across schools.
    y_i ~ N(θ_i, σ_i²)  (within-school variation)
    θ_i ~ N(μ, τ²)      (between-school variation)
    μ ~ N(0, 25)
    τ ~ HalfNormal(10)
    """
    # Data: observed effects and standard errors from 8 schools
    schools_data = {
        "school": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "effect": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "std_err": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }

    y = schools_data["effect"]
    sigma = schools_data["std_err"]

    model = pm.Model()
    with model:
        # Hyper-priors
        mu = pm.Normal("mu", mu=0.0, sigma=25.0)
        tau = pm.HalfNormal("tau", sigma=10.0)

        # School-level effects (latent)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=len(y))

        # Likelihood (observed effects)
        y_obs = pm.Normal("y_obs", mu=theta, sigma=sigma, observed=y)

        # Sample
        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True,
        )

    return model, trace


def regression_with_priors() -> Tuple[pm.Model, az.InferenceData]:
    """
    Bayesian linear regression: y = α + β*x + ε
    Priors: α ~ N(0, 10), β ~ N(0, 1), σ ~ Exp(1)
    """
    np.random.seed(42)
    x_data = np.linspace(0.0, 10.0, 50)
    y_data = 2.0 + 0.5 * x_data + np.random.normal(0.0, 1.0, len(x_data))

    model = pm.Model()
    with model:
        # Priors
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        # Linear regression
        mu = alpha + beta * x_data
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

        # Sample
        trace = pm.sample(
            2000,
            tune=1000,
            chains=2,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True,
        )

    return model, trace


def model_comparison() -> Dict[str, float]:
    """
    Compare models using LOO-CV (Leave-One-Out Cross-Validation)
    and WAIC (Widely Applicable Information Criterion).
    """
    np.random.seed(42)
    y_data = np.random.normal(loc=0.0, scale=1.5, size=100)

    # Model 1: Normal likelihood with unknown mean and variance
    model1 = pm.Model(name="normal_model")
    with model1:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.Exponential("sigma", lam=0.5)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)
        trace1 = pm.sample(
            1000, tune=500, chains=2, progressbar=False, return_inferencedata=True
        )

    # Model 2: Student-t likelihood (heavier tails, robust to outliers)
    model2 = pm.Model(name="student_t_model")
    with model2:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.Exponential("sigma", lam=0.5)
        nu = pm.Exponential("nu", lam=0.1)
        y = pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_data)
        trace2 = pm.sample(
            1000, tune=500, chains=2, progressbar=False, return_inferencedata=True
        )

    # Compute LOO and WAIC
    loo1 = az.loo(trace1)
    loo2 = az.loo(trace2)

    waic1 = az.waic(trace1)
    waic2 = az.waic(trace2)

    # Compare (negative LOO/WAIC is better)
    loo_diff = loo2.elpd_loo - loo1.elpd_loo
    waic_diff = waic2.elpd_waic - waic1.elpd_waic

    return {
        "model1_loo": float(loo1.elpd_loo),
        "model2_loo": float(loo2.elpd_loo),
        "loo_difference": float(loo_diff),
        "model1_waic": float(waic1.elpd_waic),
        "model2_waic": float(waic2.elpd_waic),
        "waic_difference": float(waic_diff),
    }


def posterior_predictive_check(model: pm.Model, trace: az.InferenceData) -> Dict:
    """
    Posterior predictive check: simulate new data from posterior
    and compare with observed data.
    """
    with model:
        ppc = pm.sample_posterior_predictive(
            trace, progressbar=False, return_inferencedata=True
        )

    # Compute statistics
    y_data = model.observed_data["y"].values
    y_pred = ppc.posterior_predictive["y"].values.flatten()

    return {
        "observed_mean": float(np.mean(y_data)),
        "predicted_mean": float(np.mean(y_pred)),
        "observed_std": float(np.std(y_data)),
        "predicted_std": float(np.std(y_pred)),
        "observed_min": float(np.min(y_data)),
        "predicted_min": float(np.min(y_pred)),
        "observed_max": float(np.max(y_data)),
        "predicted_max": float(np.max(y_pred)),
    }


def arviz_summary_example(trace: az.InferenceData) -> Dict:
    """
    Generate ArviZ summary statistics.
    Includes: mean, std, hdi (highest density interval), effective sample size, Rhat.
    """
    summary = az.summary(trace)
    return summary.to_dict()


def mixture_model_example() -> Tuple[pm.Model, az.InferenceData]:
    """
    Mixture of Gaussians: data from mixture of two normal distributions.
    y ~ (1-w)*N(μ1, σ1) + w*N(μ2, σ2)
    """
    np.random.seed(42)
    # Generate mixture data
    w_true = 0.3
    y_part1 = np.random.normal(-2.0, 0.5, size=int(100 * (1 - w_true)))
    y_part2 = np.random.normal(3.0, 1.0, size=int(100 * w_true))
    y_data = np.concatenate([y_part1, y_part2])
    np.random.shuffle(y_data)

    model = pm.Model()
    with model:
        # Component parameters
        mu1 = pm.Normal("mu1", mu=-2.0, sigma=2.0)
        mu2 = pm.Normal("mu2", mu=2.0, sigma=2.0)
        sigma1 = pm.HalfNormal("sigma1", sigma=1.0)
        sigma2 = pm.HalfNormal("sigma2", sigma=1.0)

        # Mixture weight
        w = pm.Beta("w", alpha=1.0, beta=1.0)

        # Likelihood as mixture
        y = pm.NormalMixture(
            "y",
            w=[1 - w, w],
            mu=[mu1, mu2],
            sigma=[sigma1, sigma2],
            observed=y_data,
        )

        # Sample
        trace = pm.sample(
            2000,
            tune=1000,
            chains=2,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True,
            target_accept=0.9,
        )

    return model, trace


if __name__ == "__main__":
    print("=" * 70)
    print("Bayesian Inference with PyMC")
    print("=" * 70)

    print("\n1. Simple Bayesian Inference")
    print("-" * 70)
    model_simple, trace_simple = simple_bayesian_inference()
    summary_simple = az.summary(trace_simple)
    print(summary_simple)

    print("\n2. Hierarchical Model (Schools)")
    print("-" * 70)
    model_hier, trace_hier = hierarchical_bayesian_model()
    summary_hier = az.summary(trace_hier)
    print(summary_hier)

    print("\n3. Bayesian Linear Regression")
    print("-" * 70)
    model_reg, trace_reg = regression_with_priors()
    summary_reg = az.summary(trace_reg)
    print(summary_reg)

    print("\n4. Model Comparison (LOO and WAIC)")
    print("-" * 70)
    comparison = model_comparison()
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")

    print("\n5. Posterior Predictive Check")
    print("-" * 70)
    ppc_stats = posterior_predictive_check(model_simple, trace_simple)
    for key, value in ppc_stats.items():
        print(f"  {key}: {value:.4f}")

    print("\n6. Mixture Model")
    print("-" * 70)
    model_mix, trace_mix = mixture_model_example()
    summary_mix = az.summary(trace_mix)
    print(summary_mix)

    print("\n" + "=" * 70)
    print("Inference complete. All models sampled successfully.")
    print("=" * 70)
