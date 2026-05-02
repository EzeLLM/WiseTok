"""
scipy.optimize Examples: Minimization, Curve Fitting, ODE Integration, Root Finding

Demonstrates SLSQP constrained minimization, curve_fit, solve_ivp,
and root-finding algorithms.
"""

import numpy as np
from scipy.optimize import minimize, curve_fit, solve_ivp, brentq, fsolve
import matplotlib.pyplot as plt
from typing import Callable, Tuple


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2

    Global minimum at (1.0, 1.0) with f = 0.
    Classic test function for optimization: steep valley along parabola y = x^2.
    """
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of Rosenbrock.

    ∂f/∂x = -2(1-x) - 400x(y - x^2)
    ∂f/∂y = 200(y - x^2)
    """
    dfdx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
    dfdy = 200.0 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])


def constrained_optimization_example() -> dict:
    """
    Minimize: f(x) = (x-2)^2 + (y-3)^2
    Subject to: x + 2y ≤ 4  (inequality constraint)
               x^2 + y ≥ 0  (inequality constraint)
    """
    def objective(x):
        return (x[0] - 2.0)**2 + (x[1] - 3.0)**2

    constraints = [
        {"type": "ineq", "fun": lambda x: 4.0 - x[0] - 2.0*x[1]},
        {"type": "ineq", "fun": lambda x: x[0]**2 + x[1]},
    ]

    bounds = [(0.0, 5.0), (0.0, 5.0)]

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {
        "success": result.success,
        "x_opt": result.x,
        "f_opt": result.fun,
        "iterations": result.nit,
    }


def damped_sine_model(t: np.ndarray, A: float, f: float, phi: float, tau: float) -> np.ndarray:
    """Damped oscillation model:

    y(t) = A * sin(2πft + φ) * exp(-t/τ)

    Parameters:
    - A: amplitude
    - f: frequency (Hz)
    - phi: phase shift (radians)
    - tau: damping time constant (seconds)
    """
    return A * np.sin(2.0 * np.pi * f * t + phi) * np.exp(-t / tau)


def curve_fitting_example() -> dict:
    """Generate synthetic noisy data and fit damped sine model."""
    # Generate synthetic data
    t_data = np.linspace(0.0, 5.0, 200)
    y_true = damped_sine_model(t_data, A=2.5, f=1.2, phi=0.5, tau=2.0)
    noise = np.random.normal(0.0, 0.15, len(t_data))
    y_data = y_true + noise

    # Fit model to data
    popt, pcov = curve_fit(
        damped_sine_model,
        t_data,
        y_data,
        p0=[2.0, 1.0, 0.0, 1.5],
        maxfev=5000,
    )

    A_fit, f_fit, phi_fit, tau_fit = popt
    perr = np.sqrt(np.diag(pcov))

    return {
        "parameters": {"A": A_fit, "f": f_fit, "phi": phi_fit, "tau": tau_fit},
        "std_errors": {"A": perr[0], "f": perr[1], "phi": perr[2], "tau": perr[3]},
        "covariance": pcov,
        "residuals": y_data - damped_sine_model(t_data, *popt),
    }


def damped_oscillator_ode(t: float, y: np.ndarray, damping: float = 0.3) -> np.ndarray:
    """
    Damped harmonic oscillator ODE system:

    dy0/dt = y1
    dy1/dt = -k*y0 - c*y1  (where k=1, c=damping)

    State vector: y = [position, velocity]
    """
    k = 1.0
    dydt = np.array([
        y[1],
        -k * y[0] - damping * y[1],
    ])
    return dydt


def ode_integration_example() -> dict:
    """Integrate damped oscillator ODE using RK45."""
    # Initial conditions: y0=1.0 (displacement), y1=0.0 (velocity)
    y0 = np.array([1.0, 0.0])
    t_span = (0.0, 20.0)
    t_eval = np.linspace(0.0, 20.0, 500)

    sol = solve_ivp(
        damped_oscillator_ode,
        t_span,
        y0,
        method="RK45",
        t_eval=t_eval,
        dense_output=True,
        args=(0.3,),
    )

    return {
        "success": sol.status == 0,
        "t": sol.t,
        "y": sol.y,
        "method": sol.method,
        "num_steps": len(sol.t),
    }


def nonlinear_equation_solver() -> dict:
    """
    Find roots of transcendental equation:
    x * sin(x) - 0.5 = 0

    Use both brentq (bracketed) and fsolve (Newton-like).
    """
    def equation(x):
        return x * np.sin(x) - 0.5

    # Brentq method: requires bracketing [a, b] where f(a)*f(b) < 0
    x_brentq = brentq(equation, 0.5, 2.0)

    # fsolve: Newton-like method
    x_fsolve = fsolve(equation, x0=1.0)[0]

    return {
        "brentq_root": x_brentq,
        "fsolve_root": x_fsolve,
        "f_at_brentq": equation(x_brentq),
        "f_at_fsolve": equation(x_fsolve),
    }


def unconstrained_optimization_example() -> dict:
    """Optimize Rosenbrock function with different methods."""
    x0 = np.array([-1.2, 1.0])

    methods = {
        "BFGS": {"method": "BFGS"},
        "L-BFGS-B": {"method": "L-BFGS-B"},
        "Nelder-Mead": {"method": "Nelder-Mead"},
    }

    results = {}
    for name, opts in methods.items():
        res = minimize(
            rosenbrock,
            x0,
            jac=rosenbrock_grad if name != "Nelder-Mead" else None,
            **opts,
        )
        results[name] = {
            "x_opt": res.x,
            "f_opt": res.fun,
            "nit": res.nit,
            "nfev": res.nfev,
        }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("1. Constrained Optimization (SLSQP)")
    print("=" * 60)
    const_result = constrained_optimization_example()
    for key, value in const_result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("2. Curve Fitting (Damped Sine)")
    print("=" * 60)
    fit_result = curve_fitting_example()
    params = fit_result["parameters"]
    print(f"Fitted parameters: A={params['A']:.4f}, f={params['f']:.4f}, "
          f"phi={params['phi']:.4f}, tau={params['tau']:.4f}")
    print(f"Fit residual RMS: {np.sqrt(np.mean(fit_result['residuals']**2)):.4f}")

    print("\n" + "=" * 60)
    print("3. ODE Integration (Damped Oscillator)")
    print("=" * 60)
    ode_result = ode_integration_example()
    print(f"Integration success: {ode_result['success']}")
    print(f"Time steps: {ode_result['num_steps']}")
    print(f"Final position: {ode_result['y'][0, -1]:.6f}")
    print(f"Final velocity: {ode_result['y'][1, -1]:.6f}")

    print("\n" + "=" * 60)
    print("4. Root Finding")
    print("=" * 60)
    root_result = nonlinear_equation_solver()
    for key, value in root_result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("5. Unconstrained Optimization (Rosenbrock)")
    print("=" * 60)
    uncon_result = unconstrained_optimization_example()
    for method, res in uncon_result.items():
        print(f"{method}: x={res['x_opt']}, f={res['f_opt']:.2e}, "
              f"iterations={res['nit']}")
