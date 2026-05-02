"""
SymPy Symbolic Mathematics

Solve equations, integrate, differentiate, compute Taylor series,
symbolic matrix operations, and LaTeX output.
"""

from sympy import (
    symbols, Function, Eq, solve, diff, integrate, limit,
    Matrix, simplify, expand, factor, apart, series,
    sin, cos, exp, log, sqrt, pi, I, oo, summation,
    Symbol, latex, atan2, conjugate, re, im,
)
from sympy.solvers.ode import dsolve
from typing import Dict, Any, List
import sympy as sp


def algebraic_equation_solving() -> Dict[str, Any]:
    """
    Solve system of algebraic equations:
    x^2 + y^2 = 4
    x + y = 1
    """
    x, y = symbols("x y", real=True)

    eq1 = Eq(x**2 + y**2, 4)
    eq2 = Eq(x + y, 1)

    solutions = solve([eq1, eq2], [x, y])

    return {
        "equations": [str(eq1), str(eq2)],
        "solutions": [{"x": sol[0], "y": sol[1]} for sol in solutions],
    }


def symbolic_differentiation() -> Dict[str, Any]:
    """Compute derivatives symbolically."""
    x = Symbol("x")

    # Function: f(x) = x^3 * sin(x) + exp(x)
    f = x**3 * sin(x) + exp(x)

    df_dx = diff(f, x)  # First derivative
    d2f_dx2 = diff(f, x, 2)  # Second derivative
    d3f_dx3 = diff(f, x, 3)  # Third derivative

    # Evaluate at a point
    f_at_0 = f.subs(x, 0)
    df_at_1 = df_dx.subs(x, 1)

    return {
        "function": str(f),
        "first_derivative": str(simplify(df_dx)),
        "second_derivative": str(simplify(d2f_dx2)),
        "third_derivative": str(simplify(d3f_dx3)),
        "f(0)": f_at_0,
        "f'(1)": float(df_at_1),
    }


def symbolic_integration() -> Dict[str, Any]:
    """Compute indefinite and definite integrals."""
    x = Symbol("x")

    # Indefinite integrals
    f1 = x**3 + sin(x)
    integral1 = integrate(f1, x)

    f2 = exp(x) * cos(x)
    integral2 = integrate(f2, x)

    # Definite integrals
    f3 = sin(x) / x
    # Note: integral(sin(x)/x, (x, 0, oo)) = pi/2 (known result)

    f4 = exp(-x**2)
    definite_gaussian = integrate(f4, (x, -oo, oo))

    return {
        "indefinite_1": str(integral1),
        "indefinite_2": str(simplify(integral2)),
        "definite_gaussian": definite_gaussian,
        "definite_gaussian_value": float(definite_gaussian),
    }


def taylor_series_expansion() -> Dict[str, Any]:
    """Compute Taylor series expansions."""
    x = Symbol("x")
    x0 = 0  # Expand around x=0

    # sin(x) ≈ x - x^3/6 + x^5/120 - ...
    sin_expansion = series(sin(x), x, x0, n=6)

    # exp(x) ≈ 1 + x + x^2/2 + x^3/6 + ...
    exp_expansion = series(exp(x), x, x0, n=5)

    # 1/(1-x) ≈ 1 + x + x^2 + x^3 + ...
    geometric_expansion = series(1 / (1 - x), x, x0, n=5)

    # log(1+x) ≈ x - x^2/2 + x^3/3 - ...
    log_expansion = series(log(1 + x), x, x0, n=5)

    return {
        "sin_taylor": str(sin_expansion),
        "exp_taylor": str(exp_expansion),
        "geometric_series": str(geometric_expansion),
        "log1p_taylor": str(log_expansion),
    }


def limit_computation() -> Dict[str, Any]:
    """Compute limits of functions."""
    x = Symbol("x")

    # lim_{x→0} sin(x)/x = 1
    limit_sinc = limit(sin(x) / x, x, 0)

    # lim_{x→∞} (1 + 1/x)^x = e
    limit_e = limit((1 + 1/x)**x, x, oo)

    # lim_{x→1} (x^3 - 1)/(x^2 - 1)
    limit_rational = limit((x**3 - 1) / (x**2 - 1), x, 1)

    return {
        "sin(x)/x as x→0": limit_sinc,
        "(1+1/x)^x as x→∞": limit_e,
        "(x³-1)/(x²-1) as x→1": limit_rational,
    }


def symbolic_matrix_operations() -> Dict[str, Any]:
    """Matrix algebra with symbolic elements."""
    a, b, c, d, x = symbols("a b c d x")

    # Define 2x2 matrix with symbolic entries
    M = Matrix([
        [a, b],
        [c, d],
    ])

    # Determinant: det(M) = ad - bc
    det_M = M.det()

    # Inverse
    M_inv = M.inv()

    # Eigenvalues and eigenvectors
    eigenvals = M.eigenvals()
    eigenvects = M.eigenvects()

    # Characteristic polynomial
    char_poly = M.charpoly(x)

    return {
        "matrix": str(M),
        "determinant": str(det_M),
        "inverse": str(M_inv),
        "characteristic_poly": str(char_poly),
        "eigenvalues": {str(ev): mult for ev, mult in eigenvals.items()},
    }


def differential_equation_solving() -> Dict[str, Any]:
    """Solve ordinary differential equations."""
    x = Symbol("x")
    y = Function("y")

    # ODE 1: dy/dx - 2y = 0 => y = C*exp(2x)
    ode1 = Eq(diff(y(x), x) - 2*y(x), 0)
    sol1 = dsolve(ode1, y(x))

    # ODE 2: y'' + 2y' + y = 0 (damped harmonic)
    # y = (C1 + C2*x)*exp(-x)
    ode2 = Eq(diff(y(x), x, 2) + 2*diff(y(x), x) + y(x), 0)
    sol2 = dsolve(ode2, y(x))

    return {
        "ode1": str(ode1),
        "solution1": str(sol1),
        "ode2": str(ode2),
        "solution2": str(sol2),
    }


def complex_analysis() -> Dict[str, Any]:
    """Complex number operations and functions."""
    z = Symbol("z", complex=True)
    re_part = Symbol("x", real=True)
    im_part = Symbol("y", real=True)

    # z = x + iy
    z_complex = re_part + I * im_part

    # |z| = sqrt(x^2 + y^2)
    z_magnitude = sqrt(re_part**2 + im_part**2)

    # arg(z) = atan2(y, x)
    z_argument = atan2(im_part, re_part)

    # Conjugate: z* = x - iy
    z_conj = conjugate(z_complex)

    # Complex function: f(z) = z^2 + i*z
    f_z = z**2 + I*z
    f_deriv = diff(f_z, z)

    return {
        "z": str(z_complex),
        "|z|": str(z_magnitude),
        "arg(z)": str(z_argument),
        "z*": str(z_conj),
        "f(z) = z² + iz": str(f_z),
        "df/dz": str(f_deriv),
    }


def polynomial_manipulation() -> Dict[str, Any]:
    """Factorization, expansion, and partial fractions."""
    x = Symbol("x")

    # Expand: (x+1)^3
    expanded = expand((x + 1)**3)

    # Factor: x^2 - 1
    factored = factor(x**2 - 1)

    # Partial fraction decomposition
    rational = (2*x + 3) / ((x + 1)*(x - 2))
    partial_fracs = apart(rational, x)

    return {
        "(x+1)³ expanded": str(expanded),
        "x²-1 factored": str(factored),
        "partial_fractions": str(partial_fracs),
    }


def summation_example() -> Dict[str, Any]:
    """Compute symbolic summations."""
    n = Symbol("n", integer=True, positive=True)
    k = Symbol("k", integer=True, positive=True)

    # Sum of first n natural numbers: Σ k = n(n+1)/2
    sum_natural = summation(k, (k, 1, n))

    # Sum of squares: Σ k^2 = n(n+1)(2n+1)/6
    sum_squares = summation(k**2, (k, 1, n))

    # Geometric series: Σ r^k = r(1-r^n)/(1-r)
    r = Symbol("r")
    sum_geometric = summation(r**k, (k, 0, n - 1))

    return {
        "sum_natural": str(sum_natural),
        "sum_squares": str(sum_squares),
        "sum_geometric": str(sum_geometric),
    }


def latex_output_example() -> str:
    """Generate LaTeX representation."""
    x = Symbol("x")

    expr = (x**3 + 2*x**2 + 1) / (x**2 - 1)
    latex_str = latex(expr)

    return latex_str


if __name__ == "__main__":
    print("=" * 70)
    print("SymPy Symbolic Mathematics Examples")
    print("=" * 70)

    print("\n1. Algebraic Equations")
    print("-" * 70)
    result = algebraic_equation_solving()
    for eq in result["equations"]:
        print(f"  {eq}")
    for i, sol in enumerate(result["solutions"]):
        print(f"  Solution {i+1}: x={sol['x']}, y={sol['y']}")

    print("\n2. Differentiation")
    print("-" * 70)
    result = symbolic_differentiation()
    print(f"  f(x) = {result['function']}")
    print(f"  f'(x) = {result['first_derivative']}")
    print(f"  f(0) = {result['f(0)']}")

    print("\n3. Integration")
    print("-" * 70)
    result = symbolic_integration()
    print(f"  ∫ exp(-x²) dx from -∞ to ∞ = {result['definite_gaussian']}")

    print("\n4. Taylor Series")
    print("-" * 70)
    result = taylor_series_expansion()
    print(f"  sin(x) ≈ {result['sin_taylor']}")
    print(f"  exp(x) ≈ {result['exp_taylor']}")

    print("\n5. Limits")
    print("-" * 70)
    result = limit_computation()
    for desc, val in result.items():
        print(f"  {desc}: {val}")

    print("\n6. Matrix Operations")
    print("-" * 70)
    result = symbolic_matrix_operations()
    print(f"  det(M) = {result['determinant']}")

    print("\n7. Differential Equations")
    print("-" * 70)
    result = differential_equation_solving()
    print(f"  {result['ode1']}")
    print(f"  → {result['solution1']}")

    print("\n8. Complex Analysis")
    print("-" * 70)
    result = complex_analysis()
    print(f"  |z| = {result['|z|']}")
    print(f"  arg(z) = {result['arg(z)']}")

    print("\n9. Polynomial Manipulation")
    print("-" * 70)
    result = polynomial_manipulation()
    print(f"  (x+1)³ = {result['(x+1)³ expanded']}")
    print(f"  x²-1 = {result['x²-1 factored']}")

    print("\n10. Summation")
    print("-" * 70)
    result = summation_example()
    print(f"  Σ k (k=1 to n) = {result['sum_natural']}")
    print(f"  Σ k² (k=1 to n) = {result['sum_squares']}")

    print("\n11. LaTeX Output")
    print("-" * 70)
    latex_expr = latex_output_example()
    print(f"  LaTeX: {latex_expr}")
