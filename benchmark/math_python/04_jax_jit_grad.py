"""
JAX Examples: JIT Compilation, Automatic Differentiation, Vectorization

Demonstrates jit(), grad(), vmap(), pmap(), pytrees, and differentiable
physics simulation or simple neural network training.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap
from jax.tree_util import tree_map
from functools import partial
from typing import Tuple, Dict, Any, NamedTuple
import time


class MLPParams(NamedTuple):
	"""Pytree of neural network parameters."""
	w1: jnp.ndarray
	b1: jnp.ndarray
	w2: jnp.ndarray
	b2: jnp.ndarray


def init_mlp_params(key, input_size: int, hidden_size: int, output_size: int) -> MLPParams:
	"""Initialize MLP parameters with Xavier initialization."""
	key1, key2 = jax.random.split(key)

	scale1 = jnp.sqrt(1.0 / input_size)
	w1 = jax.random.normal(key1, (input_size, hidden_size)) * scale1

	scale2 = jnp.sqrt(1.0 / hidden_size)
	w2 = jax.random.normal(key2, (hidden_size, output_size)) * scale2

	return MLPParams(
		w1=w1,
		b1=jnp.zeros(hidden_size),
		w2=w2,
		b2=jnp.zeros(output_size),
	)


def mlp_forward(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
	"""Forward pass: x -> relu(xW1 + b1) -> xW2 + b2"""
	h = jnp.dot(x, params.w1) + params.b1
	h = jnp.maximum(h, 0.0)  # ReLU
	y = jnp.dot(h, params.w2) + params.b2
	return y


def squared_error_loss(params: MLPParams, x: jnp.ndarray, y_target: jnp.ndarray) -> jnp.ndarray:
	"""MSE loss: (1/n) * sum((y_pred - y_target)^2)"""
	y_pred = mlp_forward(params, x)
	return jnp.mean((y_pred - y_target)**2)


@jit
def loss_and_grads(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, MLPParams]:
	"""Compute loss and gradients (JIT-compiled)."""
	loss_value = squared_error_loss(params, x, y)
	grads_value = grad(squared_error_loss)(params, x, y)
	return loss_value, grads_value


@jit
def update_params(params: MLPParams, grads: MLPParams, learning_rate: float = 1e-3) -> MLPParams:
	"""Parameter update step: params -= lr * grads (JIT-compiled)."""
	return tree_map(
		lambda p, g: p - learning_rate * g,
		params,
		grads,
	)


def simple_neural_net_training() -> Dict[str, Any]:
	"""Train a simple MLP on synthetic data."""
	key = jax.random.PRNGKey(42)
	key, subkey = jax.random.split(key)

	# Synthetic data: y = sin(x) + noise
	n_samples = 100
	x_train = jax.random.uniform(subkey, (n_samples, 4)) * 2.0 * jnp.pi
	y_train = jnp.sin(jnp.mean(x_train, axis=1, keepdims=True)) + 0.1 * jax.random.normal(key, (n_samples, 1))

	# Initialize parameters
	params = init_mlp_params(key, input_size=4, hidden_size=32, output_size=1)

	# Training loop
	losses = []
	learning_rate = 1e-2
	num_epochs = 50

	for epoch in range(num_epochs):
		loss_val, grads_val = loss_and_grads(params, x_train, y_train)
		params = update_params(params, grads_val, learning_rate)
		losses.append(float(loss_val))

		if (epoch + 1) % 10 == 0:
			print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.6f}")

	return {
		"final_params": params,
		"losses": losses,
		"final_loss": losses[-1],
	}


def batched_predictions(params: MLPParams, x_batch: jnp.ndarray) -> jnp.ndarray:
	"""Apply MLP to batch of inputs (vmapped)."""
	# vmap over the batch dimension (axis 0)
	return vmap(lambda x: mlp_forward(params, x))(x_batch)


def jacobian_example(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
	"""Compute Jacobian matrix: ∂y/∂x for all elements."""
	jacobian_fn = jax.jacobian(mlp_forward, argnums=1)
	return jacobian_fn(params, x)


def hessian_example(params: MLPParams, x: jnp.ndarray, y_target: jnp.ndarray) -> jnp.ndarray:
	"""Compute Hessian matrix: ∂²L/∂params² (expensive for large models)."""
	hessian_fn = jax.hessian(squared_error_loss, argnums=0)
	return hessian_fn(params, x, y_target)


class SimplePhysics(NamedTuple):
	"""State of 2D particle: position, velocity."""
	x: jnp.ndarray  # [2]
	v: jnp.ndarray  # [2]


def physics_step(state: SimplePhysics, dt: float = 0.01, g: float = 9.81) -> SimplePhysics:
	"""Single integration step: x_{n+1} = x_n + v_n*dt, v_{n+1} = v_n + a*dt

	Gravity in y-direction; no friction.
	"""
	acceleration = jnp.array([0.0, -g])
	new_x = state.x + state.v * dt
	new_v = state.v + acceleration * dt
	return SimplePhysics(x=new_x, v=new_v)


@jit
def simulate_trajectory(state: SimplePhysics, num_steps: int = 1000) -> jnp.ndarray:
	"""Simulate particle trajectory and return positions over time."""
	trajectory = [state.x]
	for _ in range(num_steps):
		state = physics_step(state)
		trajectory.append(state.x)
	return jnp.stack(trajectory)


def differentiable_physics_example() -> Dict[str, Any]:
	"""Trajectory optimization using differentiable physics."""
	# Initial state: particle at origin with initial velocity
	initial_state = SimplePhysics(
		x=jnp.array([0.0, 0.0]),
		v=jnp.array([5.0, 20.0]),  # v_x, v_y (upward throw)
	)

	# Simulate trajectory
	trajectory = simulate_trajectory(initial_state, num_steps=200)

	# Compute max height
	max_height = jnp.max(trajectory[:, 1])

	# Compute range (x when particle returns to y=0)
	final_x = trajectory[-1, 0]

	return {
		"trajectory": trajectory,
		"max_height": float(max_height),
		"final_x": float(final_x),
		"num_steps": len(trajectory),
	}


def grad_through_loop() -> jnp.ndarray:
	"""Demonstrate grad() through lax.fori_loop."""
	def loss_fn(params):
		x = params
		for _ in range(5):
			x = jnp.sin(x) * 0.9
		return jnp.sum(x**2)

	grad_fn = grad(loss_fn)
	params = jnp.array([1.0, 2.0, 3.0])
	return grad_fn(params)


if __name__ == "__main__":
	print("JAX Examples: JIT, Grad, Vmap, Pytrees\n")

	print("=" * 60)
	print("1. Simple Neural Network Training (with JIT)")
	print("=" * 60)
	training_result = simple_neural_net_training()
	print(f"Final loss: {training_result['final_loss']:.6f}\n")

	print("=" * 60)
	print("2. Batched Predictions (vmap)")
	print("=" * 60)
	key = jax.random.PRNGKey(0)
	params = init_mlp_params(key, input_size=4, hidden_size=16, output_size=1)
	x_test_batch = jax.random.normal(key, (10, 4))
	predictions = batched_predictions(params, x_test_batch)
	print(f"Batch input shape: {x_test_batch.shape}")
	print(f"Batch predictions shape: {predictions.shape}\n")

	print("=" * 60)
	print("3. Jacobian (Automatic Differentiation)")
	print("=" * 60)
	x_single = jnp.ones(4)
	jacobian = jacobian_example(params, x_single)
	print(f"Jacobian shape (∂y/∂x): {jacobian.shape}")
	print(f"Jacobian (first 2x4): {jacobian[0, :2, :]}\n")

	print("=" * 60)
	print("4. Differentiable Physics Simulation")
	print("=" * 60)
	physics_result = differentiable_physics_example()
	print(f"Max height: {physics_result['max_height']:.4f} m")
	print(f"Final range (x): {physics_result['final_x']:.4f} m")
	print(f"Trajectory points: {physics_result['num_steps']}\n")

	print("=" * 60)
	print("5. Gradients through Control Flow")
	print("=" * 60)
	grads = grad_through_loop()
	print(f"Gradients: {grads}")
