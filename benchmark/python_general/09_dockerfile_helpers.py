"""
Build automation helpers for Docker: subprocess, pathlib, env vars,
Docker SDK client interaction, shell wrappers.
"""
import subprocess
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
	"""Configuration for Docker builds."""
	dockerfile_path: Path
	context: Path
	image_name: str
	image_tag: str = "latest"
	build_args: Dict[str, str] = None
	environment: Dict[str, str] = None

	def __post_init__(self):
		if self.build_args is None:
			self.build_args = {}
		if self.environment is None:
			self.environment = {}

	@property
	def full_image_name(self) -> str:
		"""Return fully qualified image name."""
		return f"{self.image_name}:{self.image_tag}"


class ShellExecutor:
	"""Wrapper for subprocess execution with logging."""

	def __init__(self, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None):
		self.cwd = cwd or Path.cwd()
		self.env = env or os.environ.copy()

	def run(
		self,
		cmd: List[str],
		check: bool = True,
		capture_output: bool = False,
		shell: bool = False,
	) -> subprocess.CompletedProcess:
		"""Execute command and handle errors."""
		cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd

		logger.info(f"Running: {cmd_str}")

		try:
			result = subprocess.run(
				cmd,
				cwd=self.cwd,
				env=self.env,
				check=check,
				capture_output=capture_output,
				text=True,
				shell=shell,
			)
			if result.returncode == 0:
				logger.info(f"✓ Success: {cmd_str}")
			return result
		except subprocess.CalledProcessError as e:
			logger.error(f"✗ Failed: {cmd_str}")
			if e.stderr:
				logger.error(f"Error output:\n{e.stderr}")
			raise

	def run_with_output(self, cmd: List[str]) -> str:
		"""Run command and return stdout."""
		result = self.run(cmd, capture_output=True)
		return result.stdout.strip()


class DockerBuilder:
	"""Build Docker images with configuration and caching."""

	def __init__(self):
		self.executor = ShellExecutor()
		self.docker_client = None
		self._import_docker_client()

	def _import_docker_client(self):
		"""Lazy import Docker client."""
		try:
			import docker
			self.docker_client = docker.from_env()
			logger.info("Docker client initialized")
		except ImportError:
			logger.warning("docker package not installed; using CLI only")
		except Exception as e:
			logger.warning(f"Failed to initialize Docker client: {e}")

	def build_image(self, config: BuildConfig) -> bool:
		"""Build Docker image using CLI."""
		if not config.dockerfile_path.exists():
			logger.error(f"Dockerfile not found: {config.dockerfile_path}")
			return False

		if not config.context.exists():
			logger.error(f"Build context not found: {config.context}")
			return False

		cmd = [
			"docker",
			"build",
			"-f", str(config.dockerfile_path),
			"-t", config.full_image_name,
		]

		# Add build arguments
		for key, value in config.build_args.items():
			cmd.extend(["--build-arg", f"{key}={value}"])

		# Add build context
		cmd.append(str(config.context))

		try:
			self.executor.run(cmd)
			logger.info(f"Built image: {config.full_image_name}")
			return True
		except subprocess.CalledProcessError:
			return False

	def build_with_cache(self, config: BuildConfig, cache_dir: Path) -> bool:
		"""Build with BuildKit cache directory support."""
		cache_dir.mkdir(parents=True, exist_ok=True)

		env = os.environ.copy()
		env["DOCKER_BUILDKIT"] = "1"

		executor = ShellExecutor(env=env)

		cmd = [
			"docker",
			"build",
			"-f", str(config.dockerfile_path),
			"-t", config.full_image_name,
			f"--cache-from=type=local,src={cache_dir}",
			f"--cache-to=type=local,dest={cache_dir}",
		]

		for key, value in config.build_args.items():
			cmd.extend(["--build-arg", f"{key}={value}"])

		cmd.append(str(config.context))

		try:
			executor.run(cmd)
			logger.info(f"Built with cache: {config.full_image_name}")
			return True
		except subprocess.CalledProcessError:
			return False

	def push_image(self, config: BuildConfig, registry: str) -> bool:
		"""Tag and push image to registry."""
		full_name = config.full_image_name
		registry_name = f"{registry}/{config.image_name}:{config.image_tag}"

		try:
			self.executor.run(["docker", "tag", full_name, registry_name])
			self.executor.run(["docker", "push", registry_name])
			logger.info(f"Pushed: {registry_name}")
			return True
		except subprocess.CalledProcessError:
			return False

	def inspect_image(self, image_name: str) -> Optional[Dict]:
		"""Inspect image metadata using Python SDK if available."""
		if not self.docker_client:
			logger.warning("Docker SDK not available; using CLI")
			return self._inspect_image_cli(image_name)

		try:
			image = self.docker_client.images.get(image_name)
			return {
				"id": image.id,
				"size": image.attrs.get("Size"),
				"created": image.attrs.get("Created"),
				"tags": image.tags,
			}
		except Exception as e:
			logger.error(f"Failed to inspect image: {e}")
			return None

	def _inspect_image_cli(self, image_name: str) -> Optional[Dict]:
		"""Inspect image using Docker CLI."""
		try:
			output = self.executor.run_with_output(["docker", "inspect", image_name])
			return json.loads(output)[0]
		except Exception as e:
			logger.error(f"Failed to inspect via CLI: {e}")
			return None

	def list_images(self, filter_name: Optional[str] = None) -> List[str]:
		"""List images with optional filter."""
		if self.docker_client:
			return self._list_images_sdk(filter_name)
		return self._list_images_cli(filter_name)

	def _list_images_sdk(self, filter_name: Optional[str] = None) -> List[str]:
		"""List images using SDK."""
		try:
			images = self.docker_client.images.list()
			names = []
			for img in images:
				for tag in img.tags:
					if not filter_name or filter_name in tag:
						names.append(tag)
			return sorted(names)
		except Exception as e:
			logger.error(f"Failed to list images: {e}")
			return []

	def _list_images_cli(self, filter_name: Optional[str] = None) -> List[str]:
		"""List images using CLI."""
		try:
			output = self.executor.run_with_output(
				["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]
			)
			images = output.split("\n")
			if filter_name:
				images = [img for img in images if filter_name in img]
			return sorted(images)
		except Exception as e:
			logger.error(f"Failed to list images: {e}")
			return []


class DockerComposeLauncher:
	"""Helper for docker-compose operations."""

	def __init__(self, compose_file: Path, project_name: str):
		self.compose_file = compose_file
		self.project_name = project_name
		self.executor = ShellExecutor(cwd=compose_file.parent)

	def up(self, detach: bool = True, build: bool = False) -> bool:
		"""Start services."""
		cmd = ["docker-compose", "-f", str(self.compose_file), "-p", self.project_name, "up"]
		if detach:
			cmd.append("-d")
		if build:
			cmd.append("--build")

		try:
			self.executor.run(cmd)
			logger.info(f"Started compose project: {self.project_name}")
			return True
		except subprocess.CalledProcessError:
			return False

	def down(self, volumes: bool = False) -> bool:
		"""Stop and remove services."""
		cmd = ["docker-compose", "-f", str(self.compose_file), "-p", self.project_name, "down"]
		if volumes:
			cmd.append("-v")

		try:
			self.executor.run(cmd)
			logger.info(f"Stopped compose project: {self.project_name}")
			return True
		except subprocess.CalledProcessError:
			return False

	def logs(self, service: Optional[str] = None) -> str:
		"""Get logs from services."""
		cmd = ["docker-compose", "-f", str(self.compose_file), "-p", self.project_name, "logs"]
		if service:
			cmd.append(service)

		try:
			return self.executor.run_with_output(cmd)
		except subprocess.CalledProcessError:
			return ""


def build_and_push_workflow(
	dockerfile: Path,
	context: Path,
	image_name: str,
	tag: str = "latest",
	registry: str = "ghcr.io/myorg",
	build_args: Optional[Dict[str, str]] = None,
) -> bool:
	"""Complete workflow: build, inspect, and push."""
	config = BuildConfig(
		dockerfile_path=dockerfile,
		context=context,
		image_name=image_name,
		image_tag=tag,
		build_args=build_args or {},
	)

	builder = DockerBuilder()

	# Build
	if not builder.build_image(config):
		logger.error("Build failed")
		return False

	# Inspect
	inspect_result = builder.inspect_image(config.full_image_name)
	if inspect_result:
		logger.info(f"Image size: {inspect_result.get('Size', 'unknown')} bytes")

	# Push
	if not builder.push_image(config, registry):
		logger.error("Push failed")
		return False

	logger.info("Workflow completed successfully")
	return True


def main():
	"""Example usage."""
	# Build configuration
	config = BuildConfig(
		dockerfile_path=Path("Dockerfile"),
		context=Path("."),
		image_name="myapp",
		image_tag="1.0.0",
		build_args={"PYTHON_VERSION": "3.11"},
	)

	builder = DockerBuilder()

	# List existing images
	images = builder.list_images(filter_name="myapp")
	logger.info(f"Found {len(images)} matching images")

	# Build
	if builder.build_image(config):
		logger.info("✓ Build successful")

		# Inspect
		metadata = builder.inspect_image(config.full_image_name)
		if metadata:
			logger.info(f"✓ Image metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
	main()
