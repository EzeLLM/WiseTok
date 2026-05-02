"""
Advanced Python: metaclasses, descriptors, decorators.
Auto-registering subclasses, validating descriptors, dynamic injection.
"""
from typing import Type, Any, Dict, List, Callable, Optional
from functools import wraps
import inspect
import re


# ============================================================================
# Descriptors with validation
# ============================================================================

class ValidatedDescriptor:
  """Base descriptor with validation support."""

  def __init__(self, name: str, validator: Optional[Callable[[Any], bool]] = None):
    self.name = name
    self.validator = validator
    self.private_name = f'_{name}'

  def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
    if obj is None:
      return self
    return getattr(obj, self.private_name, None)

  def __set__(self, obj: Any, value: Any) -> None:
    if self.validator and not self.validator(value):
      raise ValueError(f'Validation failed for {self.name}: {value}')
    setattr(obj, self.private_name, value)

  def __delete__(self, obj: Any) -> None:
    delattr(obj, self.private_name)


class EmailDescriptor(ValidatedDescriptor):
  """Descriptor that validates email addresses."""

  def __init__(self, name: str = 'email'):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    validator = lambda x: isinstance(x, str) and re.match(pattern, x)
    super().__init__(name, validator)


class IntRangeDescriptor(ValidatedDescriptor):
  """Descriptor enforcing integer range."""

  def __init__(self, name: str, min_val: int = 0, max_val: int = 100):
    self.name = name
    self.min_val = min_val
    self.max_val = max_val
    self.private_name = f'_{name}'
    validator = lambda x: isinstance(x, int) and self.min_val <= x <= self.max_val
    self.validator = validator


class TypedDescriptor(ValidatedDescriptor):
  """Descriptor enforcing type checking."""

  def __init__(self, name: str, expected_type: Type):
    self.name = name
    self.expected_type = expected_type
    self.private_name = f'_{name}'
    validator = lambda x: isinstance(x, expected_type)
    self.validator = validator


# ============================================================================
# Decorators
# ============================================================================

def logged(func: Callable) -> Callable:
  """Decorator that logs function calls."""
  @wraps(func)
  def wrapper(*args, **kwargs):
    print(f'[LOG] Calling {func.__name__} with args={args}, kwargs={kwargs}')
    result = func(*args, **kwargs)
    print(f'[LOG] {func.__name__} returned {result}')
    return result
  return wrapper


def validate_args(**type_checks: Type) -> Callable:
  """Decorator for argument type validation."""
  def decorator(func: Callable) -> Callable:
    sig = inspect.signature(func)
    @wraps(func)
    def wrapper(*args, **kwargs):
      bound = sig.bind(*args, **kwargs)
      bound.apply_defaults()
      for param_name, expected_type in type_checks.items():
        if param_name in bound.arguments:
          value = bound.arguments[param_name]
          if not isinstance(value, expected_type):
            raise TypeError(
              f'{param_name} must be {expected_type.__name__}, got {type(value).__name__}'
            )
      return func(*args, **kwargs)
    return wrapper
  return decorator


def memoize(func: Callable) -> Callable:
  """Decorator for function result caching."""
  cache: Dict[Any, Any] = {}
  @wraps(func)
  def wrapper(*args):
    if args not in cache:
      cache[args] = func(*args)
    return cache[args]
  return wrapper


# ============================================================================
# Metaclasses
# ============================================================================

class RegistryMeta(type):
  """Metaclass that auto-registers subclasses."""

  _registry: Dict[str, Type] = {}

  def __new__(mcs, name: str, bases: tuple, namespace: dict):
    cls = super().__new__(mcs, name, bases, namespace)

    # Don't register the base class itself
    if bases and any(isinstance(b, RegistryMeta) for b in bases):
      registry_key = namespace.get('registry_key', name)
      mcs._registry[registry_key] = cls
      print(f'[Registry] Registered {registry_key} -> {name}')

    return cls

  @classmethod
  def get_subclass(mcs, key: str) -> Optional[Type]:
    """Retrieve registered subclass by key."""
    return mcs._registry.get(key)

  @classmethod
  def list_subclasses(mcs) -> List[str]:
    """List all registered subclass keys."""
    return list(mcs._registry.keys())


class ValidatorMeta(type):
  """Metaclass that auto-validates attributes."""

  def __new__(mcs, name: str, bases: tuple, namespace: dict):
    # Extract validators from docstring
    validators = {}
    for attr_name, attr_value in namespace.items():
      if isinstance(attr_value, ValidatedDescriptor):
        validators[attr_name] = attr_value

    # Store validators
    namespace['_validators'] = validators

    cls = super().__new__(mcs, name, bases, namespace)
    return cls

  def __call__(cls, *args, **kwargs):
    """Intercept instantiation to validate."""
    instance = super(ValidatorMeta, cls).__call__(*args, **kwargs)
    return instance


# ============================================================================
# __init_subclass__ approach (Python 3.6+)
# ============================================================================

class Plugin:
  """Base plugin class using __init_subclass__ for registration."""

  _plugins: Dict[str, Type['Plugin']] = {}

  def __init_subclass__(cls, plugin_type: str = None, **kwargs):
    """Called when a subclass is created."""
    super().__init_subclass__(**kwargs)

    if plugin_type:
      Plugin._plugins[plugin_type] = cls
      print(f'[Plugin] Registered {plugin_type} -> {cls.__name__}')

  @classmethod
  def get_plugin(cls, plugin_type: str) -> Optional[Type['Plugin']]:
    """Get plugin by type."""
    return cls._plugins.get(plugin_type)

  @classmethod
  def list_plugins(cls) -> List[str]:
    """List available plugin types."""
    return list(cls._plugins.keys())

  def execute(self):
    """Override in subclass."""
    raise NotImplementedError


class JSONPlugin(Plugin, plugin_type='json'):
  """Example JSON plugin."""
  def execute(self):
    return {'type': 'json', 'status': 'ok'}


class CSVPlugin(Plugin, plugin_type='csv'):
  """Example CSV plugin."""
  def execute(self):
    return {'type': 'csv', 'status': 'ok'}


class XMLPlugin(Plugin, plugin_type='xml'):
  """Example XML plugin."""
  def execute(self):
    return {'type': 'xml', 'status': 'ok'}


# ============================================================================
# Dynamic attribute injection
# ============================================================================

def inject_methods(methods: Dict[str, Callable]) -> Callable:
  """Decorator that dynamically injects methods into a class."""
  def decorator(cls: Type) -> Type:
    for method_name, method in methods.items():
      setattr(cls, method_name, method)
    return cls
  return decorator


def inject_properties(**properties: Dict[str, Any]) -> Callable:
  """Decorator that dynamically injects properties."""
  def decorator(cls: Type) -> Type:
    for prop_name, prop_value in properties.items():
      setattr(cls, prop_name, property(lambda self, v=prop_value: v))
    return cls
  return decorator


# ============================================================================
# Usage examples
# ============================================================================

class Animal(metaclass=RegistryMeta):
  """Base animal class with auto-registration."""
  registry_key = 'animal'


class Dog(Animal):
  registry_key = 'dog'
  def speak(self):
    return 'Woof!'


class Cat(Animal):
  registry_key = 'cat'
  def speak(self):
    return 'Meow!'


class Bird(Animal):
  registry_key = 'bird'
  def speak(self):
    return 'Tweet!'


class Person(metaclass=ValidatorMeta):
  """Person with validated attributes."""
  email = EmailDescriptor('email')
  age = IntRangeDescriptor('age', min_val=0, max_val=150)
  name = TypedDescriptor('name', str)

  def __init__(self, name: str, email: str, age: int):
    self.name = name
    self.email = email
    self.age = age

  def __repr__(self) -> str:
    return f'Person(name={self._name}, email={self._email}, age={self._age})'


# ============================================================================
# Builder pattern with decorators
# ============================================================================

class ConfigBuilder:
  """Configuration builder with logged methods."""

  def __init__(self):
    self.config: Dict[str, Any] = {}

  @logged
  def set_debug(self, enabled: bool) -> 'ConfigBuilder':
    self.config['debug'] = enabled
    return self

  @logged
  def set_timeout(self, seconds: int) -> 'ConfigBuilder':
    self.config['timeout'] = seconds
    return self

  @logged
  def set_host(self, host: str) -> 'ConfigBuilder':
    self.config['host'] = host
    return self

  @logged
  def build(self) -> Dict[str, Any]:
    return self.config.copy()


# ============================================================================
# Dynamic class factory
# ============================================================================

def create_model(name: str, fields: Dict[str, Type]) -> Type:
  """Dynamically create a data model class."""
  attrs = {}
  for field_name, field_type in fields.items():
    attrs[field_name] = TypedDescriptor(field_name, field_type)

  def __init__(self, **kwargs):
    for field_name in fields:
      setattr(self, field_name, kwargs.get(field_name))

  attrs['__init__'] = __init__

  return type(name, (), attrs)


# ============================================================================
# Demo
# ============================================================================

def main():
  print('=== Metaclass & Descriptor Demo ===\n')

  # 1. Auto-registration via metaclass
  print('1. Registry Metaclass:')
  for key in Animal._registry:
    print(f'   - {key}')
  dog_cls = RegistryMeta.get_subclass('dog')
  if dog_cls:
    dog = dog_cls()
    print(f'   Dog says: {dog.speak()}\n')

  # 2. Validators
  print('2. Validated Descriptors:')
  try:
    person = Person('Alice', 'alice@example.com', 30)
    print(f'   ✓ {person}')
  except ValueError as e:
    print(f'   ✗ {e}')

  try:
    bad_person = Person('Bob', 'not-an-email', 25)
  except ValueError as e:
    print(f'   ✗ {e}\n')

  # 3. Plugins via __init_subclass__
  print('3. Plugins via __init_subclass__:')
  print(f'   Available: {Plugin.list_plugins()}')
  json_plugin = Plugin.get_plugin('json')()
  print(f'   JSON Plugin: {json_plugin.execute()}\n')

  # 4. Builder pattern with logging
  print('4. Builder Pattern with @logged:')
  config = ConfigBuilder().set_host('localhost').set_debug(True).set_timeout(30).build()
  print(f'   Config: {config}\n')

  # 5. Dynamic class creation
  print('5. Dynamic Model Creation:')
  User = create_model('User', {'name': str, 'age': int, 'email': str})
  user = User(name='Charlie', age=28, email='charlie@example.com')
  print(f'   Created dynamic User: {user.name}, age={user.age}\n')

  # 6. Memoization
  print('6. Memoized Function:')
  @memoize
  def fibonacci(n: int) -> int:
    if n < 2:
      return n
    return fibonacci(n - 1) + fibonacci(n - 2)

  print(f'   fib(10) = {fibonacci(10)}')


if __name__ == '__main__':
  main()
