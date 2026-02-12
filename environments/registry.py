"""Environment registry — maps names to environment constructors."""

from typing import Any, Callable, Dict

from core.environment import EnvWrapper


class EnvRegistry:
    """Maps string names to environment wrapper factories."""

    _registry: Dict[str, Callable[..., EnvWrapper]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., EnvWrapper]) -> None:
        cls._registry[name.lower()] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> EnvWrapper:
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Unknown environment '{name}'. Available: {available}")
        return cls._registry[key](**kwargs)

    @classmethod
    def list(cls):
        return sorted(cls._registry.keys())


# --- Auto-register built-in environments ---

def _register_defaults():
    from environments.frozen_lake import create_frozen_lake
    from environments.cart_pole import create_cart_pole
    from environments.mountain_car import create_mountain_car
    from environments.pendulum import create_pendulum

    EnvRegistry.register("frozenlake", create_frozen_lake)
    EnvRegistry.register("cartpole", create_cart_pole)
    EnvRegistry.register("mountaincar", create_mountain_car)
    EnvRegistry.register("pendulum", create_pendulum)


_register_defaults()
