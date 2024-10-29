from typing import Union

from .providers import providers


def find_provider(provider_name: Union[str, None]):
    """Find a provider by name."""
    if provider_name:
        for provider_class in providers:
            if provider_class.NAME.lower() == provider_name.lower():
                # Instantiate the provider
                return provider_class()
    raise ValueError(f"Provider {provider_name} not found")
