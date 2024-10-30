import difflib
from typing import Union

from .providers import providers

_PROVIDER_NAMES = [provider.NAME.lower() for provider in providers]


def find_provider(provider_name: Union[str, None]):
    """Find a provider by name."""
    if provider_name:
        for provider_class in providers:
            if provider_class.NAME.lower() == provider_name.lower():
                # Instantiate the provider
                return provider_class()

    provider_found = difflib.get_close_matches(
        provider_name.lower(), _PROVIDER_NAMES, n=1
    )  # Show only one suggestion

    if provider_found:
        raise ValueError(
            f"Provider {provider_name!r} not found. Did you mean {provider_found[0]!r}?"
        )
    else:
        raise ValueError(f"Provider {provider_name} not found.")
