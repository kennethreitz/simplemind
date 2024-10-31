import difflib

from .providers import BaseProvider, providers

_PROVIDER_NAMES = [provider.NAME.lower() for provider in providers]


def find_provider(provider_name: str | None) -> BaseProvider:
    """
    Find and instantiate a provider by name.

    Parameters:
        provider_name (Union[str, None]): The name of the provider to find.

    Returns:
        An instance of the provider class if found.

    Raises:
        ValueError: If the provider is not specified or is not found, with a suggestion for the closest match.
    """
    if provider_name is None:
        raise ValueError("No provider specified.")

    # Find the provider by name.
    for provider_class in providers:
        if provider_class.NAME.lower() == provider_name.lower():
            # Instantiate the provider
            return provider_class()

    # Find the closest match
    provider_found = difflib.get_close_matches(
        provider_name.lower(), _PROVIDER_NAMES, n=1
    )
    if provider_found:
        raise ValueError(
            f"Provider {provider_name!r} not found. Did you mean {provider_found[0]!r}?"
        )
    raise ValueError(f"Provider {provider_name} not found.")
