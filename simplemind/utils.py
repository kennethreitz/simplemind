import difflib
from typing import Union

from .providers import providers


def find_provider(provider_name: Union[str, None]):
    """Find a provider by name."""
    if provider_name:
        for provider_class in providers:
            if provider_class.NAME.lower() == provider_name.lower():
                # Instantiate the provider
                return provider_class()
    
    providers_name = [provider.NAME.lower() for provider in providers]
    providers_founds = difflib.get_close_matches(provider_name.lower(), providers_name)
    
    if providers_founds:
        raise ValueError(f"Provider {provider_name} not found. Maybe you try to use is '{providers_founds[0]}'?")
    else:
        raise ValueError(f"Provider {provider_name} not found.")