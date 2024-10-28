from .providers import providers


def find_provider(provider_name: str):
    """Find a provider by name."""
    for provider_class in providers:
        if provider_class.__name__.lower() == provider_name.lower():
            # Instantiate the provider
            return provider_class()
    raise ValueError(f"Provider {provider_name} not found")
