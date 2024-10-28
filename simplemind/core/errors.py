class SimpleMindError(Exception):
    """Base exception for SimpleMind errors."""

    pass


class ProviderError(SimpleMindError):
    """Raised when there's an error with the AI provider."""

    pass


class ConfigurationError(SimpleMindError):
    """Raised when there's a configuration error."""

    pass


class AuthenticationError(SimpleMindError):
    """Raised when authentication fails."""

    pass
