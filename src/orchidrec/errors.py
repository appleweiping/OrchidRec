"""Exception hierarchy used throughout OrchidRec."""


class OrchidRecError(Exception):
    """Base class for all expected OrchidRec failures."""


class ValidationError(OrchidRecError, ValueError):
    """Raised when user-supplied data is invalid."""


class SplitError(OrchidRecError, ValueError):
    """Raised when an interaction dataset cannot be split as requested."""


class NotFittedError(OrchidRecError, RuntimeError):
    """Raised when inference is requested before fitting a model."""


class SerializationError(OrchidRecError, ValueError):
    """Raised when a model state cannot be encoded or decoded safely."""


class ConfigurationError(OrchidRecError, ValueError):
    """Raised when an experiment configuration is invalid."""
