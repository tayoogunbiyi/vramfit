"""Model interpretation errors, independent of the CLI and Hub transport."""


class ModelInspectionError(Exception):
    """Base class for failures to interpret a model."""


class InvalidModelConfigError(ModelInspectionError):
    """The config is malformed or contains inconsistent architecture settings."""


class UnsupportedArchitectureError(ModelInspectionError):
    """The model is outside the implemented architecture support boundary."""

    def __init__(self, model_type: str, reasons: tuple[str, ...]) -> None:
        if not reasons:
            raise ValueError(
                "An unsupported architecture requires at least one reason."
            )
        self.model_type = model_type
        self.reasons = reasons
        super().__init__(
            f"Unsupported architecture {model_type!r}: {'; '.join(reasons)}"
        )
