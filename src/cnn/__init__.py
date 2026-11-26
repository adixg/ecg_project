"""CNN and CNN-hybrid models for PTB-XL classification."""

__all__ = ["CNNBiLSTM"]


def __getattr__(name):
    """Load the PyTorch model lazily so package discovery stays lightweight."""

    if name == "CNNBiLSTM":
        from .model import CNNBiLSTM

        return CNNBiLSTM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
