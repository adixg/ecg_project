"""CNN-BiLSTM baseline for PTB-XL multi-label classification."""

__all__ = ["CNNBiLSTM"]


def __getattr__(name):
    # Lazy: importing `src` must not load torch. On macOS, torch's OpenMP runtime
    # loaded before xgboost makes xgboost segfault, and src.classical_ml imports src.metrics.
    if name == "CNNBiLSTM":
        from .model import CNNBiLSTM
        return CNNBiLSTM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
