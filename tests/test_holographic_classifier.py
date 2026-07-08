"""CI wrapper for the gradient-free prototype classifier. The module ships its own asserts in _selftest --
one-shot bundle prototypes + perceptron retraining (add/subtract, no gradients); retraining does not hurt,
accuracy beats chance, and prototypes are deterministic. This collects that check into the suite."""
from holographic.agents_and_reasoning.holographic_classifier import _selftest


def test_holographic_classifier_selftest():
    _selftest()
