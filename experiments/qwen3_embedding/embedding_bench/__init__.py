"""T3a: isolated Qwen3-Embedding-0.6B document-ranking experiment.

Pure experiment logic lives here and needs only the standard library plus
ZOMAH itself. The real embedding model is confined to ``qwen.py`` so tests
never import torch or download weights.
"""
