"""
Import context for tests.
"""
import os  # noqa E402
import sys  # noqa E402
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))  # noqa E402
import data
import dimensions
import embeddings
import support
