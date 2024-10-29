import os
import sys

# Add the parent directory to the path so we can import the module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import simplemind as sm

__all__ = ["sm"]
