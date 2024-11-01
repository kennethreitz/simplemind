import os
import sys

import pytest

# Add the project root to the Python path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplemind import Session


@pytest.fixture
def sm():
    """Fixture that provides a simplemind Session instance with default settings."""
    return Session()
