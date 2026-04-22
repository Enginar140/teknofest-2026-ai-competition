"""
Teknofest AI System
Havacılıkta Yapay Zeka Yarışması için kapsamlı AI sistemi
"""

__version__ = "1.0.0"
__author__ = "Teknofest Team"

# Modüller
from . import core
from . import data
from . import models
from . import server
from . import camera
from . import ui
from . import testing

__all__ = [
    'core',
    'data',
    'models',
    'server',
    'camera',
    'ui',
    'testing',
]
