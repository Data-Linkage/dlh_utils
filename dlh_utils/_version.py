'''script to track version number of package'''
import importlib_metadata

__version__ = importlib_metadata.version(__package__ or __name__)
