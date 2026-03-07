from pathlib import Path

directory_path = Path(__file__).parent

# Get a list of all .py files in the current directory, except this one.
modules = [f.stem for f in directory_path.glob("*.py") if f.stem != "__init__"]

# Get a list of all directories which contain a __init__.py file, they are packages.
packages = [f.parent.name for f in directory_path.glob("*/__init__.py")]

__all__ = modules + packages
