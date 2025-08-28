"""
Dynamic Utility Loader

Uses importlib.util to dynamically load utility modules for MLflow experiments.
This allows for flexible importing of utility functions without hardcoded imports.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class UtilityLoader:
    """Dynamic loader for MLflow utility modules."""
    
    def __init__(self, utils_dir: Optional[str] = None):
        """
        Initialize the utility loader.
        
        Args:
            utils_dir: Path to utils directory (defaults to current utils dir)
        """
        if utils_dir is None:
            self.utils_dir = Path(__file__).parent
        else:
            self.utils_dir = Path(utils_dir)
        
        self._loaded_modules: Dict[str, Any] = {}
    
    def load_module(self, module_name: str, reload: bool = False) -> Any:
        """
        Dynamically load a utility module.
        
        Args:
            module_name: Name of the module to load (without .py extension)
            reload: Whether to reload if already loaded
            
        Returns:
            Loaded module object
        """
        if module_name in self._loaded_modules and not reload:
            return self._loaded_modules[module_name]
        
        module_path = self.utils_dir / f"{module_name}.py"
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module {module_name} not found at {module_path}")
        
        # Create module spec
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Could not create spec for {module_name}")
        
        # Create and execute module
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules to handle internal imports
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
            self._loaded_modules[module_name] = module
            print(f"✅ Loaded utility module: {module_name}")
            return module
        except Exception as e:
            # Clean up on failure
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise ImportError(f"Failed to load module {module_name}: {e}")
    
    def get_function(self, module_name: str, function_name: str) -> Any:
        """
        Get a specific function from a utility module.
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
            
        Returns:
            Function object
        """
        module = self.load_module(module_name)
        
        if not hasattr(module, function_name):
            raise AttributeError(f"Function {function_name} not found in module {module_name}")
        
        return getattr(module, function_name)
    
    def get_class(self, module_name: str, class_name: str) -> Any:
        """
        Get a specific class from a utility module.
        
        Args:
            module_name: Name of the module
            class_name: Name of the class
            
        Returns:
            Class object
        """
        module = self.load_module(module_name)
        
        if not hasattr(module, class_name):
            raise AttributeError(f"Class {class_name} not found in module {module_name}")
        
        return getattr(module, class_name)
    
    def list_available_modules(self) -> list:
        """
        List all available utility modules.
        
        Returns:
            List of available module names
        """
        modules = []
        for file_path in self.utils_dir.glob("*.py"):
            if file_path.name not in ["__init__.py", "loader.py"]:
                modules.append(file_path.stem)
        return sorted(modules)
    
    def list_module_functions(self, module_name: str) -> list:
        """
        List all functions in a utility module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of function names
        """
        module = self.load_module(module_name)
        
        functions = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                functions.append(attr_name)
        
        return sorted(functions)
    
    def reload_all_modules(self) -> None:
        """Reload all currently loaded modules."""
        for module_name in list(self._loaded_modules.keys()):
            self.load_module(module_name, reload=True)
        print(f"🔄 Reloaded {len(self._loaded_modules)} modules")


# Convenience functions for common usage patterns
def load_mlflow_setup():
    """Load MLflow setup utilities."""
    loader = UtilityLoader()
    return loader.load_module("mlflow_setup")


def load_data_generation():
    """Load data generation utilities."""
    loader = UtilityLoader()
    return loader.load_module("data_generation")


def load_visualization():
    """Load visualization utilities."""
    loader = UtilityLoader()
    return loader.load_module("visualization")


def load_model_evaluation():
    """Load model evaluation utilities."""
    loader = UtilityLoader()
    return loader.load_module("model_evaluation")


def load_all_utils():
    """
    Load all utility modules and return them as a dictionary.
    
    Returns:
        Dictionary with module names as keys and module objects as values
    """
    loader = UtilityLoader()
    modules = {}
    
    for module_name in loader.list_available_modules():
        try:
            modules[module_name] = loader.load_module(module_name)
        except Exception as e:
            print(f"⚠️ Failed to load {module_name}: {e}")
    
    print(f"📦 Loaded {len(modules)} utility modules")
    return modules


# Example usage functions
def demo_loader_usage():
    """Demonstrate how to use the utility loader."""
    print("🔧 MLflow Utility Loader Demo")
    print("=" * 40)
    
    # Create loader
    loader = UtilityLoader()
    
    # List available modules
    print("Available modules:")
    for module in loader.list_available_modules():
        print(f"  - {module}")
    
    # Load a specific module
    mlflow_setup = loader.load_module("mlflow_setup")
    
    # Get a specific function
    setup_func = loader.get_function("mlflow_setup", "setup_mlflow_tracking")
    
    # List functions in a module
    print(f"\nFunctions in mlflow_setup:")
    for func in loader.list_module_functions("mlflow_setup"):
        print(f"  - {func}")
    
    print("\n✅ Loader demo completed!")


if __name__ == "__main__":
    demo_loader_usage()
