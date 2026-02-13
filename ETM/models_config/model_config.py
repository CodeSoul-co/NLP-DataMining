"""
Model Configuration Reader

Provides a unified interface to read model configurations from models.yaml.
Used by both Python scripts and bash scripts (via CLI).

Usage:
    # Python
    from config.model_config import ModelConfig
    config = ModelConfig()
    requirements = config.get_requirements('lda')  # ['bow']
    
    # CLI (for bash scripts)
    python -m config.model_config --model lda --query requires
    # Output: bow
"""

import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional


class ModelConfig:
    """Model configuration manager"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent / 'models.yaml'
        # Also check in models_config directory
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / 'models_config' / 'models.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get all model configurations"""
        return self._config.get('models', {})
    
    @property
    def data_types(self) -> Dict[str, Any]:
        """Get all data type definitions"""
        return self._config.get('data_types', {})
    
    @property
    def output_dirs(self) -> Dict[str, str]:
        """Get output directory configurations"""
        return self._config.get('output', {})
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model"""
        return self.models.get(model_name.lower())
    
    def get_requirements(self, model_name: str) -> List[str]:
        """Get data requirements for a model"""
        model = self.get_model(model_name)
        if model is None:
            return ['bow']  # Default to BOW only
        return model.get('requires', ['bow'])
    
    def get_optional(self, model_name: str) -> List[str]:
        """Get optional data for a model"""
        model = self.get_model(model_name)
        if model is None:
            return []
        return model.get('optional', [])
    
    def get_model_type(self, model_name: str) -> str:
        """Get model type (traditional/neural/hybrid)"""
        model = self.get_model(model_name)
        if model is None:
            return 'traditional'
        return model.get('type', 'traditional')
    
    def get_params(self, model_name: str) -> Dict[str, Any]:
        """Get all parameters for a model with their definitions"""
        model = self.get_model(model_name)
        if model is None:
            return {}
        return model.get('params', {})
    
    def get_param_names(self, model_name: str) -> List[str]:
        """Get list of parameter names for a model"""
        params = self.get_params(model_name)
        return list(params.keys())
    
    def get_param_default(self, model_name: str, param_name: str) -> Any:
        """Get default value for a specific parameter"""
        params = self.get_params(model_name)
        if param_name in params:
            return params[param_name].get('default')
        return None
    
    def get_param_help(self, model_name: str, param_name: str) -> str:
        """Get help text for a specific parameter"""
        params = self.get_params(model_name)
        if param_name in params:
            return params[param_name].get('help', '')
        return ''
    
    def has_param(self, model_name: str, param_name: str) -> bool:
        """Check if a model has a specific parameter"""
        params = self.get_params(model_name)
        return param_name in params
    
    def is_auto_topics(self, model_name: str) -> bool:
        """Check if model auto-determines topic number"""
        model = self.get_model(model_name)
        if model is None:
            return False
        return model.get('auto_topics', False)
    
    def get_description(self, model_name: str) -> str:
        """Get model description"""
        model = self.get_model(model_name)
        if model is None:
            return "Unknown model"
        return model.get('description', '')
    
    def get_full_name(self, model_name: str) -> str:
        """Get full model name"""
        model = self.get_model(model_name)
        if model is None:
            return model_name.upper()
        return model.get('name', model_name.upper())
    
    def list_models(self) -> List[str]:
        """List all supported model names"""
        return list(self.models.keys())
    
    def list_models_by_type(self, model_type: str) -> List[str]:
        """List models of a specific type"""
        return [
            name for name, config in self.models.items()
            if config.get('type') == model_type
        ]
    
    def get_data_type_info(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a data type"""
        return self.data_types.get(data_type)
    
    def get_output_dir(self, model_name: str) -> str:
        """Get output directory for a model"""
        if model_name.lower() == 'theta':
            return self.output_dirs.get('theta', '/root/autodl-tmp/result')
        return self.output_dirs.get('baseline', '/root/autodl-tmp/result/baseline')
    
    def needs_data(self, model_name: str, data_type: str) -> bool:
        """Check if a model needs a specific data type"""
        requirements = self.get_requirements(model_name)
        return data_type in requirements
    
    def print_model_info(self, model_name: str):
        """Print detailed information about a model"""
        model = self.get_model(model_name)
        if model is None:
            print(f"Unknown model: {model_name}")
            return
        
        print(f"\n{'='*50}")
        print(f"Model: {model.get('name', model_name.upper())}")
        print(f"{'='*50}")
        print(f"Type: {model.get('type', 'unknown')}")
        print(f"Description: {model.get('description', 'N/A')}")
        print(f"Requires: {', '.join(model.get('requires', []))}")
        if model.get('optional'):
            print(f"Optional: {', '.join(model.get('optional', []))}")
        if model.get('params'):
            print(f"Special Params: {', '.join(model.get('params', []))}")
        if model.get('auto_topics'):
            print("Auto Topics: Yes (topic number determined automatically)")
        print()


def main():
    """CLI interface for bash scripts"""
    parser = argparse.ArgumentParser(description='Query model configuration')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--query', type=str, 
                        choices=['requires', 'type', 'params', 'auto_topics', 
                                 'description', 'name', 'output_dir', 'list',
                                 'needs_bow', 'needs_sbert', 'needs_time', 
                                 'needs_qwen', 'needs_word2vec', 'info', 'has_param'],
                        help='What to query')
    parser.add_argument('--list-type', type=str, 
                        choices=['traditional', 'neural', 'hybrid'],
                        help='List models of specific type')
    parser.add_argument('--param', type=str, help='Parameter name for has_param query')
    
    args = parser.parse_args()
    config = ModelConfig()
    
    if args.query == 'list':
        print(' '.join(config.list_models()))
        return
    
    if args.list_type:
        print(' '.join(config.list_models_by_type(args.list_type)))
        return
    
    if not args.model:
        parser.print_help()
        return
    
    model = args.model.lower()
    
    if args.query == 'requires':
        print(' '.join(config.get_requirements(model)))
    elif args.query == 'type':
        print(config.get_model_type(model))
    elif args.query == 'params':
        print(' '.join(config.get_param_names(model)))
    elif args.query == 'has_param':
        # Check if model has a specific param (usage: --model lda --query has_param --param batch_size)
        if hasattr(args, 'param') and args.param:
            print('true' if config.has_param(model, args.param) else 'false')
        else:
            print('false')
    elif args.query == 'auto_topics':
        print('true' if config.is_auto_topics(model) else 'false')
    elif args.query == 'description':
        print(config.get_description(model))
    elif args.query == 'name':
        print(config.get_full_name(model))
    elif args.query == 'output_dir':
        print(config.get_output_dir(model))
    elif args.query == 'needs_bow':
        print('true' if config.needs_data(model, 'bow') else 'false')
    elif args.query == 'needs_sbert':
        print('true' if config.needs_data(model, 'sbert') else 'false')
    elif args.query == 'needs_time':
        print('true' if config.needs_data(model, 'time') else 'false')
    elif args.query == 'needs_qwen':
        print('true' if config.needs_data(model, 'qwen') else 'false')
    elif args.query == 'needs_word2vec':
        print('true' if config.needs_data(model, 'word2vec') else 'false')
    elif args.query == 'info':
        config.print_model_info(model)
    else:
        # Default: print requirements
        print(' '.join(config.get_requirements(model)))


if __name__ == '__main__':
    main()
