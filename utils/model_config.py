"""
模型配置加载器
用于管理和加载不同模型的优化配置
"""

import json
import os
from typing import Dict, Any

class ModelConfigManager:
    """模型配置管理器"""
    
    DEFAULT_CONFIGS = {
        'base_cnn': {
            'leaky_relu_slope': 0.01,
            'hidden_dims': [1024, 512, 256, 128, 64, 32]
        },
        'feature_concat': {
            'leaky_relu_slope': 0.1,
            'hidden_dims': [2048, 1024, 512, 256, 128, 64, 32]
        },
        'cross_attention': {
            'leaky_relu_slope': 0.1,
            'attention_dim': 8192,
            'hidden_dims': [4096, 2048, 1024, 512, 256, 128, 64, 32]
        },
        'svm_fusion': {
            'C_low': 1e-2,
            'C_high': 1e2,
            'epsilon_low': 1e-2,
            'epsilon_high': 1e2,
            'gamma_low': 1e-2,
            'gamma_high': 1e2
        }
    }
    
    def __init__(self, config_dir='model_configs'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        # 确保默认配置文件存在
        self._ensure_default_configs()
    
    def _ensure_default_configs(self):
        """确保默认配置文件存在"""
        for model_name, config in self.DEFAULT_CONFIGS.items():
            config_path = os.path.join(self.config_dir, f'{model_name}_default.json')
            if not os.path.exists(config_path):
                self.save_config(model_name, config, 'default')
    
    def save_config(self, model_name: str, config: Dict[str, Any], config_name: str = 'default'):
        """保存模型配置"""
        config_path = os.path.join(self.config_dir, f'{model_name}_{config_name}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return config_path
    
    def load_config(self, model_name: str, config_name: str = 'default'):
        """加载模型配置"""
        config_path = os.path.join(self.config_dir, f'{model_name}_{config_name}.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif model_name in self.DEFAULT_CONFIGS:
            print(f"警告: 未找到 {model_name}_{config_name}.json，使用默认配置")
            return self.DEFAULT_CONFIGS[model_name].copy()
        else:
            print(f"警告: 未找到 {model_name} 的配置，返回空配置")
            return {}
    
    def load_optimized_config(self, model_name: str):
        """加载优化后的配置"""
        # 首先尝试从优化结果加载
        optimized_path = os.path.join('optimization_results', f'{model_name}_best_params.json')
        
        if os.path.exists(optimized_path):
            with open(optimized_path, 'r') as f:
                optimized_params = json.load(f)
            
            # 将优化参数转换为模型配置
            if model_name == 'base_cnn':
                config = {
                    'leaky_relu_slope': optimized_params.get('leaky_relu_slope', 0.01),
                    'lr': optimized_params.get('lr', 0.0058),
                    'weight_decay': optimized_params.get('weight_decay', 0.0001),
                    'momentum': optimized_params.get('momentum', 0.8672)
                }
            elif model_name == 'feature_concat':
                config = {
                    'leaky_relu_slope': 0.1,
                    'lr': optimized_params.get('lr', 0.00503),
                    'weight_decay': optimized_params.get('weight_decay', 0.0015),
                    'momentum': optimized_params.get('momentum', 0.8874)
                }
            elif model_name == 'cross_attention':
                config = {
                    'attention_dim': optimized_params.get('attention_dim', 8192),
                    'leaky_relu_slope': 0.1,
                    'lr': optimized_params.get('lr', 0.01465),
                    'weight_decay': optimized_params.get('weight_decay', 0.0015),
                    'momentum': optimized_params.get('momentum', 0.8947)
                }
            elif model_name == 'svm_fusion':
                config = optimized_params.copy()
            else:
                config = optimized_params.copy()
            
            # 保存为优化配置
            self.save_config(model_name, config, 'optimized')
            return config
        
        # 如果没有优化配置，返回默认配置
        return self.load_config(model_name)
    
    def get_model_config(self, strategy: str, use_optimized: bool = True):
        """根据策略名称获取模型配置"""
        model_name_map = {
            'base': 'base_cnn',
            'concat': 'feature_concat',
            'attention': 'cross_attention',
            'decision': 'svm_fusion'
        }
        
        internal_name = model_name_map.get(strategy)
        if not internal_name:
            return {}
        
        if use_optimized:
            return self.load_optimized_config(internal_name)
        else:
            return self.load_config(internal_name)
    
    def compare_configs(self, model_name: str):
        """比较不同配置"""
        default_config = self.load_config(model_name, 'default')
        optimized_config = self.load_config(model_name, 'optimized')
        
        print(f"\n{model_name} 配置对比:")
        print("-" * 40)
        
        all_keys = set(list(default_config.keys()) + list(optimized_config.keys()))
        
        for key in sorted(all_keys):
            default_val = default_config.get(key, 'N/A')
            optimized_val = optimized_config.get(key, 'N/A')
            
            if default_val != optimized_val:
                print(f"  {key}: {default_val} -> {optimized_val} (优化后)")
            else:
                print(f"  {key}: {default_val}")


# 使用示例
if __name__ == "__main__":
    manager = ModelConfigManager()
    
    # 获取模型配置
    base_config = manager.get_model_config('base', use_optimized=True)
    print("Base CNN配置:", base_config)
    
    # 比较配置
    manager.compare_configs('base_cnn')