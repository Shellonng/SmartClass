"""
Configuration Loader
配置文件加载器
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    获取特定模型的配置
    
    Args:
        config: 总配置字典
        model_name: 模型名称 (llm/asr/embedding/etc.)
        
    Returns:
        模型配置字典
    """
    return config.get('models', {}).get(model_name, {})


def get_interview_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取面试相关配置
    
    Args:
        config: 总配置字典
        
    Returns:
        面试配置字典
    """
    return config.get('interview', {})


def get_speech_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取语音处理配置
    
    Args:
        config: 总配置字典
        
    Returns:
        语音配置字典
    """
    return config.get('speech', {})

