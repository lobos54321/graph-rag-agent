"""
数据库配置 - 支持Neo4j和内存数据库切换
"""

import os
from enum import Enum
from typing import Optional


class DatabaseType(Enum):
    NEO4J = "neo4j"
    MEMORY = "memory"


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # 从环境变量获取数据库类型，默认使用内存数据库
        self.db_type = self._get_db_type()
        
        # Neo4j配置
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        # 内存数据库配置
        self.memory_data_dir = os.getenv('GRAPH_DATA_DIR', './graph_data')
    
    def _get_db_type(self) -> DatabaseType:
        """获取数据库类型"""
        db_type_str = os.getenv('DATABASE_TYPE', 'memory').lower()
        
        # 如果设置了Neo4j连接信息，优先使用Neo4j
        if (os.getenv('NEO4J_URI') and 
            os.getenv('NEO4J_PASSWORD') and 
            db_type_str != 'memory'):
            return DatabaseType.NEO4J
        
        return DatabaseType.MEMORY
    
    def is_neo4j_available(self) -> bool:
        """检查Neo4j是否可用"""
        return (self.neo4j_uri is not None and 
                self.neo4j_password is not None)
    
    def is_memory_db(self) -> bool:
        """是否使用内存数据库"""
        return self.db_type == DatabaseType.MEMORY
    
    def get_connection_info(self) -> dict:
        """获取连接信息"""
        if self.is_memory_db():
            return {
                "type": "memory",
                "data_dir": self.memory_data_dir
            }
        else:
            return {
                "type": "neo4j",
                "uri": self.neo4j_uri,
                "username": self.neo4j_username,
                "password": self.neo4j_password
            }


# 全局配置实例
db_config = DatabaseConfig()


def get_database_config() -> DatabaseConfig:
    """获取数据库配置"""
    return db_config


def use_memory_database() -> bool:
    """是否使用内存数据库"""
    return db_config.is_memory_db()


def get_database_info() -> str:
    """获取数据库信息字符串"""
    config = db_config.get_connection_info()
    if config["type"] == "memory":
        return f"内存数据库 (数据目录: {config['data_dir']})"
    else:
        return f"Neo4j数据库 (URI: {config['uri']})"