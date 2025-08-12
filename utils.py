"""utils.py
Small helper utilities: config loader and simple logger setup.
"""
import yaml
import logging

def read_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(level='INFO'):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format='%(asctime)s | %(levelname)-7s | %(message)s')
