"""
V-SQLM: Vote-First Self-Questioning Language Models.

Main package for V-SQLM implementation.
"""

__version__ = "0.1.0"

from sqlm.vsqlm import sampler, canonicalize, aggregator, stopping, solver
from sqlm.vsqlm import verifiers
from sqlm.debate import minidebate, prompts
from sqlm.curriculum import proposer_reward
from sqlm.metrics import calibration, logging
from sqlm.stats import entropy, beta_binomial

__all__ = [
    'sampler',
    'canonicalize',
    'aggregator',
    'stopping',
    'solver',
    'verifiers',
    'minidebate',
    'prompts',
    'proposer_reward',
    'calibration',
    'logging',
    'entropy',
    'beta_binomial',
]
