"""
Centralized Crypto Trading Strategy Configuration
Ensures uniform strategy defaults across the entire application
"""

from typing import Dict, TypedDict


class StrategyConfig(TypedDict):
    """Type definition for strategy configuration"""
    name: str
    stop_pct: float
    target_pct: float
    timeframe: str
    description: str
    risk_level: str


# Centralized strategy configurations
# All crypto trading strategies defined here to ensure consistency
CRYPTO_STRATEGIES: Dict[str, StrategyConfig] = {
    'scalp': {
        'name': 'Scalping',
        'stop_pct': 1.0,
        'target_pct': 2.5,
        'timeframe': '5m',
        'description': 'Quick profits with tight stops',
        'risk_level': 'LOW'
    },
    'momentum': {
        'name': 'Momentum',
        'stop_pct': 2.0,
        'target_pct': 5.0,
        'timeframe': '15m',
        'description': 'Ride strong trends with moderate risk',
        'risk_level': 'MEDIUM'
    },
    'swing': {
        'name': 'Swing Trading',
        'stop_pct': 3.0,
        'target_pct': 8.0,
        'timeframe': '1h',
        'description': 'Larger moves with longer holds',
        'risk_level': 'MEDIUM'
    },
    'breakout': {
        'name': 'Breakout',
        'stop_pct': 2.5,
        'target_pct': 6.0,
        'timeframe': '15m',
        'description': 'Technical breakouts with volume confirmation',
        'risk_level': 'MEDIUM'
    },
    'ema_crossover': {
        'name': 'EMA Crossover + Heikin Ashi',
        'stop_pct': 1.5,
        'target_pct': 3.0,
        'timeframe': '5m',
        'description': 'EMA 20/50/100 crossovers with Heikin Ashi confirmation',
        'risk_level': 'LOW'
    },
    'rsi_stoch_hammer': {
        'name': 'RSI + Stochastic + Hammer',
        'stop_pct': 1.5,
        'target_pct': 3.5,
        'timeframe': '5m',
        'description': 'Oversold signals with Bollinger Bands and hammer patterns',
        'risk_level': 'LOW'
    },
    'fisher_rsi_multi': {
        'name': 'Fisher RSI Multi-Indicator',
        'stop_pct': 2.0,
        'target_pct': 4.0,
        'timeframe': '15m',
        'description': 'Fisher RSI with MFI, Stochastic, and EMA confirmation',
        'risk_level': 'MEDIUM'
    },
    'macd_volume': {
        'name': 'MACD + Volume + RSI',
        'stop_pct': 2.0,
        'target_pct': 4.5,
        'timeframe': '15m',
        'description': 'MACD crossovers with volume spikes and Fisher RSI',
        'risk_level': 'MEDIUM'
    },
    'aggressive_scalp': {
        'name': 'Aggressive Scalping',
        'stop_pct': 0.8,
        'target_pct': 2.0,
        'timeframe': '1m',
        'description': 'Fast EMA crosses with ultra-tight stops',
        'risk_level': 'HIGH'
    }
}


def get_strategy_config(strategy_id: str) -> StrategyConfig:
    """
    Get strategy configuration by ID with fallback to default
    
    Args:
        strategy_id: Strategy identifier (e.g., 'scalp', 'momentum', 'ema_crossover')
    
    Returns:
        StrategyConfig dictionary with stop_pct, target_pct, etc.
    """
    # Normalize strategy ID
    strategy_id = strategy_id.lower().strip()
    
    # Return strategy config or default to momentum
    return CRYPTO_STRATEGIES.get(strategy_id, CRYPTO_STRATEGIES['momentum'])


def get_all_strategy_ids() -> list:
    """Get list of all strategy IDs"""
    return list(CRYPTO_STRATEGIES.keys())


def get_all_strategy_names() -> list:
    """Get list of all strategy display names"""
    return [config['name'] for config in CRYPTO_STRATEGIES.values()]


def get_strategy_display_options() -> Dict[str, str]:
    """
    Get mapping of strategy IDs to display names
    Used for dropdowns and selectors
    """
    return {
        strategy_id: config['name']
        for strategy_id, config in CRYPTO_STRATEGIES.items()
    }


def validate_strategy_id(strategy_id: str) -> bool:
    """Check if a strategy ID is valid"""
    return strategy_id.lower() in CRYPTO_STRATEGIES
