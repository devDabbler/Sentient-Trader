"""
Risk Profile Configuration Manager
Manages user risk preferences for position sizing

Provides:
- Persistent risk profile storage
- Risk tolerance levels (Conservative, Moderate, Aggressive)
- Position sizing limits
- Account capital settings
- Integration with AICapitalAdvisor
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger


@dataclass
class RiskProfile:
    """User's risk profile configuration"""
    
    # Risk tolerance
    risk_tolerance: str = "Moderate"  # Conservative, Moderate, Aggressive
    
    # Capital settings
    total_capital: float = 10000.0
    available_capital: float = 10000.0
    reserved_pct: float = 10.0  # % to keep in reserve
    
    # Position sizing
    max_position_pct: float = 10.0  # Max % per trade
    min_position_pct: float = 2.0   # Min % per trade
    max_positions: int = 10          # Max concurrent positions
    current_positions: int = 0
    
    # Risk per trade
    risk_per_trade_pct: float = 2.0  # % of capital risked per trade
    max_loss_per_day_pct: float = 5.0  # Max daily loss %
    
    # AI settings
    use_ai_sizing: bool = True
    min_confidence_to_trade: float = 60.0
    
    # Updated timestamp
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RiskProfile':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_risk_multiplier(self) -> float:
        """Get position size multiplier based on risk tolerance"""
        multipliers = {
            "Conservative": 0.7,
            "Moderate": 1.0,
            "Aggressive": 1.3
        }
        return multipliers.get(self.risk_tolerance, 1.0)
    
    def get_max_position_value(self) -> float:
        """Calculate maximum position value in dollars"""
        return self.total_capital * (self.max_position_pct / 100.0) * self.get_risk_multiplier()
    
    def get_usable_capital(self) -> float:
        """Get capital available after reserve"""
        reserved = self.total_capital * (self.reserved_pct / 100.0)
        return max(0, self.available_capital - reserved)


# Risk tolerance presets
RISK_PRESETS = {
    "Conservative": {
        "risk_tolerance": "Conservative",
        "max_position_pct": 5.0,
        "min_position_pct": 2.0,
        "risk_per_trade_pct": 1.0,
        "max_loss_per_day_pct": 3.0,
        "reserved_pct": 20.0,
        "min_confidence_to_trade": 75.0,
        "description": "Lower risk, smaller positions, higher confidence required"
    },
    "Moderate": {
        "risk_tolerance": "Moderate",
        "max_position_pct": 10.0,
        "min_position_pct": 3.0,
        "risk_per_trade_pct": 2.0,
        "max_loss_per_day_pct": 5.0,
        "reserved_pct": 10.0,
        "min_confidence_to_trade": 60.0,
        "description": "Balanced risk/reward, standard position sizing"
    },
    "Aggressive": {
        "risk_tolerance": "Aggressive",
        "max_position_pct": 20.0,
        "min_position_pct": 5.0,
        "risk_per_trade_pct": 3.0,
        "max_loss_per_day_pct": 8.0,
        "reserved_pct": 5.0,
        "min_confidence_to_trade": 50.0,
        "description": "Higher risk, larger positions, more opportunities"
    }
}


class RiskProfileManager:
    """Manages risk profile persistence and updates"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize risk profile manager
        
        Args:
            config_file: Path to config file (default: data/risk_profile.json)
        """
        if config_file is None:
            base_dir = Path(__file__).parent.parent
            config_file = str(base_dir / "data" / "risk_profile.json")
        
        self.config_file = config_file
        self.profile = self._load_profile()
        
        logger.info(f"📊 Risk Profile Manager initialized")
        logger.info(f"   Tolerance: {self.profile.risk_tolerance}")
        logger.info(f"   Max Position: {self.profile.max_position_pct}%")
        logger.info(f"   Risk/Trade: {self.profile.risk_per_trade_pct}%")
    
    def _load_profile(self) -> RiskProfile:
        """Load profile from file or create default"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return RiskProfile.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load risk profile: {e}")
        
        return RiskProfile()
    
    def save_profile(self) -> bool:
        """Save current profile to file"""
        try:
            self.profile.updated_at = datetime.now().isoformat()
            
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.profile.to_dict(), f, indent=2)
            
            logger.info(f"💾 Risk profile saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving risk profile: {e}")
            return False
    
    def get_profile(self) -> RiskProfile:
        """Get current risk profile"""
        return self.profile
    
    def update_profile(self, **kwargs) -> RiskProfile:
        """Update profile with new values"""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        
        self.save_profile()
        return self.profile
    
    def apply_preset(self, preset_name: str) -> RiskProfile:
        """Apply a risk preset"""
        if preset_name not in RISK_PRESETS:
            logger.warning(f"Unknown preset: {preset_name}")
            return self.profile
        
        preset = RISK_PRESETS[preset_name]
        for key, value in preset.items():
            if key != "description" and hasattr(self.profile, key):
                setattr(self.profile, key, value)
        
        self.save_profile()
        logger.info(f"✅ Applied '{preset_name}' risk preset")
        return self.profile
    
    def update_capital(self, total: float = None, available: float = None) -> RiskProfile:
        """Update capital values"""
        if total is not None:
            self.profile.total_capital = total
        if available is not None:
            self.profile.available_capital = available
        
        self.save_profile()
        return self.profile
    
    def calculate_position_size(
        self,
        price: float,
        stop_loss: float = None,
        confidence: float = None
    ) -> Dict:
        """
        Calculate recommended position size based on profile
        
        Args:
            price: Entry price
            stop_loss: Stop loss price (optional, for risk-based sizing)
            confidence: Signal confidence 0-100 (optional)
            
        Returns:
            Dict with position sizing recommendation
        """
        profile = self.profile
        
        # Base position value
        max_position = profile.get_max_position_value()
        usable_capital = profile.get_usable_capital()
        
        # Start with max position or usable capital, whichever is lower
        position_value = min(max_position, usable_capital)
        
        # Apply confidence adjustment if provided
        if confidence is not None:
            if confidence >= 90:
                conf_mult = 1.3
            elif confidence >= 75:
                conf_mult = 1.15
            elif confidence >= 60:
                conf_mult = 1.0
            elif confidence >= 50:
                conf_mult = 0.8
            else:
                conf_mult = 0.5
            
            position_value *= conf_mult
        
        # Risk-based sizing if stop loss provided
        shares_by_position = int(position_value / price) if price > 0 else 0
        shares_by_risk = shares_by_position  # Default
        
        if stop_loss and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
            if risk_per_share > 0:
                max_risk_amount = profile.total_capital * (profile.risk_per_trade_pct / 100.0)
                shares_by_risk = int(max_risk_amount / risk_per_share)
        
        # Use the more conservative of the two
        recommended_shares = min(shares_by_position, shares_by_risk)
        recommended_value = recommended_shares * price
        
        # Calculate risk metrics
        risk_amount = recommended_shares * abs(price - (stop_loss or price * 0.95)) if stop_loss else recommended_value * 0.05
        risk_pct = (risk_amount / profile.total_capital) * 100 if profile.total_capital > 0 else 0
        
        return {
            'recommended_shares': recommended_shares,
            'recommended_value': recommended_value,
            'position_pct': (recommended_value / profile.total_capital * 100) if profile.total_capital > 0 else 0,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'max_position_value': max_position,
            'usable_capital': usable_capital,
            'confidence_adjustment': conf_mult if confidence else 1.0,
            'sizing_method': 'risk_based' if stop_loss else 'position_pct',
            'profile_tolerance': profile.risk_tolerance
        }
    
    def get_sizing_summary(self) -> str:
        """Get human-readable summary of current sizing settings"""
        p = self.profile
        return (
            f"📊 **Risk Profile: {p.risk_tolerance}**\n"
            f"💰 Capital: ${p.total_capital:,.2f} (${p.get_usable_capital():,.2f} usable)\n"
            f"📈 Max Position: {p.max_position_pct}% (${p.get_max_position_value():,.2f})\n"
            f"⚠️ Risk/Trade: {p.risk_per_trade_pct}%\n"
            f"🎯 Min Confidence: {p.min_confidence_to_trade}%\n"
            f"🔄 AI Sizing: {'Enabled' if p.use_ai_sizing else 'Disabled'}"
        )


# Singleton instance
_risk_profile_manager: Optional[RiskProfileManager] = None


def get_risk_profile_manager() -> RiskProfileManager:
    """Get or create singleton risk profile manager"""
    global _risk_profile_manager
    
    if _risk_profile_manager is None:
        _risk_profile_manager = RiskProfileManager()
    
    return _risk_profile_manager

