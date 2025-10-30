"""Cash account manager for PDT-safe trading with T+2 settlements.

Tracks unsettled proceeds by fill date, computes available settled cash,
and rotates capital across N buckets (default 3) to enable continuous trading
using only settled funds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    fees: float
    timestamp: datetime
    settlement_date: date

    @property
    def gross_amount(self) -> float:
        return round(self.quantity * self.price, 2)

    @property
    def net_cash_effect(self) -> float:
        # CASH EFFECT: BUY uses cash (negative), SELL adds cash (positive)
        sign = -1.0 if self.side.upper() == 'BUY' else 1.0
        return round(sign * self.gross_amount - self.fees, 2)


@dataclass
class CashBucket:
    index: int
    target_fraction: float
    last_used: Optional[date] = None
    pending_unsettled: float = 0.0


@dataclass
class CashManagerConfig:
    initial_settled_cash: float
    num_buckets: int = 3
    t_plus_days: int = 2
    use_settled_only: bool = True


class CashManager:
    def __init__(self, config: CashManagerConfig):
        self.config = config
        self._fills: List[FillRecord] = []
        self._buckets: List[CashBucket] = [
            CashBucket(index=i, target_fraction=1.0 / config.num_buckets)
            for i in range(config.num_buckets)
        ]
        self._cached_broker_settled: Optional[float] = None
        self._last_balance_sync: Optional[datetime] = None

    # ---- Settlement math ----
    def _today(self) -> date:
        return datetime.now().date()

    def _compute_unsettled_total(self, on_date: Optional[date] = None) -> float:
        if on_date is None:
            on_date = self._today()
        unsettled = sum(
            f.net_cash_effect for f in self._fills if f.settlement_date > on_date
        )
        return round(unsettled, 2)

    def get_settled_cash(self, broker_cash_available: Optional[float] = None) -> float:
        """
        Return conservative settled cash.

        If broker provides `cash_available` for cash accounts, prefer that.
        Otherwise, approximate: initial_settled_cash + settled net cash from fills.
        """
        if broker_cash_available is not None:
            return float(broker_cash_available)

        today = self._today()
        settled_effect = sum(
            f.net_cash_effect for f in self._fills if f.settlement_date <= today
        )
        return round(self.config.initial_settled_cash + settled_effect, 2)

    # ---- Bucket rotation ----
    def select_active_bucket(self, trading_day: Optional[date] = None) -> int:
        if trading_day is None:
            trading_day = self._today()

        # Choose the bucket least recently used, simple round-robin by date
        sorted_b = sorted(self._buckets, key=lambda b: (b.last_used or date(1970, 1, 1)))
        bucket = sorted_b[0]
        bucket.last_used = trading_day
        logger.debug(f"Selected cash bucket {bucket.index} for {trading_day}")
        return bucket.index

    def bucket_target_cash(self, total_settled_cash: float, bucket_index: int) -> float:
        if not (0 <= bucket_index < len(self._buckets)):
            return 0.0
        return round(total_settled_cash * self._buckets[bucket_index].target_fraction, 2)

    # ---- Recording fills ----
    def record_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        fees: float = 0.0,
        filled_at: Optional[datetime] = None,
    ) -> FillRecord:
        ts = filled_at or datetime.now()
        settle = (ts + timedelta(days=self.config.t_plus_days)).date()
        record = FillRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fees=fees,
            timestamp=ts,
            settlement_date=settle,
        )
        self._fills.append(record)
        logger.info(
            f"Recorded fill {side} {quantity} {symbol} @ {price:.2f}, settles {settle}, cash_effect {record.net_cash_effect:+.2f}"
        )
        return record

    # ---- Limits and sizing ----
    def compute_position_size_by_risk(
        self,
        account_equity: float,
        risk_perc: float,
        entry_price: float,
        stop_price: float,
        max_shares_cap: Optional[int] = None,
    ) -> int:
        risk_amount = max(0.0, account_equity * risk_perc)
        per_share_risk = max(1e-6, abs(entry_price - stop_price))
        shares = int(risk_amount // per_share_risk)
        if max_shares_cap is not None:
            shares = min(shares, max_shares_cap)
        return max(0, shares)

    def clamp_to_settled_cash(self, shares: int, entry_price: float, 
                               settled_cash: float, reserve_pct: float = 0.0) -> int:
        if shares <= 0:
            return 0
        max_cash = max(0.0, settled_cash * (1.0 - reserve_pct))
        cost = shares * entry_price
        if cost <= max_cash:
            return shares
        affordable = int(max_cash // entry_price)
        return max(0, affordable)

    # ---- Journal helpers ----
    def get_fills(self) -> List[FillRecord]:
        return list(self._fills)

    def get_unsettled_breakdown(self) -> List[Tuple[date, float]]:
        by_date: Dict[date, float] = {}
        for f in self._fills:
            if f.settlement_date > self._today():
                by_date[f.settlement_date] = by_date.get(f.settlement_date, 0.0) + f.net_cash_effect
        return sorted(by_date.items(), key=lambda x: x[0])


