"""ProjectX / TopstepX REST API client.

Handles authentication, contract search, and historical bar retrieval
via the ProjectX Gateway API used by TopstepX.

API Documentation: https://gateway.docs.projectx.com/
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import structlog

from scalper.models import Candle

logger = structlog.get_logger()


# --- Connection URLs ---
# TopstepX / ProjectX LIVE endpoints
TOPSTEPX_API_URL = "https://api.topstepx.com"
TOPSTEPX_MARKET_HUB = "https://rtc.topstepx.com/hubs/market"
TOPSTEPX_USER_HUB = "https://rtc.topstepx.com/hubs/user"

# TopstepX / ProjectX DEMO endpoints (free with eval account)
DEMO_API_URL = "https://gateway-api-demo.s2f.projectx.com"
DEMO_MARKET_HUB = "https://gateway-rtc-demo.s2f.projectx.com/hubs/market"
DEMO_USER_HUB = "https://gateway-rtc-demo.s2f.projectx.com/hubs/user"

# The Futures Desk (alternative/legacy)
TFD_API_URL = "https://api.thefuturesdesk.projectx.com"
TFD_MARKET_HUB = "https://rtc.thefuturesdesk.projectx.com/hubs/market"
TFD_USER_HUB = "https://rtc.thefuturesdesk.projectx.com/hubs/user"


def get_urls(environment: str = "live") -> tuple[str, str, str]:
    """Get API URLs for the given environment.

    Args:
        environment: "live", "demo", or "tfd" (The Futures Desk)

    Returns:
        (api_url, market_hub_url, user_hub_url)
    """
    if environment == "demo":
        return DEMO_API_URL, DEMO_MARKET_HUB, DEMO_USER_HUB
    elif environment == "tfd":
        return TFD_API_URL, TFD_MARKET_HUB, TFD_USER_HUB
    else:
        return TOPSTEPX_API_URL, TOPSTEPX_MARKET_HUB, TOPSTEPX_USER_HUB


@dataclass
class ProjectXConfig:
    """Configuration for ProjectX API connection."""
    username: str
    api_key: str
    api_url: str = TOPSTEPX_API_URL
    market_hub_url: str = TOPSTEPX_MARKET_HUB
    user_hub_url: str = TOPSTEPX_USER_HUB


@dataclass
class ContractInfo:
    """Futures contract metadata."""
    id: str           # e.g., "CON.F.US.ENQ.M26"
    name: str         # e.g., "ENQM26"
    description: str  # e.g., "E-mini NASDAQ-100 Jun 2026"
    tick_size: float
    tick_value: float
    active: bool
    symbol_id: str    # e.g., "F.US.ENQ"


class ProjectXClient:
    """Async REST client for the ProjectX Gateway API.

    Usage:
        client = ProjectXClient(config)
        await client.authenticate()
        contracts = await client.search_contracts("NQ")
        bars = await client.get_bars(contract_id, ...)
    """

    def __init__(self, config: ProjectXConfig):
        self.config = config
        self._token: Optional[str] = None
        self._token_acquired: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def token(self) -> Optional[str]:
        return self._token

    @property
    def is_authenticated(self) -> bool:
        if not self._token:
            return False
        # Tokens valid for 24 hours
        return (time.time() - self._token_acquired) < 23 * 3600

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def authenticate(self) -> str:
        """Authenticate with API key and get JWT token.

        POST /api/Auth/loginKey
        Body: {"userName": "...", "apiKey": "..."}
        Returns: JWT token string
        """
        session = await self._ensure_session()
        url = f"{self.config.api_url}/api/Auth/loginKey"

        payload = {
            "userName": self.config.username,
            "apiKey": self.config.api_key,
        }

        logger.info("authenticating", url=url, username=self.config.username)

        async with session.post(url, json=payload) as resp:
            data = await resp.json()

            if not data.get("success", False):
                error = data.get("errorMessage", "Unknown auth error")
                logger.error("auth_failed", error=error, code=data.get("errorCode"))
                raise AuthenticationError(f"Authentication failed: {error}")

            self._token = data["token"]
            self._token_acquired = time.time()
            logger.info("authenticated", token_preview=self._token[:20] + "...")
            return self._token

    async def ensure_authenticated(self) -> str:
        """Ensure we have a valid token, re-authenticating if needed."""
        if not self.is_authenticated:
            await self.authenticate()
        return self._token

    def _auth_headers(self) -> dict:
        """Build headers with JWT auth."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def search_contracts(self, search_text: str, live: bool = False) -> list[ContractInfo]:
        """Search for futures contracts.

        POST /api/Contract/search
        Body: {"searchText": "NQ", "live": false}
        """
        await self.ensure_authenticated()
        session = await self._ensure_session()

        url = f"{self.config.api_url}/api/Contract/search"
        payload = {"searchText": search_text, "live": live}

        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
            data = await resp.json()

            if not data.get("success", True):
                raise APIError(f"Contract search failed: {data.get('errorMessage')}")

            contracts = []
            for c in data.get("contracts", data if isinstance(data, list) else []):
                if isinstance(c, dict):
                    contracts.append(ContractInfo(
                        id=c.get("id", ""),
                        name=c.get("name", ""),
                        description=c.get("description", ""),
                        tick_size=float(c.get("tickSize", 0.25)),
                        tick_value=float(c.get("tickValue", 5.0)),
                        active=c.get("activeContract", False),
                        symbol_id=c.get("symbolId", ""),
                    ))

            logger.info("contracts_found", count=len(contracts), search=search_text)
            return contracts

    async def find_active_nq_contract(self) -> ContractInfo:
        """Find the currently active NQ (E-mini NASDAQ-100) front-month contract."""
        contracts = await self.search_contracts("NQ")

        # Filter for active E-mini NASDAQ contracts
        nq_contracts = [
            c for c in contracts
            if "ENQ" in c.id or "NQ" in c.name.upper()
        ]

        # Prefer active contracts
        active = [c for c in nq_contracts if c.active]
        if active:
            logger.info("active_nq_contract", id=active[0].id, name=active[0].name)
            return active[0]

        if nq_contracts:
            logger.info("nq_contract_fallback", id=nq_contracts[0].id)
            return nq_contracts[0]

        raise APIError("No NQ contracts found")

    async def get_bars(
        self,
        contract_id: str,
        start_time: datetime,
        end_time: datetime,
        unit: int = 2,  # 1=Second, 2=Minute, 3=Hour, 4=Day
        unit_number: int = 1,
        limit: int = 5000,
        live: bool = False,
        include_partial: bool = False,
    ) -> list[Candle]:
        """Retrieve historical OHLCV bars.

        POST /api/History/retrieveBars
        """
        await self.ensure_authenticated()
        session = await self._ensure_session()

        url = f"{self.config.api_url}/api/History/retrieveBars"
        payload = {
            "contractId": contract_id,
            "live": live,
            "startTime": start_time.isoformat(),
            "endTime": end_time.isoformat(),
            "unit": unit,
            "unitNumber": unit_number,
            "limit": limit,
            "includePartialBar": include_partial,
        }

        logger.info(
            "fetching_bars",
            contract=contract_id,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            unit=unit,
            limit=limit,
        )

        async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
            data = await resp.json()

            if not data.get("success", True):
                raise APIError(f"Bar retrieval failed: {data.get('errorMessage')}")

            bars_data = data.get("bars", [])
            candles = []

            for b in bars_data:
                ts = b.get("t", "")
                # Parse ISO timestamp to epoch
                if isinstance(ts, str) and ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        epoch = dt.timestamp()
                    except ValueError:
                        epoch = time.time()
                else:
                    epoch = float(ts) if ts else time.time()

                candles.append(Candle(
                    timestamp=epoch,
                    open=float(b.get("o", 0)),
                    high=float(b.get("h", 0)),
                    low=float(b.get("l", 0)),
                    close=float(b.get("c", 0)),
                    volume=int(b.get("v", 0)),
                    is_complete=True,
                ))

            logger.info("bars_received", count=len(candles), contract=contract_id)
            return candles

    async def get_recent_bars(
        self,
        contract_id: str,
        count: int = 200,
        unit: int = 2,
        unit_number: int = 1,
        live: bool = False,
    ) -> list[Candle]:
        """Convenience: get the most recent N bars."""
        from datetime import timedelta

        end = datetime.now(timezone.utc)
        # Estimate how far back to go (generous for market hours gaps)
        if unit == 2:  # minutes
            hours_back = max(24, (count * unit_number) // 60 * 3)
        elif unit == 3:  # hours
            hours_back = count * unit_number * 3
        else:
            hours_back = count * 24

        start = end - timedelta(hours=hours_back)
        bars = await self.get_bars(
            contract_id, start, end,
            unit=unit, unit_number=unit_number,
            limit=count, live=live,
        )
        return bars[-count:] if len(bars) > count else bars


class AuthenticationError(Exception):
    pass


class APIError(Exception):
    pass
