"""Tests for ProjectX/TopstepX API client."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from scalper.feeds.projectx_client import (
    ProjectXClient,
    ProjectXConfig,
    ContractInfo,
    AuthenticationError,
    APIError,
    TOPSTEPX_API_URL,
    TOPSTEPX_MARKET_HUB,
)
from scalper.models import Candle


@pytest.fixture
def config():
    return ProjectXConfig(
        username="testuser",
        api_key="test-api-key-123",
    )


@pytest.fixture
def client(config):
    return ProjectXClient(config)


class TestProjectXConfig:
    def test_default_urls(self):
        cfg = ProjectXConfig(username="u", api_key="k")
        assert cfg.api_url == TOPSTEPX_API_URL
        assert cfg.market_hub_url == TOPSTEPX_MARKET_HUB

    def test_custom_urls(self):
        cfg = ProjectXConfig(
            username="u",
            api_key="k",
            api_url="https://custom.api.com",
        )
        assert cfg.api_url == "https://custom.api.com"


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_authenticate_success(self, client):
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "token": "jwt-token-abc123",
            "success": True,
            "errorCode": 0,
            "errorMessage": None,
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        client._session = mock_session
        token = await client.authenticate()

        assert token == "jwt-token-abc123"
        assert client.is_authenticated

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, client):
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "success": False,
            "errorCode": 401,
            "errorMessage": "Invalid credentials",
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        client._session = mock_session

        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await client.authenticate()


class TestContractSearch:
    @pytest.mark.asyncio
    async def test_search_contracts(self, client):
        client._token = "test-token"
        client._token_acquired = __import__("time").time()

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "contracts": [
                {
                    "id": "CON.F.US.ENQ.M26",
                    "name": "ENQM26",
                    "description": "E-mini NASDAQ-100 Jun 2026",
                    "tickSize": 0.25,
                    "tickValue": 5.0,
                    "activeContract": True,
                    "symbolId": "F.US.ENQ",
                },
            ],
            "success": True,
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        client._session = mock_session
        contracts = await client.search_contracts("NQ")

        assert len(contracts) == 1
        assert contracts[0].id == "CON.F.US.ENQ.M26"
        assert contracts[0].tick_size == 0.25
        assert contracts[0].tick_value == 5.0
        assert contracts[0].active is True


class TestBarRetrieval:
    @pytest.mark.asyncio
    async def test_get_bars(self, client):
        client._token = "test-token"
        client._token_acquired = __import__("time").time()

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "bars": [
                {"t": "2026-03-30T14:00:00Z", "o": 20100.0, "h": 20105.0, "l": 20098.0, "c": 20103.0, "v": 1500},
                {"t": "2026-03-30T14:01:00Z", "o": 20103.0, "h": 20108.0, "l": 20101.0, "c": 20106.0, "v": 1200},
                {"t": "2026-03-30T14:02:00Z", "o": 20106.0, "h": 20110.0, "l": 20104.0, "c": 20107.0, "v": 900},
            ],
            "success": True,
        })

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))
        mock_session.closed = False

        client._session = mock_session

        start = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        candles = await client.get_bars("CON.F.US.ENQ.M26", start, end)

        assert len(candles) == 3
        assert candles[0].open == 20100.0
        assert candles[0].high == 20105.0
        assert candles[0].close == 20103.0
        assert candles[0].volume == 1500
        assert candles[0].is_complete is True
        assert candles[1].close == 20106.0
