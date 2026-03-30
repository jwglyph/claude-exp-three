#!/usr/bin/env python3
"""Test TopstepX API endpoints with real credentials.

Run this on your local machine (needs internet access):

    python scripts/test_live_api.py

Or with explicit credentials:

    python scripts/test_live_api.py --username EMAIL --api-key KEY

Set environment variables to avoid passing on command line:
    export NQ_SCALPER_USERNAME="wujacky1369@gmail.com"
    export NQ_SCALPER_API_KEY="EhozDLrskNuGKI2j0REeoLAlGjiyL4lbn5Bu+NkpIz4="
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalper.feeds.projectx_client import (
    ProjectXClient, ProjectXConfig, get_urls,
    AuthenticationError, APIError,
)


def header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


async def test_all(username: str, api_key: str):
    results = {"pass": 0, "fail": 0}

    def ok(test: str, detail: str = ""):
        results["pass"] += 1
        print(f"  [PASS] {test}" + (f" - {detail}" if detail else ""))

    def fail(test: str, error: str):
        results["fail"] += 1
        print(f"  [FAIL] {test} - {error}")

    # =============================================
    # Test 1: Authentication on LIVE environment
    # =============================================
    header("1. AUTHENTICATION")

    client = None
    env_used = None

    for env in ["live", "demo"]:
        api_url, market_hub, user_hub = get_urls(env)
        config = ProjectXConfig(
            username=username, api_key=api_key,
            api_url=api_url, market_hub_url=market_hub, user_hub_url=user_hub,
        )
        client = ProjectXClient(config)
        print(f"\n  Trying {env.upper()}: {api_url}")

        try:
            token = await client.authenticate()
            ok(f"Auth ({env})", f"Token: {token[:30]}...")
            env_used = env
            break
        except AuthenticationError as e:
            fail(f"Auth ({env})", str(e))
        except Exception as e:
            fail(f"Auth ({env})", f"Connection error: {e}")

    if not client or not client.is_authenticated:
        print("\n  FATAL: Could not authenticate on any environment.")
        print("  Check your username and API key.")
        await client.close() if client else None
        return results

    print(f"\n  Using environment: {env_used.upper()}")

    # =============================================
    # Test 2: Search contracts
    # =============================================
    header("2. CONTRACT SEARCH")

    try:
        contracts = await client.search_contracts("NQ")
        if contracts:
            ok("Search 'NQ'", f"Found {len(contracts)} contracts")
            print()
            print(f"  {'ID':<35} {'Name':<15} {'Description':<40} {'Active':<7} {'Tick':<6} {'Value'}")
            print(f"  {'-'*35} {'-'*15} {'-'*40} {'-'*7} {'-'*6} {'-'*6}")
            for c in contracts[:10]:
                print(f"  {c.id:<35} {c.name:<15} {c.description:<40} {str(c.active):<7} {c.tick_size:<6} ${c.tick_value}")
        else:
            fail("Search 'NQ'", "No contracts returned")
    except Exception as e:
        fail("Search 'NQ'", str(e))

    # Also search for ES and MNQ
    for sym in ["ES", "MNQ"]:
        try:
            c = await client.search_contracts(sym)
            ok(f"Search '{sym}'", f"Found {len(c)} contracts")
        except Exception as e:
            fail(f"Search '{sym}'", str(e))

    # =============================================
    # Test 3: Find active NQ contract
    # =============================================
    header("3. ACTIVE NQ CONTRACT")

    nq = None
    try:
        nq = await client.find_active_nq_contract()
        ok("Find active NQ", f"{nq.id} - {nq.description}")
        print(f"\n  Contract ID:  {nq.id}")
        print(f"  Name:         {nq.name}")
        print(f"  Description:  {nq.description}")
        print(f"  Symbol ID:    {nq.symbol_id}")
        print(f"  Tick size:    {nq.tick_size}")
        print(f"  Tick value:   ${nq.tick_value}")
        print(f"  Active:       {nq.active}")
    except Exception as e:
        fail("Find active NQ", str(e))

    # =============================================
    # Test 4: Historical bars
    # =============================================
    header("4. HISTORICAL 1-MINUTE BARS")

    if nq:
        try:
            bars = await client.get_recent_bars(nq.id, count=50, unit=2, unit_number=1)
            if bars:
                ok(f"Fetch 1-min bars", f"Got {len(bars)} bars")

                print(f"\n  Last 15 bars:")
                print(f"  {'Time (UTC)':<20} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>8}")
                print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
                for b in bars[-15:]:
                    ts = datetime.fromtimestamp(b.timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
                    print(f"  {ts:<20} {b.open:10.2f} {b.high:10.2f} {b.low:10.2f} {b.close:10.2f} {b.volume:8d}")

                price = bars[-1].close
                low = min(b.low for b in bars)
                high = max(b.high for b in bars)
                avg_vol = sum(b.volume for b in bars) / len(bars)
                avg_range = sum(b.high - b.low for b in bars) / len(bars)

                print(f"\n  Summary:")
                print(f"    Latest price:  {price:.2f}")
                print(f"    Price range:   {low:.2f} - {high:.2f} ({high - low:.2f} points)")
                print(f"    Avg volume:    {avg_vol:.0f}")
                print(f"    Avg bar range: {avg_range:.2f} points")
            else:
                fail("Fetch 1-min bars", "No bars returned (market may be closed)")
        except Exception as e:
            fail("Fetch 1-min bars", str(e))

        # Also test 5-min and hourly bars
        for unit, unit_num, label in [(2, 5, "5-min"), (3, 1, "1-hour")]:
            try:
                b = await client.get_recent_bars(nq.id, count=20, unit=unit, unit_number=unit_num)
                ok(f"Fetch {label} bars", f"Got {len(b)} bars")
            except Exception as e:
                fail(f"Fetch {label} bars", str(e))

    # =============================================
    # Test 5: Token validation
    # =============================================
    header("5. TOKEN STATE")
    print(f"  Token valid:      {client.is_authenticated}")
    print(f"  Token preview:    {client.token[:30]}...")

    # =============================================
    # Test 6: Account search (if available)
    # =============================================
    header("6. ACCOUNT SEARCH")
    try:
        session = await client._ensure_session()
        url = f"{client.config.api_url}/api/Account/search"
        async with session.post(url, json={"onlyActive": True}, headers=client._auth_headers()) as resp:
            data = await resp.json()
            if isinstance(data, list):
                ok("Account search", f"Found {len(data)} accounts")
                for acc in data[:5]:
                    print(f"    Account: {acc.get('id', 'N/A')} - {acc.get('name', 'N/A')} - Balance: ${acc.get('balance', 0):.2f}")
            elif isinstance(data, dict) and data.get("accounts"):
                accs = data["accounts"]
                ok("Account search", f"Found {len(accs)} accounts")
                for acc in accs[:5]:
                    print(f"    Account: {acc.get('id', 'N/A')} - {acc.get('name', 'N/A')}")
            else:
                ok("Account search", f"Response: {str(data)[:200]}")
    except Exception as e:
        fail("Account search", str(e))

    # =============================================
    # Summary
    # =============================================
    await client.close()

    header("RESULTS")
    total = results["pass"] + results["fail"]
    print(f"  Passed: {results['pass']}/{total}")
    print(f"  Failed: {results['fail']}/{total}")
    if results["fail"] == 0:
        print("\n  ALL TESTS PASSED! Ready to trade with real NQ data.")
        print(f"\n  Quick start:")
        print(f"    nq-scalper trade --username '{username}' --api-key '{api_key}' --feed-type topstepx --paper")
        print(f"\n  Backtest on real data:")
        print(f"    nq-scalper backtest-live --username '{username}' --api-key '{api_key}' --bars 500")
    else:
        print(f"\n  {results['fail']} test(s) failed. Check errors above.")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test TopstepX API endpoints")
    parser.add_argument("--username", default=os.environ.get("NQ_SCALPER_USERNAME", "wujacky1369@gmail.com"))
    parser.add_argument("--api-key", default=os.environ.get("NQ_SCALPER_API_KEY", "EhozDLrskNuGKI2j0REeoLAlGjiyL4lbn5Bu+NkpIz4="))
    args = parser.parse_args()

    print("TopstepX API Endpoint Tester")
    print(f"Username: {args.username}")
    print(f"API Key:  {args.api_key[:15]}...")

    asyncio.run(test_all(args.username, args.api_key))


if __name__ == "__main__":
    main()
