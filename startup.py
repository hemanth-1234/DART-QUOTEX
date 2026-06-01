#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
startup.py — DART-Quotex  ·  Single-file launcher
==================================================
Run this ONE file.  It handles everything automatically:

  1.  First run  → ask credentials, save to accounts.json
  2.  Next run   → show saved accounts, choose or add new
  3.  Account type  → DEMO / REAL (with REAL warning)
  4.  Pair selection → auto-discover from Quotex OR manual pick
  5.  Data harvest   → auto-pull if DB has < 7 days of candles
  6.  Backtest       → walk-forward on stored history
  7.  Quality gate   → WR ≥ 55%, PF ≥ 1.2, trades ≥ 10
  8.  Money management setup:
        · Masaniello (capital / events / wins / payout)
        · Compounding mode (compound gains each cycle)
        · Fixed capital mode (always same base stake)
        · Martingale on/off
  9.  Start live trading session (LiveTrader)

All settings persist in accounts.json between runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── suppress INFO noise until we're ready ────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("startup.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("startup")

# ── ANSI colour helpers ───────────────────────────────────────────────────────
if os.name == "nt":
    os.system("")   # enable VT100 on Windows

_USE_C = sys.stdout.isatty()
def _a(code, t): return f"\033[{code}m{t}\033[0m" if _USE_C else t
def GR(t):  return _a("92", t)
def RD(t):  return _a("91", t)
def YL(t):  return _a("93", t)
def CY(t):  return _a("96", t)
def MG(t):  return _a("95", t)
def DM(t):  return _a("2",  t)
def BD(t):  return _a("1",  t)
def WT(t):  return _a("97", t)
def BL(t):  return _a("94", t)

ACCOUNTS_FILE  = Path("accounts.json")
TRADE_LOG_FILE = Path("trade_log.txt")

# ──────────────────────────────────────────────────────────────────────────────
# Console helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner():
    print()
    print(BD(CY("  ╔══════════════════════════════════════════════════════╗")))
    print(BD(CY("  ║")) + BD(WT("         ◈  DART-QUOTEX  ·  AI TRADING BOT            ")) + BD(CY("║")))
    print(BD(CY("  ║")) + DM("      Powered by SAC · Ensemble ML · Masaniello       ") + BD(CY("║")))
    print(BD(CY("  ╚══════════════════════════════════════════════════════╝")))
    print()

def _box(title: str, colour=None):
    c = colour or CY
    w = 56
    pad = (w - len(title)) // 2
    print()
    print(BD(c("  ╔" + "═" * w + "╗")))
    print(BD(c("  ║")) + BD(WT(" " * pad + title + " " * (w - pad - len(title)))) + BD(c("║")))
    print(BD(c("  ╚" + "═" * w + "╝")))
    print()

def _sep(colour=None):
    c = colour or DM
    print("  " + c("─" * 60))

def _row(label: str, value: str, lc=None, vc=None):
    lc = lc or DM; vc = vc or WT
    print("    " + BD(lc(f"{'  ' + label + ' ':.<32}")) + BD(vc(str(value))))

def _ask(prompt: str, valid: Optional[List[str]] = None, colour=None) -> str:
    c = colour or CY
    while True:
        try:
            ans = input(BD(c(f"  ➤  {prompt} "))).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if valid is None:
            return ans
        if ans.lower() in [v.lower() for v in valid]:
            return ans.lower()
        print(RD(f"     Please enter one of: {', '.join(valid)}"))

def _confirm(prompt: str) -> bool:
    return _ask(prompt + " [y/n]", ["y", "n"]) == "y"

def _input_float(prompt: str, lo: float, hi: float, default: float) -> float:
    while True:
        raw = _ask(f"{prompt} [{lo}–{hi}, default={default}]: ", colour=YL)
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(RD(f"     Must be between {lo} and {hi}"))
        except ValueError:
            print(RD("     Enter a number"))

def _input_int(prompt: str, lo: int, hi: int, default: int) -> int:
    while True:
        raw = _ask(f"{prompt} [{lo}–{hi}, default={default}]: ", colour=YL)
        if raw == "":
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(RD(f"     Must be between {lo} and {hi}"))
        except ValueError:
            print(RD("     Enter a whole number"))


# ──────────────────────────────────────────────────────────────────────────────
# Accounts storage
# ──────────────────────────────────────────────────────────────────────────────

def _load_accounts() -> List[dict]:
    if not ACCOUNTS_FILE.exists():
        return []
    try:
        data = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_accounts(accounts: List[dict]) -> None:
    ACCOUNTS_FILE.write_text(
        json.dumps(accounts, indent=2), encoding="utf-8"
    )

def _select_account() -> dict:
    """First-run or account selection flow. Returns chosen account dict."""
    accounts = _load_accounts()

    _box("ACCOUNT LOGIN", CY)

    if not accounts:
        # ── First time setup ──────────────────────────────────────────────────
        print(GR(BD("  Welcome!  First-time setup — enter your Quotex credentials.")))
        print()
        email    = _ask("Quotex email:    ")
        password = _ask("Quotex password: ", colour=MG)
        nickname = _ask("Nickname for this account (e.g. 'Main'): ")

        account = {
            "nickname": nickname or email.split("@")[0],
            "email":    email,
            "password": password,
            "last_mode": "DEMO",
            "last_pairs": [],
            "mm": {},
        }
        accounts.append(account)
        _save_accounts(accounts)
        print()
        print(GR(BD(f"  ✔  Saved account: {account['nickname']} ({email})")))
        return account

    # ── Show saved accounts ───────────────────────────────────────────────────
    print(BD(DM("  Saved accounts:")))
    print()
    for i, acc in enumerate(accounts, 1):
        print(f"    {BD(CY(str(i)))}  {BD(WT(acc['nickname']))}  {DM(acc['email'])}  {DM('[' + acc.get('last_mode','DEMO') + ']')}")
    print()
    print(f"    {BD(CY(str(len(accounts) + 1)))}  {DM('Add a new account')}")
    print()

    choice = _input_int("Choose account", 1, len(accounts) + 1, 1) - 1

    if choice >= len(accounts):
        # ── Add new ───────────────────────────────────────────────────────────
        print()
        email    = _ask("Quotex email:    ")
        password = _ask("Quotex password: ", colour=MG)
        nickname = _ask("Nickname:        ")
        account  = {
            "nickname": nickname or email.split("@")[0],
            "email":    email,
            "password": password,
            "last_mode": "DEMO",
            "last_pairs": [],
            "mm": {},
        }
        accounts.append(account)
        _save_accounts(accounts)
        print(GR(BD("  ✔  Account added.")))
        return account

    account = accounts[choice]

    # ── Offer to change credentials ───────────────────────────────────────────
    print()
    print(DM(f"  Account: {account['email']}"))
    if _confirm("  Use this account?"):
        return account

    account["email"]    = _ask("New email:    ")
    account["password"] = _ask("New password: ", colour=MG)
    accounts[choice] = account
    _save_accounts(accounts)
    print(GR(BD("  ✔  Updated.")))
    return account


def _save_account_field(account: dict, **kwargs) -> None:
    """Persist field changes back to accounts.json."""
    accounts = _load_accounts()
    for i, acc in enumerate(accounts):
        if acc.get("email") == account["email"]:
            acc.update(kwargs)
            account.update(kwargs)
            accounts[i] = acc
            break
    _save_accounts(accounts)


# ──────────────────────────────────────────────────────────────────────────────
# Account-type selection (DEMO / REAL)
# ──────────────────────────────────────────────────────────────────────────────

def _select_mode(account: dict) -> str:
    last = account.get("last_mode", "DEMO")

    _box("SELECT ACCOUNT TYPE", CY)
    print(f"    {BD(CY('1'))}  {BD(GR('DEMO'))}   {DM('practice · no real money')}")
    print(f"    {BD(CY('2'))}  {BD(RD('REAL'))}   {DM('live money · real trades')}")
    print()
    print(DM(f"  Last used: {last}"))
    print()

    choice = _ask("Account type [1=DEMO / 2=REAL]", ["1", "2"])
    mode   = "DEMO" if choice == "1" else "REAL"

    if mode == "REAL":
        _sep(RD)
        print(RD(BD("  ⚠  WARNING — REAL MONEY")))
        print(RD("  You are about to trade with real money."))
        print(RD("  Losses are permanent.  Only continue if you accept the risk."))
        _sep(RD)
        print()
        if not _confirm(RD("  Type 'yes' — I accept the risk, continue with REAL")):
            print(YL("  Switched to DEMO."))
            mode = "DEMO"

    colour = GR if mode == "DEMO" else RD
    print(colour(BD(f"\n  ✔  Mode: {mode}")))
    _save_account_field(account, last_mode=mode)
    return mode


# ──────────────────────────────────────────────────────────────────────────────
# Pair selection
# ──────────────────────────────────────────────────────────────────────────────

OTC_DEFAULTS = [
    "EURUSD_OTC","GBPUSD_OTC","USDJPY_OTC","AUDUSD_OTC",
    "USDCAD_OTC","EURJPY_OTC","GBPJPY_OTC","EURGBP_OTC",
    "USDCHF_OTC","EURAUD_OTC",
]

def _select_pairs(account: dict) -> List[str]:
    last_pairs = account.get("last_pairs", [])

    _box("PAIR SELECTION", CY)

    print(f"    {BD(CY('1'))}  {WT('Use last session pairs')}"
          + (DM(f"  ({', '.join(last_pairs[:4])}{'…' if len(last_pairs) > 4 else ''})") if last_pairs else DM("  (none saved)")))
    print(f"    {BD(CY('2'))}  {WT('Choose from popular OTC pairs')}")
    print(f"    {BD(CY('3'))}  {WT('Enter pairs manually')}")
    print()

    choice = _ask("Pair source [1/2/3]", ["1","2","3"])

    if choice == "1" and last_pairs:
        pairs = last_pairs
        print(GR(BD(f"\n  ✔  Using {len(pairs)} saved pairs.")))
        return pairs

    if choice == "2" or (choice == "1" and not last_pairs):
        _sep()
        print(BD(WT("\n  Popular OTC pairs:")))
        print()
        for i, p in enumerate(OTC_DEFAULTS, 1):
            print(f"    {BD(CY(str(i))):>6}  {WT(p)}")
        print()
        print(DM("  Enter pair numbers (e.g. 1,3,5) or 'all' for all:"))
        raw = _ask("Pairs [numbers / 'all']: ", colour=YL)
        if raw.strip().lower() == "all":
            pairs = list(OTC_DEFAULTS)
        else:
            chosen = []
            for token in raw.replace(",", " ").split():
                try:
                    idx = int(token) - 1
                    if 0 <= idx < len(OTC_DEFAULTS):
                        chosen.append(OTC_DEFAULTS[idx])
                except ValueError:
                    pass
            pairs = chosen if chosen else [OTC_DEFAULTS[0]]
    else:
        _sep()
        print(DM("  Enter pairs separated by commas."))
        print(DM("  Example: EURUSD_OTC,GBPUSD_OTC,USDJPY_OTC"))
        print()
        raw = _ask("Pairs: ", colour=YL)
        pairs = [p.strip().upper() for p in raw.split(",") if p.strip()]
        if not pairs:
            pairs = [OTC_DEFAULTS[0]]

    # Normalise: add _OTC suffix if missing
    normalised = []
    for p in pairs:
        if not p.upper().endswith("_OTC") and not p.upper().endswith("_otc"):
            p = p + "_OTC"
        normalised.append(p)
    pairs = normalised

    print(GR(BD(f"\n  ✔  Selected {len(pairs)} pair(s):")))
    for p in pairs:
        print(f"       {CY('·')} {WT(p)}")

    _save_account_field(account, last_pairs=pairs)
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Masaniello money management setup  (same formula as main.py)
# ──────────────────────────────────────────────────────────────────────────────

class Masaniello:
    """
    Exact Masaniello implementation from main.py.
    Computes optimal stake from remaining events/wins needed.
    """

    def __init__(
        self,
        capital:       float,
        events:        int,
        wins_needed:   int,
        payout_decimal: float,
        min_bet:       float = 1.0,
    ) -> None:
        self.start_capital = float(capital)
        self.capital       = float(capital)
        self.total_events  = int(events)
        self.target_wins   = int(wins_needed)
        self.payout        = float(payout_decimal)
        self.events_left   = self.total_events
        self.wins_left     = self.target_wins
        self.status        = "ACTIVE"
        self.p             = 1.0 / self.payout
        self.min_bet       = float(min_bet)

    def _prob(self, n: int, k: int) -> float:
        if k <= 0: return 1.0
        if k > n:  return 0.0
        return sum(
            math.comb(n, i) * (self.p ** i) * ((1 - self.p) ** (n - i))
            for i in range(k, n + 1)
        )

    def get_next_stake(self) -> Tuple[float, str]:
        if self.status != "ACTIVE":
            return 0.0, self.status
        pc = self._prob(self.events_left,     self.wins_left)
        pl = self._prob(self.events_left - 1, self.wins_left)
        if pc == 0:
            return 0.0, "Math Error"
        raw   = self.capital * (1.0 - pl / pc)
        stake = max(round(raw, 2), self.min_bet) if self.capital >= self.min_bet else 0.0
        stake = min(stake, self.capital)
        return stake, "OK"

    def update(self, won: bool, stake: float, payout_dec: float) -> None:
        self.events_left -= 1
        if won:
            self.wins_left -= 1
            self.capital   += stake * (payout_dec - 1.0)
        else:
            self.capital   -= stake
        if   self.wins_left   <= 0:              self.status = "GOAL REACHED"
        elif self.events_left <  self.wins_left: self.status = "MATH IMPOSSIBLE"
        elif self.capital     <  self.min_bet:   self.status = "BANKRUPT"

    def reset_cycle(self) -> None:
        self.events_left = self.total_events
        self.wins_left   = self.target_wins
        self.status      = "ACTIVE"

    def summary(self) -> str:
        stake, _ = self.get_next_stake()
        return (
            f"Capital={self.capital:.2f}  "
            f"Events={self.events_left}/{self.total_events}  "
            f"Wins={self.wins_left}/{self.target_wins}  "
            f"NextStake={stake:.2f}  "
            f"Status={self.status}"
        )


def _setup_money_management(account: dict, balance: float) -> dict:
    """
    Interactive money management setup.
    Returns a config dict that is also saved to the account.
    """
    saved_mm = account.get("mm", {})

    _box("MONEY MANAGEMENT", YL)

    print(f"  Current balance : {BD(GR(f'₹{balance:,.2f}'))}")
    print()
    print(DM("  Choose a money management method:"))
    print()
    print(f"    {BD(CY('1'))}  {BD(WT('Masaniello'))}  {DM('mathematical stake sizing  (recommended)')}")
    print(f"    {BD(CY('2'))}  {BD(WT('Compounding'))} {DM('re-invest % of gains each cycle')}")
    print(f"    {BD(CY('3'))}  {BD(WT('Fixed Stake'))} {DM('same amount every trade')}")
    print()

    mm_choice = _ask("Method [1/2/3]", ["1","2","3"])

    mm_cfg: Dict[str, Any] = {"method": mm_choice}

    if mm_choice == "1":
        # ── Masaniello ────────────────────────────────────────────────────────
        _sep()
        print(BD(CY("\n  Masaniello Setup\n")))
        print(DM("  Capital = fraction of balance you risk in this cycle."))
        print(DM("  Events  = total number of trades in the cycle."))
        print(DM("  Wins    = how many wins you need to be profitable.\n"))

        default_capital = float(saved_mm.get("capital", round(balance * 0.20, 2)))
        default_events  = int(saved_mm.get("events",  10))
        default_wins    = int(saved_mm.get("wins",     6))
        default_payout  = float(saved_mm.get("payout", 1.80))
        default_minbet  = float(saved_mm.get("min_bet", 1.0))

        capital = _input_float("Cycle capital  (₹)", 1.0, balance, default_capital)
        events  = _input_int  ("Total events   ",     2,   100,    default_events)
        wins    = _input_int  ("Wins needed    ",     1,   events, min(default_wins, events))
        payout  = _input_float("Payout decimal ", 1.01, 3.0,       default_payout)
        min_bet = _input_float("Minimum bet    ", 1.0,  capital,   default_minbet)

        masa    = Masaniello(capital, events, wins, payout, min_bet)
        first_stake, _ = masa.get_next_stake()

        print()
        _sep(GR)
        _row("Method",          "Masaniello",   vc=CY)
        _row("Cycle capital",   f"₹{capital:,.2f}")
        _row("Events / Wins",   f"{events} total · {wins} wins needed")
        _row("Payout decimal",  f"{payout:.2f}")
        _row("First stake",     f"₹{first_stake:,.2f}", vc=YL)
        _sep(GR)

        mm_cfg.update({
            "capital": capital, "events": events,
            "wins": wins, "payout": payout, "min_bet": min_bet,
        })

    elif mm_choice == "2":
        # ── Compounding ───────────────────────────────────────────────────────
        _sep()
        print(BD(CY("\n  Compounding Setup\n")))

        default_base   = float(saved_mm.get("base_stake", round(balance * 0.01, 2)))
        default_reinvest = float(saved_mm.get("reinvest_pct", 50.0))
        default_max    = float(saved_mm.get("max_stake_pct", 5.0))

        base_stake    = _input_float("Base stake (₹)",         1.0, balance * 0.5, default_base)
        reinvest_pct  = _input_float("Reinvest % of gains",    0.0, 100.0,          default_reinvest)
        max_stake_pct = _input_float("Max stake (% of balance)", 1.0, 20.0,         default_max)

        print()
        _sep(GR)
        _row("Method",          "Compounding",  vc=CY)
        _row("Base stake",      f"₹{base_stake:,.2f}")
        _row("Reinvest gains",  f"{reinvest_pct:.0f}%")
        _row("Max stake",       f"{max_stake_pct:.0f}% of balance")
        _sep(GR)

        mm_cfg.update({
            "base_stake": base_stake,
            "reinvest_pct": reinvest_pct,
            "max_stake_pct": max_stake_pct,
        })

    else:
        # ── Fixed stake ───────────────────────────────────────────────────────
        _sep()
        print(BD(CY("\n  Fixed Stake Setup\n")))

        default_stake = float(saved_mm.get("fixed_stake", round(balance * 0.01, 2)))
        fixed_stake   = _input_float("Fixed stake (₹)", 1.0, balance * 0.5, default_stake)

        print()
        _sep(GR)
        _row("Method",       "Fixed Stake", vc=CY)
        _row("Stake",        f"₹{fixed_stake:,.2f}")
        _sep(GR)

        mm_cfg.update({"fixed_stake": fixed_stake})

    # ── Martingale add-on ─────────────────────────────────────────────────────
    print()
    mtg_on = _confirm("  Enable Martingale on loss?")
    if mtg_on:
        default_mult  = float(saved_mm.get("mtg_mult",   2.0))
        default_steps = int(saved_mm.get("mtg_steps",    3))
        mtg_mult  = _input_float("Martingale multiplier", 1.1, 5.0,  default_mult)
        mtg_steps = _input_int  ("Max martingale steps",  1,   10,   default_steps)
        mm_cfg.update({"martingale": True, "mtg_mult": mtg_mult, "mtg_steps": mtg_steps})
        print(YL(f"  ✔  Martingale: ×{mtg_mult} for up to {mtg_steps} steps"))
    else:
        mm_cfg["martingale"] = False

    # ── Daily profit lock ─────────────────────────────────────────────────────
    print()
    lock_on = _confirm("  Enable daily profit lock (stop when target reached)?")
    if lock_on:
        default_lock = float(saved_mm.get("daily_lock_pct", 10.0))
        lock_pct = _input_float("Stop when profit reaches (% of balance)", 1.0, 100.0, default_lock)
        mm_cfg.update({"daily_lock": True, "daily_lock_pct": lock_pct})
        print(GR(f"  ✔  Daily lock at +{lock_pct:.0f}%"))
    else:
        mm_cfg["daily_lock"] = False

    # ── Session drawdown stop ─────────────────────────────────────────────────
    print()
    dd_on = _confirm("  Enable session drawdown stop (stop on big loss)?")
    if dd_on:
        default_dd = float(saved_mm.get("max_dd_pct", 10.0))
        dd_pct = _input_float("Stop trading if loss reaches (% of balance)", 1.0, 50.0, default_dd)
        mm_cfg.update({"max_dd": True, "max_dd_pct": dd_pct})
        print(RD(f"  ✔  Drawdown stop at -{dd_pct:.0f}%"))
    else:
        mm_cfg["max_dd"] = False

    _save_account_field(account, mm=mm_cfg)
    print()
    print(GR(BD("  ✔  Money management saved.")))
    return mm_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Masaniello + compounding stake calculator used during live trading
# ──────────────────────────────────────────────────────────────────────────────

class StakeEngine:
    """
    Real-time stake calculator wrapping Masaniello / compounding / fixed.
    Called by the session loop after every trade result.
    """

    def __init__(self, mm_cfg: dict, balance: float) -> None:
        self.cfg        = mm_cfg
        self.method     = mm_cfg.get("method", "3")
        self.balance    = balance
        self.start_bal  = balance
        self.total_profit = 0.0
        self.wins         = 0
        self.losses       = 0
        self.mtg_step     = 0       # current martingale step
        self.mtg_base     = 0.0     # base stake before current martingale run
        self._masa: Optional[Masaniello] = None
        self._compound_capital: float = 0.0

        if self.method == "1":
            self._masa = Masaniello(
                capital=float(mm_cfg["capital"]),
                events=int(mm_cfg["events"]),
                wins_needed=int(mm_cfg["wins"]),
                payout_decimal=float(mm_cfg.get("payout", 1.80)),
                min_bet=float(mm_cfg.get("min_bet", 1.0)),
            )
        elif self.method == "2":
            self._compound_capital = float(mm_cfg.get("base_stake", balance * 0.01))

    def next_stake(self) -> float:
        """Return stake for the next trade."""
        if self.cfg.get("martingale") and self.mtg_step > 0:
            mult  = float(self.cfg.get("mtg_mult", 2.0))
            stake = self.mtg_base * (mult ** self.mtg_step)
            return round(max(stake, 1.0), 2)

        if self.method == "1" and self._masa:
            s, _ = self._masa.get_next_stake()
            self.mtg_base = s
            return s

        if self.method == "2":
            pct = float(self.cfg.get("max_stake_pct", 5.0)) / 100.0
            s   = min(self._compound_capital, self.balance * pct)
            self.mtg_base = max(round(s, 2), 1.0)
            return self.mtg_base

        # Fixed
        s = float(self.cfg.get("fixed_stake", 1.0))
        self.mtg_base = s
        return s

    def record(self, won: bool, stake: float, payout_dec: float) -> None:
        """Update engine state after a trade settles."""
        if won:
            net = stake * (payout_dec - 1.0)
            self.balance      += net
            self.total_profit += net
            self.wins         += 1
            self.mtg_step      = 0   # reset martingale

            if self.method == "1" and self._masa:
                self._masa.update(True, stake, payout_dec)
                if self._masa.status != "ACTIVE":
                    self._masa.reset_cycle()

            if self.method == "2":
                reinvest = float(self.cfg.get("reinvest_pct", 50.0)) / 100.0
                self._compound_capital += net * reinvest
        else:
            self.balance      -= stake
            self.total_profit -= stake
            self.losses       += 1

            max_steps = int(self.cfg.get("mtg_steps", 3))
            if self.cfg.get("martingale") and self.mtg_step < max_steps:
                self.mtg_step += 1
            else:
                self.mtg_step = 0

            if self.method == "1" and self._masa:
                self._masa.update(False, stake, payout_dec)
                if self._masa.status != "ACTIVE":
                    self._masa.reset_cycle()

    def should_stop(self) -> Tuple[bool, str]:
        """Return (stop, reason) based on daily lock / drawdown rules."""
        if self.cfg.get("daily_lock") and self.total_profit > 0:
            lock_pct  = float(self.cfg.get("daily_lock_pct", 10.0)) / 100.0
            threshold = self.start_bal * lock_pct
            if self.total_profit >= threshold:
                return True, f"Daily profit lock hit: +₹{self.total_profit:.2f} ≥ +₹{threshold:.2f}"

        if self.cfg.get("max_dd") and self.total_profit < 0:
            dd_pct    = float(self.cfg.get("max_dd_pct", 10.0)) / 100.0
            threshold = self.start_bal * dd_pct
            if abs(self.total_profit) >= threshold:
                return True, f"Drawdown stop hit: -₹{abs(self.total_profit):.2f} ≥ -₹{threshold:.2f}"

        return False, ""

    def status_line(self) -> str:
        tot   = self.wins + self.losses
        wr    = self.wins / tot * 100 if tot > 0 else 0.0
        stake = self.next_stake()
        pnl_colour = GR if self.total_profit >= 0 else RD
        return (
            f"  {DM('Bal')} {BD(CY(f'₹{self.balance:,.2f}'))}  "
            f"{DM('P&L')} {pnl_colour(BD(f'{self.total_profit:+.2f}'))}  "
            f"{DM('W')} {GR(BD(str(self.wins)))}  "
            f"{DM('L')} {RD(BD(str(self.losses)))}  "
            f"{DM('WR')} {BD(f'{wr:.0f}%')}  "
            f"{DM('NextStake')} {YL(BD(f'₹{stake:.2f}'))}  "
            + (f"{DM('MTG-step')} {YL(str(self.mtg_step))}" if self.mtg_step > 0 else "")
        )


# ──────────────────────────────────────────────────────────────────────────────
# Data pipeline
# ──────────────────────────────────────────────────────────────────────────────

MIN_CANDLES_7D = 7 * 24 * 60   # 1-min candles for 7 days

async def _ensure_data(advisor, pairs: List[str], gran: int = 60) -> None:
    from dart_quotex.data.database import Database
    from dart_quotex.config import cfg
    db = Database(cfg.data.db_path)

    _box("DATA CHECK", CY)

    needs_harvest = []
    for pair in pairs:
        count = db.count_candles(pair, gran)
        ok    = count >= MIN_CANDLES_7D
        sym   = GR("✔") if ok else YL("⚠")
        print(f"  {sym}  {WT(pair):25}  {DM(f'{count:,} candles')}  "
              + (GR("OK") if ok else YL(f"need ~{MIN_CANDLES_7D - count:,} more")))
        if not ok:
            needs_harvest.append(pair)

    if not needs_harvest:
        print()
        print(GR(BD("  ✔  All pairs have sufficient data.")))
        return

    print()
    print(YL(f"  Harvesting data for {len(needs_harvest)} pair(s)…"))
    print(DM("  This may take a few minutes on first run."))
    print()

    for pair in needs_harvest:
        print(f"  {CY('⟳')} {WT(pair)}…", end="", flush=True)
        try:
            stored = await advisor.harvest_history(
                asset=pair,
                total_candles=MIN_CANDLES_7D + 500,
            )
            count = db.count_candles(pair, gran)
            print(f"\r  {GR('✔')} {WT(pair):25} {DM(f'+{stored:,} new · total {count:,}')}")
        except Exception as exc:
            print(f"\r  {RD('✖')} {WT(pair):25} {RD(str(exc))}")


# ──────────────────────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────────────────────

async def _run_backtest(
    pairs: List[str],
    min_wr:     float = 0.55,
    min_pf:     float = 1.2,
    min_trades: int   = 10,
) -> bool:
    """Run per-pair backtest.  Returns True if user wants to continue."""
    from dart_quotex.data.database import Database
    from dart_quotex.advisor import AIAdvisor
    from dart_quotex.backtester import Backtester
    from dart_quotex.config import cfg

    _box("BACKTEST", CY)
    print(YL("  Running walk-forward backtest on stored history…"))
    print()

    db      = Database(cfg.data.db_path)
    advisor = AIAdvisor(use_robust_client=False)
    bt      = Backtester(db=db, advisor=advisor, lookback=cfg.ml.lookback,
                         train_online=True)

    all_pass = True
    for pair in pairs:
        count = db.count_candles(pair, cfg.data.granularity)
        if count < cfg.ml.lookback + 20:
            print(f"  {YL('⚠')}  {WT(pair):25}  {DM('insufficient candles — skipping')}")
            continue

        try:
            result = bt.run(
                asset=pair,
                granularity=cfg.data.granularity,
                start_balance=1000.0,
                payout=0.80,
                min_confidence=0.55,
                limit=3_000,
            )
        except Exception as exc:
            print(f"  {RD('✖')}  {WT(pair):25}  {RD(str(exc))}")
            continue

        wr_ok  = result.win_rate    >= min_wr
        pf_ok  = result.profit_factor >= min_pf
        tr_ok  = result.n_trades    >= min_trades
        passed = wr_ok and pf_ok and tr_ok
        if not passed:
            all_pass = False

        sym    = GR("✔") if passed else RD("✖")
        wr_c   = GR if wr_ok  else RD
        pf_c   = GR if pf_ok  else RD
        tr_c   = GR if tr_ok  else RD
        roi_c  = GR if result.roi >= 0 else RD

        print(
            f"  {sym}  {WT(pair):25}  "
            f"{DM('WR')} {wr_c(f'{result.win_rate:.0%}'):>6}  "
            f"{DM('PF')} {pf_c(f'{result.profit_factor:.2f}'):>6}  "
            f"{DM('Trades')} {tr_c(str(result.n_trades)):>5}  "
            f"{DM('ROI')} {roi_c(f'{result.roi:+.1%}'):>7}  "
            f"{DM('DD')} {f'{result.max_drawdown:.1%}':>5}"
        )

    print()
    _sep()
    if all_pass:
        print(GR(BD("  ✔  All pairs passed the quality gate.")))
    else:
        print(YL(BD("  ⚠  Some pairs failed the quality gate.")))
        print(DM(f"     Thresholds: WR ≥ {min_wr:.0%}  ·  PF ≥ {min_pf}  ·  Trades ≥ {min_trades}"))
    _sep()
    print()

    return _confirm("  Continue to live trading?")


# ──────────────────────────────────────────────────────────────────────────────
# Live session loop
# ──────────────────────────────────────────────────────────────────────────────

async def _live_session(
    account:      dict,
    mode:         str,
    pairs:        List[str],
    mm_cfg:       dict,
    session_min:  int = 60,
    interval_s:   int = 65,
) -> None:
    """
    Full live trading session.
    Wraps LiveTrader but injects our StakeEngine for money management.
    """
    from dart_quotex.advisor import AIAdvisor
    from dart_quotex.config import cfg

    _box("LIVE SESSION", GR if mode == "DEMO" else RD)

    os.environ["QUOTEX_EMAIL"]    = account["email"]
    os.environ["QUOTEX_PASSWORD"] = account["password"]
    os.environ["QUOTEX_MODE"]     = mode.lower()

    # ── Connect and get balance ───────────────────────────────────────────────
    print(YL("  Connecting to Quotex…"))
    advisor = AIAdvisor(use_robust_client=True)
    await advisor.connect()

    balance = await advisor.client.get_balance()
    print(GR(BD(f"  ✔  Connected  ·  Balance: ₹{balance:,.2f}  ·  Mode: {mode}")))
    print()

    # ── Initialise stake engine ───────────────────────────────────────────────
    engine = StakeEngine(mm_cfg, balance)

    _sep(CY)
    print(BD(WT("  Session started")))
    print(f"  Pairs   : {CY(', '.join(pairs))}")
    print(f"  Duration: {YL(str(session_min))} min  ·  Interval: {YL(str(interval_s))} s")
    print(f"  Mode    : {(GR if mode=='DEMO' else RD)(BD(mode))}")
    _sep(CY)
    print()

    session_end = time.time() + session_min * 60
    trade_n     = 0

    try:
        while time.time() < session_end:
            # ── Check stop conditions ─────────────────────────────────────────
            stop, reason = engine.should_stop()
            if stop:
                print()
                print(YL(BD(f"  ■  Session stopped: {reason}")))
                break

            # ── Print live status line ────────────────────────────────────────
            remaining = int(session_end - time.time())
            mins, secs = divmod(remaining, 60)
            print(f"\r{engine.status_line()}  {DM(f'time left {mins:02d}:{secs:02d}')}  ", end="", flush=True)

            # ── Trade each pair ───────────────────────────────────────────────
            for pair in pairs:
                stake = engine.next_stake()
                if stake <= 0:
                    continue

                try:
                    result = await advisor.trade(
                        asset=pair,
                        duration=cfg.quotex.duration,
                    )
                except Exception as exc:
                    log.warning("Trade error %s: %s", pair, exc)
                    continue

                if result is None:
                    continue

                trade_n += 1
                payout_dec = 1.0 + (result.get("payout", 0) / stake if stake > 0 else 0.8)
                engine.record(result["won"], stake, payout_dec)

                # ── Print trade result ────────────────────────────────────────
                won_str = GR(BD("WIN  ")) if result["won"] else RD(BD("LOSS "))
                dir_str = GR("▲ CALL") if result["direction"] == "call" else RD("▼ PUT ")
                print(f"\n  [{trade_n:>3}]  {WT(pair):20}  {dir_str}  {YL(f'₹{stake:.2f}'):>10}  "
                      f"{won_str}  {CY(f'₹{engine.balance:,.2f}')}"
                      + (f"  {BL(result.get('regime',''))}" if result.get("regime") else ""))

                # Log to file
                _log_trade(trade_n, pair, result["direction"],
                            stake, result["won"], engine.balance)

                # ── Check stop again after trade ──────────────────────────────
                stop, reason = engine.should_stop()
                if stop:
                    print()
                    print(YL(BD(f"  ■  {reason}")))
                    break

            await asyncio.sleep(max(1, interval_s - len(pairs) * 2))

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print()
        print(YL("  Interrupted by user."))
    finally:
        print()
        _session_summary(engine, trade_n)
        await advisor.disconnect()


def _log_trade(n, pair, direction, stake, won, balance):
    try:
        with open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
            ts  = time.strftime("%H:%M:%S")
            res = "WIN" if won else "LOSS"
            f.write(f"[{ts}] #{n:>3} {pair:<22} {direction.upper():<5} "
                    f"₹{stake:>8.2f}  {res:<5}  bal=₹{balance:,.2f}\n")
    except Exception:
        pass


def _session_summary(engine: StakeEngine, n_trades: int) -> None:
    _sep(CY)
    tot = engine.wins + engine.losses
    wr  = engine.wins / tot * 100 if tot > 0 else 0.0
    pnl_colour = GR if engine.total_profit >= 0 else RD

    print(BD(WT("  SESSION SUMMARY")))
    print()
    _row("Trades",          str(n_trades))
    _row("Wins / Losses",   f"{engine.wins} / {engine.losses}")
    _row("Win Rate",        f"{wr:.1f}%",  vc=GR if wr >= 55 else RD)
    _row("Net P&L",         f"₹{engine.total_profit:+.2f}", vc=pnl_colour)
    _row("End Balance",     f"₹{engine.balance:,.2f}")
    _sep(CY)


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    _banner()

    # ── 1. Account ────────────────────────────────────────────────────────────
    account = _select_account()

    # ── 2. Mode ───────────────────────────────────────────────────────────────
    mode = _select_mode(account)

    # ── 3. Pairs ──────────────────────────────────────────────────────────────
    pairs = _select_pairs(account)

    # ── 4. Connect (temporary, for data + balance read) ───────────────────────
    _box("CONNECTING", CY)
    os.environ["QUOTEX_EMAIL"]    = account["email"]
    os.environ["QUOTEX_PASSWORD"] = account["password"]
    os.environ["QUOTEX_MODE"]     = mode.lower()

    print(YL("  Connecting to Quotex to check balance…"))
    from dart_quotex.advisor import AIAdvisor
    advisor_tmp = AIAdvisor(use_robust_client=True)
    try:
        await advisor_tmp.connect()
        balance = await advisor_tmp.client.get_balance()
        print(GR(BD(f"  ✔  Balance: ₹{balance:,.2f}")))
    except Exception as exc:
        print(RD(f"  ✖  Connection failed: {exc}"))
        print(DM("     Using balance = 1000 for setup."))
        balance = 1000.0
    finally:
        await advisor_tmp.disconnect()

    # ── 5. Data check + harvest ───────────────────────────────────────────────
    advisor_harvest = AIAdvisor(use_robust_client=True)
    await advisor_harvest.connect()
    await _ensure_data(advisor_harvest, pairs)
    await advisor_harvest.disconnect()

    # ── 6. Backtest ───────────────────────────────────────────────────────────
    if not await _run_backtest(pairs):
        print(YL("  Pipeline aborted."))
        return

    # ── 7. Money management ───────────────────────────────────────────────────
    mm_cfg = _setup_money_management(account, balance)

    # ── 8. Session settings ───────────────────────────────────────────────────
    _box("SESSION SETTINGS", CY)
    session_min = _input_int("Session length (minutes)", 5,  480, 60)
    interval_s  = _input_int("Interval between scans (s)", 10, 300, 65)

    # ── 9. Final confirmation ─────────────────────────────────────────────────
    print()
    _sep(CY)
    print(BD(WT("  READY TO START")))
    print()
    _row("Account",   account["nickname"])
    _row("Email",     account["email"])
    _row("Mode",      mode,                 vc=GR if mode == "DEMO" else RD)
    _row("Pairs",     str(len(pairs)))
    _row("Session",   f"{session_min} min")
    _row("Interval",  f"{interval_s} s")
    _row("MM Method", {"1":"Masaniello","2":"Compounding","3":"Fixed"}.get(mm_cfg.get("method","3"),"?"))
    _sep(CY)
    print()

    if not _confirm(f"  Start {(GR if mode=='DEMO' else RD)(BD(mode))} session now?"):
        print(YL("  Cancelled."))
        return

    # ── 10. Live trading ──────────────────────────────────────────────────────
    await _live_session(account, mode, pairs, mm_cfg, session_min, interval_s)

    print()
    print(GR(BD("  All done.  Models saved.  See trade_log.txt for history.")))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YL('  Interrupted.')}")
        sys.exit(0)
