#!/usr/bin/env python3
"""
scripts/build_tcn_library.py
Build the TCN spoofing-detection pattern library offline.

Run this ONCE after harvesting historical data:
    python scripts/build_tcn_library.py --asset EURUSD_OTC --epochs 20

The trained library is saved to models/ and loaded automatically
at runtime when ENABLE_TCN_SPOOFING=true.
"""
import argparse, asyncio, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_tcn")

async def main(args: argparse.Namespace) -> None:
    from dart_quotex.config import cfg
    from dart_quotex.data.database import Database
    from dart_quotex.manipulation.tcn_spoofing import TCNSpoofingDetector

    db  = Database(cfg.data.db_path)
    df  = db.get_candles(args.asset, args.granularity, limit=args.candles)

    if len(df) < 200:
        log.error("Not enough candles (%d). Run harvest first:\n"
                  "  python main.py harvest --asset %s", len(df), args.asset)
        sys.exit(1)

    log.info("Building TCN library: %d candles, %d epochs", len(df), args.epochs)

    det = TCNSpoofingDetector(
        seq_len=args.seq_len,
        threshold=args.threshold,
        n_patterns=args.patterns,
    )
    det.build_pattern_library(df, epochs=args.epochs)

    save_path = Path(args.model_dir)
    det.save(save_path)

    log.info("Done. Library saved to %s (%d patterns).",
             save_path, len(det._library))
    log.info("Activate with: ENABLE_TCN_SPOOFING=true in .env")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build TCN spoofing pattern library")
    p.add_argument("--asset",       default="EURUSD_OTC")
    p.add_argument("--granularity", type=int,   default=60)
    p.add_argument("--candles",     type=int,   default=5000)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--seq-len",     type=int,   default=30)
    p.add_argument("--threshold",   type=float, default=0.80)
    p.add_argument("--patterns",    type=int,   default=500)
    p.add_argument("--model-dir",   default="models")
    asyncio.run(main(p.parse_args()))
