"""market_series: REAL market/economic data only, from FRED (free CSV endpoint) or Yahoo Finance (the chart API
that yfinance itself uses; yfinance is used instead when it is installed). Nothing is ever synthesized: a failed
fetch returns the error. Every series is cached as CSV under data/market/ with a JSON sidecar recording the source
URL, retrieval time (UTC) and SHA-256 of the CSV, and every call returns that provenance.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import urllib.parse
from pathlib import Path

from tools.mesh_mcp.common import cap
from tools.mesh_mcp.literature import Fetcher

REPO = Path(__file__).resolve().parents[2]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
_ID = re.compile(r"^[A-Za-z0-9.^=_-]{1,32}$")


def cache_dir() -> Path:
    d = Path(os.environ.get("MESH_MARKET_DIR", str(REPO / "data" / "market")))
    d.mkdir(parents=True, exist_ok=True)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n", encoding="utf-8")  # cached third-party data never gets committed
    return d


def _date(s: str, name: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(s))
    except ValueError:
        raise ValueError("%s must be YYYY-MM-DD, got %r" % (name, s)) from None


def fetch_fred(series_id: str, start: dt.date, end: dt.date, fetch) -> tuple[list[tuple[str, float]], str]:
    url = FRED_URL + "?" + urllib.parse.urlencode({"id": series_id, "cosd": start.isoformat(),
                                                   "coed": end.isoformat()})
    text = fetch(url).decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) < 2 or "date" not in rows[0][0].lower():
        raise RuntimeError("FRED did not return a CSV for %r (unknown series id?)" % series_id)
    out = []
    for r in rows[1:]:
        if len(r) >= 2 and r[1] not in ("", "."):  # "." marks a missing observation in FRED CSVs
            out.append((r[0], float(r[1])))
    return out, url


def fetch_yahoo(ticker: str, start: dt.date, end: dt.date, fetch) -> tuple[list[tuple[str, float]], str]:
    p1 = int(dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc).timestamp()) + 86400
    url = YAHOO_URL + urllib.parse.quote(ticker) + "?" + urllib.parse.urlencode(
        {"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"})
    data = json.loads(fetch(url))
    chart = data.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError("Yahoo: %s" % (chart["error"].get("description") or chart["error"]))
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        raise RuntimeError("Yahoo returned no prices for %r in %s..%s" % (ticker, start, end))
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    close = ((res.get("indicators") or {}).get("quote") or [{}])[0].get("close")
    vals = adj or close
    out = [(dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), float(v))
           for t, v in zip(res["timestamp"], vals) if v is not None]
    return out, url


def _yfinance(ticker, start, end):
    import yfinance as yf  # optional; only used when installed
    df = yf.download(ticker, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no prices for %r" % ticker)
    col = df["Close"]
    col = col.iloc[:, 0] if getattr(col, "ndim", 1) > 1 else col
    return [(i.date().isoformat(), float(v)) for i, v in col.dropna().items()], "yfinance:%s" % ticker


def market_series(ticker: str | None = None, fred_id: str | None = None, start: str | None = None,
                  end: str | None = None, fetch=None, refresh: bool = False) -> str:
    if bool(ticker) == bool(fred_id):
        raise ValueError("pass exactly one of ticker= (Yahoo Finance, e.g. SPY, ^GSPC, EURUSD=X) or fred_id= "
                         "(FRED, e.g. DGS10, CPIAUCSL)")
    source, series = ("yahoo", str(ticker).strip()) if ticker else ("fred", str(fred_id).strip())
    if not _ID.match(series):
        raise ValueError("invalid %s identifier %r" % ("ticker" if ticker else "FRED id", series))
    if not start:
        raise ValueError("start is required (YYYY-MM-DD)")
    s, e = _date(start, "start"), _date(end or dt.date.today().isoformat(), "end")
    if e < s:
        raise ValueError("end is before start")
    key = "%s_%s_%s_%s" % (source, re.sub(r"[^A-Za-z0-9]", "_", series), s, e)
    d = cache_dir()
    csv_path, meta_path = d / (key + ".csv"), d / (key + ".json")
    cached = csv_path.exists() and meta_path.exists() and not refresh
    if cached:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        body = csv_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != meta.get("sha256"):
            cached = False  # tampered or partial cache: fetch again
    if not cached:
        fetch = fetch or Fetcher()
        try:
            if source == "fred":
                rows, url = fetch_fred(series, s, e, fetch)
            else:
                try:
                    rows, url = fetch_yahoo(series, s, e, fetch)
                except Exception as first:  # noqa: BLE001 - try yfinance when installed, else report this error
                    try:
                        import yfinance  # noqa: F401
                    except ImportError:
                        raise first from None
                    rows, url = _yfinance(series, s, e)
        except Exception as ex:  # noqa: BLE001
            return cap("FETCH FAILED (%s %s %s..%s): %s: %s\nNo data returned; nothing was synthesized."
                       % (source, series, s, e, type(ex).__name__, str(ex)[:300]))
        if not rows:
            return cap("FETCH FAILED (%s %s %s..%s): the source returned no observations in this range.\n"
                       "No data returned; nothing was synthesized." % (source, series, s, e))
        body = ("date,value\n" + "".join("%s,%r\n" % r for r in rows)).encode("utf-8")
        csv_path.write_bytes(body)
        meta = {"series": series, "source": source, "url": url, "start": s.isoformat(), "end": e.isoformat(),
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": hashlib.sha256(body).hexdigest(), "rows": len(rows),
                "field": "adjusted close" if source == "yahoo" else "value"}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rows = [line.split(",") for line in body.decode("utf-8").splitlines()[1:]]
    vals = [float(v) for _, v in rows]
    lines = ["%s %s (%s), %d observations %s .. %s" % (meta["source"].upper(), meta["series"], meta["field"],
                                                         len(rows), rows[0][0], rows[-1][0]),
             "PROVENANCE: url=%s retrieved_utc=%s sha256=%s cached=%s" % (meta["url"], meta["retrieved_utc"],
                                                                          meta["sha256"], cached),
             "CSV: %s" % csv_path,
             "min %.6g  max %.6g  first %.6g  last %.6g" % (min(vals), max(vals), vals[0], vals[-1]),
             "first rows: " + "; ".join("%s %s" % tuple(r) for r in rows[:5]),
             "last rows: " + "; ".join("%s %s" % tuple(r) for r in rows[-5:]),
             "Pass the CSV path to stats_test/causal_analysis; cite the provenance line."]
    return cap("\n".join(lines))
