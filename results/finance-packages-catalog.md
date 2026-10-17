# Finance / Accounting / VC / Business-Plan Package Catalog

Date: 2026-10-04. Read-only research. NOTHING was installed, built, cloned or downloaded.
Verification method: registry JSON APIs only (pypi.org, registry.npmjs.org + api.npmjs.org, crates.io).
"Verified" = the package exists on the registry, with the version/license/date shown at query time.
Items marked "(from memory)" were NOT confirmed this session - audit before trusting.
Popularity: PyPI = downloads/month (pypistats, where the call returned data), npm = downloads/week, crates = all-time downloads.

Ground rules applied: no package here scrapes paid/licensed data or bypasses a provider's terms.
Every install decision still needs the user's own audit (lockfile, hash pinning, check maintainer/repo, use a venv/container).

---------------------------------------------------------------------
## 1. Financial data APIs / databases

HONEST SUMMARY: serious fundamentals, estimates, private-company and VC deal data are PAID/LICENSED
(PitchBook, Crunchbase API, S&P Capital IQ, FactSet, Refinitiv/LSEG, Bloomberg). Free sources cover
public-company filings (SEC EDGAR), macro data (FRED, ECB, World Bank, IMF) and delayed/limited prices.
Free tiers of commercial APIs are rate-limited and usually personal/non-commercial use only. Do not
work around those limits or scrape the paid sites.

| Source | Cost / access | Wrapper (registry-verified) | Notes / safety |
|---|---|---|---|
| SEC EDGAR (data.sec.gov, XBRL "company facts" JSON) | FREE, no key. Must send a descriptive User-Agent with contact email; ~10 req/s fair-use cap | `edgartools` (PyPI 5.60.0, MIT, uploaded 2026-10-02, very active); `sec-edgar-downloader` (PyPI 5.1.0, MIT, ~234k/mo); crate `sec-edgar` 1.0.5 (~10k dl, last update 2023 - stale) | Best free source for US public-company financial statements. Official docs: sec.gov/edgar/sec-api-documentation. Prefer calling the JSON API directly or edgartools. |
| FRED (St. Louis Fed) | FREE but needs a free API key (registered account) | `fredapi` (PyPI 0.5.2, license field blank in metadata, last release 2024-05, ~1.1M/mo - stable but quiet); also in `pandas-datareader` | Key = credential, keep in env var, never commit. |
| World Bank / OECD / Eurostat / ECB / IMF | FREE, mostly no key | `pandas-datareader` (PyPI 0.11.1, BSD-3, 2026-06, ~461k/mo) covers several; others via plain HTTP/SDMX | Some pandas-datareader sources break over time (Yahoo/Google readers historically dead) - test each. |
| Alpha Vantage | FREE tier (small daily call cap) + PAID tiers; key required | `alpha_vantage` (PyPI 3.0.0, MIT, last release 2024-07, ~127k/mo - lightly maintained) | Free tier is tight; wrapper is third-party. |
| Financial Modeling Prep (FMP) | Limited FREE tier; PAID for most fundamentals/history; key required | `fmpsdk` (PyPI 20260824.0, BSD, tiny: ~5.7k/mo) | Small user base - verify repo/maintainer before use; calling REST directly is an alternative. |
| Polygon.io | Limited FREE tier; PAID for real-time/depth; key required | `polygon-api-client` (PyPI 1.16.3, MIT, official, ~558k/mo) | Reputable. |
| Finnhub | FREE tier + PAID; key required | `finnhub-python` (PyPI 2.4.29, Apache-2.0, 2026-06); npm `finnhub` 2.0.15 (license field = None in registry metadata - check repo) | Official-looking client. |
| Yahoo Finance (unofficial) | "Free" but NOT an official API; Yahoo ToS limits use to personal; can break or rate-limit without notice | `yfinance` (PyPI 1.7.0, Apache-2.0, ~18.7M/mo); npm `yahoo-finance2` 4.0.2 (MIT, ~928k/wk); crate `yahoo_finance_api` 5.0.0 (~222k dl) | Popular but ToS-gray. Fine for personal prototyping; not a basis for a product or investor-facing numbers. |
| OpenBB Platform | Open-source aggregator; underlying providers each need THEIR OWN keys (many paid) | `openbb` (PyPI 5.0.0, Apache-2.0, uploaded 2026-09-29). npm `openbb` NOT FOUND | Large dependency tree - audit carefully. It is a router, not a free data source. |
| Nasdaq Data Link (ex-Quandl) | Mixed: some free datasets, premium paid; key required | (not registry-checked) | From memory. |
| Plaid | Bank-account aggregation for your own/your users' accounts; sandbox free, production needs approval | `plaid-python` (PyPI 45.0.0, MIT), npm `plaid` 47.0.0 (MIT) | Handles real credentials/tokens - highest-risk item here. Only for deliberate bank-linking. |
| sec-api.io | PAID/limited-free commercial wrapper over EDGAR | PyPI `sec-api` 1.0.36 (MIT), npm `sec-api` 4.0.8 (MIT) | Third-party commercial service; EDGAR itself is free, so you probably do not need it. |

### VC / startup / private-company data
- Crunchbase: API is PAID (enterprise/API licence). No `crunchbase` package on PyPI (NOT FOUND). Do not scrape the site (violates ToS). Limited free web UI exists (from memory - verify).
- PitchBook, CB Insights, Dealroom, Preqin, Affinity: PAID licensed only. Not available via free libraries; do NOT use scrapers.
- Free/legit proxies for startup analysis (from memory, verify each): SEC EDGAR Form D filings (private raise disclosures, free public record); national company registers (UK Companies House has a free REST API with key; Nordic registers such as Norway's Bronnoysund are open or low-cost); USPTO/EPO patent data (free); data the user already pays for, exported manually from their own login.
- Claude Code skills already listed in this environment (not packages) touching this area: `founder-toolkit`, `valuation-reviewer`, `private-credit`, `model-builder`, `pitch-agent`, `chronograph-gp/lp` (needs Chronograph MCP access), `strategy-frameworks`, `lean-pm-skills`. Several agent plugins (model-builder, earnings-reviewer, pitch-agent, market-researcher) reference CapIQ / FactSet / Daloopa MCP servers = PAID licensed data the user must already hold.

---------------------------------------------------------------------
## 2. Financial analysis / modeling libraries

### Python (all registry-verified)
| Package | What | License | Popularity / freshness | Note |
|---|---|---|---|---|
| pandas | tabular data core | BSD | 3.0.6, 2026-09-30, ~543M/mo | Standard. |
| numpy | arrays | BSD-3 (+ bundled permissive) | 2.5.3, 2026-09-06 | Standard. |
| scipy | optimization/stats | BSD | 1.18.1 | Standard. |
| statsmodels | regression/time series | BSD-3 | 0.15.0 | Standard. |
| numpy-financial | NPV, IRR, PMT etc. (split out of numpy) | BSD-3 | 1.1.0, 2026-09-15 | Good for business-plan math. |
| financetoolkit | ratios, statement analysis, 150+ metrics; pulls data via FMP (needs FMP key) | MIT | 2.2.1, 2026-10-01 | Active. Analysis free, data fetch needs a key. |
| QuantLib | derivatives/curves/bond pricing (C++ bindings) | BSD-3 | 1.43, 2026-07 | Heavy; overkill unless pricing instruments. |
| riskfolio-lib | portfolio optimisation | BSD-3 | 7.4.0, 2026-10-04 | Active. |
| skfolio | scikit-learn style portfolio optimisation | BSD | 1.4.11, 2026-10-03 | Active. |
| PyPortfolioOpt | mean-variance etc. | MIT | 1.6.0, 2026-02 | Fine. |
| quantstats | portfolio/return analytics, tear sheets | Apache-2.0 | 0.0.86, ~576k/mo | Pre-1.0 versioning, still active. |
| py-moneyed | Money/Currency types | BSD | 3.0, last upload 2022-11 | Stable but not recently updated. |
| forex-python | FX rates | MIT | 1.9.2, 2025-05 | Relies on external free rate endpoints; check source reliability. |
| beancount | plain-text double-entry accounting | GPL-2.0-only | 3.2.3, 2026-05 | GPL - fine as a tool; careful if embedding in closed code. |

Not recommended / flagged: `pycel` (GPLv3, last release 2021, beta - abandoned); `xlcalculator` (MIT, last release 2023-02 - stale). `formulas` (EUPL 1.1, 1.3.4, 2026-03) is the better-maintained Excel-formula evaluator but EUPL is copyleft - read terms. `pybliometrics` (Scopus) is academic, off-topic.

### JavaScript / TypeScript (registry-verified)
| Package | What | License | Weekly dl | Note |
|---|---|---|---|---|
| decimal.js / big.js / bignumber.js | arbitrary precision decimals | MIT | decimal.js ~110M | Use for money math; never floats. |
| dinero.js 2.0.2 | immutable money objects | MIT | ~776k | v2 reasonable. |
| currency.js 2.0.4 | tiny currency math | MIT | ~906k | Simple. |
| financial 0.2.4 | IRR/NPV/PMT (numpy-financial port) | MIT | ~40k | Small, 0.x. |
| technicalindicators 3.1.0 | TA indicators | MIT | ~57k | Niche. |
| lightweight-charts 5.2.1 | TradingView financial charts | Apache-2.0 | ~1.3M | Reputable; has an attribution requirement - read license/NOTICE. |
| chart.js 4.5.1 / d3 7.9.0 | general charts | MIT / ISC | - | Standard. |
| hyperformula 3.4.0 | headless spreadsheet formula engine | GPL-3.0-only (commercial licence sold separately) | ~447k | GPL-3: license-check before use in anything distributed. |
| ledger-cli 0.2.0 | wrapper for ledger CLI | MIT | - | Obscure; low value. |

NOT FOUND on npm: `alpha-vantage`, `openbb`, `accounting.js`.

### Rust crates (registry-verified; downloads = all-time)
| Crate | What | Dl | Last update | Note |
|---|---|---|---|---|
| rust_decimal 1.43.0 | exact decimal type (+ `rust_decimal_macros`) | 152.7M | 2026-09 | Standard for money. |
| bigdecimal 0.4.11 | arbitrary-precision decimal | 124M | 2026-09 | Alternative. |
| rusty-money 0.5.1 | Money/currency types | 1.0M | 2026-09 | Active. |
| iso_currency 0.7.1 | ISO 4217 codes | 2.3M | 2026-09 | Active. |
| rust_finprim 0.5.1 | time-value-of-money, bonds, depreciation | 12.8k | 2026-07 | Active but small/young - audit. |
| financial 1.1.5 | Excel-like financial fns (PMT, NPV...) | 53k | 2023-02 | Stale. |
| quantlib 0.1.0 | near-empty placeholder | 3.3k | 2020-03 | ABANDONED. Do not use. |
| ta 0.5.0 | technical indicators | 363k | 2021-06 | Stale but widely used. |
| polars 0.55.2 | DataFrames (pandas alternative) | 15.2M | 2026-08 | Active. |
| yahoo_finance_api 5.0.0 | unofficial Yahoo client | 222k | 2026-08 | ToS-gray (see above). |
| alpha_vantage 0.11.0 | Alpha Vantage client | 27k | 2024-09 | Small. |
| fred-rs 0.1.1 | FRED client | 8.6k | 2020-02 | ABANDONED. |
| barter 0.14.0 | algo-trading framework | 92k | 2026-08 | Trading, not accounting. |
| rust-ledger 0.8.0 | ledger-style | 27k | 2021-09 | Stale. |

NOT FOUND: `finance-rs`, `edgar`.
Rust takeaway: great primitives (decimal, money, dataframes, xlsx/pdf writers) but no mature "financial analysis" or data-API ecosystem. For modeling, Python is far ahead.

Other languages (from memory, not registry-checked): R `quantmod`/`tidyquant`; Go `shopspring/decimal`; Java OpenGamma Strata; C# EPPlus (commercial licence terms for commercial use - check).

---------------------------------------------------------------------
## 3. Accounting / financial statements / document + spreadsheet generation

| Package | Eco | Purpose | License | Popularity / freshness | Note |
|---|---|---|---|---|---|
| openpyxl 3.1.5 | PyPI | read/write .xlsx incl. formulas | MIT | ~280M/mo; last release 2024-06 | Stable de-facto standard; slow release cadence. |
| XlsxWriter 3.2.9 | PyPI | write-only .xlsx with charts/formatting | BSD | 2025-09 | Best for polished statement workbooks. |
| python-docx 1.2.0 | PyPI | .docx | MIT | ~94M/mo | Good for plan documents. |
| python-pptx 1.0.2 | PyPI | .pptx | MIT | last release 2024-08 | Pitch decks; quiet maintenance. |
| reportlab 5.0.1 | PyPI | PDF generation | BSD (open-source edition) | 2026-08 | Mature. |
| weasyprint 70.0 | PyPI | HTML/CSS to PDF | BSD | 2026-09 | Nice for statements from HTML templates; native deps. |
| fpdf2 2.8.9 | PyPI | simple PDF | LGPL-3.0 | 2026-09 | LGPL - fine as a dependency. |
| tabula-py 2.10.0 | PyPI | extract tables from PDF (needs Java) | MIT | 2024-10 | For reading existing statements. |
| beancount 3.2.3 | PyPI | double-entry bookkeeping engine | GPL-2.0 | 2026-05 | Real accounting engine, plain text. |
| exceljs 4.4.0 | npm | xlsx read/write | MIT | ~19.9M/wk | Older release but widely used. |
| xlsx (SheetJS) 0.18.5 | npm | spreadsheets | Apache-2.0 | ~15.1M/wk | CAUTION: npm copy is stale (SheetJS distributes newer builds from its own CDN; old npm versions have known advisories - from memory). Prefer exceljs, or audit SheetJS's official build. |
| docx 9.8.1 | npm | .docx generation | MIT | ~7.7M/wk | Good. |
| pdf-lib 1.17.1 | npm | PDF create/modify | MIT | ~19M/wk | Old version number but ubiquitous. |
| pdfkit 0.20.2 / jspdf 4.2.1 | npm | PDF generation | MIT | - | jsPDF has had security advisories historically (from memory) - pin and audit. |
| rust_xlsxwriter 0.99.1 | crates | xlsx writer (by the XlsxWriter author) | (licence not captured; from memory MIT/Apache) | 4.9M dl, 2026-09 | Excellent, active. |
| calamine 0.36.1 | crates | read xlsx/ods | (from memory MIT) | 14.2M dl | Reader only. |
| umya-spreadsheet 3.1.0 | crates | read/write xlsx | (from memory MIT) | 1.2M dl | Active. |
| printpdf 0.12.8 | crates | PDF gen | (from memory MIT) | 3.2M dl | Active. |
| docx-rs 0.4.22 | crates | .docx gen | (from memory MIT) | 3.6M dl | Active. |
| genpdf 0.2.0 | crates | high-level PDF | - | 613k dl, 2021 | Stale. |

Accounting SOFTWARE/APIs (not libraries, need accounts, paid): QuickBooks, Xero, Fiken, Visma, Merit Aktiva, NetSuite. Skills already listed here for several (intuit-quickbooks, merit-aktiva-skills, netsuite-*, statutory-reporting, month-end-closer, gl-reconciler). They operate on the user's own authenticated company data.
Open-source accounting/ERP apps (apps, not libs; from memory): ERPNext, Odoo Community, GnuCash, Firefly III, Akaunting, ledger/hledger/beancount.
Structured reporting: XBRL/iXBRL tooling (e.g. Arelle - from memory, not registry-checked). `edgartools` parses XBRL facts.

---------------------------------------------------------------------
## 4. Business-plan / startup tooling - blunt assessment

There is essentially NO real library category here. What exists:
- Templates and spreadsheets (free: SBA/SCORE templates, Strategyzer canvas, YC and VC memo outlines; paid closed SaaS like LivePlan/Upmetrics-type tools).
- LLM-generated documents (where Claude helps most): plan narrative, market sizing logic, unit economics, risk sections.
- Real modeling = spreadsheets. The honest "stack" is a generated xlsx (XlsxWriter/openpyxl) with live formulas the user can audit + numpy-financial for IRR/NPV + docx/pptx for narrative.
- Startup-finance formulas people reimplement themselves: runway/burn, CAC/LTV, cohort retention, cap table/dilution, SAFE/convertible math, VC-method valuation, DCF. No dominant maintained library for cap-table math was found or confirmed; treat any obscure PyPI/npm "cap table" or "startup valuation" package as unvetted (not checked here).
- Claude skills already listed in this environment that cover the planning side (not installs): `strategy-frameworks`, `founder-toolkit`, `lean-pm-skills` (business-case, financial-analysis, pricing), `revops` (arr-bridge, funnel math, revenue-plan), `ycofficehour`, `pitch-agent`, `model-builder` (DCF/LBO/3-statement; some depend on paid CapIQ/FactSet MCPs).
Do not pad this with vague "AI business plan generator" packages: unvetted, often thin wrappers around an LLM API that want your key.

---------------------------------------------------------------------
## Safety / red flags summary
- Credential-bearing: Plaid (bank tokens), and every API-key client (Alpha Vantage, FMP, Polygon, Finnhub, FRED, sec-api, OpenBB providers). Keys in env vars / secret manager only; never in repo or prompts.
- ToS-gray: yfinance / yahoo-finance2 / yahoo_finance_api (unofficial). Personal prototyping only.
- Licensed-data trap: Crunchbase, PitchBook, CapIQ, FactSet, Daloopa, Chronograph require the user's own paid licence. Scraping them is out of scope and not recommended.
- Copyleft: hyperformula (GPL-3 unless commercial licence), beancount (GPL-2), pycel (GPLv3), formulas (EUPL). fpdf2 is LGPL.
- Stale/abandoned: Rust `quantlib` (2020), `fred-rs` (2020), `genpdf` (2021), `ta` (2021), `financial` crate (2023), `rust-ledger` (2021); PyPI `pycel` (2021), `xlcalculator` (2023); npm `xlsx` (stale npm build).
- Low popularity / harder to audit: `fmpsdk` (~5.7k/mo), `rust_finprim`, `ledger-cli` (npm).
- Big dependency trees: `openbb`, `QuantLib` - install only in isolated environments.

---------------------------------------------------------------------
## Recommended starter set (mostly free, reputable)

Core (no API key):
1. Python: `pandas`, `numpy`, `scipy`, `numpy-financial`, `statsmodels`
2. Output: `XlsxWriter` (new workbooks) + `openpyxl` (read/edit), `python-docx`, `python-pptx`, `reportlab` or `weasyprint`
3. `edgartools` (SEC EDGAR - free, needs only a contact email in the User-Agent)
4. `pandas-datareader` for World Bank/OECD/Eurostat-type public series
5. `financetoolkit` for ratio analysis once data is loaded (analysis free; default data fetch uses FMP)
6. Optional: `quantstats` / `riskfolio-lib` for portfolio work
7. JS/TS: `decimal.js` (or `dinero.js`), `exceljs`, `docx`, `lightweight-charts` / `chart.js`
8. Rust: `rust_decimal`, `rusty-money`, `rust_xlsxwriter`, `calamine`, `polars`, `printpdf`

Needs the USER to obtain their own key / account:
- FRED (free key) - `fredapi`
- Alpha Vantage / FMP / Polygon / Finnhub (free tiers with keys; paid for real use)
- OpenBB (each underlying provider needs its own key)
- Plaid (bank linking; sandbox free, production approval needed)
- Crunchbase, PitchBook, CapIQ, FactSet, CB Insights (paid licences; no legitimate free bulk route)
- Accounting systems (QuickBooks/Xero/etc.) via the user's own OAuth

Suggested next step: audit and pin only the "Core" list in an isolated venv/container, then add keyed data sources one at a time.
