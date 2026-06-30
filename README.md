# Python Data Analysis Utilities — AI-Augmented Data Analysis Portfolio

## What This Does

A production-grade Python toolkit for end-to-end data analysis — from messy CSV ingestion to actionable business insights. Built as Week 2 of a 12-week AI-augmented data analysis curriculum.

**Business value demonstrated:** Analyzed 100K+ orders from the Brazilian Olist e-commerce platform and found that late-delivery rates more than double during the Nov–Mar peak season (8% → 21%), and that a small, identifiable subset of sellers — 47% of the lowest performers — fall below a 70% on-time delivery floor. Both findings translate directly into concrete operational fixes (seasonal capacity planning, a seller scorecard system).

## Key Findings

Full analysis with implications and recommendations: [`notebooks/week2_eda.ipynb`](notebooks/week2_eda.ipynb)

- Late deliveries spike sharply during peak season, climbing from ~14% (Nov 2017) to ~21% (Mar 2018)
- Order growth plateaued after Nov 2017 and hasn't broken its prior peak since
- 47% of the worst-performing sellers (20+ orders) fall below a 70% on-time delivery floor
- Even top-ranked sellers average 21–31 days per delivery — well above the platform median of 10 days
- Order activity concentrates between 10am–9pm, peaking 11am–4pm; overnight hours are largely idle

## Portfolio Deliverables

| Deliverable            | Description                                                               | Link                        |
| ---------------------- | ------------------------------------------------------------------------- | --------------------------- |
| EDA Notebook           | Full analysis — 5 visualizations (8 chart panels) and 5 business findings | `notebooks/week2_eda.ipynb` |
| Data Utilities Library | 36 typed, documented Python functions across 4 modules                    | `src/`                      |
| Cleaning Pipeline      | Auto-encoding detection, null audit, memory optimization                  | `src/data_cleaner.py`       |
| EDA Pipeline           | Method-chained transformation pipeline                                    | `src/eda_pipeline.py`       |
| Analysis Functions     | GroupBy and time-series analysis functions                                | `src/data_analyzer.py`      |
| Loading Layer          | Robust multi-format CSV ingestion                                         | `src/data_loader.py`        |
| Test Suite             | Unit tests covering cleaning and analysis modules                         | `tests/`                    |

## Quick Start (macOS)

```bash
git clone https://github.com/AHTISHAM327/python-data-utils.git
cd python-data-utils
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Download Olist dataset to data/raw/ (see notebooks/week2_eda.ipynb Section 1)
jupyter notebook notebooks/week2_eda.ipynb
```

## Repository Structure

```
python-data-utils/
├── src/                          # All analysis modules
│   ├── data_loader.py            # Robust CSV loading with encoding detection (11 functions)
│   ├── data_cleaner.py           # Null audit, datetime conversion, memory optimization (7 functions)
│   ├── data_analyzer.py          # GroupBy + time-series analysis functions (14 functions)
│   ├── eda_pipeline.py           # Method-chained end-to-end pipeline (4 functions)
│   └── reporter.py               # Formatted report generation
├── notebooks/
│   └── week2_eda.ipynb           # Portfolio EDA notebook (100K+ rows, 5 visualizations)
├── tests/                        # Unit tests for all modules
│   ├── test_all.py
│   ├── test_analyzer.py
│   └── test_cleaning_pipeline.py
└── data/                         # raw/ and processed/ (not committed — see .gitignore)
```

## Professional Standards

All source code in `src/` follows:

- **Type hints** on every function (`mypy src/` → 0 errors)
- **Google-style docstrings** (Args, Returns, Raises, Example)
- **Structured logging** (`logging` module, no `print()` statements)
- **Error handling** (specific exceptions, never bare `except:`)
- **PEP 8 compliance** (`black` + `flake8` → 0 warnings)

## Key Technical Skills Demonstrated

- Pandas method chaining (`.assign()`, `.pipe()`, `.query()`)
- Multi-level `GroupBy` with named aggregation
- Time-series resampling (`.resample("ME")`) and rolling windows
- Memory optimization via `category` dtype conversion
- Automatic encoding detection
- Professional module architecture (`src/` package with `__init__.py`)

## About

Built as part of a 12-week AI-augmented data analysis curriculum targeting Upwork freelance clients at $50–$120/hr. Week 3 adds SQL (SQLite + SQLAlchemy). Week 6 adds OpenAI API integration for natural-language insight generation.
