"""Lifelong ingestion of Claude Code transcripts for neural-mesh (owner: ingestor; contract: LLM-DESIGN.md "Pipeline").

Reads every transcript incrementally, scrubs it, and writes git-ignored data under data/lifelong/:
outcomes.jsonl (router feedback), sft.jsonl (fine-tuning pairs), judge.jsonl (local-LLM grades, cache), memory/ (credited knowledge).
"""
__version__ = "0.1.0"
