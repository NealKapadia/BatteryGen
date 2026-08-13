"""Molport membership index.

Two structures, built during preprocessing from the *same* canonical SMILES used
for training:

* a pure-Python **Bloom filter** (fast, ~no false negatives) for the
  ``--molport-only`` generation toggle and cheap ``in_molport`` checks, and
* a **SQLite** table ``molecules(smiles PRIMARY KEY, molport_id)`` for the exact
  answer + the Molport-xxx id.

The Bloom filter is only a pre-filter: a positive is confirmed against SQLite, so
its false positives never produce a wrong "yes". Canonicalization lives in
``data.canonical_smiles`` so generated and catalog molecules are compared the
same way.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from pathlib import Path
from typing import Iterable, Optional

from batterygen.core import config


_BLOOM_MAGIC = b"MVBLOOM1"


# --------------------------------------------------------------------------- #
# Bloom filter
# --------------------------------------------------------------------------- #
class BloomFilter:
    """Classic Bloom filter with Kirsch–Mitzenmacher double hashing."""

    def __init__(self, n_items: int, fp_rate: float = 1e-4):
        n = max(1, int(n_items))
        m = math.ceil(-n * math.log(fp_rate) / (math.log(2) ** 2))
        # round up to a whole byte
        self.m_bits = max(8, ((m + 7) // 8) * 8)
        self.k = max(1, round(self.m_bits / n * math.log(2)))
        self.bits = bytearray(self.m_bits // 8)
        self.n_added = 0

    # -- hashing -------------------------------------------------------------
    def _indices(self, key: str):
        d = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(d[:8], "little")
        h2 = int.from_bytes(d[8:], "little") | 1  # keep odd -> good stride
        for i in range(self.k):
            yield (h1 + i * h2) % self.m_bits

    # -- API -----------------------------------------------------------------
    def add(self, key: str) -> None:
        for idx in self._indices(key):
            self.bits[idx >> 3] |= 1 << (idx & 7)
        self.n_added += 1

    def __contains__(self, key: str) -> bool:
        return all(self.bits[idx >> 3] & (1 << (idx & 7)) for idx in self._indices(key))

    def update(self, other: "BloomFilter") -> None:
        """OR another filter of identical geometry into this one (for merging)."""
        if other.m_bits != self.m_bits or other.k != self.k:
            raise ValueError("cannot merge Bloom filters of different geometry")
        for i in range(len(self.bits)):
            self.bits[i] |= other.bits[i]
        self.n_added += other.n_added

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(_BLOOM_MAGIC)
            f.write(struct.pack("<QIQ", self.m_bits, self.k, self.n_added))
            f.write(self.bits)

    @classmethod
    def load(cls, path: Path) -> "BloomFilter":
        with open(path, "rb") as f:
            if f.read(len(_BLOOM_MAGIC)) != _BLOOM_MAGIC:
                raise ValueError(f"{path} is not a batterygen Bloom file")
            m_bits, k, n_added = struct.unpack("<QIQ", f.read(20))
            bits = bytearray(f.read())
        obj = cls.__new__(cls)
        obj.m_bits, obj.k, obj.n_added, obj.bits = m_bits, k, n_added, bits
        return obj


# --------------------------------------------------------------------------- #
# SQLite exact index
# --------------------------------------------------------------------------- #
def open_db(path: Path = config.MOLPORT_DB, *, write: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    if write:
        conn.execute("CREATE TABLE IF NOT EXISTS molecules (smiles TEXT PRIMARY KEY, molport_id TEXT)")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")     # bulk load; we re-open read-only after
    return conn


def add_many(conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    """rows: iterable of (canonical_smiles, molport_id)."""
    conn.executemany("INSERT OR IGNORE INTO molecules (smiles, molport_id) VALUES (?, ?)", rows)


class MolportIndex:
    """Read-side convenience wrapper used by generate.py / search.py."""

    def __init__(self):
        self.bloom: Optional[BloomFilter] = None
        self.conn: Optional[sqlite3.Connection] = None
        if config.MOLPORT_BLOOM.exists():
            self.bloom = BloomFilter.load(config.MOLPORT_BLOOM)
        if config.MOLPORT_DB.exists():
            self.conn = sqlite3.connect(f"file:{config.MOLPORT_DB}?mode=ro", uri=True)

    @property
    def available(self) -> bool:
        return self.conn is not None

    def _canon(self, raw_or_canon: str, already_canonical: bool) -> Optional[str]:
        if already_canonical:
            return raw_or_canon
        from batterygen.core.data import canonical_smiles  # lazy: avoids importing rdkit unless needed

        return canonical_smiles(raw_or_canon)

    def contains(self, smiles: str, *, already_canonical: bool = False) -> bool:
        canon = self._canon(smiles, already_canonical)
        if canon is None:
            return False
        if self.bloom is not None and canon not in self.bloom:
            return False  # Bloom has no false negatives -> definitely absent
        if self.conn is None:
            return self.bloom is not None and canon in self.bloom  # bloom-only fallback
        cur = self.conn.execute("SELECT 1 FROM molecules WHERE smiles=? LIMIT 1", (canon,))
        return cur.fetchone() is not None

    def get_id(self, smiles: str, *, already_canonical: bool = False) -> Optional[str]:
        canon = self._canon(smiles, already_canonical)
        if canon is None or self.conn is None:
            return None
        if self.bloom is not None and canon not in self.bloom:
            return None
        cur = self.conn.execute("SELECT molport_id FROM molecules WHERE smiles=? LIMIT 1", (canon,))
        row = cur.fetchone()
        return row[0] if row else None

    def count(self) -> int:
        if self.conn is None:
            return 0
        return self.conn.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]

    def close(self) -> None:
        """Release the SQLite handle. Required on Windows before the DB file can be
        deleted/rebuilt (e.g. preprocess.finalize) — an open handle blocks unlink."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
