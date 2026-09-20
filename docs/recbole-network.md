# Local `.net` directed-edge interchange

OrchidRec can preserve one local social edge table as a checksummed, bounded
JSON snapshot. The frozen RecBole atomic-file documentation identifies `.net`
as source/target social graph data and pairs it with `.inter` for social
recommendation. It does **not** establish a concrete `.net` header/type schema
or working model loader. Accordingly, `source_id:token` and `target_id:token`
are OrchidRec's explicit, narrow import contract, not a claim that every
RecBole `.net` file is compatible. This adapter neither trains a social
recommender nor creates a train/validation/test graph split.

```bash
orchidrec import-recbole-network \
  --net examples/recbole-network-synthetic.net \
  --inter examples/recbole-synthetic.inter --minimum-rating 4 \
  --output artifacts/network.json
```

The synthetic file is independently authored. The two typed header columns
may appear in either order; extra or duplicate columns fail. Edges are
directed, so `A -> B` and `B -> A` are distinct. Self-loops are preserved.
Duplicate directed edges fail. IDs are non-empty UTF-8 tokens, remain strings,
and cannot contain whitespace or controls; `001` and `1` remain distinct.
Unseen users are retained, not silently filtered.

`--inter` is optional, read-only, and uses the already strict `.inter` reader.
Rated interactions require an explicit finite `--minimum-rating`. The
resulting reference reports exact-byte `.inter` SHA-256, its normalized
interaction fingerprint, the retained user count, and how many distinct
network users occur in that retained catalog. It does not store interaction
rows, join them to edges, or require every network user to appear in the
catalog. The graph input must independently follow your training-data policy:
the overlap does not prove the edge file excludes held-out facts or users.

Default limits: 16 MiB source, 100,000 edges, 64 KiB per physical line,
512 characters / 2,048 UTF-8 bytes per token, 64 MiB artifact. CLI `--max-*`
flags are individually bounded by hard ceilings. The importer accepts LF or
CRLF and rejects bare CR, blank lines, malformed UTF-8 and tab counts.
Expanded records are checked against the output cap before materializing the
whole state. Loading reads a bounded JSON document into memory; the cap is
not a constant-memory parsing guarantee.

The artifact stores sorted edges, unique-user count, source byte hash/count,
order-independent normalized edge fingerprint, optional catalog reference,
and a state checksum. Canonical output is independent of source row/column
order and filename; LF versus CRLF changes the source hash. A checksum detects
accidental corruption, not malicious provenance forgery. Publication is
same-directory atomic and refuses existing destinations. The CLI also rejects
input/output path aliases.
