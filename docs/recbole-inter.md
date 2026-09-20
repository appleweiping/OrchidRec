# Local RecBole `.inter` interchange

OrchidRec can import a locally supplied RecBole atomic **interaction** file as
implicit-feedback events. It neither downloads data nor bundles RecBole or its
sample datasets. This is one interoperable format, not support for RecBole's
other atomic files in this workflow or its full feature semantics. A separate
[local `.user` / `.item` side-feature adapter](recbole-side.md) imports typed
feature tables without joining them to interactions. A separate
[`.kg` / `.link` knowledge adapter](recbole-knowledge.md) preserves triplets
and item links with optional read-only interaction overlap; `.net` remains
unsupported.

```bash
orchidrec dataset-summary examples/recbole-synthetic.inter --format recbole-inter --minimum-rating 4
orchidrec benchmark examples/recbole-benchmark-config.json --output-dir artifacts/recbole-demo
```

The example data is synthetic and independently authored. The header must use
tabs and declare exactly one each of `user_id:token` and `item_id:token`; it may
also declare `rating:float` and/or `timestamp:float`, in any order. Extra
fields, even otherwise valid RecBole types, are rejected because the current
interaction model cannot preserve them. `001` and `1` remain distinct string
IDs; the adapter never interprets token IDs as numbers. Duplicate user-item
events, missing IDs, non-finite floats, and negative timestamps are rejected.

If a rating column is present, specify a finite `minimum_rating` explicitly.
The threshold uses the source's native rating scale, so 1–5 is **not** assumed.
Rows at or above it become unit-valued implicit events; lower rows are counted
as dropped, not treated as negatives. If there is no rating column, do not set
a threshold: every row becomes a unit-valued implicit event. `rating_min` and
`rating_max` in the summary describe source ratings when present and otherwise
the normalized unit value 1.0. Timestamps retain their finite, non-negative
numeric values and input row order is retained; downstream temporal splitting
uses the timestamps. A threshold removing all events fails.

The importer caps a source at 16 MiB, 100,000 data rows, and 64 KiB per
physical line *before* expanding decoded rows. It accepts LF or CRLF line
endings and rejects bare CR (including CR-only files). It validates UTF-8, tab field
count, and duplicate pairs. Point `dataset-summary` or the benchmark `data.path`
to a `.inter` file or a directory containing exactly one `.inter` file.
Ambiguous directories fail; suffix-named directories are not candidates. The
summary stores SHA-256 of the exact source bytes and of ordered normalized
interaction records. For this adapter, `source_name` is a content-addressed
`sha256-<digest>.inter` label rather than the input basename, so renamed
hardlinks or copies of the same bytes yield the same summary and downstream
split. Changing line endings changes the source hash even if normalized events
stay equivalent.

For a benchmark config use `"data": {"path": "/local/events.inter", "format":
"recbole-inter", "minimum_rating": 4}`. Omit `minimum_rating` only for an
unrated `.inter` file. Paths in JSON configs are resolved relative to the
config file. Existing OrchidRec JSON and MovieLens adapters retain their
previous contracts and defaults.
