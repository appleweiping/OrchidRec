# Local `.kg` / `.link` knowledge interchange

OrchidRec can preserve one locally supplied RecBole-style knowledge graph
(`.kg`) and its item-to-entity mapping (`.link`) as a deterministic, typed JSON
artifact. Both files are required together. This is a strict, original
interchange path: it does **not** implement RecBole's entity remapping,
frequency filtering, reverse-edge generation, graph libraries, knowledge-aware
model training, or automatic data downloads.

```bash
orchidrec import-recbole-knowledge \
  --kg examples/recbole-knowledge-synthetic.kg \
  --link examples/recbole-knowledge-synthetic.link \
  --output artifacts/knowledge.json
```

The example files are synthetic and independently authored. The `.kg` header
must contain exactly `head_id:token`, `relation_id:token`, and
`tail_id:token`; the `.link` header must contain exactly `item_id:token` and
`entity_id:token`. Either header can permute its columns, but no extra or
duplicate columns are accepted. All values remain non-empty string tokens;
`001` and `1` are distinct IDs. Duplicate triplets fail. The `.link` table
must be one-to-one: neither an item nor an entity may appear twice. A linked
entity absent from the KG is **retained** and counted, not silently dropped;
this permits a virtual item-linked entity while making the coverage gap
visible. A KG entity need not have an item link.

For read-only overlap and provenance against existing inputs, add either or
both optional references:

```bash
orchidrec import-recbole-knowledge \
  --kg examples/recbole-knowledge-synthetic.kg \
  --link examples/recbole-knowledge-synthetic.link \
  --inter examples/recbole-synthetic.inter --minimum-rating 4 \
  --side-features artifacts/side-features.json \
  --output artifacts/knowledge-with-references.json
```

`--inter` uses the existing strict `.inter` importer; rated interactions
require an explicit finite `--minimum-rating`. `--side-features` accepts only
the existing checksummed `.user`/`.item` artifact, not raw atomic files. The
knowledge artifact records the `.inter` file's exact-byte source SHA-256 or
the side-feature artifact's canonical state checksum (not a hash of its raw
JSON bytes), plus the normalized dataset hash, catalog size, and number of
linked items also present in that catalog. It does not include interaction or
feature rows and does not
reject a valid link merely because its item is absent from an optional
reference. There is no ID coercion, graph/interaction join, or catalog
completion.

Prepare the KG/link snapshot according to your experiment's training-data
policy. This importer does not split a graph into train/validation/test and
cannot prove that an input graph excludes held-out facts; referencing `.inter`
or side features does not make a graph leakage-safe. Never fit or select a
model using relationships derived from held-out labels. The stored reference
overlaps are descriptive checks only.

The adapter accepts LF or CRLF, and rejects bare CR, blank rows, malformed
UTF-8, wrong tab counts, whitespace/control-containing tokens, ambiguous
links, and duplicate triplets. Defaults cap each source at 16 MiB, 100,000
data rows, and 64 KiB per physical line; the pair at 200,000 rows; tokens at
512 characters / 2,048 UTF-8 bytes; and the artifact at 64 MiB. `--max-*`
flags may tighten or loosen these defaults within hard ceilings. Expanded records are
checked against the output cap before full-state materialization.
Loading materializes the bounded JSON document in memory; for untrusted
artifacts, choose a lower `max_output_bytes` in the Python loader where
appropriate. The byte cap is not a constant-memory JSON parser guarantee.

The versioned artifact stores exact-byte source SHA-256 hashes and counts,
canonically sorted triples and links, an order-independent normalized
fingerprint, optional catalog references, and a checksum. Renaming the files
does not change it. Changing line endings changes the source hash but not the
normalized fingerprint. Checksums detect accidental corruption, not
malicious provenance forgery. Publication is same-directory atomic and
no-overwrite; an existing output or path alias is never replaced.

`.net` is still unsupported: the frozen RecBole source identifies its social
`source`/`target` role but does not establish a concrete header/type contract
we can validate without guessing. This artifact must not be described as
full RecBole knowledge-aware or social recommendation compatibility.
