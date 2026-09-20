# Bounded local atomic dataset registry

This registry composes OrchidRec's six atomic file families through existing
strict local adapters for explicitly named dataset directories. The frozen RecBole reference snapshot
(`7b02be5ec80a88310f2d04a27a82adfcbb5dc211`) documents atomic families
`.inter`, `.user`, `.item`, `.kg`, `.link`, and `.net`, with dataset names matching
directory and filename prefixes. OrchidRec adopts that narrow naming convention,
not the full RecBole dataset/config/model/dataloader behavior. In particular,
the accepted `.net` header is OrchidRec's documented local contract.

```bash
orchidrec register-recbole-datasets \
  --dataset alpha=examples/registry/alpha --minimum-rating alpha=3 \
  --dataset beta=examples/registry/beta \
  --registry artifacts/registries

orchidrec verify-recbole-registry \
  --dataset alpha=examples/registry/alpha --minimum-rating alpha=3 \
  --dataset beta=examples/registry/beta \
  --record artifacts/registries/<registry_id>.json
```

Both fixtures are independently authored. `alpha` exercises all six families
and a rated threshold; `beta` is an unrated `.inter`-only dataset. The command
prints the actual content-addressed record path to use in the verify command.
Python callers may use `NamedAtomicDataset`, `RegistryLimits`,
`register_atomic_datasets`, `save_atomic_registry`, and
`verify_atomic_registry` directly. Verification reimports current files and
requires the exact canonical record bytes and ID; `false` means changed input,
wrong plan, or record tampering.

Each name must match `[A-Za-z][A-Za-z0-9_-]{0,63}`, the directory basename,
and the recognized file prefix. Names are case-insensitively unique; resolved
dataset directories must differ. Every dataset requires a nonempty `.inter`.
`.kg` and `.link` must be supplied as a pair and the `.link` table must be a
one-to-one item/entity mapping. Optional `.user`, `.item`, and `.net` can occur
independently. Unrecognized files are ignored. Recognized files and the
dataset directory cannot be symlinks. Rated `.inter` requires an explicit
finite threshold; unrated `.inter` rejects one. Existing adapter schemas,
row limits, and strict token validation still apply.

The manifest records sorted datasets, original per-file SHA-256/byte counts,
the normalized interaction fingerprint/retained counts, side/knowledge/network
fingerprints where present, and named overlap counts. `lexical_user_item_ids`
is only a collision count between separate ID spaces: identical spelling does
not merge a user and an item. `side_users_in_inter`, `side_items_in_inter`,
`linked_items_in_inter`, `linked_entities_in_kg`, and
`network_users_in_inter` count distinct names observed and matched. Interaction
overlaps use **retained** rows after any rating threshold. Cold side, link, and
network IDs are preserved and described, not required to match. No records are
silently filtered or joined across user, item, and entity namespaces.

Defaults: at most 8 datasets, 16 MiB per recognized file, 64 MiB total input,
and 1 MiB manifest output. CLI `--max-*` flags may lower or raise limits only
within hard ceilings (32 datasets, 16 MiB per file, 256 MiB total, 16 MiB
output); source-adapter row/token limits remain in force. Capture reads each
recognized source once under those byte limits, stages those exact bytes in a
temporary directory, then runs the existing adapters on the stage. This
prevents per-adapter rereads of changing original paths. It does **not** make
the capture of multiple concurrently edited source files a filesystem-atomic
instantaneous snapshot; callers must quiesce their inputs for that guarantee.
Nor is the memory cap a total-process cap: adapters hold normalized rows too.

`registry_id` is SHA-256 over the declared protocol revision, limit values,
sorted names/thresholds, and exact original file hashes/byte counts. The
manifest also carries a canonical state checksum. An implementation change
that alters normalized semantics must bump `orchidrec-atomic-registry-v1` so
old and new records do not collide; readers should preserve older protocol
verifiers separately. These hashes detect accidental corruption and bind the
declared bytes; they are not signatures or proof of data licensing, dataset
quality, privacy, or absence of evaluation leakage. In particular, a threshold
does not prove that optional features/links/edges exclude future-derived
information.

Saving writes a complete same-directory temporary file, fsyncs it, and
publishes by no-replace hard link. Two writers of the same ID cannot overwrite
one another; a repeated save errors. Output is one JSON manifest, not a copy of
raw source data. The temporary file is cleaned on return. A failure after
publication may leave a complete published record; rerun verification to
inspect it. Do not use the output directory as an adversarial shared trust
boundary; a separate trusted storage/signature policy is needed for that.
