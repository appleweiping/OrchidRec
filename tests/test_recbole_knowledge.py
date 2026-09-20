"""Independent local .kg/.link oracles and adversarial serialization tests."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchidrec.cli import main
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.recbole_knowledge import (
    CatalogReference,
    ItemEntityLink,
    KnowledgeLimits,
    KnowledgeTriple,
    LoadedKnowledgeLinks,
    import_recbole_knowledge,
    load_recbole_knowledge,
    save_recbole_knowledge,
)
from orchidrec.recbole_side import import_recbole_side_features, save_recbole_side_features

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
KG = (
    b"head_id:token\trelation_id:token\ttail_id:token\n"
    b"m.film1\thas_genre\tm.book\n"
    b"m.film2\thas_genre\tm.software\n"
    b"m.book\trelated_to\tm.software\n"
)
LINK = b"item_id:token\tentity_id:token\nitem-01\tm.film1\nitem-02\tm.film2\nA\tm.virtual\n"


def _tables(root: Path, kg: bytes = KG, link: bytes = LINK) -> tuple[Path, Path]:
    kg_path = root / "tiny.kg"
    link_path = root / "tiny.link"
    kg_path.write_bytes(kg)
    link_path.write_bytes(link)
    return kg_path, link_path


def _cli(argv: list[str]) -> int:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return main(argv)


class RecBoleKnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_independent_knowledge_oracle_and_roundtrip(self) -> None:
        kg_path, link_path = _tables(self.root)
        loaded = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        self.assertEqual(
            [triple.to_state() for triple in loaded.triples],
            [
                ["m.book", "related_to", "m.software"],
                ["m.film1", "has_genre", "m.book"],
                ["m.film2", "has_genre", "m.software"],
            ],
        )
        self.assertEqual(
            [link.to_state() for link in loaded.links],
            [["A", "m.virtual"], ["item-01", "m.film1"], ["item-02", "m.film2"]],
        )
        self.assertEqual(loaded.linked_entities_in_kg, 2)
        expected = hashlib.sha256(
            json.dumps(
                {
                    "triples": [
                        ["m.book", "related_to", "m.software"],
                        ["m.film1", "has_genre", "m.book"],
                        ["m.film2", "has_genre", "m.software"],
                    ],
                    "links": [["A", "m.virtual"], ["item-01", "m.film1"], ["item-02", "m.film2"]],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(loaded.fingerprint, expected)
        self.assertEqual(loaded.sources[0].sha256, hashlib.sha256(KG).hexdigest())
        self.assertEqual(loaded.sources[1].sha256, hashlib.sha256(LINK).hexdigest())
        output = self.root / "knowledge.json"
        save_recbole_knowledge(loaded, output)
        self.assertEqual(load_recbole_knowledge(output).to_state(), loaded.to_state())

    def test_raw_hash_changes_but_normalized_fingerprint_does_not(self) -> None:
        kg_path, link_path = _tables(self.root)
        first = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        kg_path.write_bytes(KG.replace(b"\n", b"\r\n"))
        link_path.write_bytes(LINK.replace(b"\n", b"\r\n"))
        newline = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        self.assertEqual(first.fingerprint, newline.fingerprint)
        self.assertNotEqual(first.sources[0].sha256, newline.sources[0].sha256)
        self.assertNotEqual(first.sources[1].sha256, newline.sources[1].sha256)
        kg_path.write_bytes(
            b"tail_id:token\thead_id:token\trelation_id:token\n"
            b"m.software\tm.book\trelated_to\n"
            b"m.book\tm.film1\thas_genre\n"
            b"m.software\tm.film2\thas_genre\n"
        )
        link_path.write_bytes(
            b"entity_id:token\titem_id:token\nm.virtual\tA\nm.film2\titem-02\nm.film1\titem-01\n"
        )
        reordered = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        self.assertEqual(first.fingerprint, reordered.fingerprint)

    def test_optional_catalog_composition_is_read_only(self) -> None:
        kg_path, link_path = _tables(self.root)
        side = import_recbole_side_features(item_path=EXAMPLES / "recbole-side-synthetic.item")
        side_artifact = self.root / "side.json"
        save_recbole_side_features(side, side_artifact)
        loaded = import_recbole_knowledge(
            kg_path=kg_path,
            link_path=link_path,
            inter_path=EXAMPLES / "recbole-synthetic.inter",
            minimum_rating=4.0,
            side_features_path=side_artifact,
        )
        self.assertEqual(
            [reference.kind for reference in loaded.references], ["recbole-inter", "recbole-side"]
        )
        self.assertEqual(
            [reference.linked_items_in_catalog for reference in loaded.references], [1, 2]
        )
        self.assertEqual([reference.catalog_items for reference in loaded.references], [4, 2])
        self.assertEqual(
            loaded.references[0].source_sha256,
            hashlib.sha256((EXAMPLES / "recbole-synthetic.inter").read_bytes()).hexdigest(),
        )
        self.assertEqual(loaded.references[1].source_sha256, side.to_state()["state_sha256"])
        self.assertEqual([link.item_id for link in loaded.links], ["A", "item-01", "item-02"])
        output = self.root / "combined.json"
        save_recbole_knowledge(loaded, output)
        self.assertEqual(load_recbole_knowledge(output).to_state(), loaded.to_state())

    def test_missing_linked_kg_entity_is_counted_not_silently_dropped(self) -> None:
        kg_path, link_path = _tables(self.root)
        loaded = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        self.assertEqual(len(loaded.links), 3)
        self.assertEqual(loaded.linked_entities_in_kg, 2)
        self.assertIn("m.virtual", [link.entity_id for link in loaded.links])

    def test_rejects_bad_headers_rows_and_tokens(self) -> None:
        bad_kg = (
            b"head_id:token\trelation_id:token\nA\tr\n",
            b"head_id:token\trelation_id:token\ttail_id:float\nA\tr\tB\n",
            b"head_id:token\trelation_id:token\ttail_id:token\textra:token\nA\tr\tB\tx\n",
            b"head_id:token\thead_id:token\ttail_id:token\nA\tr\tB\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\tB\nA\tr\tB\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\tbad value\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\t\xff\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\tB\rC\n",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\tB\r",
            b"head_id:token\trelation_id:token\ttail_id:token\nA\tr\tB\n\n",
        )
        kg_path, link_path = _tables(self.root)
        for content in bad_kg:
            with self.subTest(content=content):
                kg_path.write_bytes(content)
                with self.assertRaises(DatasetError):
                    import_recbole_knowledge(kg_path=kg_path, link_path=link_path)

    def test_link_mapping_must_be_bijective(self) -> None:
        kg_path, link_path = _tables(self.root)
        for content in (
            b"item_id:token\tentity_id:token\nA\te1\nA\te2\n",
            b"item_id:token\tentity_id:token\nA\te1\nB\te1\n",
            b"item_id:token\tentity_id:token\nA\te1\nA\te1\n",
        ):
            with self.subTest(content=content):
                link_path.write_bytes(content)
                with self.assertRaisesRegex(DatasetError, "one-to-one"):
                    import_recbole_knowledge(kg_path=kg_path, link_path=link_path)

    def test_file_row_line_token_and_output_limits(self) -> None:
        kg_path, link_path = _tables(self.root)
        for override in (
            {"max_file_bytes": len(KG) - 1},
            {"max_rows_per_file": 2},
            {"max_total_rows": 5},
            {"max_line_bytes": 10},
            {"max_token_chars": 2},
            {"max_token_bytes": 2},
        ):
            with self.subTest(override=override), self.assertRaises(DatasetError):
                import_recbole_knowledge(
                    kg_path=kg_path, link_path=link_path, limits=KnowledgeLimits(**override)
                )
        with self.assertRaises(ValidationError):
            KnowledgeLimits(max_rows_per_file=0)
        loaded = import_recbole_knowledge(
            kg_path=kg_path, link_path=link_path, limits=KnowledgeLimits(max_output_bytes=1)
        )
        output = self.root / "bounded.json"
        with (
            patch.object(
                LoadedKnowledgeLinks, "to_state", side_effect=AssertionError("materialized")
            ),
            self.assertRaisesRegex(SerializationError, "max_output_bytes"),
        ):
            save_recbole_knowledge(loaded, output)
        self.assertFalse(output.exists())

    def test_checksum_semantics_and_no_overwrite(self) -> None:
        kg_path, link_path = _tables(self.root)
        loaded = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        output = self.root / "knowledge.json"
        save_recbole_knowledge(loaded, output)
        raw = output.read_bytes()
        with self.assertRaisesRegex(ValidationError, "exists"):
            save_recbole_knowledge(loaded, output)
        self.assertEqual(output.read_bytes(), raw)
        self.assertFalse(list(self.root.glob(".knowledge.json.*.tmp")))
        output.write_bytes(raw.replace(b'"fingerprint_sha256":"', b'"fingerprint_sha256":"0', 1))
        with self.assertRaisesRegex(SerializationError, "checksum"):
            load_recbole_knowledge(output)
        state = json.loads(raw)
        state["sources"][0]["rows"] = 99
        state["state_sha256"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in state.items() if key != "state_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        output.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(SerializationError, "row counts"):
            load_recbole_knowledge(output)

    def test_public_constructor_invalid_states_never_create_output(self) -> None:
        kg_path, link_path = _tables(self.root)
        good = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        digest = hashlib.sha256(b"catalog").hexdigest()
        reference = CatalogReference("recbole-side", digest, digest, 2, 1)
        bad_states = (
            (replace(good, triples=(*good.triples, good.triples[-1])), "row limits|sorted"),
            (replace(good, triples=tuple(reversed(good.triples))), "sorted"),
            (replace(good, triples=(KnowledgeTriple("", "r", "e"),)), "token"),
            (replace(good, links=(*good.links, ItemEntityLink("zzz", "m.film1"))), "one-to-one"),
            (replace(good, sources=(good.sources[1], good.sources[0])), "ordered kg"),
            (
                replace(good, sources=(replace(good.sources[0], sha256="bad"), good.sources[1])),
                "provenance",
            ),
            (
                replace(good, sources=(replace(good.sources[0], rows=99), good.sources[1])),
                "row counts",
            ),
            (
                replace(
                    good, sources=(replace(good.sources[0], bytes=1_000_000_000), good.sources[1])
                ),
                "source bytes",
            ),
            (replace(good, references=(reference, reference)), "duplicated or unordered"),
            (
                replace(good, references=(replace(reference, linked_items_in_catalog=3),)),
                "overlap counts",
            ),
            (replace(good, references=(replace(reference, source_sha256="bad"),)), "source_sha256"),
            (replace(good, limits=KnowledgeLimits(max_rows_per_file=2)), "row limits"),
        )
        for index, (loaded, expected) in enumerate(bad_states):
            with self.subTest(index=index):
                output = self.root / f"invalid-{index}" / "knowledge.json"
                with self.assertRaisesRegex(SerializationError, expected):
                    save_recbole_knowledge(loaded, output)
                self.assertFalse(output.parent.exists())

        rebuilt = LoadedKnowledgeLinks(
            good.triples, good.links, good.sources, good.limits, good.references
        )
        output = self.root / "manual-valid.json"
        save_recbole_knowledge(rebuilt, output)
        self.assertEqual(load_recbole_knowledge(output).to_state(), rebuilt.to_state())

    def test_rehashed_artifact_still_enforces_semantic_invariants(self) -> None:
        kg_path, link_path = _tables(self.root)
        side = import_recbole_side_features(item_path=EXAMPLES / "recbole-side-synthetic.item")
        side_artifact = self.root / "side.json"
        save_recbole_side_features(side, side_artifact)
        loaded = import_recbole_knowledge(
            kg_path=kg_path, link_path=link_path, side_features_path=side_artifact
        )
        output = self.root / "knowledge.json"
        save_recbole_knowledge(loaded, output)
        valid = json.loads(output.read_bytes())

        def rewrite(state: dict[str, object]) -> None:
            state["state_sha256"] = hashlib.sha256(
                json.dumps(
                    {key: value for key, value in state.items() if key != "state_sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            output.write_text(json.dumps(state), encoding="utf-8")

        tamper_cases = (
            (lambda state: state["links"][2].__setitem__(1, "m.film1"), "one-to-one"),
            (lambda state: state["triples"][0].__setitem__(0, ""), "invalid knowledge token"),
            (lambda state: state["sources"][0].__setitem__("kind", "link"), "ordered kg"),
            (
                lambda state: state["references"][0].__setitem__("linked_items_in_catalog", 99),
                "overlap counts",
            ),
            (
                lambda state: state.__setitem__("fingerprint_sha256", "0" * 64),
                "fingerprint mismatch",
            ),
            (lambda state: state.__setitem__("linked_entities_in_kg", 0), "linked-entity count"),
        )
        for mutate, expected in tamper_cases:
            with self.subTest(expected=expected):
                state = json.loads(json.dumps(valid))
                mutate(state)
                rewrite(state)
                with self.assertRaisesRegex(SerializationError, expected):
                    load_recbole_knowledge(output)

        output.write_bytes(
            output.read_bytes().replace(b'"format":', b'"format":"duplicate","format":', 1)
        )
        with self.assertRaisesRegex(SerializationError, "duplicate"):
            load_recbole_knowledge(output)
        with self.assertRaisesRegex(SerializationError, "max_output_bytes"):
            load_recbole_knowledge(output, max_output_bytes=1)

    def test_string_ids_and_unrated_reference(self) -> None:
        kg_path, link_path = _tables(
            self.root,
            b"head_id:token\trelation_id:token\ttail_id:token\ne1\tr\te2\n",
            b"item_id:token\tentity_id:token\n001\te1\n1\te2\n",
        )
        inter_path = self.root / "tiny.inter"
        inter_path.write_bytes(b"user_id:token\titem_id:token\nu\t001\n")
        loaded = import_recbole_knowledge(
            kg_path=kg_path, link_path=link_path, inter_path=inter_path
        )
        self.assertEqual([link.item_id for link in loaded.links], ["001", "1"])
        self.assertEqual(loaded.references[0].catalog_items, 1)
        self.assertEqual(loaded.references[0].linked_items_in_catalog, 1)
        self.assertIsNone(loaded.references[0].minimum_rating)

    def test_atomic_cleanup_on_link_error_and_short_write(self) -> None:
        kg_path, link_path = _tables(self.root)
        loaded = import_recbole_knowledge(kg_path=kg_path, link_path=link_path)
        with (
            patch(
                "orchidrec.recbole_knowledge.os.link", side_effect=OSError("injected link error")
            ),
            self.assertRaisesRegex(SerializationError, "link error"),
        ):
            save_recbole_knowledge(loaded, self.root / "failed.json")
        self.assertFalse((self.root / "failed.json").exists())
        self.assertFalse(list(self.root.glob(".failed.json.*.tmp")))

        original = tempfile.NamedTemporaryFile

        class ShortWriter:
            def __init__(self, **kwargs: object) -> None:
                self.stream = original(**kwargs)
                self.name = self.stream.name

            def __enter__(self) -> ShortWriter:
                return self

            def __exit__(self, *args: object) -> None:
                self.stream.close()

            def write(self, data: bytes) -> int:
                return self.stream.write(data[:1])

        with (
            patch("orchidrec.recbole_knowledge.tempfile.NamedTemporaryFile", ShortWriter),
            self.assertRaisesRegex(SerializationError, "short write"),
        ):
            save_recbole_knowledge(loaded, self.root / "short.json")
        self.assertFalse((self.root / "short.json").exists())
        self.assertFalse(list(self.root.glob(".short.json.*.tmp")))

    def test_cli_import_and_alias_guard(self) -> None:
        kg_path, link_path = _tables(self.root)
        side = import_recbole_side_features(item_path=EXAMPLES / "recbole-side-synthetic.item")
        side_artifact = self.root / "side.json"
        save_recbole_side_features(side, side_artifact)
        output = self.root / "knowledge.json"
        self.assertEqual(
            _cli(
                [
                    "import-recbole-knowledge",
                    "--kg",
                    str(kg_path),
                    "--link",
                    str(link_path),
                    "--inter",
                    str(EXAMPLES / "recbole-synthetic.inter"),
                    "--minimum-rating",
                    "4",
                    "--side-features",
                    str(side_artifact),
                    "--output",
                    str(output),
                ]
            ),
            0,
        )
        self.assertEqual(len(load_recbole_knowledge(output).references), 2)
        self.assertEqual(
            _cli(
                [
                    "import-recbole-knowledge",
                    "--kg",
                    str(kg_path),
                    "--link",
                    str(link_path),
                    "--output",
                    str(kg_path),
                ]
            ),
            2,
        )
        self.assertEqual(
            _cli(
                [
                    "import-recbole-knowledge",
                    "--kg",
                    str(kg_path),
                    "--link",
                    str(link_path),
                    "--output",
                    str(output),
                ]
            ),
            2,
        )
        with self.assertRaisesRegex(ValidationError, "requires inter_path"):
            import_recbole_knowledge(kg_path=kg_path, link_path=link_path, minimum_rating=4.0)


if __name__ == "__main__":
    unittest.main()
