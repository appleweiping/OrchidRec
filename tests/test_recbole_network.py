"""Independent directed-edge oracle and adversarial `.net` interchange tests."""

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

from orchidrec import recbole_network as network_module
from orchidrec.cli import main
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.recbole_network import (
    LoadedSocialNetwork,
    NetworkCatalogReference,
    NetworkLimits,
    NetworkSource,
    SocialEdge,
    import_recbole_network,
    load_recbole_network,
    save_recbole_network,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NET = b"source_id:token\ttarget_id:token\n001\t1\n1\t001\n1\tu3\nu3\tu3\nu4\t001\n"


def _cli(argv: list[str]) -> int:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return main(argv)


def _checksum(state: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in state.items() if key != "state_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


class RecBoleNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.net = self.root / "tiny.net"
        self.net.write_bytes(NET)

    def test_independent_directed_oracle_and_roundtrip(self) -> None:
        loaded = import_recbole_network(net_path=self.net)
        expected = [["001", "1"], ["1", "001"], ["1", "u3"], ["u3", "u3"], ["u4", "001"]]
        self.assertEqual([edge.to_state() for edge in loaded.edges], expected)
        self.assertEqual(loaded.users, ("001", "1", "u3", "u4"))
        self.assertEqual(loaded.source.sha256, hashlib.sha256(NET).hexdigest())
        self.assertEqual(loaded.source.bytes, len(NET))
        self.assertEqual(loaded.source.rows, 5)
        self.assertEqual(
            loaded.fingerprint,
            hashlib.sha256(
                json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        output = self.root / "network.json"
        save_recbole_network(loaded, output)
        self.assertEqual(load_recbole_network(output).to_state(), loaded.to_state())
        self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_permuted_columns_rows_crlf_and_filename(self) -> None:
        first = import_recbole_network(net_path=self.net)
        self.net.write_bytes(NET.replace(b"\n", b"\r\n"))
        newline = import_recbole_network(net_path=self.net)
        self.assertEqual(first.fingerprint, newline.fingerprint)
        self.assertNotEqual(first.source.sha256, newline.source.sha256)
        second_path = self.root / "renamed.net"
        second_path.write_bytes(
            b"target_id:token\tsource_id:token\n001\tu4\nu3\tu3\nu3\t1\n001\t1\n1\t001\n"
        )
        reordered = import_recbole_network(net_path=second_path)
        self.assertEqual(first.fingerprint, reordered.fingerprint)
        self.assertEqual(first.edges, reordered.edges)

    def test_optional_inter_user_overlap_is_read_only(self) -> None:
        inter = EXAMPLES / "recbole-synthetic.inter"
        original = inter.read_bytes()
        loaded = import_recbole_network(net_path=self.net, inter_path=inter, minimum_rating=4.0)
        self.assertEqual(loaded.reference.catalog_users, 3)
        self.assertEqual(loaded.reference.network_users_in_catalog, 3)
        self.assertEqual(loaded.reference.minimum_rating, 4.0)
        self.assertEqual(loaded.reference.source_sha256, hashlib.sha256(original).hexdigest())
        self.assertEqual([edge.target_id for edge in loaded.edges][-1], "001")
        self.assertEqual(inter.read_bytes(), original)
        output = self.root / "with-reference.json"
        save_recbole_network(loaded, output)
        self.assertEqual(load_recbole_network(output).to_state(), loaded.to_state())
        with self.assertRaisesRegex(ValidationError, "requires inter_path"):
            import_recbole_network(net_path=self.net, minimum_rating=4.0)
        with self.assertRaisesRegex(DatasetError, "minimum_rating"):
            import_recbole_network(net_path=self.net, inter_path=inter)

    def test_input_header_token_and_line_failures(self) -> None:
        bad = (
            b"source_id:token\ttarget_id:float\nA\tB\n",
            b"source_id:token\tsource_id:token\nA\tB\n",
            b"source_id:token\ttarget_id:token\textra:token\nA\tB\tC\n",
            b"source_id:token\ttarget_id:token\n",
            b"source_id:token\ttarget_id:token\nA\n",
            b"source_id:token\ttarget_id:token\nA\tB\tC\n",
            b"source_id:token\ttarget_id:token\nA\t\n",
            b"source_id:token\ttarget_id:token\nA\tbad value\n",
            b"source_id:token\ttarget_id:token\nA\t\xff\n",
            b"source_id:token\ttarget_id:token\nA\tB\rC\n",
            b"source_id:token\ttarget_id:token\nA\tB\r",
            b"source_id:token\ttarget_id:token\nA\tB\n\n",
            b"source_id:token\ttarget_id:token\nA\tB\nA\tB\n",
        )
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(DatasetError):
                self.net.write_bytes(payload)
                import_recbole_network(net_path=self.net)
        self.net.write_bytes(b"")
        with self.assertRaises(DatasetError):
            import_recbole_network(net_path=self.net)
        with self.assertRaises(DatasetError):
            import_recbole_network(net_path=self.root / "missing.net")
        other_suffix = self.root / "tiny.txt"
        other_suffix.write_bytes(NET)
        with self.assertRaises(DatasetError):
            import_recbole_network(net_path=other_suffix)

    def test_source_and_output_limits_and_invalid_types(self) -> None:
        for override in (
            {"max_file_bytes": len(NET) - 1},
            {"max_line_bytes": 20},
            {"max_rows": 4},
            {"max_token_chars": 2},
            {"max_token_bytes": 2},
        ):
            with self.subTest(override=override), self.assertRaises(DatasetError):
                import_recbole_network(net_path=self.net, limits=NetworkLimits(**override))
        for override in ({"max_rows": 0}, {"max_rows": True}, {"max_rows": 1_000_001}):
            with self.subTest(override=override), self.assertRaises(ValidationError):
                NetworkLimits(**override)
        with self.assertRaises(ValidationError):
            import_recbole_network(net_path=self.net, limits=object())  # type: ignore[arg-type]
        loaded = import_recbole_network(net_path=self.net, limits=NetworkLimits(max_output_bytes=1))
        output = self.root / "bounded.json"
        with (
            patch.object(
                LoadedSocialNetwork, "to_state", side_effect=AssertionError("materialized")
            ),
            self.assertRaisesRegex(SerializationError, "max_output_bytes"),
        ):
            save_recbole_network(loaded, output)
        self.assertFalse(output.exists())

    def test_no_overwrite_checksum_and_rehashed_semantics(self) -> None:
        loaded = import_recbole_network(net_path=self.net)
        output = self.root / "network.json"
        save_recbole_network(loaded, output)
        raw = output.read_bytes()
        with self.assertRaisesRegex(ValidationError, "exists"):
            save_recbole_network(loaded, output)
        self.assertEqual(output.read_bytes(), raw)
        self.assertFalse(list(self.root.glob(".network.json.*.tmp")))
        output.write_bytes(raw.replace(b'"fingerprint_sha256":"', b'"fingerprint_sha256":"0', 1))
        with self.assertRaisesRegex(SerializationError, "checksum"):
            load_recbole_network(output)

        valid = json.loads(raw)
        cases = (
            (lambda state: state["edges"].reverse(), "sorted"),
            (lambda state: state["edges"].append(["1", "001"]), "unique"),
            (lambda state: state["edges"][0].__setitem__(1, ""), "invalid network token"),
            (lambda state: state["edges"][0].append("third"), "wrong width"),
            (lambda state: state["source"].__setitem__("rows", 99), "row or byte limits"),
            (lambda state: state["source"].__setitem__("sha256", "bad"), "provenance"),
            (lambda state: state.__setitem__("users", 9), "unique-user count"),
            (lambda state: state.__setitem__("fingerprint_sha256", "0" * 64), "fingerprint"),
        )
        for mutate, expected in cases:
            with self.subTest(expected=expected):
                state = json.loads(json.dumps(valid))
                mutate(state)
                state["state_sha256"] = _checksum(state)
                output.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(SerializationError, expected):
                    load_recbole_network(output)
        output.write_bytes(raw.replace(b'"format":', b'"format":"duplicate","format":', 1))
        with self.assertRaisesRegex(SerializationError, "duplicate"):
            load_recbole_network(output)
        with self.assertRaisesRegex(SerializationError, "max_output_bytes"):
            load_recbole_network(output, max_output_bytes=1)
        with self.assertRaises(ValidationError):
            load_recbole_network(output, max_output_bytes=True)

    def test_constructor_semantics_checked_before_output_directory_creation(self) -> None:
        good = import_recbole_network(net_path=self.net)
        sha = "a" * 64
        reference = NetworkCatalogReference(sha, sha, 2, 1, None)
        invalid = (
            (replace(good, edges=(*good.edges, good.edges[-1])), "unique"),
            (replace(good, edges=tuple(reversed(good.edges))), "sorted"),
            (replace(good, edges=(SocialEdge("", "B"),)), "token"),
            (replace(good, edges=(SocialEdge("A B", "C"),)), "token"),
            (replace(good, edges=()), "row limits"),
            (replace(good, source=NetworkSource("bad", 10, 5)), "provenance"),
            (replace(good, source=replace(good.source, rows=99)), "row or byte limits"),
            (replace(good, limits=NetworkLimits(max_rows=2)), "row limits"),
            (replace(good, reference=replace(reference, catalog_users=0)), "overlap counts"),
            (replace(good, reference=replace(reference, source_sha256="bad")), "source_sha256"),
            (replace(good, reference=replace(reference, minimum_rating=float("nan"))), "threshold"),
        )
        for index, (loaded, expected) in enumerate(invalid):
            with self.subTest(index=index):
                output = self.root / f"invalid-{index}" / "network.json"
                with self.assertRaisesRegex(SerializationError, expected):
                    save_recbole_network(loaded, output)
                self.assertFalse(output.parent.exists())
        rebuilt = LoadedSocialNetwork(good.edges, good.source, good.limits, good.reference)
        output = self.root / "rebuilt.json"
        save_recbole_network(rebuilt, output)
        self.assertEqual(load_recbole_network(output).to_state(), rebuilt.to_state())

    def test_cli_smoke_alias_and_failure_do_not_mutate_source(self) -> None:
        output = self.root / "network.json"
        before = self.net.read_bytes()
        self.assertEqual(
            _cli(
                [
                    "import-recbole-network",
                    "--net",
                    str(self.net),
                    "--inter",
                    str(EXAMPLES / "recbole-synthetic.inter"),
                    "--minimum-rating",
                    "4",
                    "--output",
                    str(output),
                ]
            ),
            0,
        )
        self.assertEqual(load_recbole_network(output).reference.network_users_in_catalog, 3)
        self.assertEqual(self.net.read_bytes(), before)
        self.assertEqual(
            _cli(["import-recbole-network", "--net", str(self.net), "--output", str(self.net)]),
            2,
        )
        self.assertEqual(self.net.read_bytes(), before)
        self.assertEqual(
            _cli(["import-recbole-network", "--net", str(self.net), "--output", str(output)]),
            2,
        )

    def test_atomic_cleanup_on_link_failure(self) -> None:
        loaded = import_recbole_network(net_path=self.net)
        output = self.root / "failed.json"
        with (
            patch("orchidrec.recbole_network.os.link", side_effect=OSError("injected")),
            self.assertRaisesRegex(SerializationError, "injected"),
        ):
            save_recbole_network(loaded, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".failed.json.*.tmp")))

    def test_no_final_newline_utf8_bytes_and_nonstandard_separators(self) -> None:
        self.net.write_bytes(NET.rstrip(b"\n"))
        self.assertEqual(len(import_recbole_network(net_path=self.net).edges), 5)
        self.net.write_bytes(b"source_id:token\ttarget_id:token\nA\t\xc3\xa9\n")
        with self.assertRaisesRegex(DatasetError, "token length"):
            import_recbole_network(net_path=self.net, limits=NetworkLimits(max_token_bytes=1))
        self.net.write_bytes(b"source_id:token\ttarget_id:token\nA\tB\r\n")
        self.assertEqual(len(import_recbole_network(net_path=self.net).edges), 1)

    def test_limit_source_reference_state_decoders(self) -> None:
        good_limits = NetworkLimits().to_state()
        self.assertEqual(NetworkLimits.from_state(good_limits), NetworkLimits())
        for value in (None, {}, {**good_limits, "unknown": 1}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(SerializationError, "limits have missing"),
            ):
                NetworkLimits.from_state(value)
        with self.assertRaisesRegex(SerializationError, "invalid network limits"):
            NetworkLimits.from_state({**good_limits, "max_rows": False})

        source = import_recbole_network(net_path=self.net).source
        self.assertEqual(NetworkSource.from_state(source.to_state()), source)
        for value in (
            None,
            {"sha256": source.sha256, "bytes": 1},
            {"sha256": "BAD", "bytes": 1, "rows": 1},
            {"sha256": source.sha256, "bytes": True, "rows": 1},
            {"sha256": source.sha256, "bytes": 0, "rows": 1},
            {"sha256": source.sha256, "bytes": 1, "rows": False},
            {"sha256": source.sha256, "bytes": 1, "rows": 0},
        ):
            with self.subTest(value=value), self.assertRaises(SerializationError):
                NetworkSource.from_state(value)

        sha = source.sha256
        good = NetworkCatalogReference(sha, sha, 3, 2, None)
        self.assertEqual(NetworkCatalogReference.from_state(good.to_state(), 3), good)
        reference_cases = (
            None,
            {"source_sha256": sha},
            {**good.to_state(), "source_sha256": "bad"},
            {**good.to_state(), "normalized_sha256": "bad"},
            {**good.to_state(), "catalog_users": True},
            {**good.to_state(), "catalog_users": 0},
            {**good.to_state(), "network_users_in_catalog": True},
            {**good.to_state(), "network_users_in_catalog": 4},
            {**good.to_state(), "minimum_rating": "4"},
            {**good.to_state(), "minimum_rating": float("inf")},
        )
        for value in reference_cases:
            with self.subTest(value=value), self.assertRaises(SerializationError):
                NetworkCatalogReference.from_state(value, 3)
        with self.assertRaisesRegex(SerializationError, "overlap counts"):
            NetworkCatalogReference.from_state(good.to_state(), 1)

    def test_public_constructor_type_and_size_failures(self) -> None:
        good = import_recbole_network(net_path=self.net)
        malformed = (
            replace(good, limits=object()),  # type: ignore[arg-type]
            replace(good, source=object()),  # type: ignore[arg-type]
            replace(good, edges=[*good.edges]),  # type: ignore[arg-type]
            replace(good, edges=("edge",)),  # type: ignore[arg-type]
            replace(good, source=replace(good.source, bytes=100_000_000)),
            replace(good, reference="reference"),  # type: ignore[arg-type]
        )
        for index, loaded in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(SerializationError):
                save_recbole_network(loaded, self.root / f"invalid-type-{index}" / "out.json")
            self.assertFalse((self.root / f"invalid-type-{index}").exists())
        with self.assertRaises(ValidationError):
            save_recbole_network(object(), self.root / "never.json")  # type: ignore[arg-type]
        with self.assertRaisesRegex(SerializationError, "max_output_bytes"):
            network_module._canonical(good.to_state(), 1)
        with self.assertRaisesRegex(SerializationError, "canonical JSON"):
            network_module._digest(object())
        with self.assertRaisesRegex(SerializationError, "canonical JSON"):
            network_module._canonical(object(), 100)

    def test_rehashed_artifact_structure_and_declared_limit_failures(self) -> None:
        good = import_recbole_network(net_path=self.net)
        output = self.root / "mutated.json"
        save_recbole_network(good, output)
        valid = json.loads(output.read_bytes())
        mutations = (
            (lambda s: s.__setitem__("format", "other"), "unsupported"),
            (lambda s: s.__setitem__("schema_version", True), "unsupported"),
            (lambda s: s.__setitem__("edges", None), "row limits"),
            (lambda s: s.__setitem__("edges", []), "row limits"),
            (lambda s: s["edges"].__setitem__(0, "bad"), "wrong width"),
            (lambda s: s["edges"].__setitem__(0, ["A", 1]), "invalid network token"),
            (lambda s: s["source"].__setitem__("bytes", 100_000_000), "row or byte limits"),
            (lambda s: s.__setitem__("reference", {}), "reference has missing"),
            (lambda s: s["limits"].__setitem__("max_rows", 1), "row limits"),
            (lambda s: s["limits"].__setitem__("max_output_bytes", 1), "max_output_bytes"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                state = json.loads(json.dumps(valid))
                mutate(state)
                state["state_sha256"] = _checksum(state)
                output.write_text(json.dumps(state), encoding="utf-8")
                with self.assertRaisesRegex(SerializationError, expected):
                    load_recbole_network(output)
        for state in (
            {**valid, "unknown": 1},
            {key: value for key, value in valid.items() if key != "source"},
        ):
            output.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(SerializationError, "missing or unknown"):
                load_recbole_network(output)
        output.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(SerializationError, "invalid network JSON"):
            load_recbole_network(output)
        with self.assertRaisesRegex(SerializationError, "could not read"):
            load_recbole_network(self.root / "missing.json")

    def test_source_read_and_atomic_write_io_errors(self) -> None:
        with (
            patch("orchidrec.recbole_network.Path.open", side_effect=OSError("read denied")),
            self.assertRaisesRegex(DatasetError, "read denied"),
        ):
            import_recbole_network(net_path=self.net)
        loaded = import_recbole_network(net_path=self.net)
        output = self.root / "write-error" / "out.json"
        with (
            patch("orchidrec.recbole_network.os.fsync", side_effect=OSError("sync denied")),
            self.assertRaisesRegex(SerializationError, "sync denied"),
        ):
            save_recbole_network(loaded, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob(".out.json.*.tmp")))

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
            patch("orchidrec.recbole_network.tempfile.NamedTemporaryFile", ShortWriter),
            self.assertRaisesRegex(SerializationError, "short write"),
        ):
            save_recbole_network(loaded, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob(".out.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
