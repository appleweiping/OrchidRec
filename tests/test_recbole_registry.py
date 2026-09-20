"""Independent snapshot/provenance oracle and adversarial atomic-registry tests."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orchidrec import recbole_registry as registry_module
from orchidrec.cli import main
from orchidrec.errors import DatasetError, SerializationError, ValidationError
from orchidrec.recbole_registry import (
    AtomicFileProvenance,
    AtomicRegistryManifest,
    NamedAtomicDataset,
    NamespaceOverlap,
    RegistryLimits,
    register_atomic_datasets,
    save_atomic_registry,
    verify_atomic_registry,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "registry"
INTER = b"user_id:token\titem_id:token\nu1\ti1\n"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class AtomicRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def dataset(
        self, name: str = "tiny", files: dict[str, bytes] | None = None
    ) -> NamedAtomicDataset:
        directory = self.root / name
        directory.mkdir()
        for suffix, payload in (files or {"inter": INTER}).items():
            (directory / f"{name}.{suffix}").write_bytes(payload)
        return NamedAtomicDataset(name, directory)

    def test_two_dataset_independent_provenance_and_namespaces(self) -> None:
        alpha = NamedAtomicDataset("alpha", EXAMPLES / "alpha", 3)
        beta = NamedAtomicDataset("beta", EXAMPLES / "beta")
        manifest = register_atomic_datasets([beta, alpha])
        self.assertEqual([item.name for item in manifest.datasets], ["alpha", "beta"])
        self.assertEqual(manifest.registry_id, register_atomic_datasets([alpha, beta]).registry_id)
        first = manifest.datasets[0]
        self.assertEqual(
            [file.suffix for file in first.files], ["inter", "user", "item", "kg", "link", "net"]
        )
        for file in first.files:
            source = (EXAMPLES / "alpha" / f"alpha.{file.suffix}").read_bytes()
            self.assertEqual((file.bytes, file.sha256), (len(source), _sha(source)))
        self.assertEqual(first.interactions.retained_interactions, 2)
        self.assertEqual(first.interactions.dropped_interactions, 1)
        self.assertEqual(first.interactions.users, 2)
        self.assertEqual(first.interactions.items, 2)
        self.assertEqual(
            {row.name: (row.observed, row.matched) for row in first.overlaps},
            {
                "lexical_user_item_ids": (2, 1),
                "side_users_in_inter": (2, 1),
                "side_items_in_inter": (2, 1),
                "linked_items_in_inter": (2, 1),
                "linked_entities_in_kg": (2, 1),
                "network_users_in_inter": (2, 1),
            },
        )
        self.assertEqual(
            [(row.name, row.observed, row.matched) for row in manifest.datasets[1].overlaps],
            [("lexical_user_item_ids", 2, 1)],
        )
        identity = {
            "protocol": manifest.protocol,
            "limits": manifest.limits.to_state(),
            "datasets": [
                {
                    "name": dataset.name,
                    "minimum_rating": dataset.minimum_rating,
                    "files": [file.to_state() for file in dataset.files],
                }
                for dataset in manifest.datasets
            ],
        }
        self.assertEqual(manifest.registry_id, _sha(_canonical(identity)))
        state = manifest.to_state()
        self.assertEqual(
            state["state_sha256"],
            _sha(_canonical({k: v for k, v in state.items() if k != "state_sha256"})),
        )
        output = save_atomic_registry(manifest, self.root / "registry")
        self.assertEqual(output.name, f"{manifest.registry_id}.json")
        self.assertEqual(output.read_bytes(), _canonical(state) + b"\n")
        self.assertTrue(verify_atomic_registry(output, [beta, alpha]))

    def test_identity_changes_with_source_threshold_limits_and_protocol(self) -> None:
        spec = self.dataset(
            files={"inter": b"user_id:token\titem_id:token\trating:float\nu1\ti1\t5\n"}
        )
        rated = NamedAtomicDataset(spec.name, spec.directory, 4)
        first = register_atomic_datasets([rated])
        self.assertNotEqual(
            first.registry_id,
            register_atomic_datasets([replace(rated, minimum_rating=3)]).registry_id,
        )
        self.assertNotEqual(
            first.registry_id, replace(first, limits=RegistryLimits(max_datasets=7)).registry_id
        )
        self.assertNotEqual(
            first.registry_id, replace(first, protocol="orchidrec-atomic-registry-v2").registry_id
        )
        (spec.directory / "tiny.inter").write_bytes(
            b"user_id:token\titem_id:token\trating:float\nu1\ti1\t6\n"
        )
        self.assertNotEqual(first.registry_id, register_atomic_datasets([rated]).registry_id)

    def test_no_replace_mutation_and_tamper_detection(self) -> None:
        spec = self.dataset()
        manifest = register_atomic_datasets([spec])
        output = save_atomic_registry(manifest, self.root / "registry")
        original = output.read_bytes()
        with self.assertRaisesRegex(ValidationError, "already exists"):
            save_atomic_registry(manifest, self.root / "registry")
        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(list(output.parent.glob("*.tmp")))
        output.write_bytes(original.replace(b"registry_id", b"registry_ix", 1))
        self.assertFalse(verify_atomic_registry(output, [spec]))
        output.write_bytes(original)
        (spec.directory / "tiny.inter").write_bytes(INTER.replace(b"i1", b"i2"))
        self.assertFalse(verify_atomic_registry(output, [spec]))

    def test_concurrent_no_replace_has_one_complete_winner(self) -> None:
        manifest = register_atomic_datasets([self.dataset()])
        directory = self.root / "registry"
        barrier = threading.Barrier(2)
        real_link = os.link

        def synchronized_link(source: Path, destination: Path) -> None:
            barrier.wait(timeout=10)
            real_link(source, destination)

        with (
            patch("orchidrec.recbole_registry.os.link", side_effect=synchronized_link),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(lambda _: self._save_result(manifest, directory), range(2)))
        self.assertEqual(sorted(results), ["exists", "saved"])
        output = directory / f"{manifest.registry_id}.json"
        self.assertEqual(output.read_bytes(), _canonical(manifest.to_state()) + b"\n")
        self.assertFalse(list(directory.glob("*.tmp")))

    @staticmethod
    def _save_result(manifest: AtomicRegistryManifest, directory: Path) -> str:
        try:
            save_atomic_registry(manifest, directory)
        except ValidationError:
            return "exists"
        return "saved"

    def test_structure_name_namespace_and_invalid_inputs(self) -> None:
        with self.assertRaises(ValidationError):
            NamedAtomicDataset("1bad", self.root / "1bad")
        with self.assertRaises(ValidationError):
            NamedAtomicDataset("good", self.root / "other")
        with self.assertRaises(ValidationError):
            NamedAtomicDataset("tiny", self.root / "tiny", float("nan"))
        with self.assertRaises(ValidationError):
            NamedAtomicDataset("tiny", self.root / "tiny", True)
        with self.assertRaises(ValidationError):
            NamedAtomicDataset("tiny", self.root / "tiny", 10**1000)
        spec = self.dataset()
        with self.assertRaisesRegex(ValidationError, "unique"):
            register_atomic_datasets([spec, spec])
        with self.assertRaises(ValidationError):
            register_atomic_datasets([])
        with self.assertRaises(ValidationError):
            register_atomic_datasets([spec], limits=RegistryLimits(max_datasets=0))
        with self.assertRaises(ValidationError):
            RegistryLimits(max_file_bytes=17 * 1024 * 1024)
        with self.assertRaises(ValidationError):
            RegistryLimits(max_file_bytes=100, max_total_bytes=99)
        with self.assertRaises(ValidationError):
            RegistryLimits(max_datasets=True)
        with self.assertRaises(ValidationError):
            AtomicFileProvenance("bad", "x" * 64, 1)
        with self.assertRaises(ValidationError):
            AtomicFileProvenance("inter", "0" * 64, 0)
        with self.assertRaises(ValidationError):
            NamespaceOverlap("bad", 1, 2)
        with self.assertRaises(ValidationError):
            NamespaceOverlap("", 1, 0)
        with self.assertRaises(ValidationError):
            replace(register_atomic_datasets([spec]), protocol="unknown")
        manifest = register_atomic_datasets([spec])
        dataset = manifest.datasets[0]
        with self.assertRaises(ValidationError):
            replace(dataset, interactions=replace(dataset.interactions, source_bytes=99))
        with self.assertRaises(ValidationError):
            replace(dataset, side_sha256="0" * 64)
        with self.assertRaises(ValidationError):
            replace(dataset, overlaps=(NamespaceOverlap("z", 0, 0), NamespaceOverlap("a", 0, 0)))
        with self.assertRaises(ValidationError):
            replace(manifest, datasets=())
        with self.assertRaises(SerializationError):
            registry_module._ids([1])

    def test_optional_families_and_casefold_duplicate(self) -> None:
        user = self.dataset(
            "userOnly",
            {"inter": INTER, "user": b"user_id:token\tage:float\nu2\t21\n"},
        )
        item = self.dataset(
            "itemOnly",
            {"inter": INTER, "item": b"item_id:token\tgenre:token\ni2\tfolk\n"},
        )
        social = self.dataset(
            "social",
            {"inter": INTER, "net": b"source_id:token\ttarget_id:token\nu1\tu9\n"},
        )
        manifest = register_atomic_datasets([social, item, user])
        self.assertEqual(
            [part.name for part in manifest.datasets], ["itemOnly", "social", "userOnly"]
        )
        self.assertEqual(
            [(part.name, part.observed, part.matched) for part in manifest.datasets[0].overlaps],
            [("lexical_user_item_ids", 1, 0), ("side_items_in_inter", 1, 0)],
        )
        self.assertEqual(
            [(part.name, part.observed, part.matched) for part in manifest.datasets[1].overlaps],
            [("lexical_user_item_ids", 1, 0), ("network_users_in_inter", 2, 1)],
        )
        self.assertEqual(
            [(part.name, part.observed, part.matched) for part in manifest.datasets[2].overlaps],
            [("lexical_user_item_ids", 1, 0), ("side_users_in_inter", 1, 0)],
        )
        alternate_parent = self.root / "alternate"
        alternate_parent.mkdir()
        alternate = alternate_parent / "USERONLY"
        alternate.mkdir()
        (alternate / "USERONLY.inter").write_bytes(INTER)
        with self.assertRaisesRegex(ValidationError, "ignoring case"):
            register_atomic_datasets([user, NamedAtomicDataset("USERONLY", alternate)])

    def test_missing_pair_symlink_and_prefix_semantics(self) -> None:
        missing = self.dataset("missing", {"user": b"user_id:token\tage:float\nu1\t20\n"})
        with self.assertRaisesRegex(DatasetError, "required"):
            register_atomic_datasets([missing])
        one = self.dataset(
            "one",
            {"inter": INTER, "kg": b"head_id:token\trelation_id:token\ttail_id:token\ne1\tr\te2\n"},
        )
        with self.assertRaisesRegex(DatasetError, "together"):
            register_atomic_datasets([one])
        empty = self.dataset("empty", {"inter": b""})
        with self.assertRaisesRegex(DatasetError, "empty"):
            register_atomic_datasets([empty])
        extra = self.dataset("extra")
        (extra.directory / "not-extra.net").write_text("ignored", encoding="utf-8")
        self.assertEqual(len(register_atomic_datasets([extra]).datasets[0].files), 1)
        try:
            (extra.directory / "extra.net").symlink_to(extra.directory / "extra.inter")
        except (OSError, NotImplementedError):
            pass
        else:
            with self.assertRaisesRegex(DatasetError, "non-symlink"):
                register_atomic_datasets([extra])

    def test_bounded_input_output_and_corrupt_adapters(self) -> None:
        spec = self.dataset()
        with self.assertRaisesRegex(DatasetError, "max_file_bytes"):
            register_atomic_datasets([spec], limits=RegistryLimits(max_file_bytes=len(INTER) - 1))
        (spec.directory / "tiny.user").write_bytes(b"user_id:token\tage:float\nu1\t20\n")
        with self.assertRaisesRegex(DatasetError, "max_total_bytes"):
            register_atomic_datasets(
                [spec],
                limits=RegistryLimits(max_total_bytes=len(INTER), max_file_bytes=len(INTER)),
            )
        with self.assertRaisesRegex(SerializationError, "max_output_bytes"):
            save_atomic_registry(
                register_atomic_datasets([spec], limits=RegistryLimits(max_output_bytes=100)),
                self.root / "too-small",
            )
        self.assertFalse((self.root / "too-small").exists())
        (spec.directory / "tiny.inter").write_bytes(b"bad-header\nu1\ti1\n")
        with self.assertRaises(DatasetError):
            register_atomic_datasets([spec])

    def test_cli_register_verify_and_rejections(self) -> None:
        spec = self.dataset()
        registry = self.root / "records"
        args = ["--dataset", f"tiny={spec.directory}"]
        code, out, err = _cli(["register-recbole-datasets", *args, "--registry", str(registry)])
        self.assertEqual((code, err), (0, ""))
        record = Path(json.loads(out)["output"])
        self.assertTrue(record.is_file())
        self.assertEqual(_cli(["verify-recbole-registry", *args, "--record", str(record)])[0], 0)
        self.assertEqual(
            _cli(["register-recbole-datasets", *args, "--registry", str(registry)])[0], 2
        )
        self.assertEqual(
            _cli(
                ["register-recbole-datasets", "--dataset", "no-equals", "--registry", str(registry)]
            )[0],
            2,
        )
        self.assertEqual(
            _cli(
                [
                    "register-recbole-datasets",
                    *args,
                    "--minimum-rating",
                    "other=4",
                    "--registry",
                    str(registry),
                ]
            )[0],
            2,
        )
        self.assertEqual(
            _cli(["verify-recbole-registry", *args, "--record", str(self.root / "missing.json")])[
                0
            ],
            2,
        )
        self.assertEqual(
            _cli(
                [
                    "register-recbole-datasets",
                    *args,
                    "--minimum-rating",
                    "tiny=nope",
                    "--registry",
                    str(registry),
                ]
            )[0],
            2,
        )
        self.assertEqual(
            _cli(
                [
                    "register-recbole-datasets",
                    *args,
                    "--minimum-rating",
                    "tiny=1",
                    "--minimum-rating",
                    "tiny=2",
                    "--registry",
                    str(registry),
                ]
            )[0],
            2,
        )

    def test_output_path_and_artifact_size_rejections(self) -> None:
        spec = self.dataset()
        manifest = register_atomic_datasets([spec])
        occupied = self.root / "occupied"
        occupied.write_bytes(b"keep")
        with self.assertRaisesRegex(ValidationError, "directory"):
            save_atomic_registry(manifest, occupied)
        self.assertEqual(occupied.read_bytes(), b"keep")
        output = save_atomic_registry(manifest, self.root / "records")
        output.write_bytes(output.read_bytes() + b"x" * manifest.limits.max_output_bytes)
        self.assertFalse(verify_atomic_registry(output, [spec]))
        with self.assertRaises(SerializationError):
            verify_atomic_registry(self.root / f"{manifest.registry_id}.json", [spec])

    def test_staging_and_resolution_io_failures_are_domain_errors(self) -> None:
        spec = self.dataset()
        with (
            patch(
                "orchidrec.recbole_registry.tempfile.TemporaryDirectory",
                side_effect=OSError("disk full"),
            ),
            self.assertRaisesRegex(SerializationError, "could not stage"),
        ):
            register_atomic_datasets([spec])
        with (
            patch.object(Path, "resolve", side_effect=OSError("inaccessible")),
            self.assertRaisesRegex(DatasetError, "could not resolve"),
        ):
            register_atomic_datasets([spec])


if __name__ == "__main__":
    unittest.main()
