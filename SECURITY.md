# Security policy

OrchidRec is an offline experiment toolkit and does not make network calls. Interaction files and saved models are nevertheless untrusted inputs. Model loading enforces a 256 MiB file ceiling and strict JSON depth, numeric, UTF-8, Unicode-scalar, and duplicate-key checks; callers should still apply tighter deployment-specific limits where appropriate. Avoid real personal identifiers in public fixtures, and keep sensitive feedback outside the repository.

Report security-sensitive problems privately through GitHub's security advisory interface rather than a public issue. The supported version is the latest commit on the default branch.
