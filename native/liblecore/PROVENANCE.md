# liblecore native provenance

*Status: clean-room ledger for the ABI-0 preview.*

All canonical files under `native/liblecore` are newly written for this MIT-licensed repository from the public
semantic descriptions in [`docs/ISA.md`](../../docs/ISA.md), the repository's NumPy definitional reference, and the
requirements in [`PRD.md`](../../PRD.md) and [`ENG.md`](../../ENG.md). No C implementation from another project was
copied.

| Material | Role | Provenance and license decision |
|---|---|---|
| `include/lecore/*.h`, `src/*.c`, CMake/package files, tests, and examples | Canonical liblecore implementation | Independently written in this repository; covered by the repository MIT license. |
| `docs/ISA.md` and `holographic/misc/holographic_reference.py` | Normative semantics and differential oracle | Existing leCore sources in this MIT repository. |
| CRC-64/ECMA-182 parameters | Interchange checksum specification | Public algorithm parameters; implementation written directly for liblecore. |
| C and C++ standard library interfaces | Hosted allocation, math, and portability substrate | Platform interfaces; no vendored source. |

Nearby implementations found during workspace analysis are compatibility targets or research evidence only:

- Signal's holographic kernel is AGPL-licensed and may be used as an adopter/conformance oracle, but its source is
  not a source for this MIT implementation.
- `leos-c` and `asix` use normalized or otherwise different HRR semantics and have incomplete or unclear packaging
  provenance; no source was copied.
- Holonet's MAP algebra, NoSQLite and Zero LM text hypervectors, and NSRL's integer associative memory are distinct
  profiles, not implementations of this HRR profile.

Any future imported, generated, or vendored native material must be added to this ledger with its exact origin,
license, modifications, and affected files before release.
