# I/O formats

**Source:** `mpsio.c`, `cbf.c`

Four formats read, three written. `PRIMAL_readdata` and `PRIMAL_writedata`
dispatch on the file extension; `PRIMAL_readdataformat` takes an explicit
format code instead.

| Format | Read | Write | Notes |
|---|---|---|---|
| MPS | yes | yes | `QSECTION`/`QUADOBJ`/`QMATRIX` for a quadratic objective, `QCMATRIX` for quadratic rows, `CSECTION` for cones |
| LP | yes | yes | linear part only |
| OPF | yes | yes | linear/quadratic/conic/integer core |
| CBF | yes | yes | v4, including the `CHANGE` section |
| TASK, PTF | no | no | no reader, and `readdataformat` refuses them explicitly |

## Choosing the format

`PRIMAL_readdataautoformat` detects an **OPF from its content** (a file
starting with `[`) and otherwise dispatches on the extension.

`PRIMAL_readdataformat` respects the format and compression codes it is given.
Accepted format codes: extension-detect, MPS, LP, OPF, CBF; the remaining
codes are refused with `ERR_ARG` because there is no reader for them. GZIP and
ZSTD compression codes are **also** refused: nothing is decompressed, since no
decompression library is linked. That is a declared limitation, not a
configuration.

## MPS

The most complete reader here.

- **Quadratic objective**: `QSECTION` with `QMATRIX` rows, also accepted as
  `QUADOBJ` and `QUADOBJ` blocks.
- **Quadratic rows**: `QCMATRIX`. One `L` row is read; a `G` or `E` row is
  **refused** rather than having its quadratic part silently dropped — a
  refused model is better than a wrong one.
- **Cones**: `CSECTION` (a MOSEK extension) reads and writes `QUAD`, `RQUAD`,
  `PEXP`, `DEXP`, `PPOW`, `DPOW`. `ZERO` is refused. A full round-trip of a
  cone is asserted by the test suite.
- **Rows**: `N`, `L`, `G`, `E`, and free rows are all supported, and a free row
  round-trips without loss.
- **Names**: the readers fill the task's name tables at the index each label
  names, so after a read `PRIMAL_getvarnameidx("x7")` finds column 7. A file
  with **two rows of the same name** is refused *before* anything reaches the
  task: the parser looks rows up by name, so a second row with an existing
  label would silently receive no coefficients at all.
- **Writer**: names rows and columns as the task names them, with a positional
  fallback for an empty name or one containing whitespace (MPS/LP names are
  whitespace-delimited), and writes the `OBJNAME` section.

## LP

- The reader refuses the `[ ... ]/2` bracket form. The quadratic LP format is
  not read, rather than being read into a junk variable called `[`.
- The writer emits the linear part only and **refuses** a model that has
  `qobj` or `qcon` terms, instead of writing a model that is not the one it
  was given.

Reading the full quadratic LP syntax is a declared gap: it would need a shared
tokenizer with the MPS reader.

## OPF

The linear, quadratic, conic and integer core, with `opf_read` / `opf_write`
and the data callbacks `READ_OPF`, `READ_OPF_SECTION`, `WRITE_OPF` routed
through `PRIMAL_putcallbackfunc`.

- Convex quadratic constraints are read: an "up" form is passed through
  directly, and `q(x) ≥ b` is rewritten as `−q(x) ≤ −b`.
- Equality and ranged quadratic rows are refused.
- Declared deviations: the `zero` cone is not read, `[solutions]` and
  `[vendor]` sections are not read, and `dpow` maps to `RPOW`.

## CBF

CBF v4, cones plus SDP, including the `CHANGE` section.

CBF is **not** subject to the one-name-one-index rule that MPS, LP and OPF
follow, and deliberately so: a CBF group label names a *set* of indices
together, so a name there does not identify a single index and populating a
table with it would give CBF a rule the other three do not have.

A `POW*CONES` section is refused rather than skipped — skipping it would
produce an incomplete model that looks fine.

## Solution files

Text, native binary and JSON solution imports require complete, finite data
matching the task dimensions. Malformed imports leave the previous result
unchanged. Imported vectors can be inspected, but statuses are `UNKNOWN` and
objective getters remain unavailable until optimization. Binary compression
is unsupported and nonzero compression arguments are rejected.
The original ten-key JSON format remains readable for scalar solutions;
it contains no bar-matrix data.

## Limits

- No compression, on read or write.
- No LP quadratic bracket syntax; no TASK or PTF.
- The MPS `G`/`E` quadratic row and the OPF equality/ranged quadratic row are
  refused, not supported.
- `PRIMAL_getsymmatinfo` refuses a null `dim`.
