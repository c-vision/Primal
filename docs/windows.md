# Building on Windows

PrimalSolver builds on Windows with **MinGW-w64**. Microsoft Visual Studio
(`cl.exe`) is **not** supported yet; see [issue #1](https://github.com/c-vision/Primal/issues/1)
for what that would take.

Verified (see below): the library, `example_lp`, all 171 samples and the test
suite compile warning-free with `-std=c99 -Wall -Wextra -pedantic -O2` and link
as PE32+ executables with `x86_64-w64-mingw32-gcc`. The verification is
compile-and-link; the `.exe` were not executed here (no Wine on the build
machine). Running them on Windows itself is the same `make` you use elsewhere.

## Native build (MSYS2 or w64devkit)

The Makefile needs GNU `make` and a POSIX shell (`mkdir`, `rm`, `for`) — both
come with these environments.

**MSYS2** — install the toolchain, then use the "MinGW x64" shell:

```sh
pacman -S --needed mingw-w64-x86_64-gcc make
make                          # builds out/example_lp.exe and out/run_tests.exe
out/run_tests.exe             # the reliability suite
make samples                  # the 171 examples -> out/c_examples/<name>.exe
out/c_examples/logistic_large.exe
```

The Makefile is POSIX-oriented, but MSYS2 provides `make`, the shell and
`mkdir`/`rm`; Windows just appends `.exe` to every binary, so the outputs are
`out/*.exe` rather than `out/*`.

**w64devkit** — unpack it, open its shell (it ships gcc, make and busybox), and
run the same commands.

## Cross-compiling from Linux or macOS

Useful for a Windows binary without a Windows machine.

```sh
# Debian/Ubuntu:   apt install mingw-w64
# macOS (Homebrew): brew install mingw-w64

make CC=x86_64-w64-mingw32-gcc OUT=out-win     # -> out-win/example_lp.exe, out-win/run_tests.exe
```

The produced files are PE32+ executables with a `.exe` suffix; run them on
Windows, or under Wine. A single-shot build without `make` compiles the
`LIBSRCS` list from the Makefile plus a driver:

```sh
x86_64-w64-mingw32-gcc -std=c99 -Wall -Wextra -pedantic -O2 -I. \
    -o example_lp.exe example_lp.c <LIBSRCS> -lm
```

`<LIBSRCS>` is the `LIBSRCS` variable in the Makefile (the engine files plus the
`primal_*.c` units). winpthread is linked automatically; no `-lpthread` is
needed.

## What Windows needed

- **`setenv` / `unsetenv` do not exist on Windows** (neither MinGW nor MSVC).
  `test_primal.c` maps them onto the CRT `_putenv_s` under `_WIN32`. This was
  the only change the Windows build required — the library and the samples were
  already plain C99.
- `strdup` and `M_PI` are POSIX/BSD, not ISO C; the sources declare `strdup`
  (via `_POSIX_C_SOURCE`) and define `M_PI` under `#ifndef` where they use it.

## Why not MSVC (yet)

Three blockers remain, tracked in issue #1:

- **VLAs** in `socp.c`: MSVC's C compiler rejects runtime-sized arrays (`C2057`).
- **POSIX threads** (`<pthread.h>`) for the parallel MIP probing / strong
  branching / concurrent LP: MinGW ships winpthreads, MSVC does not.
- **Build system**: GNU make plus a POSIX shell, not `nmake`/`.vcxproj`.

Closing them (a portable threading layer, no VLAs, an MSVC project) is planned;
the MinGW path above works today.
