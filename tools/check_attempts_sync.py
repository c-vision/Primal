#!/usr/bin/env python3
"""Controlla che lo specchio versionato di attempts.md sia allineato all'originale.

`attempts.md` sta nel repo PrimalSolver ma e' escluso via `.gitignore`: il
registro completo dei tentativi (inclusi i fallimenti e le ragioni per cui sono
stati respinti) esiste solo sulla macchina che ha prodotto il lavoro. Per questo
nel vault ne e' stato messo uno specchio versionato. Uno specchio che puo
divergere in silenzio e' peggio di nessuno specchio, quindi qui si verifica.

Uso:
    python3 tools/check_attempts_sync.py            # exit 1 se divergono
    python3 tools/check_attempts_sync.py --quiet    # solo il codice di uscita

Exit codes: 0 allineati, 1 divergenti, 2 file illeggibili.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MIRROR = Path(
    "/Users/gaetano/Documents/AiWiki/01-Projects/PrimalSolver/Knowledge"
    "/PrimalSolver-Log-Completo.md"
)
ORIGINAL = Path(__file__).resolve().parent.parent / "attempts.md"

# Le voci sono intestazioni di secondo livello "## F<n> — ..." oppure "## F<n>: ...".
# Gli slashes sono parte della voce ("## F38/F39 ..."): senza questo, "F38/F39" verrebbe
# letto come "F38" e una voce "F38" separata risulterebbe duplicata.
ENTRY = re.compile(r"^##\s+(F\d+(?:/F\d+)*)\b", re.MULTILINE)


def _sort_key(key: str) -> tuple[int, ...]:
    return tuple(int(p[1:]) for p in key.split("/"))


def entries(text: str) -> dict[str, int]:
    """Mappa voce -> numero di righe, per trovare le voci e il loro peso."""
    out: dict[str, int] = {}
    matches = list(ENTRY.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text.count("\n", m.start(), end)
    return out


def strip_header(text: str) -> str:
    """Toglie l'intestazione aggiunta dallo specchio, che non sta nell'originale."""
    marker = "\n---\n\n"
    i = text.find(marker)
    return text[i + len(marker):] if i != -1 else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not ORIGINAL.exists():
        print(f"originale non trovato: {ORIGINAL}", file=sys.stderr)
        return 2
    if not MIRROR.exists():
        print(f"specchio non trovato: {MIRROR}", file=sys.stderr)
        return 2

    src = ORIGINAL.read_text()
    mirror_body = strip_header(MIRROR.read_text())

    same_body = src.rstrip() == mirror_body.rstrip()
    a, b = entries(src), entries(mirror_body)

    only_src = sorted(set(a) - set(b), key=_sort_key)
    only_mirror = sorted(set(b) - set(a), key=_sort_key)
    grew = sorted((k for k in set(a) & set(b) if a[k] != b[k]), key=_sort_key)

    diverged = not same_body or only_src or only_mirror or grew
    if not args.quiet:
        print(f"originale : {ORIGINAL} ({len(src.splitlines())} righe, {len(a)} voci)")
        print(f"specchio  : {MIRROR} ({len(mirror_body.splitlines())} righe, {len(b)} voci)")
        if only_src:
            print(f"  voci solo nell'originale (da specchiare): {', '.join(only_src)}")
        if only_mirror:
            print(f"  voci solo nello specchio (da riportare): {', '.join(only_mirror)}")
        if grew:
            print("  voci con lunghezza diversa: "
                  + ", ".join(f"{k} ({a[k]}->{b[k]} righe)" for k in grew))
        if diverged:
            print("DIVERGENTI: rigenera lo specchio vault.")
        else:
            print("allineati.")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
