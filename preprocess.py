"""
Preprocessing pipeline: malware binaries -> opcode sequence text files.

Expected directory structure:
    samples/
        FamilyName1/
            binary1
            binary2
        FamilyName2/
            binary3

Output structure:
    opcodes/
        FamilyName1/
            binary1.txt   <- one opcode per line
            binary2.txt
        FamilyName2/
            binary3.txt

Usage:
    python preprocess.py --input samples/ --output opcodes/
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path


def extract_opcodes(binary_path: Path) -> list[str]:
    """
    Disassemble a binary with objdump and return a list of mnemonic opcodes.
    Filters out <unknown> entries and empty lines.
    """
    try:
        result = subprocess.run(
            ["objdump", "-d", str(binary_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {binary_path.name}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  [ERROR] {binary_path.name}: {e}", file=sys.stderr)
        return []

    opcodes = []
    for line in result.stdout.splitlines():
        # Instruction lines start with whitespace followed by a hex address and colon
        stripped = line.lstrip()
        if not stripped or stripped[0] not in "0123456789abcdef":
            continue
        if ":" not in stripped:
            continue

        # LLVM objdump format: "  address: bytes\topcode\toperands"
        # Split on tab — field index 1 is the mnemonic
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        opcode = parts[1].strip()
        if not opcode or opcode == "<unknown>":
            continue

        opcodes.append(opcode)

    return opcodes


def preprocess(input_dir: Path, output_dir: Path):
    families = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]

    if not families:
        # Flat layout — treat all files as one unnamed family
        print(f"No subdirectories found. Treating all files in {input_dir} as one family.")
        families_map = {"unknown_family": list(input_dir.iterdir())}
    else:
        families_map = {d.name: list(d.iterdir()) for d in families}

    total_files = sum(len(v) for v in families_map.values())
    processed = 0
    skipped = 0

    for family, binaries in families_map.items():
        family_out = output_dir / family
        family_out.mkdir(parents=True, exist_ok=True)

        for binary in sorted(binaries):
            if not binary.is_file():
                continue

            out_file = family_out / (binary.name + ".txt")

            # Skip if already processed
            if out_file.exists():
                processed += 1
                continue

            opcodes = extract_opcodes(binary)

            if not opcodes:
                print(f"  [SKIP] {family}/{binary.name} — no opcodes extracted")
                skipped += 1
                continue

            out_file.write_text("\n".join(opcodes))
            processed += 1
            print(f"  [OK] {family}/{binary.name} — {len(opcodes)} opcodes")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total: {total_files}")


def main():
    parser = argparse.ArgumentParser(description="Extract opcode sequences from malware binaries.")
    parser.add_argument("--input", required=True, type=Path, help="Input directory (family subdirs or flat)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for opcode .txt files")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
