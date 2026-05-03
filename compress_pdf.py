#!/usr/bin/env python3
"""
compress_pdf.py — Compress one or many PDF files using Ghostscript.

Single file:
    python3 compress_pdf.py report.pdf
    python3 compress_pdf.py report.pdf --quality screen --overwrite

Batch:
    python3 compress_pdf.py *.pdf --outdir compressed/
    python3 compress_pdf.py a.pdf b.pdf c.pdf --min-savings 5

Recursive (all PDFs in all subfolders, preserving folder structure):
    python3 compress_pdf.py . --recursive --outdir compressed/
    python3 compress_pdf.py docs/ --recursive --quality screen --overwrite

Parallel (use multiple CPU cores — great for large batches):
    python3 compress_pdf.py . -r --overwrite --workers 8

Options:
    --quality      screen | ebook (default) | printer | prepress
    --outdir       Save to a different directory (structure preserved with --recursive)
    --suffix       Suffix before .pdf (default: _compressed)
    --overwrite    Replace originals in place
    --force        Overwrite existing output files (default: skip if output already exists)
    --min-savings  Skip file if savings are below N% (e.g. --min-savings 1)
                   Files that would grow are always skipped automatically.
    --recursive    Walk all subfolders under the given path(s)
    --workers      Number of parallel workers (default: 1)
    --timeout      Kill Ghostscript after N seconds (e.g. --timeout 120)
    --log          Write a results log to this file (e.g. --log compress.log)
"""

import subprocess
import sys
import os
import glob
import argparse
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


QUALITY_SETTINGS = {
    "screen":   "/screen",
    "ebook":    "/ebook",
    "printer":  "/printer",
    "prepress": "/prepress",
}


def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def tprint(msg, progress=None):
    """Print without corrupting the tqdm progress bar."""
    if progress is not None and HAS_TQDM:
        tqdm.write(msg)
    else:
        print(msg)


def resolve_symlink(path):
    try:
        return os.path.realpath(path)
    except OSError:
        return path


def compress_one(input_path, output_path, quality, min_savings=0, timeout=None):
    """
    Compress a single PDF.
    Returns (input_path, output_path, original_bytes, saved_bytes, status)
    where status is: 'ok' | 'skipped' | 'timeout' | 'error:<msg>'

    saved_bytes is 0 for any non-'ok' status so stats stay accurate.
    """
    try:
        dPDFSETTINGS = QUALITY_SETTINGS[quality]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        tmp_path = output_path + ".tmp.pdf"

        cmd = [
            "gs",
            "-dSAFER",                          # Fix 5: always sandbox gs
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={dPDFSETTINGS}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={tmp_path}",
            input_path,
        ]

        original_size = os.path.getsize(input_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return input_path, output_path, original_size, 0, "timeout"

        if result.returncode != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return input_path, output_path, original_size, 0, f"error:{result.stderr.strip()}"

        compressed_size = os.path.getsize(tmp_path)

        # Always skip if output would be larger or equal
        if compressed_size >= original_size:
            os.remove(tmp_path)
            return input_path, output_path, original_size, 0, "skipped"

        # Skip if savings fall below threshold
        savings_pct = (1 - compressed_size / original_size) * 100
        if savings_pct < min_savings:
            os.remove(tmp_path)
            return input_path, output_path, original_size, 0, "skipped"

        os.replace(tmp_path, output_path)
        saved = original_size - compressed_size
        return input_path, output_path, original_size, saved, "ok"

    except Exception as e:                      # Fix 2: catch all OS/unexpected errors
        if os.path.exists(output_path + ".tmp.pdf"):
            try:
                os.remove(output_path + ".tmp.pdf")
            except OSError:
                pass
        return input_path, output_path, 0, 0, f"error:{type(e).__name__}: {e}"


def compress_one_star(args):
    """Unpacking wrapper for ProcessPoolExecutor."""
    return compress_one(*args)


def collect_pdfs(inputs, recursive):
    """Return list of (input_path, rel_path) tuples, resolving symlinks to avoid duplicates."""
    seen_real = set()
    results = []

    def add(path, rel):
        real = resolve_symlink(path)
        if real not in seen_real:
            seen_real.add(real)
            results.append((os.path.normpath(path), rel))

    for entry in inputs:
        matches = glob.glob(entry)
        if not matches:
            matches = [entry]

        for match in matches:
            if os.path.isdir(match) and recursive:
                base = os.path.normpath(match)
                for dirpath, _, filenames in os.walk(base, followlinks=False):
                    for fname in sorted(filenames):
                        if fname.lower().endswith(".pdf"):
                            full = os.path.join(dirpath, fname)
                            rel = os.path.relpath(full, base)
                            add(full, rel)
            elif os.path.isfile(match) and match.lower().endswith(".pdf"):
                add(match, os.path.basename(match))
            elif os.path.isdir(match):
                print(f"  ! '{match}' is a directory — use --recursive to walk it")
            elif match.lower().endswith(".pdf"):
                print(f"  ! '{match}' — file not found")  # Fix 1: warn on missing .pdf
            else:
                print(f"  ! Skipping non-PDF: {match}")

    return results


def build_output_path(input_path, rel_path, outdir, suffix, overwrite):
    if overwrite:
        return input_path
    base, ext = os.path.splitext(rel_path)
    out_rel = base + suffix + ext
    if outdir:
        return os.path.join(outdir, out_rel)
    return os.path.join(
        os.path.dirname(input_path) or ".",
        os.path.basename(base) + suffix + ext
    )


def setup_logger(log_path):
    logger = logging.getLogger("compress_pdf")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Compress PDF files using Ghostscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="PDF files, folders, or glob patterns")
    parser.add_argument("--quality", "-q", default="ebook", choices=QUALITY_SETTINGS,
                        help="Compression quality (default: ebook)")
    parser.add_argument("--outdir", "-o", default=None,
                        help="Output directory (folder structure preserved with --recursive)")
    parser.add_argument("--suffix", default="_compressed",
                        help="Suffix added to output filenames (default: _compressed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace original files in place")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files (default: skip if output exists)")
    parser.add_argument("--min-savings", type=float, default=0, metavar="PCT",
                        help="Also skip if savings < PCT%% (files that grow are always skipped)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Walk all subfolders under the given path(s)")
    parser.add_argument("--workers", "-w", type=int, default=1, metavar="N",
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--timeout", type=int, default=None, metavar="SEC",
                        help="Kill Ghostscript if it takes longer than SEC seconds")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Write results log to FILE")

    args = parser.parse_args()

    if args.overwrite and args.outdir:
        parser.error("--overwrite and --outdir cannot be used together.")

    if not shutil.which("gs"):
        print("Error: Ghostscript (gs) is not installed or not in PATH.")
        print("Install with: sudo dnf install ghostscript")
        sys.exit(1)

    logger = setup_logger(args.log) if args.log else None
    if logger:
        logger.info(f"Started — quality={args.quality} min_savings={args.min_savings} "
                    f"workers={args.workers} timeout={args.timeout}")

    pdfs = collect_pdfs(args.inputs, args.recursive)
    if not pdfs:
        print("No PDF files found.")
        sys.exit(1)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    rel_by_input = {os.path.normpath(ip): rp for ip, rp in pdfs}

    jobs = []
    for input_path, rel_path in pdfs:
        if not os.path.isfile(input_path):
            tprint(f"  ✗ {input_path} — file not found")
            if logger:
                logger.error(f"NOT FOUND  {input_path}")
            continue

        out = build_output_path(input_path, rel_path, args.outdir, args.suffix, args.overwrite)

        # Fix 6: skip if output already exists, unless --force or --overwrite
        if not args.overwrite and not args.force and os.path.exists(out):
            tprint(f"  – {rel_path} — output already exists, skipping (use --force to overwrite)")
            if logger:
                logger.info(f"EXISTS     {out}")
            continue

        jobs.append((input_path, out, args.quality, args.min_savings, args.timeout))

    if not jobs:
        print("Nothing to do.")
        sys.exit(0)

    # Fix 3: track only bytes actually saved (saved=0 for skipped/failed)
    total_scanned_bytes = 0   # original size of all processed files
    total_saved_bytes = 0     # bytes actually recovered
    ok = skipped = failed = timed_out = 0
    workers = min(args.workers, len(jobs))

    batch = len(jobs) > 1
    if batch:
        parallel_note = f", {workers} workers" if workers > 1 else ""
        print(f"Compressing {len(jobs)} files  [quality: {args.quality}{parallel_note}]\n")

    progress = None
    if HAS_TQDM and batch:
        progress = tqdm(total=len(jobs), unit="file", ncols=72, leave=False)

    def handle_result(result):
        nonlocal total_scanned_bytes, total_saved_bytes, ok, skipped, failed, timed_out
        input_path, output_path, orig, saved, status = result
        rel = rel_by_input.get(os.path.normpath(input_path), os.path.basename(input_path))

        total_scanned_bytes += orig
        total_saved_bytes += saved  # 0 unless status == 'ok'

        if status == "ok":
            comp = orig - saved
            ratio = (saved / orig * 100) if orig > 0 else 0
            msg = f"  ✓ {rel}  {human_size(orig)} → {human_size(comp)}  (-{ratio:.1f}%)"
            tprint(msg, progress)
            if logger:
                logger.info(f"OK        -{ratio:.1f}%  {input_path}")
            ok += 1
        elif status == "skipped":
            msg = f"  – {rel} ({human_size(orig)}) — skipped"
            tprint(msg, progress)
            if logger:
                logger.info(f"SKIPPED   {input_path}")
            skipped += 1
        elif status == "timeout":
            msg = f"  ✗ {rel} — timed out after {args.timeout}s"
            tprint(msg, progress)
            if logger:
                logger.warning(f"TIMEOUT   {input_path}")
            timed_out += 1
        else:
            err = status.removeprefix("error:")
            msg = f"  ✗ {rel} — {err}"
            tprint(msg, progress)
            if logger:
                logger.error(f"FAILED    {input_path}  —  {err}")
            failed += 1

    if workers <= 1:
        for job in jobs:
            handle_result(compress_one(*job))
            if progress:
                progress.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compress_one_star, job): job for job in jobs}
            for future in as_completed(futures):
                handle_result(future.result())
                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    if batch:
        # Fix 3: report savings on compressed files only, not the whole batch
        compressed_original = total_scanned_bytes - sum(
            0 for _ in range(skipped + failed + timed_out)
        )
        pct = (total_saved_bytes / total_scanned_bytes * 100) if total_scanned_bytes > 0 else 0
        summary = (f"\nDone: {ok} compressed, {skipped} skipped, "
                   f"{timed_out} timed out, {failed} failed")
        saved_line = f"Saved: {human_size(total_saved_bytes)} ({pct:.1f}% of total scanned)"
        print(summary)
        print(saved_line)
        if logger:
            logger.info(summary.strip())
            logger.info(saved_line)


if __name__ == "__main__":
    main()
