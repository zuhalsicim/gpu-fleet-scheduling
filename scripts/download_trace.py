"""Download the tables of the Alibaba PAI GPU trace (cluster-trace-gpu-v2020) that this project reads.

The trace is published by Alibaba under CC BY 4.0 at github.com/alibaba/clusterdata. It is not committed
here because of its size (about 150 MB compressed for the four tables below). Every archive and every
extracted CSV is checked against the SHA-256 sums published in the trace's README.

    python scripts/download_trace.py            # job, task, machine and group-tag tables into data/raw
    python scripts/download_trace.py --check    # only verify files that are already there
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/"

# name: (sha256 of .tar.gz, sha256 of the extracted .csv, compressed size in bytes)
TABLES: dict[str, tuple[str, str, int]] = {
    "pai_job_table": (
        "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
        "379ecb3becaba347f44a53bf7eb53e54b185221b2a0338a3f828828d269ba96c",
        62_065_432,
    ),
    "pai_task_table": (
        "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
        "6954802b457305f8a9e480ef97c40060baee59649fd3adc62c5a1e048aa058de",
        35_514_117,
    ),
    "pai_machine_spec": (
        "cc0d38a4045af1b1af8179de8b1b54b1ddd995e6160d6d061a6b1000f1276c2d",
        "251e9e33649155a6a935fdc1dc0e90cd4698206936b60bd8fce60ff2ccd51439",
        30_449,
    ),
    "pai_group_tag_table": (
        "722fef30b7fb7aa50dabd79155614b5423a9d65cf45a9b26c590d57725423a14",
        "d23fcb56d88b2976fda4a62708d77b2a8c98635c4b443f100ed5553d4cd07fbe",
        55_064_781,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    partial.replace(target)


def fetch_table(name: str, raw_dir: Path, check_only: bool) -> bool:
    archive_sha, csv_sha, size = TABLES[name]
    archive = raw_dir / f"{name}.tar.gz"
    csv = raw_dir / f"{name}.csv"
    if csv.exists() and sha256(csv) == csv_sha:
        print(f"ok        {csv.name}")
        return True
    if check_only:
        print(f"missing   {csv.name}")
        return False
    if not (archive.exists() and sha256(archive) == archive_sha):
        print(f"download  {archive.name} ({size / 1e6:.1f} MB)")
        download(BASE_URL + archive.name, archive)
        if sha256(archive) != archive_sha:
            print(f"checksum mismatch for {archive.name}", file=sys.stderr)
            return False
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith(f"{name}.csv"))
        member.name = csv.name  # flatten any directory inside the archive
        tar.extract(member, raw_dir, filter="data")
    if sha256(csv) != csv_sha:
        print(f"checksum mismatch for {csv.name}", file=sys.stderr)
        return False
    print(f"ok        {csv.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--check", action="store_true", help="verify existing files, download nothing")
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    results = [fetch_table(name, args.raw_dir, args.check) for name in TABLES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
