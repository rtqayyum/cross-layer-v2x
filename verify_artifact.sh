#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
shasum -a 256 -c CHECKSUMS.sha256

if rg -n '/Users/|/home/|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+' \
  --glob '!CHECKSUMS.sha256' --glob '!verify_artifact.sh' .; then
  echo "Potential private path or secret found." >&2
  exit 2
fi

echo "Artifact integrity and public-path checks passed."
