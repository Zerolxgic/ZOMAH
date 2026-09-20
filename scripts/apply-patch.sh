#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <patch-file-or-name>"
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
patch="$1"

if [[ "$patch" != /* ]]; then
    if [[ -f "$patch" ]]; then
        patch="$(realpath "$patch")"
    elif [[ -f "$HOME/Downloads/$patch" ]]; then
        patch="$HOME/Downloads/$patch"
    else
        echo "Patch not found: $patch"
        exit 1
    fi
fi

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit or stash changes first."
    exit 1
fi

echo "Checking: $patch"
git -C "$repo_root" apply --check "$patch"

echo "Applying: $patch"
git -C "$repo_root" apply "$patch"

echo
echo "Patch applied successfully."
git -C "$repo_root" status --short
