#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    printf 'Python interpreter not found or not executable: %s\n' "${PYTHON}" >&2
    exit 1
fi

if ! cd "${PROJECT_ROOT}"; then
    printf 'Could not change to project directory: %s\n' "${PROJECT_ROOT}" >&2
    exit 1
fi

if "${PYTHON}" -m 5Gsim.demo "$@"; then
    exit 0
else
    status=$?
    printf '5G demo failed with exit code %s.\n' "${status}" >&2
    exit "${status}"
fi
