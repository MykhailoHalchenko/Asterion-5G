#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"

if [[ ! -x "${PYTHON}" ]]; then
    printf 'Python interpreter not found or not executable: %s\n' "${PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/app.py" ]]; then
    printf 'Application entrypoint not found: %s\n' "${PROJECT_ROOT}/app.py" >&2
    exit 1
fi

if ! cd "${PROJECT_ROOT}"; then
    printf 'Could not change to project directory: %s\n' "${PROJECT_ROOT}" >&2
    exit 1
fi

if "${PYTHON}" app.py "$@"; then
    exit 0
else
    status=$?
    printf 'GUI failed with exit code %s.\n' "${status}" >&2
    exit "${status}"
fi
