#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
COMMAND="${1:-train}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"

if [[ ! -x "${PYTHON}" ]]; then
    printf 'Python interpreter not found or not executable: %s\n' "${PYTHON}" >&2
    exit 1
fi

if ! cd "${PROJECT_ROOT}"; then
    printf 'Could not change to project directory: %s\n' "${PROJECT_ROOT}" >&2
    exit 1
fi

case "${COMMAND}" in
    train)
        module="Model.train_model"
        ;;
    predict)
        module="Model.inference"
        ;;
    evaluate)
        module="Model.evaluator"
        ;;
    *)
        printf 'Usage: %s [train|predict|evaluate] [arguments...]\n' "$0" >&2
        exit 2
        ;;
esac

if "${PYTHON}" -m "${module}" "${@:2}"; then
    exit 0
else
    status=$?
    printf 'Model command "%s" failed with exit code %s.\n' "${COMMAND}" "${status}" >&2
    exit "${status}"
fi