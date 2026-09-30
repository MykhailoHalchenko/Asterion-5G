#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-gui}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    printf 'Project directory not found: %s\n' "${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ $# -gt 0 && "${COMMAND}" == "" ]]; then
    printf 'Command must not be empty.\n' >&2
    exit 2
fi

case "${COMMAND}" in
    gui|app)
        exec "${PROJECT_ROOT}/run_app.sh" "${@:2}"
        ;;
    5g|demo)
        exec "${PROJECT_ROOT}/run_5g.sh" "${@:2}"
        ;;
    train|predict|evaluate)
        exec "${PROJECT_ROOT}/run_model.sh" "${COMMAND}" "${@:2}"
        ;;
    help|--help|-h)
        printf 'Usage: %s [gui|5g|train|predict|evaluate] [arguments...]\n' "$0"
        printf '\nCommands:\n'
        printf '  gui       Start the CustomTkinter interface (default)\n'
        printf '  5g        Start the continuous 5G sensing demo\n'
        printf '  train     Train the model on training_*.parquet files\n'
        printf '  predict   Generate predictions from a saved model\n'
        printf '  evaluate  Evaluate the saved model\n'
        exit 0
        ;;
    *)
        printf 'Unknown command: %s\n' "${COMMAND}" >&2
        printf 'Use "%s --help" to list available commands.\n' "$0" >&2
        exit 2
        ;;
esac
