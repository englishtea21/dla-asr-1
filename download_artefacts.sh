#!/usr/bin/env bash
set -e

# Скрипт скачивания артефактов: IR-датасет FLAIR и (может быть) чекпоинты LM.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# FLAIR RIR dataset 
FLAIR_DIR="_data_for_augmentations"
FLAIR_FILE="$FLAIR_DIR/data_FLAIR.mat"
FLAIR_URL="https://zenodo.org/api/records/17037517/files/data_FLAIR.mat/content"

mkdir -p "$FLAIR_DIR"

if [ -f "$FLAIR_FILE" ]; then
    echo "FLAIR уже скачан: $FLAIR_FILE"
else
    echo "Скачиваю FLAIR RIR dataset (~110 MB)..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "$FLAIR_FILE" "$FLAIR_URL"
    else
        curl -L -o "$FLAIR_FILE" "$FLAIR_URL"
    fi
    echo "Готово: $FLAIR_FILE"
fi

# (мб) чекпоинты LM

# Чекпоинты моделей из hf в saved/
echo "Скачиваю чекпоинты из hf (kdemyokhin/dla-asr-1)..."
mkdir -p saved
HF_TOKEN="${HF_TOKEN:-}" uv run python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('kdemyokhin/dla-asr-1', local_dir='saved', token=os.environ.get('HF_TOKEN') or None)"
echo "Чекпоинты готовы: saved/"

echo "Все артефакты готовы."
