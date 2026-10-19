#!/usr/bin/env bash
set -e

# 1. Установить uv, если его нет
if ! command -v uv >/dev/null 2>&1; then
    echo "uv не найден, устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Создать виртуальное окружение
uv venv .venv --python 3.13

# 3. Установить зависимости (из uv.lock / pyproject.toml)
uv sync

# 4. Скачать артефакты (IR-датасет и т.д.)
./download_artefacts.sh

echo "Готово. Активируйте окружение: source .venv/bin/activate"
echo "Или запускайте через: uv run train.py"