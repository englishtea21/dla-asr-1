# ASR (CTC) — HSE DLA homework

## Установка

```bash
./setup.sh
```

`setup.sh` ставит `uv`, создаёт `.venv` (Python 3.13), устанавливает зависимости и скачивает артефакты.

Ручная установка:
```bash
uv venv .venv --python 3.13
uv sync
./download_artefacts.sh
```

## Артефакты

`download_artefacts.sh` скачивает FLAIR RIR dataset (~110 MB) в `_data_for_augmentations/data_FLAIR.mat`. 
`ImpulseResponseAugmentation` ожидает файл по этому пути.

## Запуск

```bash
# обучение
uv run train.py -cn=baseline

# overfit-тест (2 примера)
uv run train.py -cn=baseline datasets=overfit trainer.epoch_len=1 trainer.n_epochs=50 \
  trainer.override=True dataloader.batch_size=2 +trainer.log_predictions_on_train=True

# инференс
uv run inference.py
```