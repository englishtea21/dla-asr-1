# HSE DLA HW-1 ASR (CTC)

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

`download_artefacts.sh` скачивает FLAIR RIR dataset (~110 MB) и чекпоинты моделей из Hugging Face (`kdemyokhin/dla-asr-1`) в `saved/`.

## Запуск

```bash
# обучение финальной модели
uv run train.py -cn=conformer_deeper3_ascii_exp_4 writer.run_name="conformer_deeper3_ascii_exp_4"

# инференс (argmax / beam search)
uv run inference.py -cn=inference_argmax model=conformer_deeper3 datasets=demo_librispeech transforms=basic dataloader=example inferencer.from_pretrained=saved/conformer_deeper3_ascii_exp_4/model_best.pth
uv run inference.py -cn=inference_beam model=conformer_deeper3 datasets=demo_librispeech transforms=basic dataloader=example inferencer.from_pretrained=saved/conformer_deeper3_ascii_exp_4/model_best.pth

# метрики по предсказаниям
uv run python calc_metrics.py --pred_dir data/saved/argmax/test
```

## Demo
Запускать в Colab
См. `DEMO.ipynb`.

## Report

Логи: [comet_ml](https://www.comet.com/liskirill37/dla-asr-hw-1)

Interactive Report: [comet_ml](https://www.comet.com/liskirill37/dla-asr-hw-1/reports/x3Kb4L1q1SaV36XKhQoHv5NRG)

