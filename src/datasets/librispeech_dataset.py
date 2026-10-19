import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torchaudio
import soundfile
import wget
from datasets import load_dataset
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}

HF_CONFIG_SPLITS = {
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test"),
    "train-clean-100": ("clean", "train.100"),
    "train-clean-360": ("clean", "train.360"),
    "train-other-500": ("other", "train.500"),
}

HF_DATASET_NAME = "openslr/librispeech_asr"


class LibrispeechDataset(BaseDataset):
    def __init__(
        self,
        part,
        data_dir=None,
        use_hf=False,
        hf_cache_dir=None,
        force_redownload=False,
        take=None,
        *args,
        **kwargs,
    ):
        assert part in URL_LINKS or part == "train_all", f"Unknown part: {part}"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(exist_ok=True, parents=True)

        self._use_hf = use_hf
        self._force_redownload = force_redownload
        self._take = take
        self._hf_cache_dir = (
            Path(hf_cache_dir)
            if hf_cache_dir
            else ROOT_PATH / "data" / "hf_cache" / "librispeech"
        )

        if part == "train_all":
            parts = [p for p in URL_LINKS if "train" in p]
        else:
            parts = [part]

        if use_hf:
            index = sum([self._get_or_load_hf_index(p) for p in parts], [])
        else:
            index = sum([self._get_or_load_index(p) for p in parts], [])

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        wget.download(URL_LINKS[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)

        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))

        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"

        if index_path.exists() and not self._force_redownload:
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part

        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            for f in filenames:
                if f.endswith(".flac"):
                    flac_dirs.add(dirpath)

        for flac_dir in tqdm(
            list(flac_dirs),
            desc=f"Preparing librispeech folders: {part}",
            file=sys.stderr,
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]

            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"

                    t_info = soundfile.info(str(flac_path))
                    length = t_info.duration

                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )

        return index

    def _get_or_load_hf_index(self, part):
        index_path = self._data_dir / f"{part}_hf_index.json"

        if index_path.exists() and not self._force_redownload:
            with index_path.open() as f:
                return json.load(f)

        index = self._create_hf_index(part)

        with index_path.open("w") as f:
            json.dump(index, f, indent=2)

        return index

    def _create_hf_index(self, part):
        config, split = HF_CONFIG_SPLITS[part]

        dataset = load_dataset(
            HF_DATASET_NAME,
            config,
            split=split,
            streaming=True,
            cache_dir=str(self._hf_cache_dir),
        )

        if self._take is not None:
            dataset = dataset.take(self._take)

        index = []
        audio_dir = self._data_dir / "hf_audio" / part
        audio_dir.mkdir(exist_ok=True, parents=True)

        for item in tqdm(
            dataset, desc=f"Streaming librispeech: {part}", file=sys.stderr
        ):
            audio = item["audio"]
            audio_id = item.get("id")

            if not audio_id:
                raise ValueError("HF dataset item has no 'id' field.")

            audio_path = audio_dir / f"{audio_id}.wav"

            if not audio_path.exists():
                waveform = torch.as_tensor(audio["array"], dtype=torch.float32)

                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)

                torchaudio.save(
                    str(audio_path),
                    waveform,
                    audio["sampling_rate"],
                )

            audio_info = soundfile.info(str(audio_path))
            length = audio_info.duration

            index.append(
                {
                    "path": str(Path(audio_path).absolute().resolve()),
                    "text": item["text"].lower(),
                    "audio_len": length,
                }
            )

        return index
