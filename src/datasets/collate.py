import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch = {}
    batch["text"] = [item["text"] for item in dataset_items]
    batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    specs = [item["spectrogram"].squeeze(0) for item in dataset_items]  # [n_mels, T_i]
    batch["spectrogram_length"] = torch.tensor([s.shape[-1] for s in specs])  # [B]
    specs = [s.transpose(0, 1) for s in specs]  # [T_i, n_mels]
    specs = torch.nn.utils.rnn.pad_sequence(
        specs, batch_first=True
    )  # [B, T_max, n_mels]
    batch["spectrogram"] = specs.transpose(1, 2)  # [B, n_mels, T_max]

    targets = [
        item["text_encoded"].squeeze(0).long() for item in dataset_items
    ]  # [L_i]
    batch["text_encoded_length"] = torch.tensor([t.shape[0] for t in targets])  # [B]
    batch["text_encoded"] = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True
    )  # [B, L_max]

    # аудио для логов и возможных будущих батч аугментаций на waveform
    waves = [item["audio"].squeeze(0) for item in dataset_items]  # [T_i]
    batch["audio_length"] = torch.tensor([w.shape[0] for w in waves])  # [B]
    batch["audio"] = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True).unsqueeze(
        1
    )  # [B, 1, T_max]

    return batch
