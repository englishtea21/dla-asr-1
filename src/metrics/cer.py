from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


from src.text_encoder.beam_search import TextEncoderBeamSearch


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_width=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.decoder = TextEncoderBeamSearch(text_encoder, beam_width=beam_width)

    def __call__(self, log_probs, log_probs_length, text, **kwargs):
        cers = []
        predictions = log_probs.cpu() if hasattr(log_probs, "cpu") else log_probs
        lengths = (
            log_probs_length.detach().cpu().numpy()
            if hasattr(log_probs_length, "detach")
            else log_probs_length
        )
        for i in range(predictions.shape[0]):
            pred_text = self.decoder.decode(predictions[i, : int(lengths[i])])
            target_text = self.text_encoder.normalize_text(text[i])
            from src.metrics.utils import calc_cer

            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
