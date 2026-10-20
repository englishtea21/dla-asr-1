import math

import numpy as np
import torch


def _logaddexp(a, b):
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class TextEncoderBeamSearch:
    def __init__(self, text_encoder, beam_width=5, **kwargs):
        self.text_encoder = text_encoder
        self.beam_width = beam_width

    def decode(self, log_probs):
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        log_probs = np.asarray(log_probs)
        if log_probs.ndim == 3:
            log_probs = log_probs[0]

        time_steps, vocab_size = log_probs.shape
        blank = 0
        beams = {(): [0.0, float("-inf")]}

        for t in range(time_steps):
            new_beams = {}
            for prefix, (pb, pnb) in beams.items():
                for v in range(vocab_size):
                    p = float(log_probs[t, v])
                    if v == blank:
                        cur_pb, cur_pnb = new_beams.get(
                            prefix, [float("-inf"), float("-inf")]
                        )
                        cur_pb = _logaddexp(cur_pb, _logaddexp(pb, pnb) + p)
                        new_beams[prefix] = [cur_pb, cur_pnb]
                    elif prefix and v == prefix[-1]:
                        cur_pb, cur_pnb = new_beams.get(
                            prefix, [float("-inf"), float("-inf")]
                        )
                        cur_pnb = _logaddexp(cur_pnb, pnb + p)
                        new_beams[prefix] = [cur_pb, cur_pnb]
                        new_prefix = prefix + (v,)
                        cur_pb2, cur_pnb2 = new_beams.get(
                            new_prefix, [float("-inf"), float("-inf")]
                        )
                        cur_pnb2 = _logaddexp(cur_pnb2, pb + p)
                        new_beams[new_prefix] = [cur_pb2, cur_pnb2]
                    else:
                        new_prefix = prefix + (v,)
                        cur_pb2, cur_pnb2 = new_beams.get(
                            new_prefix, [float("-inf"), float("-inf")]
                        )
                        cur_pnb2 = _logaddexp(cur_pnb2, _logaddexp(pb, pnb) + p)
                        new_beams[new_prefix] = [cur_pb2, cur_pnb2]

            scored = []
            for prefix, (pb, pnb) in new_beams.items():
                scored.append((_logaddexp(pb, pnb), prefix, pb, pnb))
            scored.sort(reverse=True)
            beams = {}
            for total, prefix, pb, pnb in scored[: self.beam_width]:
                beams[prefix] = [pb, pnb]

        if not beams:
            return ""
        best_prefix = max(beams.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]))[
            0
        ]
        return "".join(
            self.text_encoder.ind2char[int(i)] for i in best_prefix if int(i) != 0
        ).strip()

    def __call__(self, log_probs):
        return self.decode(log_probs)
