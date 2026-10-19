import editdistance
import jiwer

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text):
        return editdistance.eval(target_text, predicted_text) / len(target_text)
    return 0.0


def calc_wer(target_text, predicted_text) -> float:
    return jiwer.wer(target_text, predicted_text)
