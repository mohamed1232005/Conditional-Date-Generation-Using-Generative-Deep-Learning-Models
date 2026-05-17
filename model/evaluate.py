# evaluate.py
from datetime import date
import calendar
from typing import List


def validate_date(pred_str: str, conditions: List[str]) -> bool:
    """
    Returns True if pred_str satisfies all 4 conditions.

    Args:
        pred_str   : predicted date string e.g. '10-1-1810'
        conditions : list of 4 strings [day, month, leap, decade]
                     e.g. ['WED', 'JAN', 'False', '181']
    """
    try:
        d_str, m_str, y_str = pred_str.split("-")
        d, m, y = int(d_str), int(m_str), int(y_str)
        if not (1800 <= y <= 2200):
            return False
        dt = date(y, m, d)   # raises ValueError if date is invalid (e.g. Feb 30)
    except Exception:
        return False

    day_cond, mon_cond, leap_cond, dec_cond = conditions

    DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    if DAYS[dt.weekday()] != day_cond:   return False
    if MONTHS[m - 1]       != mon_cond:  return False

    is_leap = calendar.isleap(y)
    if str(is_leap) != leap_cond:        return False
    if str(y // 10) != dec_cond:         return False

    return True


def constraint_satisfaction_rate(
    preds: List[str],
    cond_list: List[List[str]],
) -> float:
    """
    Compute CSR: fraction of predictions that satisfy all 4 conditions.

    This is the main evaluation metric used during training (in train.py).

    Args:
        preds     : list of predicted date strings e.g. ['10-1-1810', ...]
        cond_list : list of condition string lists e.g. [['WED','JAN','False','181'], ...]

    Returns:
        float in [0, 1] — higher is better
    """
    if not preds:
        return 0.0
    correct = sum(
        validate_date(pred, conds)
        for pred, conds in zip(preds, cond_list)
    )
    return correct / len(preds)


def evaluate_model(predict_fn, val_data, tokenizer) -> float:
    """
    Evaluate a model using CSR on a val/test dataset.

    Args:
        predict_fn : callable that takes cond_tokens (list of ints) and
                     returns a date string e.g. '10-1-1810'
        val_data   : iterable of (cond_tokens_tensor, date_tokens_tensor)
        tokenizer  : Tokenizer instance

    Returns:
        CSR float in [0, 1]
    """
    correct = 0
    total   = 0
    for cond_tokens, _ in val_data:
        pred         = predict_fn(cond_tokens)
        cond_strings = tokenizer.decode_conditions(cond_tokens.tolist())
        correct     += validate_date(pred, cond_strings)
        total       += 1
    return correct / total if total > 0 else 0.0