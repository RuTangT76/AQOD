"""Sampled-token OPD signal; no ATOD schedule, turn weights, or RL reward."""

import numpy as np


def sampled_opd_signal(student_logp, teacher_logp, response_mask):
    student = np.asarray(student_logp, dtype=np.float32)
    teacher = np.asarray(teacher_logp, dtype=np.float32)
    mask = np.asarray(response_mask)
    if student.ndim != 2 or student.shape != teacher.shape or student.shape != mask.shape:
        raise ValueError('Log probabilities and mask must share [turn, token] shape')
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('A nonempty binary response mask is required')
    active = mask.astype(bool)
    if not np.isfinite(student[active]).all() or not np.isfinite(teacher[active]).all():
        raise ValueError('Nonfinite log probability on a generated token')
    signal = np.zeros_like(student)
    signal[active] = teacher[active] - student[active]
    return signal
