"""Fail-closed model separation for the teacher-sourced AQOD mainline."""
from pathlib import Path

MODE = 'aqod_mvp_teacher_sourced'


def validate_mainline(*, mode, student_model, vanilla_student_model,
                      student_adapter=None, diagnostic_only=False):
    if mode != MODE:
        raise ValueError('Wrong AQOD experiment mode')
    if Path(student_model).resolve() != Path(vanilla_student_model).resolve():
        raise ValueError('Mainline student must start from vanilla Qwen3.5-2B')
    if student_adapter is not None and not diagnostic_only:
        raise ValueError('Task LoRA is diagnostic only; mainline C0 must be vanilla')
    return {'mode': mode, 'student_model': str(Path(student_model).resolve()),
            'student_adapter': student_adapter, 'diagnostic_only': diagnostic_only}


def assert_teacher_frozen(model):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('AQOD teacher must be frozen')
