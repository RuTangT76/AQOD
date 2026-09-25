import unittest
from trustlab.aqod_mainline import MODE, assert_teacher_frozen, validate_mainline


class TestMainline(unittest.TestCase):
    def test_vanilla_c0(self):
        result = validate_mainline(mode=MODE, student_model='/models/2b',
                                   vanilla_student_model='/models/2b')
        self.assertIsNone(result['student_adapter'])

    def test_task_lora_rejected(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic only'):
            validate_mainline(mode=MODE, student_model='/models/2b',
                              vanilla_student_model='/models/2b',
                              student_adapter='/old/task-lora')

    def test_wrong_base_rejected(self):
        with self.assertRaisesRegex(ValueError, 'vanilla'):
            validate_mainline(mode=MODE, student_model='/models/4b',
                              vanilla_student_model='/models/2b')

    def test_diagnostic_explicit(self):
        result = validate_mainline(mode=MODE, student_model='/models/2b',
                                   vanilla_student_model='/models/2b',
                                   student_adapter='/old/task-lora', diagnostic_only=True)
        self.assertTrue(result['diagnostic_only'])

    def test_teacher_freeze(self):
        class P:
            def __init__(self, trainable): self.requires_grad = trainable
        class M:
            def __init__(self, trainable): self.trainable = trainable
            def parameters(self): return [P(self.trainable)]
        assert_teacher_frozen(M(False))
        with self.assertRaisesRegex(ValueError, 'frozen'):
            assert_teacher_frozen(M(True))
