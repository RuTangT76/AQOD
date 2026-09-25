"""veRL FSDP worker with an independent frozen-teacher role."""

from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch

from .contracts import teacher_worker_config


class AQODWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        plain = OmegaConf.to_container(config, resolve=True)
        isolated = OmegaConf.create(teacher_worker_config(plain, role))
        super().__init__(isolated, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self.role == 'ref':
            self.ref_module_fsdp.eval()
            self.ref_module_fsdp.requires_grad_(False)
            if self._is_lora or any(p.requires_grad for p in self.ref_module_fsdp.parameters()):
                raise RuntimeError('External teacher must be frozen and independent of student LoRA')
