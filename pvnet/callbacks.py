"""Custom callbacks developed to be able to use early stopping and learning rate finder even when
pretraining parts of the network.
"""
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BaseFinetuning, EarlyStopping, LearningRateFinder
from lightning.pytorch.trainer.states import TrainerFn


class PhaseEarlyStopping(EarlyStopping):
    training_phase = None

    def switch_phase(self, phase: str):
        if phase == self.training_phase:
            self.activate()
        else:
            self.deactivate()

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True

    def _should_skip_check(self, trainer: Trainer) -> bool:
        return (
            (trainer.state.fn != TrainerFn.FITTING) or (trainer.sanity_checking) or not self.active
        )


class PretrainEarlyStopping(EarlyStopping):
    training_phase = "pretrain"


class MainEarlyStopping(EarlyStopping):
    training_phase = "main"


class PretrainFreeze(BaseFinetuning):
    training_phase = "pretrain"

    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        modules = []
        if pl_module.include_sat:
            modules += [pl_module.sat_encoder]
        if pl_module.include_nwp:
            modules += [pl_module.nwp_encoder]
        self.freeze(modules)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if not self.active:
            modules = []
            if pl_module.include_sat:
                modules += [pl_module.sat_encoder]
            if pl_module.include_nwp:
                modules += [pl_module.nwp_encoder]
            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                train_bn=True,
            )

    def switch_phase(self, phase: str):
        if phase == self.training_phase:
            self.activate()
        else:
            self.deactivate()

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True


class PhasedLearningRateFinder(LearningRateFinder):
    """Finds a learning rate at the start of each phase of learning"""

    active = True

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if self.active:
            self.lr_find(trainer, pl_module)
            self.deactivate()

    def switch_phase(self, phase: str):
        self.activate()

    def deactivate(self):
        self.active = False

    def activate(self):
        self.active = True
