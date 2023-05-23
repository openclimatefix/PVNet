"""Custom callbacks
"""
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BaseFinetuning, EarlyStopping, LearningRateFinder
from lightning.pytorch.trainer.states import TrainerFn


class PhaseEarlyStopping(EarlyStopping):
    """Monitor a validation metric and stop training when it stops improving.

    Only functions in a specific phase of training.
    """

    training_phase = None

    def switch_phase(self, phase: str):
        """Switch phase of callback"""
        if phase == self.training_phase:
            self.activate()
        else:
            self.deactivate()

    def deactivate(self):
        """Deactivate callback"""
        self.active = False

    def activate(self):
        """Activate callback"""
        self.active = True

    def _should_skip_check(self, trainer: Trainer) -> bool:
        return (
            (trainer.state.fn != TrainerFn.FITTING) or (trainer.sanity_checking) or not self.active
        )


class PretrainEarlyStopping(EarlyStopping):
    """Monitor a validation metric and stop training when it stops improving.

    Only functions in the 'pretrain' phase of training.
    """

    training_phase = "pretrain"


class MainEarlyStopping(EarlyStopping):
    """Monitor a validation metric and stop training when it stops improving.

    Only functions in the 'main' phase of training.
    """

    training_phase = "main"


class PretrainFreeze(BaseFinetuning):
    """Freeze the satellite and NWP encoders during pretraining"""

    training_phase = "pretrain"

    def __init__(self):
        """Freeze the satellite and NWP encoders during pretraining"""
        super().__init__()

    def freeze_before_training(self, pl_module):
        """Freeze satellite and NWP encoders before training start"""
        # freeze any module you want
        modules = []
        if pl_module.include_sat:
            modules += [pl_module.sat_encoder]
        if pl_module.include_nwp:
            modules += [pl_module.nwp_encoder]
        self.freeze(modules)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        """Unfreeze satellite and NWP encoders"""
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
        """Switch phase of callback"""
        if phase == self.training_phase:
            self.activate()
        else:
            self.deactivate()

    def deactivate(self):
        """Deactivate callback"""
        self.active = False

    def activate(self):
        """Activate callback"""
        self.active = True


class PhasedLearningRateFinder(LearningRateFinder):
    """Finds a learning rate at the start of each phase of learning"""

    active = True

    def on_fit_start(self, *args, **kwargs):
        """Do nothing"""
        return

    def on_train_epoch_start(self, trainer, pl_module):
        """Run learning rate finder on epoch start and then deactivate"""
        if self.active:
            self.lr_find(trainer, pl_module)
            self.deactivate()

    def switch_phase(self, phase: str):
        """Switch training phase"""
        self.activate()

    def deactivate(self):
        """Deactivate callback"""
        self.active = False

    def activate(self):
        """Activate callback"""
        self.active = True
