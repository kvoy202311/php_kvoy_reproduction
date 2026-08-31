"""Multi-teacher visuomotor distillation for ELF3.

The modules in this package intentionally do not modify RSL-RL itself.  They
implement the flat-tensor interfaces used by the RSL-RL 2.3.x version shipped
with the current Isaac Lab environment.
"""

from .action_transform import CanonicalActionTransform
from .action_contract import validate_runtime_action_contract
from .observation import BlockwiseObservationNormalizer, VisionObservationLayout
from .option_controller import OptionSelection, OptionStateController
from .schedules import DistillationWeights, PhpLossSchedule
from .teacher_manifest import TeacherManifest

__all__ = [
    "BlockwiseObservationNormalizer",
    "CanonicalActionTransform",
    "DistillationWeights",
    "OptionSelection",
    "OptionStateController",
    "PhpLossSchedule",
    "TeacherManifest",
    "VisionObservationLayout",
    "validate_runtime_action_contract",
]
