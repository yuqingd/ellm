from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_cos_eor_sim():
    from text_housekeep.cos_eor.sim.sim import CosRearrangementSim  # noqa: F401
