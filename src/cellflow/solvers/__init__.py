from cellflow.networks._velocity_field import ConditionalVelocityField, GENOTConditionalVelocityField
from cellflow.solvers._genot import GENOT
from cellflow.solvers._otfm import ClassifierFreeGuidance, Guidance, OTFlowMatching
from cellflow.solvers._otrfm import OTRFlowMatching
from cellflow.solvers._registry import SOLVER_REGISTRY, register_solver

# Built-in solvers, resolved by name in `CellFlow(solver=...)`.
register_solver("otfm", OTFlowMatching, ConditionalVelocityField)
register_solver("genot", GENOT, GENOTConditionalVelocityField)
register_solver("otrfm", OTRFlowMatching, ConditionalVelocityField)

__all__ = ["GENOT", "OTFlowMatching", "OTRFlowMatching", "ClassifierFreeGuidance", "Guidance", "SOLVER_REGISTRY", "register_solver"]
