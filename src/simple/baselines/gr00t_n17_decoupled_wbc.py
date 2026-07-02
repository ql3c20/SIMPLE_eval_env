"""GR00T N1.7 alias for the shared 36D decoupled-WBC SIMPLE agent."""

from .gr00t_n16_decoupled_wbc import Gr00tN16DecoupledWbcAgent


class Gr00tN17DecoupledWbcAgent(Gr00tN16DecoupledWbcAgent):
    """N1.7 uses the same 32D state and 36D WBC command protocol as N1.6."""

