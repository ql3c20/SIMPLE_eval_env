from __future__ import annotations

import numpy as np

from simple.baselines.gr00t_n17_sonic import Gr00tN17SonicAgent


def test_native_sonic_prefers_arena_mono_camera() -> None:
    front = np.full((4, 6, 3), 1, dtype=np.uint8)
    legacy = np.full((4, 6, 3), 2, dtype=np.uint8)

    selected = Gr00tN17SonicAgent._policy_image(
        {"front_camera": front, "head_stereo_left": legacy}
    )

    assert selected is front


def test_native_sonic_keeps_legacy_stereo_compatibility() -> None:
    legacy = np.zeros((4, 6, 3), dtype=np.uint8)

    selected = Gr00tN17SonicAgent._policy_image({"head_stereo_left": legacy})

    assert selected is legacy
