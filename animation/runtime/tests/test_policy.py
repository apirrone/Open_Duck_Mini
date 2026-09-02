import numpy as np

from animation.runtime.policy import WalkPolicy


class FakeInput:
    name = "obs"
    shape = [1, 74]


class FakeSession:
    def __init__(self) -> None:
        self.input = None

    def get_inputs(self):
        return [FakeInput()]

    def run(self, _outputs, inputs):
        self.input = inputs["obs"].copy()
        return [np.arange(16, dtype=np.float32)[None, :]]


def test_previous_action_is_raw_policy_action_not_blended_output() -> None:
    session = FakeSession()
    policy = WalkPolicy("unused.onnx", session=session)
    observation = np.zeros(56, dtype=np.float32)
    raw_action = policy.infer(observation)
    blended_output = policy.action_to_targets(raw_action) + 0.4

    policy.infer(observation)

    assert np.array_equal(session.input[0, 37:53], raw_action)
    assert not np.array_equal(session.input[0, 37:53], blended_output)
