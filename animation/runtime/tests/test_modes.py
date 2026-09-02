from animation.runtime.modes import ModeStateMachine, RobotMode


def test_illegal_mode_transitions_are_rejected() -> None:
    machine = ModeStateMachine()
    assert not machine.request(RobotMode.HYBRID_WALK)
    assert machine.mode is RobotMode.IDLE
    assert machine.request(RobotMode.STAND)
    machine.update(1.0)
    assert machine.request(RobotMode.DEMO_DOCK)
    machine.update(1.0)
    assert not machine.request(RobotMode.WALK)


def test_emergency_stop_is_reachable_from_every_mode() -> None:
    for mode in RobotMode:
        if mode is RobotMode.EMERGENCY_STOP:
            continue
        machine = ModeStateMachine(mode)
        assert machine.request(RobotMode.EMERGENCY_STOP)
        assert machine.target_mode is RobotMode.EMERGENCY_STOP
        assert machine.mode is RobotMode.EMERGENCY_STOP
