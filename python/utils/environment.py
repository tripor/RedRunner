from mlagents_envs.environment import UnityEnvironment


def load_environment_editor() -> UnityEnvironment:
    unity_timeout = 1000000
    print("Waiting for Play on Editor")
    unity_env = UnityEnvironment(
        timeout_wait=100000, no_graphics=False, side_channels=[])
    return unity_env
