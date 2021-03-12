from mlagents_envs.environment import UnityEnvironment


def load_environment_editor(graphics=True) -> UnityEnvironment:
    unity_timeout = 1000000
    print("Waiting for Play on Editor")
    unity_env = UnityEnvironment(
        timeout_wait=100000, no_graphics=not(graphics), side_channels=[])
    return unity_env


def load_environment(graphics=True) -> UnityEnvironment:
    unity_timeout = 1000000
    unity_env = UnityEnvironment(file_name="./game_build/Red Runner",
                                 timeout_wait=100000, no_graphics=not(graphics), side_channels=[])
    return unity_env
