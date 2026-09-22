"""Smoke test: leisaac on the native IsaacLab 3.0 / Isaac Sim 6.0 stack.

1. import leisaac (triggers gym registrations)
2. list all registered LeIsaac env ids
3. build one env (kitchen task, headless)
4. step a few frames and confirm sim advances

Usage (the task's wrist/front cameras require --enable_cameras, as in the
other leisaac scripts):

    ./isaaclab/isaaclab.sh -p ./leisaac/smoke_test_leisaac.py --enable_cameras
"""
import argparse

import gymnasium as gym
import torch
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for the LeIsaac environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher_args = vars(args_cli)
launcher_args["headless"] = True  # this test never opens a window

app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import leisaac  # noqa: F401  (registers envs)
from isaaclab_tasks.utils import parse_env_cfg

print("\n=== REGISTERED LeIsaac ENVS ===")
env_ids = sorted(gym.registry.keys())
for eid in env_ids:
    if eid.startswith("LeIsaac"):
        print(eid)

TARGET = "LeIsaac-SO101-PickOrange-v0"
print(f"\n=== BUILDING {TARGET} (headless) ===")
success = False
env = None
try:
    # NOTE: the registered entry point is a ManagerBasedRLEnv, which takes a cfg
    # object as its only constructor argument.  `num_envs` is not accepted as a
    # gym.make() kwarg, so it has to be applied to the env cfg first.
    env_cfg = parse_env_cfg(TARGET, num_envs=1, device="cuda:0")
    # The leisaac env template declares actions.arm_action / actions.gripper_action
    # as MISSING; they are filled in by init_action_cfg() the same way leisaac's own
    # scripts do it (env_cfg.use_teleop_device(<device>)).  "so101leader" selects the
    # plain joint-position action terms and needs no physical device at build time.
    env_cfg.use_teleop_device("so101leader")
    env = gym.make(TARGET, cfg=env_cfg)
    obs, info = env.reset(seed=42)
    print(f"reset OK: obs keys = {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}")
    # ManagerBasedRLEnv actions are a (num_envs, action_dim) tensor, not an obs dict.
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for i in range(5):
        obs, rew, term, trunc, info = env.step(actions)
        rew_val = rew.item() if hasattr(rew, "item") else rew
        sim_time = (env.unwrapped.episode_length_buf * env.unwrapped.step_dt).item()
        print(f"step {i}: rew={rew_val:.4f} sim_time={sim_time:.3f}s")
    print(f"\nSMOKE TEST PASSED: stepped 5 frames of {TARGET}")
    success = True
except Exception:
    traceback.print_exc()
    print("\nSMOKE TEST FAILED (see traceback above)")
finally:
    if env is not None:
        env.close()

# The verdict is printed *before* shutting Kit down: Kit's teardown ends the
# interpreter without flushing Python's block-buffered stdout, so a print placed
# after simulation_app.close() can be lost when the output is redirected to a file.
print(f"\n=== RESULT: {'PASSED' if success else 'FAILED'} ===", flush=True)

simulation_app.close()
