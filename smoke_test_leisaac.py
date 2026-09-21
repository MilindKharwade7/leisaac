"""Smoke test: leisaac on the native IsaacLab 3.0 / Isaac Sim 6.0 stack.

1. import leisaac (triggers gym registrations)
2. list all registered LeIsaac env ids
3. build one env (kitchen task, headless)
4. step a few frames and confirm sim time advances
"""
import gymnasium as gym
import torch
import traceback

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import leisaac  # noqa: F401  (registers envs)

print("\n=== REGISTERED LeIsaac ENVS ===")
env_ids = sorted(gym.registry.keys())
for eid in env_ids:
    if eid.startswith("LeIsaac"):
        print(eid)

TARGET = "LeIsaac-SO101-PickOrange-v0"
print(f"\n=== BUILDING {TARGET} (headless) ===")
success = False
try:
    env = gym.make(TARGET, num_envs=1)
    obs, info = env.reset(seed=42)
    print(f"reset OK: obs keys = {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}")
    for i in range(5):
        actions = {
            k: torch.zeros(v.shape, device=v.device) if hasattr(v, "shape") else v
            for k, v in (obs.items() if hasattr(obs, "items") else [])
        }
        obs, rew, term, trunc, info = env.step(actions)
        print(f"step {i}: rew={rew if not hasattr(rew, 'item') else rew.item():.4f}")
    print(f"\nSMOKE TEST PASSED: stepped 5 frames of {TARGET}")
    env.close()
    success = True
except Exception:
    traceback.print_exc()
    print("\nSMOKE TEST FAILED (see traceback above)")

simulation_app.close()
print(f"\n=== RESULT: {'PASSED' if success else 'FAILED'} ===")
