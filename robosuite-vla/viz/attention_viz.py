#!/usr/bin/env python3
"""Visualize SmolVLA attention maps over camera images.

Shows where the model "looks" when making action decisions.

Usage:
    python viz/attention_viz.py --checkpoint outputs/checkpoints/smolvla_spatial \
        --suite libero_spatial --task 0 --output attention.png
"""

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT = Path(__file__).resolve().parent.parent


def extract_attention(policy, obs_dict):
    """Extract attention weights from a SmolVLA forward pass.

    Returns attention map as (H, W) array normalized to [0, 1].
    """
    import torch

    # Hook to capture attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        if hasattr(output, "attentions") and output.attentions is not None:
            attention_weights.append(output.attentions[-1].detach().cpu())

    # Register hooks on the vision encoder's last attention layer
    hooks = []
    for name, module in policy.named_modules():
        if "attention" in name.lower() or "attn" in name.lower():
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        policy.select_action(obs_dict)

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attention_weights:
        return None

    # Average attention across heads and reshape to spatial
    attn = attention_weights[-1]  # Last layer
    attn = attn.mean(dim=1)      # Average across heads
    attn = attn[0, 0, 1:]        # First batch, CLS token attending to patches

    # Reshape to spatial grid
    n_patches = attn.shape[0]
    grid_size = int(np.sqrt(n_patches))
    if grid_size * grid_size == n_patches:
        attn_map = attn.reshape(grid_size, grid_size).numpy()
    else:
        attn_map = attn.numpy().reshape(-1)[:grid_size**2].reshape(grid_size, grid_size)

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return attn_map


def visualize_attention(checkpoint_path, suite_name, task_id, output_path,
                       device="cuda", seed=42):
    """Generate attention visualization for a single frame."""
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image as PILImage

    from eval.eval_libero import load_policy, create_env

    policy = load_policy(checkpoint_path, device=device)
    if policy is None:
        print("Cannot visualize attention without a loaded policy.")
        return

    env, task, task_suite = create_env(suite_name, task_id)
    print(f"Task: {task.language}")

    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    # Take a few steps so the scene is interesting
    for _ in range(20):
        action = np.random.uniform(-0.3, 0.3, size=7)
        obs, _, done, _ = env.step(action)
        if done:
            break

    # Get camera image
    cam_img = np.flip(obs["agentview_image"], axis=0)
    pil_img = PILImage.fromarray(cam_img).resize((256, 256))

    # Build observation for policy
    obs_dict = {
        "observation.image": torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,
        "observation.state": torch.zeros(1, 7).to(device),
    }

    # Extract attention
    attn_map = extract_attention(policy, obs_dict)

    # Plot
    fig, axes = plt.subplots(1, 3 if attn_map is not None else 1, figsize=(15, 5))
    if attn_map is None:
        axes = [axes]

    # Original image
    axes[0].imshow(pil_img)
    axes[0].set_title("Camera View")
    axes[0].axis("off")

    if attn_map is not None:
        # Attention heatmap
        from PIL import Image
        attn_resized = np.array(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize((256, 256))
        ) / 255.0

        axes[1].imshow(attn_resized, cmap="hot")
        axes[1].set_title("Attention Map")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(pil_img)
        axes[2].imshow(attn_resized, cmap="hot", alpha=0.5)
        axes[2].set_title("Attention Overlay")
        axes[2].axis("off")

    plt.suptitle(f"Task: {task.language}", fontsize=11)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    env.close()
    print(f"Attention visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model attention")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/videos/attention.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    visualize_attention(args.checkpoint, args.suite, args.task, args.output,
                       device=args.device)


if __name__ == "__main__":
    main()
