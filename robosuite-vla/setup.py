from setuptools import setup, find_packages

setup(
    name="robosuite-vla",
    version="0.1.0",
    description="SmolVLA fine-tuning on LIBERO with LIBERO-PRO evaluation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "robosuite>=1.4.1",
        "mujoco>=3.0",
        "numpy>=2.0,<2.1",
        "h5py>=3.0",
        "Pillow>=9.0",
        "imageio",
        "matplotlib>=3.5",
        "tqdm",
        "rich",
        "omegaconf>=2.3",
        "pyyaml",
        "python-dotenv",
    ],
    extras_require={
        "train": [
            "torch>=2.0",
            "torchvision>=0.15",
            "transformers>=4.40",
            "accelerate>=0.30",
            "wandb",
        ],
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
)
