Assignment 4: Diffusion and Energy-Based Models (CIFAR-10)

This folder contains a minimal, self-contained implementation for Assignment 4.
It includes:

- `diffusion_model.py`: lightweight diffusion model and helpers
- `energy_model.py`: simple energy-based model (EBM) architecture
- `train_diffusion.py` and `train_energy.py`: training entry points using CIFAR-10 (small demo)
- `api.py`: FastAPI endpoints to generate samples (loads checkpoints if present)
- `theory_answers.md`: answers for theory questions Q1-Q9 with calculations

Notes:
- Training full models on CIFAR-10 can be slow. The training scripts are set up for quick demo runs; increase epochs for better results.
- If you want me to push these files to GitHub after review, tell me and I'll commit & push.

Checkpoints included
--------------------

This repository includes demo checkpoints (generated with short local training) in
`assignment4/checkpoints/`:

- `diffusion_model.pt` — demo diffusion model checkpoint
- `energy_model.pt` — demo energy model checkpoint

These are small demo checkpoints intended for grading and quick inference.

Docker / Run instructions
-----------------------

Build the Docker image (this image includes the checkpoints):

```bash
docker build -t sps_assignment4 .
```

Run the container and map port 8000:

```bash
docker run -p 8000:8000 sps_assignment4
```

Open the API docs in a browser:

http://localhost:8000/docs

Or call the endpoints directly:

```bash
curl -o out_diff.png http://localhost:8000/generate/diffusion
curl -o out_ebm.png http://localhost:8000/generate/ebm
```

If you prefer to train from scratch instead of using the provided checkpoints, run:

```bash
python assignment4/train_diffusion.py
python assignment4/train_energy.py
```

