
# Assignment 5 — Post-Training an LLM (Submission README)

This folder contains the code and notes for Assignment 5: a small, reproducible workflow to fine-tune a GPT-2 model (openai-community/gpt2) on SQuAD-derived QA pairs and expose a generation API.

What is included
- `assignment5/fine_tune.py` — fine-tuning script (Hugging Face Trainer). Configured for a quick smoke-run; adjust args for larger experiments.
- `assignment5/main.py` — a small CLI used earlier (you can call `train` or `generate`). The project root `main.py` exposes a FastAPI endpoint POST `/assignment5/generate` that loads a tokenizer + model and returns generated text.
- `assignment5/part2_answers.md` — theory answers for reinforcement-learning questions.

Quick notes
- This implementation uses causal LM fine-tuning: prompts are formatted as `Q: <question>\nA: <answer>` and labels = input_ids so the model learns to continue the prompt with the answer.
- Training on CPU is slow; the included smoke-run uses a small dataset slice (100 examples) and 1 epoch to produce a working checkpoint quickly. For full results, run on GPU and increase dataset/epochs.

Run locally (recommended, using the project's virtualenv `.venv`)

1. Activate the project's virtual environment and install deps (already done in this repo session):

```bash
source .venv/bin/activate
pip install -r requirements.txt  # if you maintain a requirements file, else install: transformers datasets accelerate sentencepiece torch uvicorn fastapi
```

2. Quick smoke fine-tune (produces `assignment5/checkpoint/`):

```bash
python assignment5/fine_tune.py --split "train[:100]" --num_train_epochs 1 --per_device_train_batch_size 1
```

This will download the base model and SQuAD slice, train for 1 epoch, and save tokenizer + model into `assignment5/checkpoint/`.

3. Run the API locally (the repository root `main.py` exposes the endpoint):

```bash
# start the service (in the venv)
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 &

# call the endpoint (example)
curl -X POST http://localhost:8000/assignment5/generate \
	-H "Content-Type: application/json" \
	-d '{"model_dir":"assignment5/checkpoint","prompt":"Hello, how are you?","max_new_tokens":50}'
```

Docker (optional)

The repository root `Dockerfile` has been updated to install transformers/datasets and copy only `main.py` + `assignment5/` into the image. To build and run the container:

```bash
docker build -t sps-genai:assignment5 .
docker run -p 8000:8000 sps-genai:assignment5
```

What to include for submission
- `assignment5/` directory including:
	- `fine_tune.py` (training script)
	- `main.py` (CLI) and `part2_answers.md` (theory answers)
	- `README.md` (this file)
- Optionally include the `assignment5/checkpoint/` folder (it can be large). If you prefer, include a `checkpoint.zip` of the saved model, or provide instructions to reproduce it using the `fine_tune.py` command above.

Notes for graders
- The code is runnable as-is. For a fast demonstration, run the fine-tune with the small split and then call the generate endpoint. For a full evaluation, train with a larger split or more epochs (preferably on GPU).

Contact / Issues
- If you hit environment/package problems (NumPy/torch binary mismatch on macOS), try downgrading numpy: `pip install "numpy<2"`. See logs for exact errors. If you want, I can produce a zipped checkpoint to avoid re-training.
