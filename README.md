# Text Generation on Lightning AI

Loading a small open weights model and running inference from a Lightning AI Studio notebook.

## Setup

```bash
pip install "transformers>=4.45" torch accelerate
```

No Hugging Face account or token is needed for the model used here.

## Usage

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="unsloth/Llama-3.2-1B-Instruct",
    device_map="auto",
    dtype="auto",
)

out = pipe("Explain Kafka consumer groups in two sentences.", max_new_tokens=100)
print(out[0]["generated_text"])
```

`device_map="auto"` places the model on GPU when the Studio has one and falls back to CPU otherwise. `dtype="auto"` picks fp16 on GPU, which keeps the 1B model under about 2.5 GB.

## Notes

The `Instruct` suffix matters. The base `Llama-3.2-1B` checkpoint continues your text rather than answering it, so a question comes back as more questions.

`max_new_tokens` caps output length. Raise it for longer responses, keeping in mind that generation time scales with it.

## Layout

```
.
├── README.md
├── requirements.txt
└── notebooks/
    └── inference.ipynb
```
