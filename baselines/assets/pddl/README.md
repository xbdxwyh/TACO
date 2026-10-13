# Usage

This model outputs a reward for each reasoning step evaluating it.

`Babelscape/Qwen2.5-Math-PRM-7B-PDDL-r` is a **Process Reward Model (PRM)** obtained by continual fine-tuning from **Qwen/Qwen2.5-Math-PRM-7B** with the planning-based supervision introduced in **PDDL2PRM**.

Unlike the other PRM checkpoints in this release, this model is not trained from the base/instruct model with a newly added scalar reward head. Instead, it starts from the original **Qwen2.5-Math-PRM-7B** checkpoint and continues its training on PDDL2PRM data. For this reason, it follows the original Qwen PRM reward interface: reasoning steps must be separated with the `<extra_0>` marker, and rewards are obtained from the positive-class probability at marker positions.

PDDL2PRM is the dataset introduced in:

**Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards**
Raffaele Pisano and Roberto Navigli, ACL 2026

Project page & paper: https://babelscape.github.io/prm-meets-planning/

arXiv: https://arxiv.org/abs/2604.17957

The paper proposes using symbolic planning problems written in **Planning Domain Definition Language (PDDL)** to generate precise step-level rewards for reasoning trajectories. In PDDL, actions, states, preconditions, effects, and goals are explicitly defined, so intermediate reasoning steps can be evaluated automatically.

## Example

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

repo_id = "Babelscape/Qwen2.5-Math-PRM-7B-PDDL-r"

tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()


def build_messages(problem, steps):
    return [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}."
        },
        {
            "role": "user",
            "content": problem
        },
        {
            "role": "assistant",
            "content": "<extra_0>".join(steps) + "<extra_0>"
        }
    ]


def get_step_rewards(logits, marker_positions):
    probs = F.softmax(logits, dim=-1)
    # Positive-class probability at each <extra_0> marker position
    return probs[0, marker_positions, 1].detach().cpu().tolist()


problem = "If x + 3 = 10, find x."
steps = [
    "Subtract 3 from both sides: x = 10 - 3.",
    "So x = 7."
]

messages = build_messages(problem, steps)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

marker_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
marker_positions = (inputs["input_ids"][0] == marker_id).nonzero(as_tuple=True)[0]

step_scores = get_step_rewards(logits, marker_positions)

print("Step scores:", step_scores)

first_bad = next((i for i, score in enumerate(step_scores) if score < 0.5), -1)
print("First failing step index:", first_bad)
```

# Notes

* The marker `<extra_0>` must appear after every reasoning step.
* This model follows the reward format of `Qwen/Qwen2.5-Math-PRM-7B`.
* Rewards are computed from the positive-class probability at `<extra_0>` marker positions.
* A threshold such as 0.5 can be used to identify potentially incorrect steps.
* This differs from the PRM800K-based checkpoints with a scalar reward head, where `pred_scalar` is read at marker positions.

# Citation

If you use this model or the PDDL2PRM dataset in your work, please cite:

```bibtex
@inproceedings{pisano2026prmplanning,
  title={Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards},
  author={Pisano, Raffaele and Navigli, Roberto},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  note={Accepted}
}
```
