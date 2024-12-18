import json
from pathlib import Path

# Load topics from JSON file
with open("topics.json", "r") as file:
    topics = json.load(file)

# Labels for classification
labels = ["false", "half_true", "mostly_false", "mostly_true", "on_fire", "true"]

# Import necessary libraries for model handling
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import load_model, get_conversation_template

# Model configuration
model_path = "lmsys/vicuna-7b-v1.5"
device = "cuda"
num_gpus = 1
temperature = 0.7
repetition_penalty = 1.0
max_new_tokens = 512
debug = False

# Load the model and tokenizer
model, tokenizer = load_model(
    model_path,
    device=device,
    num_gpus=num_gpus,
    debug=debug,
)


@torch.inference_mode()
def get_response(msg):
    """Generate a response from the model based on the input message."""
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=(temperature > 1e-5),
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
    )

    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


# Main processing loop
for i in range(5):
    output = {}

    for topic, labels_dict in topics.items():
        for label in labels:
            print(f"Iteration {i + 1}, Topic: {topic}, Label: {label}")
            path = Path(f"./retrieval/{i + 1}/test/{topic}/{label}")
            num_samples = len(list(path.glob("*")))

            hit, total = 0, 0
            for j in range(num_samples):
                # Load claim information
                with open(
                    f"./dataset/test/{topic}/{label}/{j + 1}/info.json", "r"
                ) as file:
                    info = json.load(file)

                # Prepare evidence
                evidence_path = Path(
                    f"./retrieval/{i + 1}/test/{topic}/{label}/{j + 1}"
                )
                evidence_files = list(evidence_path.glob("*"))

                cnt = {lbl: 0 for lbl in labels if lbl != "on_fire"}
                for premise in evidence_files:
                    with open(premise, "r") as file:
                        evidence = json.load(file)

                    truncated_evidence = (
                        str(evidence)
                        if len(str(evidence)) < 3000
                        else str(evidence)[:3000]
                    )
                    prompt = (
                        f"Classify the claim into true, mostly true, half true, mostly false, or false, "
                        f"and provide reasons based on the arguments.\n"
                        f"Claim: {info['claim']}\n"
                        f"Claimant: {info['claimant']}\n"
                        f"Arguments: {truncated_evidence}\n"
                        f"Instruction: Don't forget to output reasons."
                    )
                    prediction = get_response(prompt)

                    # Count predictions
                    for lbl in cnt.keys():
                        if lbl in prediction.lower():
                            cnt[lbl] += 1

                # Determine final prediction
                sorted_cnt = dict(
                    sorted(cnt.items(), key=lambda item: item[1], reverse=True)
                )
                prediction = next(iter(sorted_cnt.keys()))
                if label == "on_fire" and prediction == "false":
                    prediction = "on_fire"

                total += 1
                if label == prediction:
                    hit += 1

            # Record results
            output.setdefault(topic, {})
            output[topic][label] = {"hit": hit, "total": total, "rate": hit / total}

    # Save output to JSON
    with open(f"./vicuna_{i + 1}.json", "w") as file:
        json.dump(output, file)
