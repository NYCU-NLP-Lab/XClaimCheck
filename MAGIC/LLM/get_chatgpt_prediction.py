import requests
import json
from pathlib import Path

# Set your API key
key = "{use your key here}"

# Load topics from JSON file
with open("topics.json", "r") as file:
    topics = json.load(file)

# Labels for classification
labels = ["false", "half_true", "mostly_false", "mostly_true", "on_fire", "true"]


def get_response(prompt):
    """Send a prompt to the OpenAI API and get a response."""
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/chat/completions"
    response = requests.post(url=url, data=json.dumps(payload), headers=headers)
    result = json.loads(response.text)
    try:
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error fetching response:", result)
        return None


# Main loop for iterations
for i in range(5):
    output = {}

    for topic in topics.keys():
        precision = {label: 0 for label in labels if label != "on_fire"}

        for label in labels:
            print(f"Iteration {i + 1}, Topic: {topic}, Label: {label}")
            path = Path(f"./retrieval/{i + 1}/test/{topic}/{label}")
            num_samples = len(list(path.glob("*")))

            hit, total = 0, 0

            for j in range(num_samples):
                try:
                    # Load claim information
                    with open(
                        f"./dataset/test/{topic}/{label}/{j + 1}/info.json", "r"
                    ) as file:
                        info = json.load(file)

                    evidence_path = Path(
                        f"./retrieval/{i + 1}/test/{topic}/{label}/{j + 1}"
                    )
                    evidence_files = list(evidence_path.glob("*"))
                    cnt = {lbl: 0 for lbl in labels if lbl != "on_fire"}
                    save = []

                    # Check if pre-existing predictions are available
                    chatgpt_output_path = (
                        f"./chatgpt/{i + 1}_{topic}_{label}_{j + 1}.json"
                    )
                    try:
                        with open(chatgpt_output_path, "r") as file:
                            exist = json.load(file)
                        for k, content in enumerate(exist):
                            if isinstance(content, str):
                                predict = content
                            else:
                                with open(evidence_files[k], "r") as file:
                                    evidence = json.load(file)
                                prompt = (
                                    f"You are a fact-checking machine, classify the claim into true, mostly true, "
                                    f"half true, mostly false or false based on the reference.\n"
                                    f"Claim: {info['claim']}\nClaimant: {info['claimant']}\n"
                                    f"Reference: {evidence if len(str(evidence)) < 3000 else str(evidence)[:3000]}\n"
                                    f"Instruction: Simply output the classification label. If the provided reference "
                                    f"doesn't help classifying the claim, output 'not related'."
                                )
                                predict = get_response(prompt)
                                while not isinstance(predict, str):
                                    predict = get_response(prompt)
                                save.append(predict)
                    except FileNotFoundError:
                        for premise in evidence_files:
                            with open(premise, "r") as file:
                                evidence = json.load(file)
                            prompt = (
                                f"You are a fact-checking machine, classify the claim into true, mostly true, "
                                f"half true, mostly false or false based on the reference.\n"
                                f"Claim: {info['claim']}\nClaimant: {info['claimant']}\n"
                                f"Reference: {evidence if len(str(evidence)) < 3000 else str(evidence)[:3000]}\n"
                                f"Instruction: Simply output the classification label. If the provided reference "
                                f"doesn't help classifying the claim, output 'not related'."
                            )
                            predict = get_response(prompt)
                            while not isinstance(predict, str):
                                predict = get_response(prompt)
                            save.append(predict)

                    # Update counts based on predictions
                    for lbl in cnt.keys():
                        if lbl in predict.lower():
                            cnt[lbl] += 1

                    # Determine final prediction
                    sorted_cnt = dict(
                        sorted(cnt.items(), key=lambda item: item[1], reverse=True)
                    )
                    prediction = next(iter(sorted_cnt.keys()))
                    if label == "on_fire" and prediction == "false":
                        prediction = "on_fire"

                    precision[prediction] += 1
                    total += 1
                    if label == prediction:
                        hit += 1

                    # Save predictions
                    with open(chatgpt_output_path, "w") as file:
                        json.dump(save, file)

                except Exception as e:
                    print(f"Error processing sample {j + 1} in {topic}/{label}: {e}")

            if topic not in output:
                output[topic] = {}
            output[topic][label] = {"hit": hit, "total": total, "rate": hit / total}

        # Add precision to output
        output[topic]["precision"] = precision

    # Save the output for the current iteration
    with open(f"./openai_{i + 1}.json", "w") as file:
        json.dump(output, file)
