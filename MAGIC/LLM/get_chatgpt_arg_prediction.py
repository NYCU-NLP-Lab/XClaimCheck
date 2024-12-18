import requests
import json
from pathlib import Path

# Set your API key
key = "{use your key here}"

# Load topics from a JSON file
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
    try:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error fetching response:", response.text)
        return None


# Main loop for iterations
for i in range(3, 5):
    output = {}

    for topic in topics.keys():
        precision = {label: 0 for label in labels if label != "on_fire"}

        for label in labels:
            print(f"Iteration {i + 1}, Topic: {topic}, Label: {label}")
            path = Path(f"./argument/retrieve/gpt-3.5/{i + 1}/{topic}/{label}")
            num_samples = len(list(path.glob("*")))

            hit, total = 0, 0

            for j in range(num_samples):
                try:
                    # Load predictions, claim info, and arguments
                    with open(
                        f"./chatgpt_arg/{i + 1}_{topic}_{label}_{j + 1}.json", "r"
                    ) as file:
                        predict = json.load(file)[0]

                    with open(
                        f"./dataset/test/{topic}/{label}/{j + 1}/info.json", "r"
                    ) as file:
                        info = json.load(file)

                    with open(
                        f"./argument/retrieve/gpt-3.5/{i + 1}/{topic}/{label}/{j + 1}.json",
                        "r",
                    ) as file:
                        arguments = json.load(file)

                    # Prepare arguments for the prompt
                    valid_arguments = [
                        f"Argument {idx + 1}: {arg}"
                        for idx, arg in enumerate(arguments)
                        if "not related" not in arg.lower()
                    ]
                    argument_text = (
                        "\n".join(valid_arguments)
                        if valid_arguments
                        else "No argument for this claim, classify based on your knowledge."
                    )

                    prompt = (
                        f"You are a fact-checking machine, classify the claim into true, mostly true, half true, mostly false or false "
                        f"based on some arguments.\nClaim: {info['claim']}\nClaimant: {info['claimant']}\nArguments: "
                        f"{argument_text[:3000]}\nInstruction: Simply output the classification label."
                    )

                    # Get prediction from the API
                    prediction = get_response(prompt)
                    while not isinstance(prediction, str):
                        prediction = get_response(prompt)

                    save = [prediction]

                    # Map prediction to standardized labels
                    if "mostly true" in prediction.lower():
                        prediction = "mostly_true"
                    elif "half true" in prediction.lower():
                        prediction = "half_true"
                    elif "true" in prediction.lower():
                        prediction = "true"
                    elif "mostly false" in prediction.lower():
                        prediction = "mostly_false"
                    elif "false" in prediction.lower():
                        prediction = "false"

                    precision[prediction] += 1
                    if label == "on_fire" and prediction == "false":
                        prediction = "on_fire"

                    total += 1
                    if label == prediction:
                        hit += 1

                    # Save prediction
                    with open(
                        f"./chatgpt_arg/{i + 1}_{topic}_{label}_{j + 1}.json", "w"
                    ) as file:
                        json.dump(save, file)

                except Exception as e:
                    print(f"Error processing sample {j + 1} in {topic}/{label}: {e}")

            if topic not in output:
                output[topic] = {}
            output[topic][label] = {
                "hit": hit,
                "total": total,
                "rate": hit / total if total > 0 else 0,
            }

        # Add precision data
        output[topic]["precision"] = precision

    # Save the results for the current iteration
    with open(f"./openai_arg-{i + 1}.json", "w") as file:
        json.dump(output, file)
