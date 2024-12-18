from pathlib import Path
import json
import requests

with open("../topics.json", "r") as file:
    topic = json.load(file)

label = ["true", "false", "mostly_true", "mostly_false", "half_true", "on_fire"]

# Set your API key
key = "{use your key here}"


def get_response(prompt):
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/chat/completions"
    res = requests.post(url=url, data=json.dumps(payload), headers=headers)
    try:
        res = json.loads(res.text)
        return res["choices"][0]["message"]["content"]
    except:
        print(res)
        return res


with open("../retriever/dataset/split.json", "r") as file:
    split = json.load(file)

for t in topic.keys():
    # educate, economic...
    for l in label:
        print(t, l)
        path = Path(f"../dataset/train/{t}/{l}/")
        num = len(list(path.glob("*")))
        for i in range(num):
            idx = i + 1
            with open(f"../dataset/train/{t}/{l}/{idx}/info.json", "r") as file:
                info = json.load(file)
            claim = info["claim"]
            claimant = info["claimant"]
            date = info["claim_date"]

            for group in range(5):
                if t not in split[group]:
                    continue
                path = Path(f"../retrieval/{group+1}/train/{t}/{l}/{idx}/")
                sub = path.glob("*")
                output = []
                for article in sub:
                    with open(article, "r") as file:
                        data = json.load(file)
                        prompt = f"""Output a no more than 50-words argument assembling evidences in the reference that help evaluating the authenticity of the claim.\nClaim: {claim}\nClaimant: {claimant}\nReference: {data if len(str(data))<3000 else str(data)[:3000]}\nNote: If the provided reference doesn't help evaluating the claim, simply output not related."""

                        result = get_response(prompt)
                        while type(result) is not str:
                            result = get_response(prompt)
                        output.append(result)
                with open(
                    f"./retrieve/gpt-3.5/train/{group+1}/{t}/{l}/{idx}.json", "w"
                ) as file:
                    json.dump(output, file)
