import json
from pathlib import Path

with open("id_topics.json", "r") as file:
    data = json.load(file)

unit = len(data) // 5
train_ids = list(data.keys())[: unit * 3]
valid_ids = list(data.keys())[unit * 3: unit * 4]
test_ids = list(data.keys())[unit * 4:]

with open("domain.json", "r") as file:
    domain = json.load(file)

path = Path("XClaimCheck_dataset")
path.mkdir()
for i in range(5):
    path = Path(f"XClaimCheck_dataset/Domain_{i+1}")
    path.mkdir()

    for in_domain_topic in domain[i]:
        path = Path(f"XClaimCheck_dataset/Domain_{i+1}/{in_domain_topic}")
        path.mkdir()
        output = []
        for id in train_ids:
            for topic in data[id]["topics"]:
                if topic in domain[i]:
                    output.append(id)
        with open(f"XClaimCheck_dataset/Domain_{i+1}/{in_domain_topic}/training.json", "w") as file:
            json.dump(output, file)

        output = []
        for id in valid_ids:
            for topic in data[id]["topics"]:
                if topic in domain[i]:
                    output.append(id)
        with open(f"XClaimCheck_dataset/Domain_{i+1}/{in_domain_topic}/validation.json", "w") as file:
            json.dump(output, file)

        output = []
        for id in test_ids:
            for topic in data[id]["topics"]:
                if topic in domain[i]:
                    output.append(id)
        with open(f"XClaimCheck_dataset/Domain_{i+1}/{in_domain_topic}/test.json", "w") as file:
            json.dump(output, file)
