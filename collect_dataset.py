import json
import requests
import re
import sys


id_topics = {}
selected_topics = []


def get_politifact_topic(id, url):
    response = requests.get(url)
    html_content = response.text
    pattern = r"Editions\': \[(.*?)\]"
    matches = re.search(pattern, html_content)

    if matches:
        tags_string = matches.group(1)
        tags = re.findall(r'"(.*?)"', tags_string)

        for topic in tags:
            if topic in selected_topics:
                if id in id_topics:
                    id_topics[id]["topics"].append(topic)
                else:
                    id_topics[id] = {"url": url, "topics": [topic]}
    return


def handle_dataset(dataset):
    with open(f"{watclaim_path}/WatClaimCheck_dataset/{dataset}.json", "r") as file:
        data = json.load(file)

    for i, claim in enumerate(data):
        print(f"{i:5.0f}/{len(data)}\r", end="")
        id = str(claim["label"]["id"])
        url = claim["label"]["review_url"]
        if claim["label"]["reviewer_name"] == "PolitiFact":
            get_politifact_topic(id, url)
    return


if __name__ == "__main__":
    watclaim_path = sys.argv[1]
    with open("topics.json", "r") as file:
        selected_topics = json.load(file)
    for dataset in ["train", "valid", "test"]:
        handle_dataset(dataset)
    with open("id_topics.json", "w") as file:
        json.dump(id_topics, file)
