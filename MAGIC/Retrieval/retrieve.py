from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
from pathlib import Path
import json

# Directory for saving models
save_dir = "./saved_models"

# Load the group information from the split.json file
with open("./dataset/split.json", "r") as file:
    group = json.load(file)

# Loop through iterations
for i in range(5):
    # Load the retriever model for the current iteration
    retriever = DensePassageRetriever.load(
        load_dir=f"{save_dir}/{i + 1}", document_store=None
    )

    for split in ["test"]:
        dataset_path = Path(f"../dataset/{split}/")
        topics = dataset_path.glob("*")

        for topic_path in topics:
            topic_name = topic_path.name

            # Skip topics that belong to the current group
            if topic_name in group[i]:
                continue

            # Create necessary directories for retrieval output
            retrieval_topic_dir = Path(f"../retrieval/{i + 1}/{split}/{topic_name}")
            retrieval_topic_dir.mkdir(parents=True, exist_ok=True)

            # Process each label within the topic
            label_paths = topic_path.glob("*")
            for label_path in label_paths:
                label_name = label_path.name
                retrieval_label_dir = Path(
                    f"../retrieval/{i + 1}/{split}/{topic_name}/{label_name}"
                )
                retrieval_label_dir.mkdir(parents=True, exist_ok=True)

                # Process each index within the label
                index_paths = label_path.glob("*")
                for index_path in index_paths:
                    index_name = index_path.name
                    retrieval_index_dir = Path(
                        f"../retrieval/{i + 1}/{split}/{topic_name}/{label_name}/{index_name}"
                    )
                    retrieval_index_dir.mkdir(parents=True, exist_ok=True)

                    # Read the claim and evidence information
                    try:
                        with open(index_path / "info.json", "r") as file:
                            info = json.load(file)
                        claim = info["claim"]
                    except Exception as e:
                        print(f"Error reading claim info from {index_path}: {e}")
                        continue

                    premise_path = index_path / "premise"
                    premise_files = premise_path.glob("*")

                    for premise_file in premise_files:
                        premise_name = premise_file.name

                        # Read evidence from the premise file
                        try:
                            with open(premise_file, "r") as file:
                                evidence = json.load(file)
                        except Exception as e:
                            print(f"Error reading premise file {premise_file}: {e}")
                            continue

                        # Prepare documents for retrieval
                        docs = [{"content": e} for e in evidence]
                        document_store = InMemoryDocumentStore()
                        document_store.write_documents(docs)

                        # Update embeddings for retrieval
                        document_store.update_embeddings(retriever)
                        retriever.document_store = document_store

                        # Retrieve top documents for the claim
                        try:
                            top_k_documents = retriever.retrieve(claim, top_k=50)
                        except Exception as e:
                            print(f"Error retrieving documents for claim: {claim}: {e}")
                            continue

                        # Filter documents based on a score threshold
                        candi = [
                            doc.content for doc in top_k_documents if doc.score >= 0.625
                        ]
                        result = [e for e in evidence if e in candi]

                        # Save the retrieval results
                        output_file = retrieval_index_dir / f"{premise_name}.json"
                        with open(output_file, "w") as file:
                            json.dump(result, file)
