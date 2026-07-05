from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

dataset.save_to_disk("data/raw/hotpot")