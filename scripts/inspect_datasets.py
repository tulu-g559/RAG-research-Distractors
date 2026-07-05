import json
from datasets import load_from_disk

print("=" * 60)
print("SQuAD")

with open("data/raw/squad_v2/train-v2.0.json", "r", encoding="utf-8") as f:
    squad = json.load(f)

print(squad.keys())
print(squad["data"][0].keys())

print("=" * 60)
print("HotpotQA")

hotpot = load_from_disk("data/raw/hotpot")

print(hotpot)
print(hotpot["train"][0])