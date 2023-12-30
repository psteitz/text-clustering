"""
Compute support for sentences.
"""
import sys
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# FIXME: path hack to allow peer modules to be imported
# As is, requires text-classification to be checked out at the same level as the root of this repo.
# So, for example text-classification and this repo are both cloned into subdirectories of $HOME.
sys.path.append("..")

# import from sibling
from data import csv_to_hf, text_utils  # noqa

# Facebook BART model fine-tuned for NLI tasks
HF_MNLI = "facebook/bart-large-mnli"
nli_model = AutoModelForSequenceClassification.from_pretrained(
    HF_MNLI, truncation=True)
tokenizer = AutoTokenizer.from_pretrained(HF_MNLI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model.to(device)

TARGET_QUESTION = "ptylik_dwhatdp"
SUPPORT_THRESHOLD = 0.5
MIN_SENTENCE_LENGTH = 4
DS_NAME = "hf_dataset"
CSV_NAME = "anes20i1.csv"

print("Loading csv file...")
# load csv file, create and save hf dataset with text column from target_question column
csv_to_hf.convert_csv_to_hf(
    "../data/" + CSV_NAME, TARGET_QUESTION, "../data/" + DS_NAME)
# load hf dataset
ds = load_from_disk("../data/" + DS_NAME)
# get rid of "//" and "\\" in text column
print("Cleaning text...")
ds.map(lambda x: {"text": x["text"].replace("//", " ").replace("\\", " ")})

print("Extracting sentences...")
# get sentences from text column
sentences = text_utils.get_all_sentences(ds["text"])
# Drop sentences that are too short
sentences = [sentence for sentence in sentences if
             text_utils.word_count(sentence) >= MIN_SENTENCE_LENGTH]
print(len(sentences), "sentences", "in", ds.num_rows, "rows")
# get support for sentences
print("Computing support...")
support = {}
sentence_count = 0
for sentence in sentences:
    support_score = 0
    for row in ds:
        x = tokenizer.encode(row['text'], sentence, return_tensors='pt',
                             truncation=True)
        raw_logits = nli_model(x.to(device))[0].data
        # raw_logits are logits for [contradiction, neutral, entailment]
        # apply softmax to estimate probabilities - last is entailement
        entailment = torch.nn.Softmax(dim=-1)(raw_logits).tolist()[0][2]
        if entailment > SUPPORT_THRESHOLD:
            support_score += entailment
    support[sentence] = support_score
    sentence_count += 1
    if sentence_count % 100 == 0:
        print(sentence_count, "sentences processed out of ", len(sentences))
# sort sentences by support
print("Sorting sentences by support...")
sorted_support = sorted(support.items(), key=lambda kv: kv[1], reverse=True)
# print top 10 sentences
print("Top 10 sentences:")
for i in range(10):
    print(sorted_support[i])
