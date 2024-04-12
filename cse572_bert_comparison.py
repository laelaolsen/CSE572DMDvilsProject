import torch
from transformers import BertTokenizer, BertModel
import csv
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# Function to read CSV file and extract columns
def read_csv(file_path):
    columns = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for column in reader.fieldnames:
            columns[column] = []
        for row in reader:
            for column in reader.fieldnames:
                columns[column].append(row[column])
    return columns

if __name__ == "__main__":

    gpt_separate = read_csv("C:/Users/laela/Downloads/Separate_Pairs_FINAL_Result_wGPT.csv")["GPT"]
    cagpt_separate = read_csv("C:/Users/laela/Downloads/Separate_Pairs_FINAL_Result_wGPT.csv")["CAGPT"]
    gpt_mixed = read_csv("C:/Users/laela/Downloads/Mixed_Pairs_FINAL_Result_wGPT.csv")["GPT"]
    cagpt_mixed= read_csv("C:/Users/laela/Downloads/Mixed_Pairs_FINAL_Result_wGPT.csv")["CAGPT"]

    # Load the BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the summaries and obtain BERT embeddings
    gpt_embeddings = []
    cagpt_embeddings = []

    for summary in tqdm(gpt_separate):
        input_ids = tokenizer.encode(summary, add_special_tokens=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        gpt_embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    for summary in tqdm(cagpt_separate):
        input_ids = tokenizer.encode(summary, add_special_tokens=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        cagpt_embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Calculate cosine similarity between GPT and CAGPT embeddings
    similarities = cosine_similarity(gpt_embeddings, cagpt_embeddings)

    average_similarity_scores = []
    # Print similarity scores
    print("Similarity scores:")
    for i, score in enumerate(similarities):
        #print(f"Pair {i+1}: {sum(score) / len(score)}")
        average_similarity_scores.append(sum(score) / len(score))
    print(np.average(average_similarity_scores))
    print(np.std(average_similarity_scores))

    # Tokenize the summaries and obtain BERT embeddings
    gpt_embeddings_mixed = []
    cagpt_embeddings_mixed = []

    for summary in tqdm(gpt_mixed):
        input_ids = tokenizer.encode(summary, add_special_tokens=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        gpt_embeddings_mixed.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    for summary in tqdm(cagpt_mixed):
        input_ids = tokenizer.encode(summary, add_special_tokens=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        cagpt_embeddings_mixed.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Calculate cosine similarity between GPT and CAGPT embeddings
    similarities_mixed = cosine_similarity(gpt_embeddings_mixed, cagpt_embeddings_mixed)

    average_similarity_scores_mixed = []
    # Print similarity scores
    print("Similarity scores:")
    for i, score in enumerate(similarities_mixed):
        #print(f"Pair {i+1}: {sum(score) / len(score)}")
        average_similarity_scores_mixed.append(sum(score) / len(score))
    print(np.average(average_similarity_scores_mixed))
    print(np.std(average_similarity_scores_mixed))

