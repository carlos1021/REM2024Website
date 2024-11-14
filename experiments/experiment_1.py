from datasets import load_from_disk
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from docx import Document
from bert_score import score as bert_score
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch

nltk.download('punkt')

# Define paths
original_summary_path = './data/arxiv_summarization_document_50'  # Path to the summaries
generated_brief_path = './data/g_p_b'  # Path to the generated policy briefs

# Load the original dataset
ds_document = load_from_disk(original_summary_path)

# Initialize ROUGE scorer for ROUGE-L only
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Function to calculate ROUGE-L
def calculate_rouge_l(original_summary, generated_brief):
    rouge_scores = rouge_scorer_instance.score(original_summary, generated_brief)
    return rouge_scores['rougeL'].fmeasure

# Initialize BERTScore scorer
def calculate_bertscore(original_summary, generated_brief):
    P, R, F1 = bert_score([generated_brief], [original_summary], lang="en")
    return F1.mean().item()

# Initialize BARTScore model
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Function to calculate BARTScore
def calculate_bartscore(original_summary, generated_brief):
    inputs = bart_tokenizer([original_summary], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bart_model.generate(inputs.input_ids, max_length=512, num_beams=4, length_penalty=2.0, early_stopping=True)
    generated_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Calculate similarity based on the probability of generated text given reference
    return bart_model(inputs.input_ids, labels=outputs).loss.item()

# Function to load the generated brief from a DOCX file
def load_generated_brief(filename):
    path = os.path.join(generated_brief_path, filename)
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to evaluate scores between original summary and generated brief
def evaluate_policy_brief(original_summary, generated_brief):
    # Calculate ROUGE-L
    rouge_l = calculate_rouge_l(original_summary, generated_brief)
    
    # Calculate BERTScore
    bert_f1 = calculate_bertscore(original_summary, generated_brief)
    
    # Calculate BARTScore
    bartscore = calculate_bartscore(original_summary, generated_brief)
    
    return {
        'ROUGE-L': rouge_l,
        'BERTScore': bert_f1,
        'BARTScore': bartscore
    }

# Initialize a list to store scores for each document
results = []

# Run evaluation for the first five documents in the dataset
for index in range(5):
    # Load the abstract for the current paper
    original_summary = ds_document[index]['abstract']  # Assuming 'abstract' is the summary field
    
    # Construct the filename for the corresponding generated brief
    generated_filename = f"g_p_b_{index}.docx"
    
    # Check if the generated file exists
    if os.path.exists(os.path.join(generated_brief_path, generated_filename)):
        # Load the generated brief
        generated_brief = load_generated_brief(generated_filename)
        
        # Perform evaluation and append the scores to results
        scores = evaluate_policy_brief(original_summary, generated_brief)
        results.append(scores)
    else:
        print(f"Generated brief for Document {index} not found.")

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Calculate the average scores
average_scores = df_results.mean()

# Plotting the average scores as a bar chart
plt.figure(figsize=(10, 6))
average_scores.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.xlabel('Evaluation Metric')
plt.ylabel('Average Score')
plt.title('Average Evaluation Scores for Generated Policy Briefs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
