print('running data_collection.py')
from fpdf import FPDF
from datasets import load_from_disk
import os

# Define paths
original_papers_path = './data/arxiv_summarization_document_50'  # Path to the dataset
pdf_output_path = './data/r_P'  # Folder to save generated PDFs

# Ensure output directory exists
os.makedirs(pdf_output_path, exist_ok=True)

# Load the original dataset
ds_document = load_from_disk(original_papers_path)

# Function to save one research paper as a PDF
def save_research_paper_as_pdf(paper_index, filename="research_paper.pdf"):
    # Extract the text of the paper
    paper_text = ds_document[paper_index]['article']  # Accessing 'article' directly
    
    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Add the text to the PDF with automatic line breaks
    pdf.multi_cell(0, 10, paper_text)
    
    # Save the PDF
    pdf_path = os.path.join(pdf_output_path, filename)
    pdf.output(pdf_path)
    print(f"Research paper saved as PDF at {pdf_path}")

# Example usage
for i in range(2, 6):
    save_research_paper_as_pdf(paper_index=i, filename=f"r_p_{i}.pdf")

# from datasets import load_dataset
# import os

# # Define directory paths
# section_path = "./data/arxiv_summarization_section_50"
# document_path = "./data/arxiv_summarization_document_50"

# # Ensure the data directory exists
# os.makedirs(section_path, exist_ok=True)
# os.makedirs(document_path, exist_ok=True)

# # Load the dataset
# print("Loading the dataset...")
# ds_section = load_dataset("ccdv/arxiv-summarization", "section")
# ds_document = load_dataset("ccdv/arxiv-summarization", "document")

# # Keep only the first 50 samples
# print("Selecting the first 50 samples for each dataset...")
# ds_section_50 = ds_section['train'].select(range(50))
# ds_document_50 = ds_document['train'].select(range(50))

# # Save the reduced dataset locally
# print(f"Saving the section dataset to {section_path}...")
# ds_section_50.save_to_disk(section_path)

# print(f"Saving the document dataset to {document_path}...")
# ds_document_50.save_to_disk(document_path)

# # Confirm that data is saved by checking the directory content
# print("Data saved. Checking contents of data folder:")
# print("Contents of section folder:", os.listdir(section_path))
# print("Contents of document folder:", os.listdir(document_path))
