print("Program starting, grabbing imports ...")

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from flask import Flask, request, send_file, jsonify
from docx import Document
from docx.shared import Inches
import requests
from io import BytesIO
import os
import openai
import pybase64 as base64
import firebase_admin
from firebase_admin import credentials, storage
import fitz
from PIL import Image
import io
import json
import nltk
import ssl
from flask_cors import CORS
import sys
# sys.setrecursionlimit(5000)  # Increase recursion limit to a higher value
# ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app, origins=[
    "https://rem2024-f429b.firebaseapp.com", 
    "https://rem2024-f429b.web.app", 
    "http://localhost:8000",
    "https://rem2024website.onrender.com"
],
methods=["GET", "POST", "OPTIONS"],
allow_headers=["Content-Type"])

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

print("Imports complete")

openai.api_key = os.getenv("OPENAI_API_KEY")
# firebase_key_base64 = os.getenv("FIREBASE_SERVICE_KEY")
# firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')
# firebase_service_account = json.loads(firebase_key_json)
with open("REM_Service_Account_Key.json") as f:
    firebase_service_account = json.load(f)

print(f"Grabbed OpenAI API Key: {openai.api_key != None}")
print(f"Grabbed Firebase Service Account Key: {openai.api_key != None}")

MODEL = "gpt-4o"
llm = ChatOpenAI(model=MODEL)

# Firebase initialization
cred = credentials.Certificate(firebase_service_account)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'rem2024-f429b.appspot.com'
})


def extract_images_and_text_from_pdf(pdf_file_path):
    """
    Extracts text and images from a PDF file using the Fitz (PyMuPDF) library.
    """
    pdf_document = fitz.open(pdf_file_path)
    image_urls = []
    text_content = []
    bucket = storage.bucket()

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Extract text
        text = page.get_text("text")
        text_content.append(text)

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image to PNG format
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")

            # Save image in memory to upload to Firebase
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)

            # Upload image to Firebase Storage
            image_filename = f"image_{page_num + 1}_{img_index + 1}.png"
            blob = bucket.blob(image_filename)
            blob.upload_from_file(image_buffer, content_type="image/png")

            # Make the file public
            blob.make_public()

            # Get the public URL for the uploaded image
            img_url = blob.public_url
            image_urls.append(img_url)

    pdf_document.close()
    return text_content, image_urls


def process_text(text_content):
    """
    Simulates separation of tables and normal text. Fitz doesn't distinguish between them,
    so we use some heuristics like text structure to identify potential table-like text.
    """
    tables, texts = [], []

    for text in text_content:
        if is_table_like(text):
            tables.append(text)
        else:
            texts.append(text)

    return texts, tables


def is_table_like(text):
    """
    Heuristic to determine if a block of text is table-like.
    If the text has lots of tabular structure (e.g., many lines and columns of numbers), it's likely a table.
    """
    return "\t" in text or " | " in text or "-----" in text


# Function to generate text and table summaries using OpenAI
def make_prompt(element):
    return f"""You are an assistant tasked with summarizing tables and text from retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element}"""


def generate_text_summaries(texts, tables, summarize_texts=False):
    text_summaries, table_summaries = [], []

    if texts:
        if summarize_texts:
            for text in texts:
                prompt = make_prompt(text)
                completion = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant tasked with summarizing text."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
                text_summaries.append(response)

    if tables:
        for table in tables:
            prompt = make_prompt(table)
            completion = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant tasked with summarizing tables."},
                    {"role": "user", "content": prompt}
                ]
            )
            response = completion.choices[0].message.content
            table_summaries.append(response)

    return text_summaries, table_summaries


# Function to analyze an image URL with OpenAI
def analyze_image(img_url):
    print(f"Image URL: {img_url}")
    print("")

    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Describe concisely the characteristics (shape, color), but do not infer what the image means. \
    Only describe the characteristics of the image you see."""
    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=300,
            top_p=0.1,
            timeout=30  # Add a timeout of 15 seconds
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred during OpenAI API call: {e}")
        return None


# Function to generate image summaries from Firebase URLs
def generate_image_summaries(image_urls):
    image_summaries = []
    for img_url in image_urls:
        response = analyze_image(img_url)
        image_summaries.append(response)
    return image_summaries


# Function to filter relevant images
def filter_relevant_images(image_summaries, img_url_list):
    relevant_images = []
    relevant_urls = []

    for i, summary in enumerate(image_summaries):
        prompt = f"""
        You are an expert assistant analyzing images in a research paper.
        The following image summary is provided:

        {summary}

        Determine if this image is substantially relevant to the research paper's findings, discussions, or conclusions.
        If the image is merely aesthetic or irrelevant, it should not be included.
        Please answer with "Relevant" or "Not Relevant" and provide a one sentence explanation.
        """

        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            top_p=0.1,
            timeout=30
        )

        response_text = response.choices[0].message.content.strip()

        if "Relevant" in response_text and "Not Relevant" not in response_text:
            relevant_images.append(summary)
            relevant_urls.append(img_url_list[i])
            print(f"Image {i+1}: Relevant\nExplanation: {response_text}\n")
        else:
            print(f"Image {i+1}: Not Relevant\nExplanation: {response_text}\n")

    return relevant_urls


# Function to create a Word document
def create_policy_brief_document(context, relevant_image_urls, output_path):
    """
    Create a Word document with the policy brief text and relevant images.
    """
    doc = Document()
    doc.add_heading('Policy Brief', level=1)
    doc.add_paragraph(context)

    # Add each relevant image to the document
    for img_url in relevant_image_urls:
        response = requests.get(img_url)
        img_stream = BytesIO(response.content)

        doc.add_paragraph()
        doc.add_picture(img_stream, width=Inches(5.0))

    doc.save(output_path)


@app.route('/upload', methods=['POST'])
def upload_pdf():
    print("Request received")  # Print when the request is received

    # Check if the request has the file
    if 'file' not in request.files:
        print("No file provided in the request")  # Debug if no file is provided
        return jsonify({'error': 'No file provided'}), 400

    # Get the PDF file from the request
    file = request.files['file']
    pdf_file_path = './temp_pdf.pdf'  # Temporary storage for the uploaded PDF
    print(f"File received: {file.filename}")  # Print the name of the received file

    # Save the file locally
    file.save(pdf_file_path)
    print(f"File saved locally at {pdf_file_path}")  # Confirm file saved

    # Extract text and images from PDF
    text_content, image_urls = extract_images_and_text_from_pdf(pdf_file_path)
    print(f"Extracted text content: {len(text_content)} pages")  # How many pages of text
    print(f"Extracted image URLs: {image_urls}")  # Print the image URLs extracted

    # Separate the text and tables
    texts, tables = process_text(text_content)
    print(f"Texts found: {len(texts)}, Tables found: {len(tables)}")  # Number of texts and tables found

    # Generate summaries for the extracted text and tables
    text_summaries, table_summaries = generate_text_summaries(texts, tables, summarize_texts=False)
    print(f"Generated {len(text_summaries)} text summaries and {len(table_summaries)} table summaries")

    # Generate summaries for the extracted images
    image_summaries = generate_image_summaries(image_urls)
    print(f"Generated {len(image_summaries)} image summaries")

    # Load Hugging Face embeddings and create the vectorstore
    print("Loading Hugging Face embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
    
    documents = []
    for i, table in enumerate(tables):
        documents.append({"id": f"table_{i}", "text": table})
    for i, text in enumerate(texts):
        documents.append({"id": f"text_{i}", "text": text})
    for i, summary in enumerate(image_summaries):
        documents.append({"id": f"image_summary_{i}", "text": summary})

    print(f"Documents prepared for vector store: {len(documents)}")

    doc_texts = [doc["text"] for doc in documents]
    vectorstore = FAISS.from_texts(doc_texts, embeddings)
    print("Vectorstore created")

    retriever = vectorstore.as_retriever()

    # Set up the chain for LangChain responses
    template = """
    You are an expert assistant specialized in summarizing research papers into policy briefs.
    Use the provided context to create one section of the comprehensive policy brief:

    <context>
    {context}
    </context>

    Task: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    # Process questions and generate policy brief
    questions = {
        "Group 1": "Given the research findings, craft an executive summary that highlights the most critical insights in 3-4 sentences.",
        "Group 2": "Provide a detailed background on the issue addressed in the research paper in 3-4 sentences.",
        "Group 3": "Identify and summarize the core research question and the associated problem discussed in the paper.",
        "Group 4": "Outline the key statistical findings of the paper with 3-4 bullet points including specific numbers.",
        "Group 5": "Summarize the conclusion and policy recommendations from the research paper."
    }

    context = ""
    for group, question in questions.items():
        print(f"Asking question: {question}")  # Print the question being asked
        response = chain.invoke({"context": context, "input": question})
        if response["answer"]:
            context += f"\n\nAnswer:{response['answer']}"
            print(f"Group: {group}, Answer: {response['answer']}")
        else:
            print(f"Group: {group}, No information found")

    print("Final Policy Brief:\n", context)

    # Filter relevant images
    relevant_image_urls = filter_relevant_images(image_summaries, image_urls)
    print(f"Relevant Image URLs: {relevant_image_urls}")

    # Create and save the policy brief document
    output_path = './Policy_Brief.docx'
    create_policy_brief_document(context, relevant_image_urls, output_path)
    print(f"Document saved to {output_path}")

    # Clean up the temporary file
    os.remove(pdf_file_path)
    print(f"Temporary PDF file {pdf_file_path} removed")

    # Send the .docx file back to the user
    return send_file(output_path, as_attachment=True, download_name='policy_brief.docx')



# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

