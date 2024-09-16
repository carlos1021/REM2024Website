# PolicyBrief - A Multimodal RAG Approach to Policy Brief Creation

**PolicyBrief** is an AI-powered application designed to simplify the process of generating policy briefs from research papers. This project employs a multimodal Retrieval-Augmented Generation (RAG) approach, combining both text and image analysis to generate concise summaries of key findings and visual data.

The service is currently live and accessible through the following link:
[PolicyBrief Live Service](https://rem2024-f429b.firebaseapp.com)

## Project Structure

The repository is organized as follows:

- **server**: Contains the Flask server code, which is deployed via Render. This handles the core API functionality, including processing PDF uploads, extracting images, and generating policy briefs.
- **public**: Holds static content for the frontend, which is the web interface users interact with. This includes HTML, CSS, and JavaScript.

## Features

- Upload research papers (PDF format) via the web interface.
- Automatically extracts both text and images from the PDF document.
- Generates policy briefs using advanced text summarization techniques.
- Identifies and processes relevant images from the document, integrating them into the final brief.
- Allows users to download the final policy brief in `.docx` format.

## Live Demo

You can test out the live version of PolicyBrief at:
[https://rem2024-f429b.firebaseapp.com](https://rem2024-f429b.firebaseapp.com)

## Technologies Used

- **Flask**
- **Render**
- **Firebase**
- **OpenAI API**
- **LangChain**
- **PyMuPDF (fitz)**
- **Pillow**

## Getting Started

To get the project running locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/carlos1021/REM2024Website.git

2. Navigate to the server directory:
    ``` bash
    cd server

3. Install dependencies:
    ``` bash
    pip install -r requirements.txt

4. Set up Firebase Storage, retrieve a Service Account Key and initialize app ...

5. Step out to the root directory, and run app.py:
    ``` bash
    cd ..
    python -m app

## Contributions
...

## License
...


### Summary of Sections:
1. **Project Overview**: A brief description.
2. **Project Structure**: Organization of the project files.
3. **Features**: Main functionalities of the app.
4. **Live Demo**: Live version of the project.
5. **Technologies Used**: Details the tech stack.
6. **Getting Started**: Instructions on how to clone, install, and run the project locally.
7. **Contributions**: ...
8. **License**: ...

Let me know if you need any modifications or additional information!
