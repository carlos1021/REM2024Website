function uploadFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  const policyBriefResult = document.getElementById('policyBriefResult');

  if (!file) {
    alert('Please select a PDF file to upload.');
    return;
  }

  // Show loading message
  policyBriefResult.innerHTML = '<p>Generating Policy Brief... Please wait.</p>';

  const formData = new FormData();
  formData.append('file', file);

  // Send the PDF file to the server
  fetch('https://rem2024website.onrender.com/upload', {
    method: 'POST',
    body: formData
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('File upload failed');
      }
      return response.json();
    })
    .then(data => {
      // data contains: { doc_b64, file_name, titles (array) }

      // Convert the base64 doc data into a Blob
      const docBlob = b64toBlob(
        data.doc_b64,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      );
      const downloadUrl = URL.createObjectURL(docBlob);

      // Create a "Download Policy Brief Now" button
      const downloadButton = document.createElement('button');
      downloadButton.textContent = 'Download Policy Brief Now';
      downloadButton.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = data.file_name || 'policy_brief.docx';
        link.click();
      });

      // Display success message and add the button
      policyBriefResult.innerHTML = '<p>Policy brief generated successfully.</p>';
      policyBriefResult.appendChild(downloadButton);

      // If titles of figures/tables exist, show them
      if (data.titles && data.titles.length > 0) {
        const titlesContainer = document.createElement('div');
        titlesContainer.innerHTML = '<h2>Titles of Figures and Tables Extracted From Research Paper</h2>';

        data.titles.forEach((title) => {
          const p = document.createElement('p');
          p.textContent = title;
          titlesContainer.appendChild(p);
        });
        policyBriefResult.appendChild(titlesContainer);
      }
    })
    .catch(error => {
      console.error('Error:', error);
      policyBriefResult.innerHTML = '<p>There was an error generating the policy brief. Please try again.</p>';
    });
}

/**
 * Converts Base64 string to a Blob, so we can create a URL for downloading.
 */
function b64toBlob(b64Data, contentType, sliceSize = 512) {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  return new Blob(byteArrays, { type: contentType });
}