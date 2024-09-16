function uploadFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  const policyBriefResult = document.getElementById('policyBriefResult');

  if (!file) {
    alert('Please select a PDF file to upload.');
    return;
  }

  // Create form data to send the file
  const formData = new FormData();
  formData.append('file', file);

  // Send the PDF file to the server using fetch API
  fetch('https://your-server-endpoint.com/upload', {  // Replace with your actual server API endpoint
    method: 'POST',
    body: formData
  })
    .then(response => {
      if (!response.ok) {
        throw new Error('File upload failed');
      }
      return response.blob();  // The response is a blob (e.g., .docx)
    })
    .then(blob => {
      // Create a link to download the .docx file
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'policy_brief.docx';  // The filename for the download
      document.body.appendChild(a);
      a.click();
      a.remove();

      // Clear the result section or show a success message
      policyBriefResult.innerHTML = '<p>Policy brief generated and downloaded successfully.</p>';
    })
    .catch(error => {
      console.error('Error:', error);
      policyBriefResult.innerHTML = '<p>There was an error generating the policy brief. Please try again.</p>';
    });
}
