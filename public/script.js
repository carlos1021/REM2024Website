function uploadFile() {
  const fileInput = document.getElementById('fileInput');
  const file = fileInput.files[0];
  const policyBriefResult = document.getElementById('policyBriefResult');

  if (!file) {
    alert('Please select a PDF file to upload.');
    return;
  }

  // Clear previous messages and show loading text
  policyBriefResult.innerHTML = '<p>Generating Policy Brief...</p>';

  // Create form data to send the file
  const formData = new FormData();
  formData.append('file', file);


//   fetch('http://127.0.0.1:8080/upload', {  // Local server API endpoint
//     method: 'POST',
//     body: formData
// })
  // Send the PDF file to the server using fetch API
  fetch('https://rem2024website.onrender.com/upload', {  // Your server API endpoint
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

      // Show success message
      policyBriefResult.innerHTML = '<p>Policy brief generated and downloaded successfully.</p>';
    })
    .catch(error => {
      console.error('Error:', error);
      policyBriefResult.innerHTML = '<p>There was an error generating the policy brief. Please try again.</p>';
    });
}
