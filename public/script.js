// script.js

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const policyBriefResult = document.getElementById('policyBriefResult');
  
    if (!file) {
      alert('Please select a file to upload.');
      return;
    }
  
    const formData = new FormData();
    formData.append('file', file);
  
    fetch('https://helloworld-2gfnx6j33q-uc.a.run.app', {  // Replace with your backend API endpoint
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // Assuming the API returns a summary in a field named 'summary'
      policyBriefResult.innerHTML = `
        <h2>Generated Policy Brief</h2>
        <p>${data.summary}</p>
      `;
    })
    .catch(error => {
      console.error('Error uploading file:', error);
      policyBriefResult.innerHTML = '<p>There was an error generating the policy brief. Please try again.</p>';
    });
  }
  


// document.getElementById('uploadForm').addEventListener('submit', function(event) {
//     event.preventDefault();
//     var formData = newFormData();
//     formData.append('file', document.getElementById('fileInput').files[0]);

//     fetch('/upload', {
//         method: 'POST',
//         body: formData
//     }).then(response => response.text())
//       .then(filename => {
//           document.getElementById('generateButton').disabled = false;
//           document.getElementById('generateButton').dataset.filename = filename;
//           // Show confirmation messagedocument.getElementById('uploadStatus').textContent = 'PDF document grabbed successfully';
//       }).catch(error => {
//           console.error('Error:', error);
//       });
// });

// document.getElementById('generateButton').addEventListener('click', function() {
//     var filename = this.dataset.filename;
//     fetch('/generate_policy_brief', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ filename: filename })
//     }).then(response => response.blob())
//       .then(blob => {
//           var url = URL.createObjectURL(blob);
//           var iframe = document.getElementById('policyBriefFrame');
//           iframe.src = url;
//           iframe.style.display = 'block';
//       }).catch(error => {
//           console.error('Error:', error);
//       });
// });

