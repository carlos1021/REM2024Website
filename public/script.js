document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var formData = new FormData();
    formData.append('file', document.getElementById('fileInput').files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => response.text())
      .then(filename => {
          document.getElementById('generateButton').disabled = false;
          document.getElementById('generateButton').dataset.filename = filename;
      }).catch(error => {
          console.error('Error:', error);
      });
});

document.getElementById('generateButton').addEventListener('click', function() {
    var filename = this.dataset.filename;
    fetch('/generate_policy_brief', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: filename })
    }).then(response => response.blob())
      .then(blob => {
          var url = URL.createObjectURL(blob);
          var iframe = document.getElementById('policyBriefFrame');
          iframe.src = url;
          iframe.style.display = 'block';
      }).catch(error => {
          console.error('Error:', error);
      });
});
