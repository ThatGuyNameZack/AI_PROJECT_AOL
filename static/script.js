
function uploadImage() {
    const imageInput = document.getElementById('imageInput');
    const responseElement = document.getElementById('response');
    if (imageInput.files.length === 0) {
        responseElement.innerText = "Please select an image to upload.";
        return;
    }
    const formData = new FormData();
    formData.append('file', imageInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        responseElement.innerText = `Uploaded: ${data.filename}`;
    })
    .catch(error => {
        console.error('Error:', error);
        responseElement.innerText = "Error occurred while uploading the image.";
    });
}