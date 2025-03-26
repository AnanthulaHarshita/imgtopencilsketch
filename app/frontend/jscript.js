document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadedImage = document.getElementById('uploaded-image'); // Reference to the image tag

    // Check if the uploadStatus element exists
    if (!uploadStatus) {
        console.error("upload-status element not found!");
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const imageFile = imageInput.files[0];

        if (!imageFile) {
            alert('Please select an image to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageFile);

        try {
            console.log('Sending image upload request...');
            const response = await fetch('/upload_GAN/', {
                method: 'POST',
                body: formData,
            });

            // Check if response is OK
            if (!response.ok) {
                console.error('Upload failed with status:', response.status);
                throw new Error('Failed to upload the image');
            }

            const data = await response.json();
            console.log('Upload success:', data);
            alert('Image uploaded successfully!');

            // Update the uploadStatus element with success message
            uploadStatus.textContent = `Image uploaded successfully: ${data.filename}`;
            uploadStatus.style.color = 'green';

            // Log the image URL to console
            console.log('Image URL:', `/uploads/${data.filename}`);

            // Display the uploaded image
            uploadedImage.src = `/uploads/${data.filename}`; // Set the image source to the uploaded file's path
            uploadedImage.style.display = 'block'; // Make the image visible
            console.log('Server response:', data);

            // Optionally reset the input field after upload
            imageInput.value = '';

            // Add the new code here
            setTimeout(() => {
                alert('Image uploaded successfully!');
            }, 1000);

        } catch (error) {
            console.error('Error during upload:', error);

            // Update the uploadStatus element with error message
            uploadStatus.textContent = `Error uploading image: ${error.message}`;
            uploadStatus.style.color = 'red';

            // Show error alert
            alert(`Error uploading image: ${error.message}`);
        }
    });


const generateSketchForm = document.getElementById('generate-sketch-form');
const sketchContainer = document.getElementById('sketch-container');
const sketchImage = document.getElementById('sketch-image');

generateSketchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('image-input');
    console.log(fileInput.files);  // Log files array to check if the file is selected
    if (!fileInput.files[0]) {
        alert('Please select a file before submitting.');
        return;
    }
    const filename = fileInput.files[0].name;
    
    fetch('/generate-sketch_GAN/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filename: filename })
    })
    .then((response) => response.json())
    .then((data) => {
        const sketchFilename = data.processed_filename;
        sketchImage.src = sketchFilename;
        sketchContainer.style.display = 'block';
    })
    .catch((error) => console.error(error));
});
});