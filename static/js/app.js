document
  .getElementById('uploadButton')
  .addEventListener('click', triggerFileInput);
document
  .getElementById('predictButton')
  .addEventListener('click', handlePrediction);
document.getElementById('resetButton').addEventListener('click', resetApp);

const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const previewText = document.getElementById('previewText');
const predictButton = document.getElementById('predictButton');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resultContent = document.getElementById('resultContent');
const uploadSection = document.querySelector('.upload-section');
const origin = window.location.origin;
let selectedFile = null; // Variable to store the selected file

['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
  imagePreview.addEventListener(eventName, preventDefaults, false);
});

['dragenter', 'dragover'].forEach((eventName) => {
  imagePreview.addEventListener(
    eventName,
    () => imagePreview.classList.add('dragging'),
    false
  );
});

['dragleave', 'drop'].forEach((eventName) => {
  imagePreview.addEventListener(
    eventName,
    () => imagePreview.classList.remove('dragging'),
    false
  );
});

imagePreview.addEventListener('drop', handleDrop, false);
imagePreview.addEventListener('click', triggerFileInput);

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length) {
    selectedFile = files[0]; // Store the selected file globally
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];

    if (allowedTypes.indexOf(selectedFile.type) === -1) {
      showToastMessage(
        'error',
        'Invalid file type. Please upload an image file (PNG or JPEG)'
      );

      return resetApp();
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      const img = new Image();
      img.onload = function () {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        previewText.style.display = 'none';
        predictButton.disabled = false;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(selectedFile);
  }
}

function triggerFileInput() {
  document.getElementById('imageInput').click();
}

function handlePrediction() {
  if (selectedFile) {
    showLoading();

    const formData = new FormData();
    formData.append('image', selectedFile);

    fetch(`${origin}/process_image`, {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        showToastMessage('success', 'Prediction successful');
        showResults(`Prediction: ${data.prediction}`);
      })
      .catch((error) => {
        console.error('Error:', error);
        showToastMessage('error', 'An error occurred. Please try again.');
      });
  }
}

function showLoading() {
  uploadSection.style.display = 'none';
  loading.style.display = 'block';
}

function showResults(resultText) {
  loading.style.display = 'none';
  results.style.display = 'block';
  resultContent.textContent = resultText;
}

function resetApp() {
  results.style.display = 'none';
  uploadSection.style.display = 'flex';
  previewImg.style.display = 'none';
  previewText.style.display = 'block';
  document.getElementById('imageInput').value = '';
  predictButton.disabled = true;
  selectedFile = null; // Reset selected file when resetting the app
}

function showToastMessage(type, message) {
  const toastContainer = document.getElementById('toast-container');

  // Create a new toast message element
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerText = message;

  // Append the toast message to the container
  toastContainer.appendChild(toast);

  // Show the toast message with a slide-in animation
  setTimeout(() => {
    toast.classList.add('show');
  }, 100); // Slight delay to trigger CSS transition

  // Hide the toast message after 3 seconds
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      toastContainer.removeChild(toast);
    }, 300); // Wait for the transition to complete before removing the element
  }, 3000);
}
