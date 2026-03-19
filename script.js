const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const languageSelect = document.getElementById('languageSelect');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const statusText = document.getElementById('statusText');
const resultSection = document.getElementById('resultSection');
const audioResult = document.getElementById('audioResult');
const downloadAudio = document.getElementById('downloadAudio');
const downloadMetadata = document.getElementById('downloadMetadata');
const segmentsBody = document.querySelector('#segmentsTable tbody');

let selectedFile = null;

// Drag and drop handling
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        fileNameDisplay.textContent = `Selected: ${selectedFile.name}`;
    }
}

// Processing logic
processBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Please select an audio file first.");
        return;
    }

    const language = languageSelect.value;
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Update UI for processing
    processBtn.disabled = true;
    progressContainer.classList.remove('hidden');
    resultSection.classList.add('hidden');
    progressFill.style.width = '20%';
    statusText.textContent = "Uploading and transcribing...";

    try {
        const response = await fetch(`/translate?language=${language}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText || response.statusText}`);
        }

        const data = await response.json();

        if (data.status === 'success') {
            progressFill.style.width = '100%';
            statusText.textContent = "Translation Complete!";
            showResults(data);
        } else {
            throw new Error(data.message || "Processing failed");
        }
    } catch (error) {
        console.error(error);
        alert(`Error: ${error.message}`);
        statusText.textContent = "Error occurred during processing.";
    } finally {
        processBtn.disabled = false;
    }
});

function showResults(data) {
    resultSection.classList.remove('hidden');
    
    // Set audio and downloads
    audioResult.src = data.audio_url;
    downloadAudio.href = data.audio_url;
    downloadMetadata.href = data.metadata_url;
    
    // Fill segments table
    segmentsBody.innerHTML = '';
    data.segments.forEach(seg => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s</td>
            <td><span class="badge ${seg.emotion}">${seg.emotion}</span></td>
            <td>${seg.gender}</td>
            <td>${seg.age_group}</td>
            <td><span class="voice-tag ${seg.voice_type || 'neural'}">${seg.voice_type || 'Neural'}</span></td>
            <td class="translation-cell">${seg.translated_text}</td>
        `;
        segmentsBody.appendChild(row);
    });

    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}
