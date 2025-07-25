const chatBox = document.getElementById('chatBox');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');

// State to track if a PDF is active
let pdfIsActive = false;

function addMessage(text, sender, imageUrl = null) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    let content = text;
    if (imageUrl) {
        content += `<br><img src="${imageUrl}" alt="Generated Image">`;
    }
    messageElement.innerHTML = content;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageElement;
}

function showLoadingIndicator() {
    const loadingMessage = addMessage('', 'ai', null);
    loadingMessage.classList.add('loading');
    loadingMessage.innerHTML = '<div class="spinner"></div>';
    return loadingMessage;
}

function removeLoadingIndicator(indicator) {
    chatBox.removeChild(indicator);
}

function handleFileSelect(files) {
    if (files.length === 0) return;
    const file = files[0];
    addMessage(`Selected file: <b>${file.name}</b>`, 'user');
    processPdf(file);
}

async function processPdf(file) {
    const loadingIndicator = showLoadingIndicator();
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/process-pdf', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            addMessage(`✅ PDF processed successfully! You can now ask questions about it.`, 'system');
            pdfIsActive = true;
        } else {
            addMessage(`❌ PDF Error: ${result.detail}`, 'system');
            pdfIsActive = false;
        }
    } catch (error) {
        addMessage(`❌ Network Error: ${error.message}`, 'system');
        pdfIsActive = false;
    } finally {
        removeLoadingIndicator(loadingIndicator);
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userText = userInput.value.trim();
    if (!userText) return;
    addMessage(userText, 'user');
    userInput.value = '';
    const loadingIndicator = showLoadingIndicator();
    try {
        // Determine the correct API endpoint based on context
        const isImageRequest = userText.toLowerCase().includes('generate') || userText.toLowerCase().includes('create an image') || userText.toLowerCase().includes('draw');
        if (isImageRequest) {
            // It's an image generation request
            if (pdfIsActive) {
                await generatePdfBasedImage(userText);
            } else {
                await generateRegularImage(userText);
            }
        } else {
            // It's a question-answering request
            if (pdfIsActive) {
                await askPdfQuestion(userText);
            } else {
                addMessage("Please upload a PDF first to ask questions about it.", 'system');
            }
        }
    } catch (error) {
        addMessage(`❌ An unexpected error occurred: ${error.message}`, 'system');
    } finally {
        removeLoadingIndicator(loadingIndicator);
    }
});

async function askPdfQuestion(question) {
    const formData = new FormData();
    formData.append('question', question);
    formData.append('use_llama3', 'true'); // Defaulting to the advanced model for simplicity
    const response = await fetch('/ask-question', { method: 'POST', body: formData });
    const result = await response.json();
    if (response.ok) {
        addMessage(result.answer, 'ai');
    } else {
        addMessage(`❌ Q&A Error: ${result.detail}`, 'system');
    }
}

async function generateRegularImage(prompt) {
    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('use_stable_diffusion', 'true'); // Defaulting to SD
    const response = await fetch('/generate-image', { method: 'POST', body: formData });
    const result = await response.json();
    if (response.ok && result.image_file) {
        // Always use only the base filename for the download URL
        const imageFile = result.image_file.split('/').pop();
        addMessage("Here is the image you requested:", 'ai', `/generated_image/${imageFile}`);
    } else {
        addMessage(`❌ Image Gen Error: ${result.detail || result.status}`, 'system');
    }
}

async function generatePdfBasedImage(question) {
    const formData = new FormData();
    formData.append('question', question);
    formData.append('use_stable_diffusion', 'true'); // Defaulting to SD
    const response = await fetch('/generate-pdf-image', { method: 'POST', body: formData });
    const result = await response.json();
    if (response.ok && result.image_file) {
        // Always use only the base filename for the download URL
        const imageFile = result.image_file.split('/').pop();
        addMessage("Generated an image based on the PDF content:", 'ai', `/generated_image/${imageFile}`);
    } else {
        addMessage(`❌ PDF Image Gen Error: ${result.detail || result.status}`, 'system');
    }
}

// Initial welcome message
addMessage("Welcome! Upload a PDF to start asking questions, or type a prompt to generate an image.", 'system'); 