document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const browseBtn = document.getElementById('browseBtn');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const changeImgBtn = document.getElementById('changeImgBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnSpinner = document.getElementById('btnSpinner');

  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');

  const statusBanner = document.getElementById('statusBanner');
  const statusBadge = document.getElementById('statusBadge');
  const statusExplanation = document.getElementById('statusExplanation');

  const categoryVal = document.getElementById('categoryVal');
  const catConfFill = document.getElementById('catConfFill');
  const catConfText = document.getElementById('catConfText');

  const simVal = document.getElementById('simVal');
  const simFill = document.getElementById('simFill');
  const simText = document.getElementById('simText');

  const designCard = document.getElementById('designCard');
  const metaDesignId = document.getElementById('metaDesignId');
  const metaMaterial = document.getElementById('metaMaterial');
  const metaStone = document.getElementById('metaStone');
  const metaCollection = document.getElementById('metaCollection');
  const metaDesc = document.getElementById('metaDesc');

  const summaryText = document.getElementById('summaryText');

  const qaInput = document.getElementById('qaInput');
  const askBtn = document.getElementById('askBtn');
  const qaResponse = document.getElementById('qaResponse');
  const qaAnswerText = document.getElementById('qaAnswerText');

  let selectedFile = null;
  let activeDesignId = null;

  // File Upload Handlers
  browseBtn.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('click', (e) => {
    if (e.target === dropzone || e.target.classList.contains('upload-icon') || e.target.classList.contains('drop-text')) {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag & Drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (dt.files && dt.files[0]) {
      handleFileSelected(dt.files[0]);
    }
  });

  changeImgBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPG, PNG, WEBP).');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropzonePrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      analyzeBtn.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    selectedFile = null;
    activeDesignId = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    dropzonePrompt.classList.remove('hidden');
    analyzeBtn.classList.add('hidden');
    resultsContent.classList.add('hidden');
    resultsPlaceholder.classList.remove('hidden');
  }

  // Analyze Action
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Set UI loading
    analyzeBtn.disabled = true;
    btnSpinner.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/v1/identify', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze image.');
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      analyzeBtn.disabled = false;
      btnSpinner.classList.add('hidden');
    }
  });

  function renderResults(data) {
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    const status = data.status.toLowerCase();
    statusBanner.className = `status-banner ${status}`;
    statusBadge.textContent = data.status;
    statusExplanation.textContent = data.explanation;

    // Category
    categoryVal.textContent = data.category;
    const catPct = Math.round(data.category_confidence * 100);
    catConfFill.style.width = `${catPct}%`;
    catConfText.textContent = `${catPct}% Category Confidence`;

    // Similarity
    const simPct = Math.round(data.similarity_score * 100);
    simVal.textContent = `${simPct}%`;
    simFill.style.width = `${simPct}%`;
    if (data.status === 'KNOWN_DESIGN') {
      simText.textContent = `${simPct}% FAISS Visual Similarity`;
    } else if (data.status === 'LOW_CONFIDENCE') {
      simText.textContent = 'Low Confidence - No Match';
    } else {
      simText.textContent = 'No Catalog Match';
    }

    // Design Card
    if (data.status === 'KNOWN_DESIGN' && data.design_details) {
      activeDesignId = data.best_match_design_id;
      const d = data.design_details;
      metaDesignId.textContent = d.design_id;
      const matStr = d.material && d.karat ? `${d.material} (${d.karat}K)` : (d.material || (d.karat ? `${d.karat}K Gold` : 'Not specified'));
      metaMaterial.textContent = matStr;
      metaStone.textContent = d.stone || 'None';
      metaCollection.textContent = d.collection || 'Heritage';
      metaDesc.textContent = d.description || '';
      designCard.classList.remove('hidden');
    } else {
      activeDesignId = null;
      designCard.classList.add('hidden');
    }

    // Summary
    summaryText.innerHTML = data.explanation.replace(/\n/g, '<br>');

    // Q&A reset
    qaInput.value = '';
    qaResponse.classList.add('hidden');
  }

  // Q&A Action
  askBtn.addEventListener('click', handleAskQuestion);
  qaInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleAskQuestion();
  });

  async function handleAskQuestion() {
    const question = qaInput.value.trim();
    if (!question) return;

    if (!activeDesignId) {
      alert('Follow-up Q&A is available when a known design is identified.');
      return;
    }

    askBtn.disabled = true;
    askBtn.textContent = '...';

    try {
      const response = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          design_id: activeDesignId,
          question: question
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to get answer.');
      }

      const data = await response.json();
      qaAnswerText.textContent = data.answer;
      qaResponse.classList.remove('hidden');
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      askBtn.disabled = false;
      askBtn.textContent = 'Ask';
    }
  }
});
