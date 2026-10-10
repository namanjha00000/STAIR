/* ====================================================================
   STAIR – app.js
   Full client-side logic: document management, chat, streaming, ToC
   ==================================================================== */

const API = '';  // same-origin

// ── State ────────────────────────────────────────────────────────────────
let activeDocId = null;
let isQuerying  = false;
let currentPopover = null;

// ── DOM references ───────────────────────────────────────────────────────
const sidebar        = document.getElementById('sidebar');
const sidebarToggle  = document.getElementById('sidebarToggle');
const menuBtn        = document.getElementById('menuBtn');
const fileInput      = document.getElementById('fileInput');
const uploadZone     = document.getElementById('uploadZone');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill   = document.getElementById('progressFill');
const progressLabel  = document.getElementById('progressLabel');
const docList        = document.getElementById('docList');
const retrievalModelSel = document.getElementById('retrievalModel');
const answerModelSel    = document.getElementById('answerModel');
const topKInput      = document.getElementById('topK');
const saveSettingsBtn= document.getElementById('saveSettings');
const settingsStatus = document.getElementById('settingsStatus');
const statusDot      = document.getElementById('statusDot');
const activeDocTitle = document.getElementById('activeDocTitle');
const topbarBadges   = document.getElementById('topbarBadges');
const welcomeState   = document.getElementById('welcomeState');
const tocPanel       = document.getElementById('tocPanel');
const tocToggle      = document.getElementById('tocToggle');
const tocTree        = document.getElementById('tocTree');
const chatArea       = document.getElementById('chatArea');
const chatMessages   = document.getElementById('chatMessages');
const queryInput     = document.getElementById('queryInput');
const sendBtn        = document.getElementById('sendBtn');
const modelHint      = document.getElementById('modelHint');
const toastContainer = document.getElementById('toastContainer');

// ── Init ─────────────────────────────────────────────────────────────────
(async function init() {
  setupEventListeners();
  await checkHealth();
  await loadDocuments();
})();

// ── Health & Models ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const data = await apiFetch('/api/health');
    statusDot.classList.toggle('connected', data.ollama_connected);
    statusDot.classList.toggle('error', !data.ollama_connected);
    statusDot.title = data.ollama_connected
      ? `Ollama connected · ${data.available_models.length} models`
      : 'Ollama not reachable';

    // Populate model selects
    const models = data.available_models || [];
    [retrievalModelSel, answerModelSel].forEach(sel => {
      sel.innerHTML = '';
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        sel.appendChild(opt);
      });
      // Pre-select current model
    });
    retrievalModelSel.value = data.current_retrieval_model || models[0] || '';
    answerModelSel.value    = data.current_answer_model    || models[0] || '';
    modelHint.textContent   = data.current_answer_model || 'ollama';

    if (!data.ollama_connected) {
      showToast('⚠️ Cannot reach Ollama. Make sure it is running on port 11434.', 'warning');
    }
  } catch(e) {
    statusDot.classList.add('error');
    statusDot.title = 'Server error';
  }
}

// ── Document management ──────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const data = await apiFetch('/api/documents');
    renderDocList(data.documents || []);
  } catch(e) {
    console.warn('Could not load documents:', e);
  }
}

function renderDocList(docs) {
  if (!docs.length) {
    docList.innerHTML = '<div class="doc-empty">No documents indexed yet</div>';
    return;
  }
  docList.innerHTML = '';
  docs.forEach(doc => {
    const item = document.createElement('div');
    item.className = 'doc-item' + (doc.doc_id === activeDocId ? ' active' : '');
    item.dataset.docId = doc.doc_id;
    item.innerHTML = `
      <div class="doc-item-icon">📄</div>
      <div class="doc-item-info">
        <div class="doc-item-title">${escHtml(doc.doc_title)}</div>
        <div class="doc-item-meta">${doc.num_leaves} sections · ${doc.num_pages} pages</div>
      </div>`;
    item.addEventListener('click', () => selectDocument(doc.doc_id, doc.doc_title));
    docList.appendChild(item);
  });
}

async function selectDocument(docId, docTitle) {
  activeDocId = docId;
  activeDocTitle.textContent = docTitle;

  // Update active state in sidebar
  document.querySelectorAll('.doc-item').forEach(el => {
    el.classList.toggle('active', el.dataset.docId === docId);
  });

  // Show chat + ToC, hide welcome
  welcomeState.classList.add('hidden');
  tocPanel.classList.remove('hidden');
  chatArea.classList.remove('hidden');

  // Load & render ToC
  await loadToC(docId);

  // Focus query input
  queryInput.focus();
}

async function loadToC(docId) {
  try {
    const data = await apiFetch(`/api/documents/${docId}/toc`);
    renderToC(data.toc_string || '');
  } catch(e) {
    tocTree.textContent = 'Could not load Table of Contents.';
  }
}

function renderToC(tocString) {
  tocTree.innerHTML = '';
  const lines = tocString.split('\n');
  lines.forEach(line => {
    const div = document.createElement('div');
    const isLeaf = /^\s*leaf\s/.test(line);
    div.className = isLeaf ? 'toc-leaf' : 'toc-node';

    // indent preservation
    const indent = (line.match(/^(\s*)/) || ['',''])[1].length;
    div.style.paddingLeft = `${indent * 6}px`;

    div.textContent = line.trim();
    if (isLeaf) {
      const title = line.replace(/^\s*leaf\s+[\d.]*\s*/, '').trim();
      div.title = 'Click to search this section';
      div.addEventListener('click', () => {
        queryInput.value = `Tell me about: ${title}`;
        queryInput.dispatchEvent(new Event('input'));
        queryInput.focus();
      });
    }
    tocTree.appendChild(div);
  });
}

// ── Upload ───────────────────────────────────────────────────────────────
function setupUploadZone() {
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleUpload(fileInput.files[0]);
  });
}

async function handleUpload(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Only PDF files are supported.', 'error');
    return;
  }

  uploadZone.classList.add('hidden');
  uploadProgress.classList.remove('hidden');
  setProgress(10, 'Uploading…');

  const formData = new FormData();
  formData.append('file', file);

  try {
    setProgress(30, 'Parsing PDF & extracting ToC…');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/api/documents/upload`);

    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        setProgress(10 + Math.round(e.loaded / e.total * 20), 'Uploading…');
      }
    };

    const result = await new Promise((resolve, reject) => {
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          try { reject(new Error(JSON.parse(xhr.responseText).detail)); }
          catch { reject(new Error(`HTTP ${xhr.status}`)); }
        }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(formData);
    });

    setProgress(80, 'Building STAIR index…');
    await sleep(400);
    setProgress(100, 'Done!');
    await sleep(600);

    showToast(`✅ Indexed "${result.doc_title}" · ${result.num_leaves} sections`, 'success');

    // Refresh doc list and select new doc
    await loadDocuments();
    await selectDocument(result.doc_id, result.doc_title);

  } catch(e) {
    showToast(`Upload failed: ${e.message}`, 'error');
  } finally {
    uploadProgress.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    progressFill.style.width = '0%';
    fileInput.value = '';
  }
}

function setProgress(pct, label) {
  progressFill.style.width = `${pct}%`;
  progressLabel.textContent = label;
}

// ── Query / Chat ──────────────────────────────────────────────────────────
async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || isQuerying) return;
  if (!activeDocId) {
    showToast('Please select or upload a document first.', 'warning');
    return;
  }

  isQuerying = true;
  sendBtn.disabled = true;
  queryInput.value = '';
  autoResize();

  // Add user message
  appendMessage('user', query);

  // Add assistant placeholder with typing indicator
  const assistantEl = appendAssistantPlaceholder();
  scrollToBottom();

  try {
    const topK = parseInt(topKInput.value, 10) || 3;

    // Use streaming endpoint
    await streamQuery(query, topK, assistantEl);
  } catch(e) {
    assistantEl.querySelector('.msg-bubble').innerHTML =
      `<span style="color:var(--error)">Error: ${escHtml(e.message)}</span>`;
    showToast(`Query error: ${e.message}`, 'error');
  } finally {
    isQuerying = false;
    sendBtn.disabled = false;
    queryInput.focus();
    scrollToBottom();
  }
}

async function streamQuery(query, topK, assistantEl) {
  const bubble = assistantEl.querySelector('.msg-bubble');

  const resp = await fetch(`${API}/api/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, doc_id: activeDocId, top_k: topK }),
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || 'Stream error');
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answerHtml = '';
  let sectionsRendered = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop(); // keep incomplete chunk

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = JSON.parse(line.slice(6));

      if (payload.type === 'sections' && !sectionsRendered) {
        sectionsRendered = true;
        // Render retrieved sections pills above the answer
        const pillsHtml = renderSectionPills(payload.sections);
        bubble.innerHTML = pillsHtml + '<div class="answer-text"></div>';
        scrollToBottom();

      } else if (payload.type === 'token') {
        answerHtml += escHtml(payload.token);
        const answerEl = bubble.querySelector('.answer-text');
        if (answerEl) {
          // Convert markdown-ish formatting
          answerEl.innerHTML = formatAnswer(answerHtml);
          scrollToBottom();
        }

      } else if (payload.type === 'done') {
        break;

      } else if (payload.type === 'error') {
        throw new Error(payload.message);
      }
    }
  }
}

function renderSectionPills(sections) {
  if (!sections || !sections.length) return '';
  const pills = sections.map((s, i) => `
    <span class="section-pill" 
          onclick="showSectionPopover(event, ${JSON.stringify(escHtml(s.title)).replace(/"/g,"&quot;")}, ${JSON.stringify(escHtml(s.content || '').slice(0,2000)).replace(/"/g,"&quot;")})"
          title="p.${s.page_num} · ${s.retrieval_method}">
      <span class="section-pill-badge">${i + 1}</span>
      ${escHtml(s.title)}
      <span class="section-pill-method">${s.retrieval_method}</span>
    </span>`).join('');
  return `<div class="retrieved-sections">${pills}</div>`;
}

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  const avatar = role === 'user'
    ? `<div class="msg-avatar">👤</div>`
    : `<div class="msg-avatar">🔍</div>`;
  div.innerHTML = `
    ${avatar}
    <div class="msg-body">
      <div class="msg-bubble">${escHtml(text)}</div>
    </div>`;
  chatMessages.appendChild(div);
  return div;
}

function appendAssistantPlaceholder() {
  const div = document.createElement('div');
  div.className = 'msg msg-assistant';
  div.innerHTML = `
    <div class="msg-avatar">🔍</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  chatMessages.appendChild(div);
  return div;
}

// ── Section content popover ───────────────────────────────────────────────
window.showSectionPopover = function(event, title, content) {
  // Close existing popover
  if (currentPopover) currentPopover.remove();

  const popover = document.createElement('div');
  popover.className = 'section-popover';
  popover.innerHTML = `
    <button class="popover-close" onclick="this.parentElement.remove()">✕</button>
    <div class="popover-title">📖 ${title}</div>
    <div class="popover-content">${content || '(no content available)'}</div>`;

  // Position near click
  const x = Math.min(event.clientX, window.innerWidth - 500);
  const y = Math.min(event.clientY + 10, window.innerHeight - 320);
  popover.style.left = `${Math.max(10, x)}px`;
  popover.style.top  = `${Math.max(10, y)}px`;

  document.body.appendChild(popover);
  currentPopover = popover;

  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', outsideClick, { once: true });
  }, 50);
  function outsideClick(e) {
    if (!popover.contains(e.target)) {
      popover.remove();
      currentPopover = null;
    }
  }
};

// ── Settings ─────────────────────────────────────────────────────────────
async function saveSettings() {
  try {
    const data = await apiFetch('/api/models/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        retrieval_model: retrievalModelSel.value,
        answer_model: answerModelSel.value,
      }),
    });
    modelHint.textContent = data.answer_model;
    settingsStatus.textContent = '✓ Settings saved';
    settingsStatus.classList.remove('hidden');
    setTimeout(() => settingsStatus.classList.add('hidden'), 2500);
    showToast('Settings saved!', 'success');
  } catch(e) {
    showToast('Failed to save settings', 'error');
  }
}

// ── ToC toggle ────────────────────────────────────────────────────────────
tocToggle.addEventListener('click', () => {
  const collapsed = tocPanel.classList.toggle('collapsed');
  tocToggle.textContent = collapsed ? 'Show' : 'Hide';
});

// ── Event listeners ──────────────────────────────────────────────────────
function setupEventListeners() {
  setupUploadZone();

  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebar.classList.toggle('collapsed');
  });
  menuBtn.addEventListener('click', () => {
    sidebar.classList.add('open');
    sidebar.classList.remove('collapsed');
  });

  sendBtn.addEventListener('click', sendQuery);
  queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });
  queryInput.addEventListener('input', autoResize);

  saveSettingsBtn.addEventListener('click', saveSettings);

  // Close sidebar on outside tap (mobile)
  document.addEventListener('click', e => {
    if (window.innerWidth <= 700 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        e.target !== menuBtn) {
      sidebar.classList.remove('open');
    }
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────
function autoResize() {
  queryInput.style.height = 'auto';
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 120)}px`;
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatAnswer(raw) {
  // Very light markdown: **bold**, *italic*, `code`, \n\n paragraphs
  return raw
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`([^`]+)`/g,     '<code>$1</code>')
    .replace(/\n\n+/g,         '</p><p>')
    .replace(/\n/g,            '<br/>');
}

async function apiFetch(path, options = {}) {
  const resp = await fetch(`${API}${path}`, options);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
