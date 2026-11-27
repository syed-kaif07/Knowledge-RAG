// frontend/app.js
// Client Application for Knowledge RAG with Pitch-Black / Charcoal LinkHub Theme

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const greetingEl = document.getElementById("greeting");
  const tabButtons = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Chat Elements
  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const chatMessages = document.getElementById("chat-messages");
  const btnSend = document.getElementById("btn-send");

  // Settings & Ingestion Elements
  const modeRadios = document.querySelectorAll("input[name='index_mode']");
  const toggleSources = document.getElementById("toggle-sources");
  const toggleEval = document.getElementById("toggle-eval");
  const btnUpload = document.getElementById("btn-upload");
  const pdfUpload = document.getElementById("pdf-upload");
  const btnIndex = document.getElementById("btn-index");
  const ingestStatus = document.getElementById("ingest-status");

  // Status Elements
  const statusBox = document.getElementById("system-status-box");
  const statusText = document.getElementById("system-status-text");

  // Reviewer Elements
  const btnRefreshReviewer = document.getElementById("btn-refresh-reviewer");
  const reviewerText = document.getElementById("reviewer-briefing-text");
  const rev503 = document.getElementById("rev-503");
  const revTimeouts = document.getElementById("rev-timeouts");
  const revGroq = document.getElementById("rev-groq");
  const revUnindexed = document.getElementById("rev-unindexed");

  // Evaluation Elements
  const metricAvgFaith = document.getElementById("metric-avg-faith");
  const metricAvgRel = document.getElementById("metric-avg-rel");
  const metricAvgUtil = document.getElementById("metric-avg-util");
  const metricTotalEval = document.getElementById("metric-total-eval");
  const evalHistoryList = document.getElementById("eval-history-list");

  // State
  let evalHistory = [];

  // Greeting
  function updateGreeting() {
    const hour = new Date().getHours();
    if (hour < 12) greetingEl.textContent = "Good morning.";
    else if (hour < 17) greetingEl.textContent = "Good afternoon.";
    else greetingEl.textContent = "Good evening.";
  }
  updateGreeting();

  // Tab Navigation
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${btn.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (btn.dataset.tab === "eval") {
        loadEvaluationHistory();
      } else if (btn.dataset.tab === "reviewer") {
        loadReviewerBriefing();
      }
    });
  });

  function getActiveMode() {
    const checked = document.querySelector("input[name='index_mode']:checked");
    return checked ? checked.value : "llamaparse";
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  // System Status Check
  async function checkStatus() {
    try {
      const mode = getActiveMode();
      const res = await fetch(`/api/status?mode=${mode}`);
      const data = await res.json();

      if (!data.index_ready) {
        statusBox.className = "status-box alert";
        statusText.textContent = `No index for ${mode}. Indexing required.`;
      } else if (data.unindexed_pdfs > 0 || data.unindexed_mds > 0) {
        statusBox.className = "status-box alert";
        statusText.textContent = `Alert: ${data.unindexed_pdfs + data.unindexed_mds} unindexed document(s).`;
      } else {
        statusBox.className = "status-box";
        statusText.textContent = "System Status: Online & Ready";
      }
    } catch {
      statusBox.className = "status-box alert";
      statusText.textContent = "Status: Connection offline";
    }
  }

  function updateRadioSelection() {
    modeRadios.forEach((r) => {
      const label = r.closest(".radio-label");
      if (label) {
        label.classList.toggle("selected", r.checked);
      }
    });
  }

  modeRadios.forEach((r) => {
    r.addEventListener("change", () => {
      updateRadioSelection();
      checkStatus();
    });
  });
  updateRadioSelection();
  checkStatus();

  // Markdown & Structured Response Renderer (ChatGPT Style)
  function renderChatGPTMarkdown(text) {
    if (!text) return "";

    let raw = escapeHtml(text);

    // 1. Inline Citations
    raw = raw.replace(/\[Source:\s*([^\]]+)\]/gi, (match, src) => {
      return `<span class="inline-citation">Source: ${src.trim()}</span>`;
    });

    // 2. Fenced Code Blocks
    raw = raw.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const langLabel = lang.trim() || "code";
      return `<div class="code-block"><div class="code-header"><span>${langLabel}</span><button type="button" class="code-copy-btn">Copy code</button></div><pre><code>${code.trim()}</code></pre></div>`;
    });

    // 3. Inline Code
    raw = raw.replace(/`([^`]+)`/g, "<code>$1</code>");

    // 4. Headings
    raw = raw.replace(/^### (.*$)/gim, "<h4>$1</h4>");
    raw = raw.replace(/^## (.*$)/gim, "<h3>$1</h3>");
    raw = raw.replace(/^# (.*$)/gim, "<h2>$1</h2>");

    // 5. Bold & Italics
    raw = raw.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    raw = raw.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // 6. Structured Steps & Lifecycles (matching ChatGPT screenshot)
    const lines = raw.split("\n");
    const processed = [];
    let currentCardItems = [];
    let inStepCard = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const stepMatch = line.match(/^(\d+)[\.\s]+(.*?)(?::\s*(.*))?$/);

      if (stepMatch && !line.startsWith("<h") && stepMatch[2].length > 1 && !line.includes("<div class=\"code-block")) {
        const num = stepMatch[1];
        let title = stepMatch[2];
        let desc = stepMatch[3] || "";

        if (!desc && i + 1 < lines.length && lines[i + 1].trim() && !lines[i + 1].trim().match(/^\d+[\.\s]+/)) {
          desc = lines[i + 1].trim();
          i++;
        }

        currentCardItems.push({ num, title, desc });
        inStepCard = true;
      } else {
        if (inStepCard && currentCardItems.length >= 2) {
          const cardHtml = `
            <div class="structured-card">
              <div class="structured-card-header">Agentic Architecture Steps</div>
              ${currentCardItems.map(item => `
                <div class="step-item">
                  <span class="step-num">${item.num}</span>
                  <div class="step-content">
                    <span class="step-title">${item.title}</span>
                    ${item.desc ? `<span class="step-desc">${item.desc}</span>` : ""}
                  </div>
                </div>
              `).join("")}
            </div>
          `;
          processed.push(cardHtml);
          currentCardItems = [];
          inStepCard = false;
        } else if (inStepCard) {
          currentCardItems.forEach(item => {
            processed.push(`<p><strong>${item.num}. ${item.title}</strong>${item.desc ? ` - ${item.desc}` : ""}</p>`);
          });
          currentCardItems = [];
          inStepCard = false;
        }

        if (line) {
          if (line.startsWith("- ") || line.startsWith("* ")) {
            processed.push(`<li>${line.substring(2)}</li>`);
          } else if (!line.startsWith("<h") && !line.startsWith("<div") && !line.startsWith("</div") && !line.startsWith("<pre")) {
            processed.push(`<p>${line}</p>`);
          } else {
            processed.push(line);
          }
        }
      }
    }

    if (inStepCard && currentCardItems.length >= 2) {
      const cardHtml = `
        <div class="structured-card">
          <div class="structured-card-header">Agentic Architecture Steps</div>
          ${currentCardItems.map(item => `
            <div class="step-item">
              <span class="step-num">${item.num}</span>
              <div class="step-content">
                <span class="step-title">${item.title}</span>
                ${item.desc ? `<span class="step-desc">${item.desc}</span>` : ""}
              </div>
            </div>
          `).join("")}
        </div>
      `;
      processed.push(cardHtml);
    } else if (inStepCard) {
      currentCardItems.forEach(item => {
        processed.push(`<p><strong>${item.num}. ${item.title}</strong>${item.desc ? ` - ${item.desc}` : ""}</p>`);
      });
    }

    return processed.join("");
  }

  // Quick upload button in chat input bar
  const btnInputUpload = document.getElementById("btn-input-upload");
  if (btnInputUpload && pdfUpload) {
    btnInputUpload.addEventListener("click", () => {
      pdfUpload.click();
    });
  }

  // Copy helper
  chatMessages.addEventListener("click", (e) => {
    const copyBtn = e.target.closest(".btn-copy");
    if (copyBtn) {
      const textToCopy = copyBtn.dataset.answer || "";
      if (textToCopy) {
        navigator.clipboard.writeText(textToCopy);
        const original = copyBtn.textContent;
        copyBtn.textContent = "Copied!";
        setTimeout(() => {
          copyBtn.textContent = original;
        }, 2000);
      }
      return;
    }

    const codeCopy = e.target.closest(".code-copy-btn");
    if (codeCopy) {
      const block = codeCopy.closest(".code-block");
      const code = block ? block.querySelector("pre code")?.innerText : "";
      if (code) {
        navigator.clipboard.writeText(code);
        codeCopy.textContent = "Copied!";
        setTimeout(() => {
          codeCopy.textContent = "Copy code";
        }, 2000);
      }
    }
  });

  // Chat Submission
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query) return;

    // Append User Message (ChatGPT user pill)
    const userRow = document.createElement("div");
    userRow.className = "message-row user-row";
    userRow.innerHTML = `<div class="user-bubble">${escapeHtml(query)}</div>`;
    chatMessages.appendChild(userRow);

    userInput.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Append Loading Assistant Message
    const loaderId = "loader-" + Date.now();
    const loaderRow = document.createElement("div");
    loaderRow.id = loaderId;
    loaderRow.className = "message-row assistant-row";
    loaderRow.innerHTML = `<div class="message-body assistant-body"><p>Thinking and grounding answer...</p></div>`;
    chatMessages.appendChild(loaderRow);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    btnSend.disabled = true;

    try {
      const mode = getActiveMode();
      const enableEval = toggleEval.checked;
      const showSources = toggleSources.checked;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          mode: mode,
          enable_eval: enableEval,
        }),
      });

      const data = await res.json();
      const loader = document.getElementById(loaderId);
      if (loader) loader.remove();

      const assistantRow = document.createElement("div");
      assistantRow.className = "message-row assistant-row";

      if (data.error) {
        assistantRow.innerHTML = `
          <div class="message-body assistant-body" style="color:var(--danger);">
            <p>${escapeHtml(data.error.details || "An error occurred while generating the answer.")}</p>
          </div>
        `;
      } else {
        const formattedAnswer = renderChatGPTMarkdown(data.answer || "");

        // Grounded Sources Accordion
        let sourcesHtml = "";
        if (showSources && data.sources && data.sources.length > 0) {
          const sourceItems = data.sources
            .map((s, idx) => `
              <div class="source-chunk">
                <strong>Source [${idx + 1}]</strong> (${escapeHtml(s.source || "Document")}):
                <div class="source-content">${escapeHtml(s.content || s.text || "")}</div>
              </div>
            `)
            .join("");

          sourcesHtml = `
            <div class="source-accordion">
              <details>
                <summary>View ${data.sources.length} Grounded Source Chunks</summary>
                ${sourceItems}
              </details>
            </div>
          `;
        }

        // Live RAGAS Evaluation Badge
        let evalHtml = "";
        if (data.evaluation) {
          const f = data.evaluation.faithfulness !== undefined ? `${Math.round(data.evaluation.faithfulness * 100)}%` : "--";
          const r = data.evaluation.answer_relevancy !== undefined ? `${Math.round(data.evaluation.answer_relevancy * 100)}%` : "--";
          const u = data.evaluation.context_utilization !== undefined ? `${Math.round(data.evaluation.context_utilization * 100)}%` : "--";

          evalHtml = `
            <div class="bubble-eval">
              <span class="eval-pill">Faithfulness: ${f}</span>
              <span class="eval-pill">Relevancy: ${r}</span>
              <span class="eval-pill">Context Utilization: ${u}</span>
            </div>
          `;

          evalHistory.push({
            query: query,
            ...data.evaluation,
            latency_ms: data.latency_ms,
          });
        }

        const latencyHtml = data.latency_ms ? `<span class="msg-latency-tag">${data.latency_ms}ms</span>` : "";

        assistantRow.innerHTML = `
          <div class="message-body assistant-body">
            ${formattedAnswer}
          </div>
          <div class="assistant-footer">
            <button type="button" class="msg-action-btn btn-copy" data-answer="${escapeHtml(data.answer || "")}">Copy</button>
            ${latencyHtml}
            ${sourcesHtml}
            ${evalHtml}
          </div>
        `;
      }

      chatMessages.appendChild(assistantRow);
    } catch (err) {
      const loader = document.getElementById(loaderId);
      if (loader) loader.remove();

      const errRow = document.createElement("div");
      errRow.className = "message-row assistant-row";
      errRow.innerHTML = `
        <div class="message-body assistant-body" style="color:var(--danger);">
          <p>Network error: ${escapeHtml(err.message)}</p>
        </div>
      `;
      chatMessages.appendChild(errRow);
    } finally {
      btnSend.disabled = false;
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  });

  // PDF Upload
  btnUpload.addEventListener("click", async () => {
    if (!pdfUpload.files.length) {
      ingestStatus.textContent = "Select PDF files first.";
      ingestStatus.className = "helper-text error";
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < pdfUpload.files.length; i++) {
      formData.append("files", pdfUpload.files[i]);
    }

    ingestStatus.textContent = "Uploading PDF files...";
    ingestStatus.className = "helper-text";

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      ingestStatus.textContent = `Uploaded ${data.saved_count || data.count || 0} file(s). Ready to index.`;
      ingestStatus.className = "helper-text ok";
      pdfUpload.value = "";
      checkStatus();
    } catch (err) {
      ingestStatus.textContent = "Upload failed: " + err.message;
      ingestStatus.className = "helper-text error";
    }
  });

  // Index Documents
  btnIndex.addEventListener("click", async () => {
    const mode = getActiveMode();
    ingestStatus.textContent = `Running ${mode} indexing pipeline...`;
    ingestStatus.className = "helper-text";
    btnIndex.disabled = true;

    try {
      const formData = new FormData();
      formData.append("mode", mode);

      const res = await fetch(`/api/index?mode=${mode}`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (data.error) {
        ingestStatus.textContent = "Indexing error: " + (data.error.details || data.error);
        ingestStatus.className = "helper-text error";
      } else {
        ingestStatus.textContent = data.message || "Indexing complete.";
        ingestStatus.className = "helper-text ok";
      }
      checkStatus();
    } catch (err) {
      ingestStatus.textContent = "Indexing failed: " + err.message;
      ingestStatus.className = "helper-text error";
    } finally {
      btnIndex.disabled = false;
    }
  });

  // Load Evaluation History
  async function loadEvaluationHistory() {
    try {
      const res = await fetch("/api/evaluation/history");
      const data = await res.json();
      evalHistory = data.history || [];

      if (!evalHistory.length) {
        metricAvgFaith.textContent = "--%";
        metricAvgRel.textContent = "--%";
        metricAvgUtil.textContent = "--%";
        metricTotalEval.textContent = "0";
        evalHistoryList.innerHTML = `<p class="empty-state">No queries evaluated yet. Enable 'Run live RAGAS evaluation' in the sidebar to score responses.</p>`;
        return;
      }

      const total = evalHistory.length;
      let sumFaith = 0, sumRel = 0, sumUtil = 0;
      let validFaith = 0, validRel = 0, validUtil = 0;

      evalHistory.forEach((item) => {
        if (typeof item.faithfulness === "number" && !isNaN(item.faithfulness)) {
          sumFaith += item.faithfulness;
          validFaith++;
        }
        if (typeof item.answer_relevancy === "number" && !isNaN(item.answer_relevancy)) {
          sumRel += item.answer_relevancy;
          validRel++;
        }
        if (typeof item.context_utilization === "number" && !isNaN(item.context_utilization)) {
          sumUtil += item.context_utilization;
          validUtil++;
        }
      });

      metricAvgFaith.textContent = validFaith > 0 ? `${Math.round((sumFaith / validFaith) * 100)}%` : "--%";
      metricAvgRel.textContent = validRel > 0 ? `${Math.round((sumRel / validRel) * 100)}%` : "--%";
      metricAvgUtil.textContent = validUtil > 0 ? `${Math.round((sumUtil / validUtil) * 100)}%` : "--%";
      metricTotalEval.textContent = total;

      evalHistoryList.innerHTML = evalHistory
        .slice()
        .reverse()
        .map((entry) => `
          <div class="eval-history-card">
            <div class="eval-query-text">${escapeHtml(entry.query || "Query")}</div>
            <div class="eval-scores-row">
              <span>Faithfulness: ${entry.faithfulness !== undefined ? Math.round(entry.faithfulness * 100) + "%" : "--"}</span>
              <span>Relevancy: ${entry.answer_relevancy !== undefined ? Math.round(entry.answer_relevancy * 100) + "%" : "--"}</span>
              <span>Context: ${entry.context_utilization !== undefined ? Math.round(entry.context_utilization * 100) + "%" : "--"}</span>
              <span>Latency: ${entry.latency_ms || "--"}ms</span>
            </div>
          </div>
        `)
        .join("");
    } catch (err) {
      evalHistoryList.innerHTML = `<p class="empty-state">Error loading evaluation metrics: ${escapeHtml(err.message)}</p>`;
    }
  }

  // Load Reviewer Briefing
  async function loadReviewerBriefing() {
    reviewerText.textContent = "Scanning telemetry logs, failover events, and unindexed documents...";
    try {
      const res = await fetch("/api/reviewer/briefing");
      const data = await res.json();

      if (data.telemetry) {
        rev503.textContent = data.telemetry.http_503_count || 0;
        revTimeouts.textContent = data.telemetry.timeout_count || 0;
        revGroq.textContent = data.telemetry.groq_failover_count || 0;
      }
      if (data.corpus) {
        revUnindexed.textContent = data.corpus.unindexed_markdown || 0;
      }

      reviewerText.textContent = data.briefing || "No telemetry briefing available.";
    } catch (err) {
      reviewerText.textContent = "Failed to load reviewer briefing: " + err.message;
    }
  }

  btnRefreshReviewer.addEventListener("click", loadReviewerBriefing);
});
