/**
 * Warehouse Activity Intelligence — Frontend Application
 */

const API_BASE = window.location.origin;

// ===== Toast Notifications (replaces alert) =====
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("visible"));
    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
const WS_BASE = API_BASE.replace(/^http/, "ws");

// ===== State =====
const state = {
    ws: null,
    wsDashboard: null,
    isStreaming: false,
    videoSource: null, // "camera" | "file"
    frameInterval: null,
    currentPrediction: null,
    timelineData: [],
    distributionData: {},
    activeProjectId: null,
    activeCollectionId: null,
    currentVideoName: null,  // for label persistence
};

// ===== Activity colors =====
const COLORS = {
    sorting: "#4f8cff",
    carrying: "#a78bfa",
    loading: "#f59e0b",
    idle: "#6b7280",
    walking: "#34d399",
    no_person: "#374151",
    untrained: "#525252",
};

// ===== DOM Elements =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ===== Tab Navigation =====
$$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        $(`#tab-${btn.dataset.tab}`).classList.add("active");
    });
});

// ===== WebSocket Connection =====
function connectLiveWS() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;

    state.ws = new WebSocket(`${WS_BASE}/ws/live`);

    state.ws.onopen = () => {
        updateConnectionStatus(true);
        console.log("[WS] Live connected");
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "prediction") {
            handlePrediction(data);
        }
    };

    state.ws.onclose = () => {
        updateConnectionStatus(false);
        setTimeout(connectLiveWS, 3000);
    };

    state.ws.onerror = () => {
        state.ws.close();
    };
}

function connectDashboardWS() {
    state.wsDashboard = new WebSocket(`${WS_BASE}/ws/dashboard`);

    state.wsDashboard.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "metrics") {
            updateKPIs(data.data);
        }
    };

    state.wsDashboard.onclose = () => {
        setTimeout(connectDashboardWS, 5000);
    };

    state.wsDashboard.onerror = () => {
        state.wsDashboard.close();
    };
}

function updateConnectionStatus(connected) {
    const dot = $("#connectionStatus");
    const label = $("#connectionLabel");
    if (connected) {
        dot.classList.add("connected");
        label.textContent = "Connected";
    } else {
        dot.classList.remove("connected");
        label.textContent = "Disconnected";
    }
}

// ===== Video / Camera =====
$("#btnStartCamera").addEventListener("click", async () => {
    const btn = $("#btnStartCamera");

    if (state.isStreaming) {
        stopStreaming();
        btn.textContent = "Start Camera";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "environment" },
        });
        const video = $("#videoElement");
        video.srcObject = stream;
        state.videoSource = "camera";
        btn.textContent = "Stop Camera";
        startStreaming();
    } catch (err) {
        showToast("Camera access denied: " + err.message);
    }
});

$("#btnUploadVideo").addEventListener("click", () => {
    $("#videoFileInput").click();
});

$("#videoFileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const video = $("#videoElement");
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    video.loop = true;
    video.play();
    state.videoSource = "file";
    $("#btnStartCamera").textContent = "Start Camera";
    startStreaming();
});

function startStreaming() {
    state.isStreaming = true;
    connectLiveWS();

    const video = $("#videoElement");
    const canvas = $("#captureCanvas");
    const ctx = canvas.getContext("2d");

    state.frameInterval = setInterval(() => {
        if (video.readyState < 2) return;
        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

        canvas.width = 320;
        canvas.height = 240;
        ctx.drawImage(video, 0, 0, 320, 240);

        const b64 = canvas.toDataURL("image/jpeg", 0.7);

        state.ws.send(JSON.stringify({
            type: "frame",
            data: b64,
        }));

        // Animate pipeline
        animatePipeline();
    }, 500); // 2 FPS for demo
}

function stopStreaming() {
    state.isStreaming = false;

    if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
    }

    const video = $("#videoElement");
    if (video.srcObject) {
        video.srcObject.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
    }
    video.src = "";
}

// ===== Prediction Handling =====
function handlePrediction(data) {
    const pred = data.prediction;
    state.currentPrediction = pred;

    // Update overlay badge (if visible)
    const badge = $("#overlayActivity");
    const conf = $("#overlayConfidence");
    if (badge) {
        badge.textContent = pred.activity;
        badge.style.color = COLORS[pred.activity] || COLORS.idle;
    }
    if (conf) conf.textContent = `${(pred.confidence * 100).toFixed(0)}%`;

    // Update scores
    updateScores(pred.all_scores);

    // Update distribution
    if (pred.pose_detected && pred.activity !== "no_person") {
        state.distributionData[pred.activity] = (state.distributionData[pred.activity] || 0) + 1;
        drawDistributionChart();
    }

    // Add to timeline
    state.timelineData.push({
        timestamp: data.timestamp,
        activity: pred.activity,
        confidence: pred.confidence,
    });
    if (state.timelineData.length > 500) {
        state.timelineData = state.timelineData.slice(-500);
    }
    drawTimelineChart();
}

function updateScores(scores) {
    const container = $("#scoresContainer");
    if (!scores || Object.keys(scores).length === 0) return;

    container.innerHTML = Object.entries(scores)
        .sort(([, a], [, b]) => b - a)
        .map(([label, score]) => `
            <div class="score-bar">
                <span class="score-bar__label">${label}</span>
                <div class="score-bar__track">
                    <div class="score-bar__fill" style="width:${score * 100}%; background:${COLORS[label] || COLORS.idle}"></div>
                </div>
                <span class="score-bar__value">${(score * 100).toFixed(1)}%</span>
            </div>
        `).join("");
}

// ===== KPI Updates =====
function updateKPIs(metrics) {
    $("#kpiActiveTime").textContent = `${metrics.active_time_pct}%`;
    $("#kpiOpsHour").textContent = metrics.operations_per_hour;
    $("#kpiAvgDuration").textContent = `${metrics.avg_operation_duration}s`;
    $("#kpiConfidence").textContent = `${(metrics.avg_confidence * 100).toFixed(0)}%`;
    $("#kpiLatency").textContent = `${metrics.avg_latency_ms}ms`;

    const current = metrics.current_activity || "--";
    const el = $("#kpiCurrentActivity");
    el.textContent = current;
    el.style.color = COLORS[current] || COLORS.idle;

    // Activity breakdown in trend
    const breakdown = metrics.activity_breakdown || {};
    const parts = Object.entries(breakdown)
        .map(([k, v]) => `${k}: ${v.count}`)
        .join(" | ");
    $("#kpiActiveTrend").textContent = parts || "No data yet";
}

// ===== Charts (Canvas-based, no dependencies) =====
function drawDistributionChart() {
    const canvas = $("#distributionChart");
    const ctx = canvas.getContext("2d");
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const entries = Object.entries(state.distributionData);
    if (entries.length === 0) return;

    const total = entries.reduce((s, [, v]) => s + v, 0);
    const barHeight = Math.min(30, (canvas.height - 20) / entries.length - 8);
    const maxVal = Math.max(...entries.map(([, v]) => v));
    const labelWidth = 75;
    const barAreaWidth = canvas.width - labelWidth - 60;

    entries.sort(([, a], [, b]) => b - a);

    entries.forEach(([label, count], i) => {
        const y = 10 + i * (barHeight + 8);
        const barW = (count / maxVal) * barAreaWidth;
        const pct = ((count / total) * 100).toFixed(1);

        // Label
        ctx.fillStyle = "#9aa0a6";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(label, labelWidth - 8, y + barHeight / 2 + 4);

        // Bar
        ctx.fillStyle = COLORS[label] || COLORS.idle;
        ctx.beginPath();
        ctx.roundRect(labelWidth, y, barW, barHeight, 4);
        ctx.fill();

        // Value
        ctx.fillStyle = "#e8eaed";
        ctx.textAlign = "left";
        ctx.fillText(`${pct}%`, labelWidth + barW + 6, y + barHeight / 2 + 4);
    });
}

function drawTimelineChart() {
    const canvas = $("#timelineChart");
    const ctx = canvas.getContext("2d");
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const data = state.timelineData;
    if (data.length < 2) return;

    const activities = [...new Set(data.map((d) => d.activity))];
    const activityIndex = {};
    activities.forEach((a, i) => { activityIndex[a] = i; });

    const segmentWidth = canvas.width / data.length;
    const barHeight = canvas.height - 30;

    // Draw segments
    data.forEach((d, i) => {
        ctx.fillStyle = COLORS[d.activity] || COLORS.idle;
        ctx.globalAlpha = Math.max(d.confidence, 0.3);
        ctx.fillRect(i * segmentWidth, 5, segmentWidth + 0.5, barHeight);
    });

    ctx.globalAlpha = 1;

    // Legend
    const legendY = canvas.height - 12;
    let legendX = 10;
    activities.forEach((a) => {
        ctx.fillStyle = COLORS[a] || COLORS.idle;
        ctx.fillRect(legendX, legendY, 10, 10);
        ctx.fillStyle = "#9aa0a6";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(a, legendX + 14, legendY + 9);
        legendX += ctx.measureText(a).width + 28;
    });
}

// ===== Pipeline Animation =====
let pipelineStep = 0;

function animatePipeline() {
    const steps = [
        $("#pipeStep1"),
        $("#pipeStep2"),
        $("#pipeStep3"),
        $("#pipeStep4"),
        $("#pipeStep5"),
    ];

    steps.forEach((s) => s.classList.remove("active"));

    pipelineStep = (pipelineStep + 1) % steps.length;
    steps[pipelineStep].classList.add("active");

    // Quick cascade
    let i = 0;
    const interval = setInterval(() => {
        steps.forEach((s) => s.classList.remove("active"));
        if (i < steps.length) {
            steps[i].classList.add("active");
            i++;
        } else {
            clearInterval(interval);
        }
    }, 80);
}

// ===== Training Tab =====
let trainingSource = "current"; // "file" | "current"

// Upload source tabs
$$(".upload-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        $$(".upload-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        trainingSource = tab.dataset.source;
        $("#uploadSourceFile").style.display = trainingSource === "file" ? "block" : "none";
        $("#uploadSourceCurrent").style.display = trainingSource === "current" ? "block" : "none";
    });
});

// Populate training demo video dropdown
async function loadTrainingDemoVideos() {
    try {
        const res = await fetch(`${API_BASE}/api/demo/videos`);
        const data = await res.json();
        const select = $("#trainingDemoSelect");
        select.querySelectorAll('option:not([value=""])').forEach((o) => o.remove());
        data.videos.forEach((v) => {
            const opt = document.createElement("option");
            opt.value = v.url;
            opt.textContent = v.name.replace(/\.mp4$/, "").replace(/_/g, " ");
            select.appendChild(opt);
        });
    } catch (err) {
        console.log("Could not load demo videos for training tab");
    }
}

// Demo video selection in training tab
$("#trainingDemoSelect").addEventListener("change", async (e) => {
    const url = e.target.value;
    const statusEl = $("#currentVideoStatus");
    if (!url) {
        statusEl.textContent = "No video selected";
        statusEl.classList.remove("has-video");
        return;
    }

    // Save labels for previous video before switching
    await saveCurrentLabels();

    const videoFile = url.split("/").pop();
    const name = videoFile.replace(/\.mp4$/, "").replace(/_/g, " ");
    statusEl.textContent = `Selected: ${name}`;
    statusEl.classList.add("has-video");
    state.currentVideoName = videoFile;
    showPosePreview(url);

    // Load saved labels for this video in current project
    activityLabels.length = 0;
    renderLabelTimeline();
    await loadLabelsForCurrentVideo();
});

const uploadZone = $("#uploadZone");
const trainingInput = $("#trainingFileInput");

uploadZone.addEventListener("click", () => trainingInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
});

// Switch pose model on dropdown change
$("#poseModelMode").addEventListener("change", async () => {
    const select = $("#poseModelMode");
    const mode = select.value;
    const status = $("#poseModelSwitchStatus");
    select.disabled = true;
    status.textContent = "Switching...";
    status.style.color = "var(--text-secondary)";

    try {
        const formData = new FormData();
        formData.append("mode", mode);
        const res = await fetch(`${API_BASE}/api/pose-model/switch`, { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            showToast("Switch failed: " + data.error);
            status.textContent = data.error;
            status.style.color = "var(--danger)";
        } else {
            showToast(`Switched to ${data.description}`);
            status.textContent = "";
            loadTrainingStatus();
        }
    } catch (err) {
        showToast("Switch failed: " + err.message);
        status.textContent = "Failed";
        status.style.color = "var(--danger)";
    }

    select.disabled = false;
});

async function loadTrainingStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/train/status`);
        const data = await res.json();

        $("#modelType").textContent = data.pose_model || data.model_type;

        // Sync dropdown with current server mode
        if (data.pose_mode) {
            $("#poseModelMode").value = data.pose_mode;
        }

        // Active version
        const versionEl = $("#modelVersion");
        if (versionEl) {
            const ensLabel = data.ensemble ? ` (Ensemble: ${data.ensemble_count} models)` : "";
            versionEl.textContent = (data.active_version || (data.is_trained ? "loaded" : "untrained")) + ensLabel;

            // Sync ensemble checkbox
            const ensToggle = document.getElementById("ensembleToggle");
            if (ensToggle) ensToggle.checked = !!data.ensemble;
        }

        // Classes — shown in model status AND synced to labeling checkboxes
        const classes = data.classes || [];
        $("#modelClasses").innerHTML = classes
            .map((c) => `<span class="class-tag" style="border-color:${COLORS[c] || '#555'}">${c}</span>`)
            .join("");
        syncActivityCheckboxes(classes);

        // Version history
        const versions = data.versions || [];
        const historyEl = $("#versionHistory");
        if (historyEl) {
            if (versions.length === 0) {
                historyEl.innerHTML = '<div style="font-size:11px; color:var(--text-muted); padding:4px 0">No trained versions yet. Upload samples and click Train.</div>';
            } else {
                historyEl.innerHTML = versions.slice().reverse().map(v => {
                    const isActive = v.version_id === data.active_version;
                    const hasLabels = !!v.raw_labels;
                    return `
                        <div class="version-entry ${isActive ? 'version-entry--active' : ''}" data-version="${v.version_id}">
                            <span class="version-entry__id">${v.version_id}</span>
                            <span class="version-entry__meta">${v.samples} samples, ${(v.classes || []).join(", ") || "default"}</span>
                            ${hasLabels ? `<button class="btn-load-labels" data-version="${v.version_id}" style="background:none; border:none; color:var(--accent); cursor:pointer; font-size:9px; padding:0 3px" title="Load labels into timeline">Labels</button>` : ''}
                            ${isActive
                                ? '<span class="version-entry__badge version-entry__badge--active">Active</span>'
                                : '<span class="version-entry__badge version-entry__badge--load">Switch</span>'}
                            ${!isActive
                                ? `<button class="btn-delete-version" data-version="${v.version_id}" style="background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:10px; padding:0 2px" title="Delete version">&#10005;</button>`
                                : ''}
                        </div>
                    `;
                }).join("");
            }
        }

        // Update footer
        const footerEngine = $("#footerEngine");
        if (footerEngine) {
            const reidInfo = data.reid && data.reid.enabled
                ? `ReID: OSNet-AIN (${data.reid.gallery_size} enrolled)`
                : "ReID: off";
            footerEngine.textContent = `Pose: RTMPose [${data.pose_mode || "balanced"}] | ${reidInfo} | Classifier: RandomForest | Apache 2.0`;
        }

        // Load dataset info
        loadDatasetInfo();
    } catch (err) {
        console.log("Could not load training status");
    }
}

function syncActivityCheckboxes(classes) {
    const container = document.getElementById("labelCheckboxes");
    if (!container) return;
    // Get currently checked values to preserve state
    const checked = new Set([...container.querySelectorAll(".label-cb:checked")].map(cb => cb.value));
    // Get existing checkbox values
    const existing = new Set([...container.querySelectorAll(".label-cb")].map(cb => cb.value));
    // Add missing classes from model
    (classes || []).forEach(cls => {
        if (!existing.has(cls)) {
            const lbl = document.createElement("label");
            lbl.className = "label-cb-item";
            lbl.innerHTML = `<input type="checkbox" value="${cls}" class="label-cb"${checked.has(cls) ? " checked" : ""}> ${cls}`;
            container.appendChild(lbl);
        }
    });
    // If empty (untrained model), show hint
    if (container.children.length === 0) {
        container.innerHTML = '<span style="font-size:11px; color:var(--text-muted)">No activities yet. Type a custom activity below to start.</span>';
    }
}


// Load dataset breakdown
async function loadDatasetInfo() {
    try {
        const res = await fetch(`${API_BASE}/api/train/dataset`);
        const data = await res.json();

        const countEl = document.getElementById("datasetSampleCount");
        if (countEl) countEl.textContent = data.sample_count || 0;

        // Activities breakdown
        const actEl = document.getElementById("datasetActivities");
        if (actEl) {
            const activities = data.activities || [];
            if (activities.length === 0) {
                actEl.innerHTML = '<span style="color:var(--text-muted)">No labeled data yet.</span>';
            } else {
                actEl.innerHTML = activities.map(a =>
                    `<div style="display:flex; justify-content:space-between; padding:1px 0">
                        <span style="color:${COLORS[a.name] || 'var(--text-primary)'}">${a.name}</span>
                        <span style="color:var(--text-muted)">${a.count} samples</span>
                    </div>`
                ).join("");
            }
        }

        // Sources breakdown
        const srcEl = document.getElementById("datasetSources");
        if (srcEl) {
            const sources = data.sources || [];
            if (sources.length === 0) {
                srcEl.innerHTML = "";
            } else {
                srcEl.innerHTML = '<div style="font-size:10px; color:var(--text-muted); margin-bottom:2px">Sources:</div>' +
                    sources.map(s =>
                        `<div style="display:flex; align-items:center; gap:4px; padding:2px 0; font-size:10px">
                            <span style="flex:1; color:var(--text-secondary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap" title="${s.name}">${s.name}</span>
                            <span style="color:var(--text-muted)">${s.count}</span>
                            <button class="btn-remove-source" data-source="${s.name}" style="background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:10px; padding:0 2px" title="Remove samples from ${s.name}">&#10005;</button>
                        </div>`
                    ).join("");
            }
        }
    } catch (err) {
        console.log("Could not load dataset info");
    }
}

// Remove source from dataset
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-remove-source");
    if (!btn) return;
    const source = btn.dataset.source;
    btn.textContent = "...";
    try {
        await fetch(`${API_BASE}/api/train/dataset/source/${encodeURIComponent(source)}`, { method: "DELETE" });
        showToast(`Removed samples from ${source}`);
        loadDatasetInfo();
        loadTrainingStatus();
    } catch (err) {
        showToast("Remove failed: " + err.message);
    }
});

// Ensemble toggle
document.addEventListener("change", async (e) => {
    if (e.target.id !== "ensembleToggle") return;
    const enabled = e.target.checked;
    const formData = new FormData();
    formData.append("enabled", enabled);
    try {
        const res = await fetch(`${API_BASE}/api/train/ensemble`, { method: "POST", body: formData });
        const data = await res.json();
        if (data.status === "enabled") {
            showToast(`Ensemble ON: ${data.models_loaded} models loaded`);
        } else {
            showToast("Ensemble OFF: using single active model");
        }
        loadTrainingStatus();
    } catch (err) {
        showToast("Ensemble toggle failed: " + err.message);
        e.target.checked = !enabled;
    }
});

// Re-train on full dataset
document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnRetrain") return;
    const btn = e.target;
    btn.textContent = "Training...";
    btn.disabled = true;
    try {
        const res = await fetch(`${API_BASE}/api/train/retrain`, { method: "POST" });
        const data = await res.json();
        if (data.status === "success") {
            showToast(`Re-trained: ${data.samples} samples, v${data.version_id}`);
            loadTrainingStatus();
            loadDatasetInfo();
        } else {
            showToast(data.message || "Training failed");
        }
    } catch (err) {
        showToast("Training failed: " + err.message);
    }
    btn.textContent = "Rebuild";
    btn.disabled = false;
});

// Load version labels into Activity Labels timeline
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-load-labels");
    if (!btn) return;
    e.stopPropagation();
    const versionId = btn.dataset.version;
    btn.textContent = "...";

    try {
        const res = await fetch(`${API_BASE}/api/train/versions/${encodeURIComponent(versionId)}/labels`);
        const data = await res.json();

        if (!data.has_labels || !data.labels || data.labels.length === 0) {
            showToast("No label data saved for this version");
            btn.textContent = "Labels";
            return;
        }

        // Load the video if available
        const videoUrl = data.video_url;
        if (videoUrl) {
            // Switch to Demo tab and load the video
            const demoSelect = document.getElementById("trainingDemoSelect");
            if (demoSelect) {
                // Try to find the video in the demo dropdown
                const videoFile = data.video_file || videoUrl.split("/").pop();
                const matchingOption = [...demoSelect.options].find(o => o.value.includes(videoFile));
                if (matchingOption) {
                    document.querySelector('[data-source="current"]').click();
                    demoSelect.value = matchingOption.value;
                    demoSelect.dispatchEvent(new Event("change", { bubbles: true }));
                }
            }
        }

        // Load labels into the Activity Labels timeline
        activityLabels.length = 0;
        const slider = document.getElementById("frameSlider");
        const video = document.getElementById("previewVideo");
        const maxFrame = slider ? parseInt(slider.max) || 100 : 100;
        const duration = video ? video.duration || 1 : 1;

        data.labels.forEach(l => {
            const startFrame = Math.round((l.start_time / duration) * maxFrame);
            const endFrame = Math.round((l.end_time / duration) * maxFrame);
            const activities = l.activities || [l.activity || "unknown"];

            activityLabels.push({
                start: startFrame,
                end: endFrame,
                activities: activities,
                activity: activities.join("+"),
                person: l.person || "all",
                id: Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
            });

            // Add any new activities as checkboxes
            activities.forEach(act => {
                const container = document.getElementById("labelCheckboxes");
                if (container && ![...container.querySelectorAll(".label-cb")].some(cb => cb.value === act)) {
                    const hint = container.querySelector("span");
                    if (hint) hint.remove();
                    const lbl = document.createElement("label");
                    lbl.className = "label-cb-item";
                    lbl.innerHTML = `<input type="checkbox" value="${act}" class="label-cb"> ${act}`;
                    container.appendChild(lbl);
                }
            });
        });

        renderLabelTimeline();
        showToast(`Loaded ${data.labels.length} labels from version ${versionId}`);
    } catch (err) {
        showToast("Failed to load labels: " + err.message);
    }

    btn.textContent = "Labels";
});

// Delete model version
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-delete-version");
    if (!btn) return;
    e.stopPropagation();
    const versionId = btn.dataset.version;
    btn.textContent = "...";
    try {
        const res = await fetch(`${API_BASE}/api/train/versions/${encodeURIComponent(versionId)}`, { method: "DELETE" });
        const data = await res.json();
        if (data.status === "deleted") {
            showToast(`Deleted version ${versionId}`);
            loadTrainingStatus();
        } else {
            showToast(data.message || "Delete failed");
        }
    } catch (err) {
        showToast("Delete failed: " + err.message);
    }
});

// Switch model version by clicking on version entry
document.addEventListener("click", async (e) => {
    if (e.target.closest(".btn-delete-version")) return; // handled above
    const entry = e.target.closest(".version-entry");
    if (!entry || entry.classList.contains("version-entry--active")) return;

    const versionId = entry.dataset.version;
    const badge = entry.querySelector(".version-entry__badge");
    if (badge) badge.textContent = "...";

    try {
        const formData = new FormData();
        formData.append("version_id", versionId);
        const res = await fetch(`${API_BASE}/api/train/switch-version`, { method: "POST", body: formData });
        const data = await res.json();
        if (data.status === "switched") {
            showToast(`Switched to version ${versionId}`);
            loadTrainingStatus();
        } else {
            showToast(data.message || "Switch failed");
        }
    } catch (err) {
        showToast("Switch failed: " + err.message);
    }
});

// ===== Events Tab =====
async function loadEvents() {
    try {
        const res = await fetch(`${API_BASE}/api/events?limit=200`);
        const data = await res.json();
        renderEventsTable(data.events);
    } catch (err) {
        console.log("Could not load events");
    }
}

function renderEventsTable(events) {
    const tbody = $("#eventsTableBody");
    tbody.innerHTML = events.map((e) => `
        <tr class="slide-in">
            <td><code>${e.id}</code></td>
            <td>${new Date(e.timestamp * 1000).toLocaleTimeString()}</td>
            <td><span class="activity-tag activity-tag--${e.activity}">${e.activity}</span></td>
            <td>
                <div class="conf-bar">
                    <div class="conf-bar__track">
                        <div class="conf-bar__fill" style="width:${e.confidence * 100}%"></div>
                    </div>
                    <span>${(e.confidence * 100).toFixed(0)}%</span>
                </div>
            </td>
            <td>${e.duration ? e.duration.toFixed(1) + "s" : "--"}</td>
            <td>${e.worker_id}</td>
            <td>${e.zone}</td>
        </tr>
    `).join("");
}

// ===== Export =====
$("#btnExportCSV").addEventListener("click", async () => {
    try {
        const res = await fetch(`${API_BASE}/api/events?limit=10000`);
        const data = await res.json();
        const events = data.events;

        const header = "id,timestamp,activity,confidence,duration,worker_id,zone\n";
        const rows = events.map((e) =>
            `${e.id},${new Date(e.timestamp * 1000).toISOString()},${e.activity},${e.confidence},${e.duration || 0},${e.worker_id},${e.zone}`
        ).join("\n");

        downloadFile(header + rows, "warehouse_events.csv", "text/csv");
    } catch (err) {
        showToast("Export failed: " + err.message);
    }
});

$("#btnExportJSON").addEventListener("click", async () => {
    try {
        const res = await fetch(`${API_BASE}/api/events?limit=10000`);
        const data = await res.json();
        downloadFile(JSON.stringify(data.events, null, 2), "warehouse_events.json", "application/json");
    } catch (err) {
        showToast("Export failed: " + err.message);
    }
});

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ===== Timeline Range =====
$("#timelineRange").addEventListener("change", async (e) => {
    const minutes = parseInt(e.target.value);
    try {
        const res = await fetch(`${API_BASE}/api/metrics/timeline?minutes=${minutes}`);
        const data = await res.json();
        state.timelineData = data.timeline.map((t) => ({
            timestamp: t.timestamp,
            activity: t.activity,
            confidence: t.confidence,
        }));
        drawTimelineChart();
    } catch (err) {
        console.log("Could not load timeline");
    }
});

// ===== Demo Videos =====
async function loadDemoVideos() {
    try {
        const res = await fetch(`${API_BASE}/api/demo/videos`);
        const data = await res.json();
        const select = $("#demoVideoSelect");
        select.querySelectorAll('option:not([value=""])').forEach((o) => o.remove());
        if (data.videos.length > 0) {
            select.style.display = "inline-block";
            data.videos.forEach((v) => {
                const opt = document.createElement("option");
                opt.value = v.url;
                opt.textContent = v.name.replace(/\.mp4$/, "").replace(/_/g, " ");
                select.appendChild(opt);
            });
        }
    } catch (err) {
        console.log("No demo videos available");
    }
}

$("#demoVideoSelect").addEventListener("change", (e) => {
    const url = e.target.value;
    if (!url) return;

    stopStreaming();
    const video = $("#videoElement");
    video.srcObject = null;
    video.muted = true;
    video.src = url;
    video.loop = true;
    video.play();
    state.videoSource = "file";
    $("#btnStartCamera").textContent = "Start Camera";
    startStreaming();
});

// ===== Workers Gallery (ReID) =====

async function loadGallery() {
    try {
        // Use collection API if a collection is selected, otherwise global
        const collId = state.activeCollectionId;
        const galleryUrl = collId
            ? `${API_BASE}/api/workers/collections/${collId}/gallery`
            : `${API_BASE}/api/reid/gallery`;
        const res = await fetch(galleryUrl);
        const data = await res.json();
        const workers = data.gallery || [];
        const container = $("#reidWorkers");
        const empty = $("#reidEmpty");
        const badge = $("#reidBadge");

        if (badge) {
            if (data.enabled) {
                badge.textContent = `ReID on (${workers.length})`;
                badge.className = "reid-badge reid-badge--on";
            } else {
                badge.textContent = "ReID off";
                badge.className = "reid-badge";
            }
        }

        if (!workers.length) {
            if (empty) empty.style.display = "block";
            if (container) container.innerHTML = "";
            return;
        }

        if (empty) empty.style.display = "none";

        const avatarColors = ["#4f8cff", "#a78bfa", "#f59e0b", "#34d399", "#f87171", "#38bdf8"];
        container.innerHTML = workers.map((w, i) => {
            const initials = w.name.split(/[\s_-]+/).map(p => p[0]).join("").toUpperCase().slice(0, 2);
            const color = avatarColors[i % avatarColors.length];
            const embCount = w.embeddings_count || 1;
            return `
                <div class="reid-worker" data-name="${w.name}">
                    <div class="reid-worker__avatar" style="background:${color}">${initials}</div>
                    <div class="reid-worker__info">
                        <div class="reid-worker__name">${w.name}</div>
                        <div class="reid-worker__meta">${embCount} view${embCount !== 1 ? "s" : ""} &middot; ${w.updated_at || w.enrolled_at || ""}</div>
                        <div class="reid-worker__thumbs" id="thumbs-${i}"></div>
                    </div>
                    <button class="reid-worker__delete" data-name="${w.name}" title="Remove ${w.name}">&#10005;</button>
                </div>
            `;
        }).join("");

        // Load thumbnails for each worker
        workers.forEach(async (w, i) => {
            try {
                const thumbUrl = state.activeCollectionId
                    ? `${API_BASE}/api/workers/collections/${state.activeCollectionId}/gallery/${encodeURIComponent(w.name)}/thumbnails`
                    : `${API_BASE}/api/reid/gallery/${encodeURIComponent(w.name)}/thumbnails`;
                const res = await fetch(thumbUrl);
                const data = await res.json();
                const el = document.getElementById(`thumbs-${i}`);
                if (el && data.thumbnails && data.thumbnails.length > 0) {
                    el.innerHTML = data.thumbnails.map(url =>
                        `<img src="${url}" style="height:40px; border-radius:3px; border:1px solid var(--border); object-fit:cover">`
                    ).join("");
                }
            } catch (_) {}
        });
    } catch (err) {
        console.log("Could not load ReID gallery");
    }
}

// Delete worker from gallery
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".reid-worker__delete");
    if (!btn) return;
    const name = btn.dataset.name;
    if (!name) return;

    btn.textContent = "...";
    try {
        await fetch(`${API_BASE}/api/reid/gallery/${encodeURIComponent(name)}`, { method: "DELETE" });
        showToast(`Removed ${name} from gallery`);
        loadGallery();
        loadTrainingStatus();
    } catch (err) {
        showToast("Remove failed: " + err.message);
    }
});

// Enroll from photo upload
$("#btnEnrollPhoto").addEventListener("click", () => {
    $("#reidPhotoInput").click();
});

$("#reidPhotoInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const name = prompt("Enter worker name:");
    if (!name || !name.trim()) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name.trim());

    try {
        const res = await fetch(`${API_BASE}/api/reid/enroll`, { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) {
            showToast("Enroll failed: " + data.error);
        } else {
            showToast(`${data.status === "enrolled" ? "Enrolled" : "Updated"} ${data.name} (gallery: ${data.gallery_size})`);
            loadGallery();
            loadTrainingStatus();
        }
    } catch (err) {
        showToast("Enroll failed: " + err.message);
    }

    e.target.value = "";
});

// Enroll person from Pose Preview frame
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-enroll-person");
    if (!btn) return;
    const personIdx = parseInt(btn.dataset.personIndex);
    if (isNaN(personIdx)) return;

    const card = btn.closest("[data-person-card]");
    if (!card) return;

    let enrollRow = card.querySelector(".enroll-inline");
    if (enrollRow) { enrollRow.remove(); return; }

    // Build gallery options for "Link to existing"
    let galleryOptions = '<option value="">-- Link to existing --</option>';
    try {
        const res = await fetch(`${API_BASE}/api/reid/gallery`);
        const data = await res.json();
        (data.gallery || []).forEach(w => {
            galleryOptions += `<option value="${w.name}">${w.name}</option>`;
        });
    } catch (_) {}

    enrollRow = document.createElement("div");
    enrollRow.className = "enroll-inline";
    enrollRow.innerHTML = `
        <select class="enroll-existing-select" style="padding:4px 6px; font-size:11px; background:var(--bg-primary); border:1px solid var(--border); border-radius:4px; color:var(--text-primary)">
            ${galleryOptions}
        </select>
        <span style="font-size:10px; color:var(--text-muted)">or</span>
        <input type="text" placeholder="New name..." class="enroll-name-input">
        <button class="btn btn--primary btn-enroll-confirm" data-person-index="${personIdx}">Save</button>
        <button class="btn btn--secondary btn-enroll-cancel">&#10005;</button>
    `;
    card.appendChild(enrollRow);

    // When selecting existing, clear new name input and vice versa
    const select = enrollRow.querySelector(".enroll-existing-select");
    const input = enrollRow.querySelector(".enroll-name-input");
    select.addEventListener("change", () => { if (select.value) input.value = ""; });
    input.addEventListener("input", () => { if (input.value) select.value = ""; });
    input.focus();
});

// Confirm enrollment (handles both new and link-to-existing)
document.addEventListener("click", async (e) => {
    if (!e.target.classList.contains("btn-enroll-confirm")) return;
    const personIdx = parseInt(e.target.dataset.personIndex);
    const row = e.target.closest(".enroll-inline");
    const select = row.querySelector(".enroll-existing-select");
    const input = row.querySelector(".enroll-name-input");

    // Get name from dropdown or text input
    const name = (select && select.value) || (input && input.value.trim());
    if (!name) { input.focus(); showToast("Enter a name or select existing"); return; }

    const isExisting = select && select.value === name;
    e.target.textContent = "...";
    e.target.disabled = true;

    try {
        const video = document.getElementById("previewVideo");
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        tempCanvas.getContext("2d").drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        const blob = await new Promise(r => tempCanvas.toBlob(r, "image/jpeg", 0.9));

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");
        formData.append("name", name);
        formData.append("person_index", personIdx);

        const res = await fetch(`${API_BASE}/api/reid/enroll-from-frame`, { method: "POST", body: formData });
        const data = await res.json();

        if (data.error) {
            showToast("Enroll failed: " + data.error);
        } else if (isExisting) {
            showToast(`Linked to ${data.name} — now ${data.embeddings_count} views stored`);
        } else {
            showToast(`Enrolled ${data.name} (${data.embeddings_count} view)`);
        }
        loadGallery();
        loadTrainingStatus();
    } catch (err) {
        showToast("Enroll failed: " + err.message);
    }

    row.remove();
});

// Cancel enrollment
document.addEventListener("click", (e) => {
    if (e.target.classList.contains("btn-enroll-cancel")) {
        e.target.closest(".enroll-inline").remove();
    }
});

// Enter key to confirm enrollment
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.classList.contains("enroll-name-input")) {
        e.target.closest(".enroll-inline").querySelector(".btn-enroll-confirm").click();
    }
});


// Fix wrong identity — shows correction dropdown
document.addEventListener("click", async (e) => {
    const btn = e.target.closest(".btn-fix-identity");
    if (!btn) return;
    const personIdx = parseInt(btn.dataset.personIndex);
    const wrongName = btn.dataset.wrongName;
    if (isNaN(personIdx)) return;

    const card = btn.closest("[data-person-card]");
    if (!card) return;

    let fixRow = card.querySelector(".enroll-inline");
    if (fixRow) { fixRow.remove(); return; }

    // Build gallery options (exclude the wrong match)
    let galleryOptions = `<option value="">-- This is actually... --</option>`;
    try {
        const res = await fetch(`${API_BASE}/api/reid/gallery`);
        const data = await res.json();
        (data.gallery || []).forEach(w => {
            if (w.name !== wrongName) {
                galleryOptions += `<option value="${w.name}">${w.name}</option>`;
            }
        });
    } catch (_) {}

    fixRow = document.createElement("div");
    fixRow.className = "enroll-inline";
    fixRow.innerHTML = `
        <span style="font-size:10px; color:var(--warning)">Wrong ID?</span>
        <select class="enroll-existing-select" style="padding:4px 6px; font-size:11px; background:var(--bg-primary); border:1px solid var(--border); border-radius:4px; color:var(--text-primary)">
            ${galleryOptions}
        </select>
        <span style="font-size:10px; color:var(--text-muted)">or</span>
        <input type="text" placeholder="New name..." class="enroll-name-input">
        <button class="btn btn--primary btn-enroll-confirm" data-person-index="${personIdx}">Save</button>
        <button class="btn btn--secondary btn-enroll-cancel">&#10005;</button>
    `;
    card.appendChild(fixRow);
});


// ===== Pose Preview (Training Tab) =====
// COCO 17-point skeleton connections (RTMPose output)
const POSE_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],           // face
    [5, 6],                                     // shoulders
    [5, 7], [7, 9], [6, 8], [8, 10],          // arms
    [5, 11], [6, 12], [11, 12],               // torso
    [11, 13], [13, 15], [12, 14], [14, 16],   // legs
];

// First 7: per-frame pose features, 0..1 (shown as %). The rest: motion
// features from services/motion.py, in torso lengths / seconds (shown raw).
const POSE_FEATURE_COUNT = 7;
const formatFeature = (i, val) => i < POSE_FEATURE_COUNT ? `${(val * 100).toFixed(0)}%` : val.toFixed(2);
const FEATURE_NAMES = [
    "Torso Angle", "Arm Spread", "Vert. Move",
    "Hand Height", "Stride Len.", "Body Speed", "Symmetry",
    "Velocity", "Wrist Motion", "Gait", "Hands Ahead",
    "Wrist-Head", "Head Down", "Wrist Rise", "Nearest"
];

let poseAnalyzeTimeout = null;

function showPosePreview(videoSrc) {
    const video = $("#previewVideo");
    const slider = $("#frameSlider");

    // Remove old listeners to avoid stacking
    const newVideo = video.cloneNode(false);
    video.parentNode.replaceChild(newVideo, video);

    newVideo.src = videoSrc;
    newVideo.load();

    newVideo.addEventListener("loadedmetadata", function onMeta() {
        newVideo.removeEventListener("loadedmetadata", onMeta);
        const duration = newVideo.duration;
        slider.max = Math.floor(duration * 30);
        slider.value = 0;
        $("#previewDuration").textContent = formatTime(duration);
        $("#previewCurrentTime").textContent = "0:00";
        $("#poseFrameInfo").textContent = `Frame 0 / ${slider.max}`;
        newVideo.currentTime = 0;
    });

    newVideo.addEventListener("seeked", () => {
        // Skip auto-analyze during playback — playback handles its own analysis
        if (posePlayActive) return;
        if ($("#poseAutoAnalyze").checked) {
            clearTimeout(poseAnalyzeTimeout);
            poseAnalyzeTimeout = setTimeout(analyzeCurrentFrame, 150);
        }
    });
}

function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
}

// Slider — seek video on any interaction (drag, click, keyboard)
function handleSliderSeek() {
    const slider = $("#frameSlider");
    const video = document.getElementById("previewVideo");
    if (!video || !video.duration) return;
    const t = (slider.value / slider.max) * video.duration;
    video.currentTime = t;
    $("#previewCurrentTime").textContent = formatTime(t);
    $("#poseFrameInfo").textContent = `Frame ${slider.value} / ${slider.max}`;
}

// "input" fires while dragging, "change" fires on release/click
document.addEventListener("input", (e) => { if (e.target.id === "frameSlider") handleSliderSeek(); });
document.addEventListener("change", (e) => { if (e.target.id === "frameSlider") handleSliderSeek(); });

// Fix: clicking on or near the slider track should seek.
// Use mousedown on the slider itself for reliable click-to-position.
(function initSliderClickFix() {
    // Wait for slider to exist (it's created dynamically when video loads)
    const observer = new MutationObserver(() => {
        const slider = document.getElementById("frameSlider");
        if (!slider || slider.__clickFixed) return;
        slider.__clickFixed = true;

        slider.addEventListener("mousedown", (e) => {
            const rect = slider.getBoundingClientRect();
            const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            slider.value = Math.round(pct * Number(slider.max));
            handleSliderSeek();
        });

        // Also make the wrapper div clickable as a bigger hit target
        const wrap = slider.closest(".pose-preview__slider-wrap");
        if (wrap) {
            wrap.addEventListener("click", (e) => {
                if (e.target === slider) return; // already handled by mousedown
                const rect = slider.getBoundingClientRect();
                const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                slider.value = Math.round(pct * Number(slider.max));
                handleSliderSeek();
            });
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();

// Manual analyze button
document.addEventListener("click", (e) => {
    if (e.target.id === "btnAnalyzeFrame") {
        analyzeCurrentFrame();
    }
});

// Play/Pause: step through frames, analyze each, wait for completion + delay
let posePlayActive = false;

document.addEventListener("click", (e) => {
    if (e.target.id !== "btnPosePlay") return;
    if (posePlayActive) {
        stopPosePlay();
    } else {
        startPosePlay();
    }
});

function startPosePlay() {
    posePlayActive = true;
    const btn = document.getElementById("btnPosePlay");
    btn.innerHTML = "&#9724; Stop";
    posePlayStep();
}

function stopPosePlay() {
    posePlayActive = false;
    const btn = document.getElementById("btnPosePlay");
    btn.innerHTML = "&#9654; Play";
}

async function posePlayStep() {
    if (!posePlayActive) return;

    const slider = document.getElementById("frameSlider");
    const video = document.getElementById("previewVideo");
    if (!slider || !video || !video.duration) { stopPosePlay(); return; }

    // Analyze current frame and wait for it to complete
    await analyzeCurrentFrame();

    if (!posePlayActive) return; // stopped during analysis

    // Wait user-configured delay
    const delay = parseInt(document.getElementById("posePlayDelay").value) || 0;
    if (delay > 0) {
        await new Promise((r) => setTimeout(r, delay));
    }

    if (!posePlayActive) return; // stopped during delay

    // Advance to next frame
    const nextVal = parseInt(slider.value) + 1;
    if (nextVal > parseInt(slider.max)) {
        stopPosePlay(); // reached end
        return;
    }
    slider.value = nextVal;
    handleSliderSeek();

    // Wait for video to actually seek to the new position
    await new Promise((resolve) => {
        const onSeeked = () => { video.removeEventListener("seeked", onSeeked); resolve(); };
        video.addEventListener("seeked", onSeeked);
        // Fallback timeout in case seeked doesn't fire
        setTimeout(resolve, 500);
    });

    // Continue to next step
    posePlayStep();
}

let lastAnalyzeTime = 0;

async function analyzeCurrentFrame() {
    const video = document.getElementById("previewVideo");
    if (!video || !video.duration || video.readyState < 2) return;

    const frameStart = performance.now();

    // Capture frame to a temp canvas
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    // Convert to blob and send
    const blob = await new Promise((r) => tempCanvas.toBlob(r, "image/jpeg", 0.9));
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("video_time", String(video.currentTime));

    try {
        const res = await fetch(`${API_BASE}/api/predict/landmarks`, { method: "POST", body: formData });
        const data = await res.json();
        drawAllPersons(data.persons || [], video.videoWidth, video.videoHeight);
        updatePoseBadge(data);
        updatePersonsStats(data.persons || []);
        updateLabelPersonDropdown(data.persons || []);
        if (activityLabels.length > 0) renderLabelTimeline();

        // Update FPS display
        const elapsed = performance.now() - frameStart;
        const fps = 1000 / elapsed;
        const fpsEl = document.getElementById("poseFps");
        if (fpsEl) {
            fpsEl.textContent = `${fps.toFixed(1)} FPS | ${Math.round(elapsed)}ms`;
        }
    } catch (err) {
        console.error("Pose analysis failed:", err);
    }
}

// Per-person colors for multi-person skeleton rendering
const PERSON_COLORS = [
    { line: "79, 140, 255",  dot: "52, 211, 153",  label: "#4f8cff", name: "Person 1" },  // blue/green
    { line: "251, 146, 60",  dot: "251, 191, 36",   label: "#fb923c", name: "Person 2" },  // orange/yellow
    { line: "167, 139, 250", dot: "232, 121, 249",  label: "#a78bfa", name: "Person 3" },  // purple/pink
    { line: "248, 113, 113", dot: "252, 165, 165",  label: "#f87171", name: "Person 4" },  // red
    { line: "52, 211, 153",  dot: "110, 231, 183",  label: "#34d399", name: "Person 5" },  // green
    { line: "56, 189, 248",  dot: "125, 211, 252",  label: "#38bdf8", name: "Person 6" },  // sky
];

function drawAllPersons(persons, videoW, videoH) {
    const canvas = $("#poseCanvas");
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!persons || persons.length === 0) return;

    // Calculate scaling to match object-fit: contain
    const displayW = canvas.width;
    const displayH = canvas.height;
    const videoAspect = videoW / videoH;
    const displayAspect = displayW / displayH;
    let scaleW, scaleH, offsetX, offsetY;
    if (videoAspect > displayAspect) {
        scaleW = displayW; scaleH = displayW / videoAspect;
        offsetX = 0; offsetY = (displayH - scaleH) / 2;
    } else {
        scaleH = displayH; scaleW = displayH * videoAspect;
        offsetX = (displayW - scaleW) / 2; offsetY = 0;
    }
    function toPixel(lm) {
        return { x: offsetX + lm.x * scaleW, y: offsetY + lm.y * scaleH };
    }

    persons.forEach((person, pIdx) => {
        const landmarks = person.landmarks;
        if (!landmarks || landmarks.length === 0) return;
        // Use stable person_id for consistent color (not array index which can change)
        const colorIdx = person.person_id != null ? (person.person_id - 1) : pIdx;
        const colors = PERSON_COLORS[colorIdx % PERSON_COLORS.length];

        // Draw connections
        ctx.lineWidth = 2.5;
        POSE_CONNECTIONS.forEach(([i, j]) => {
            if (i >= landmarks.length || j >= landmarks.length) return;
            const a = landmarks[i], b = landmarks[j];
            if (a.visibility < 0.4 || b.visibility < 0.4) return;
            const pa = toPixel(a), pb = toPixel(b);
            ctx.strokeStyle = `rgba(${colors.line}, ${Math.min(a.visibility, b.visibility)})`;
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.stroke();
        });

        // Draw keypoints
        landmarks.forEach((lm, idx) => {
            if (lm.visibility < 0.4) return;
            const p = toPixel(lm);
            const isKey = [5,6,7,8,9,10,11,12,13,14,15,16].includes(idx);
            const radius = isKey ? 5 : 3;
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = isKey
                ? `rgba(${colors.dot}, ${lm.visibility})`
                : `rgba(${colors.line}, ${lm.visibility * 0.7})`;
            ctx.fill();
            if (isKey) {
                ctx.strokeStyle = `rgba(255, 255, 255, ${lm.visibility * 0.6})`;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        });

        // Draw person label near nose or first visible keypoint
        const labelKp = landmarks.find(lm => lm.visibility > 0.4) || landmarks[0];
        if (labelKp) {
            const lp = toPixel(labelKp);
            const pid = person.person_id != null ? person.person_id : pIdx + 1;
            const identity = person.identity;
            const idLabel = identity && identity.matched ? identity.name : `#${pid}`;
            const acts = person.activities && person.activities.length > 0 ? person.activities.join("+") : person.activity;
            const label = `${idLabel} ${acts}`;
            ctx.font = "bold 12px Inter, sans-serif";
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = "rgba(0,0,0,0.7)";
            ctx.beginPath();
            ctx.roundRect(lp.x - tw / 2 - 4, lp.y - 28, tw + 8, 20, 4);
            ctx.fill();
            ctx.fillStyle = colors.label;
            ctx.textAlign = "center";
            ctx.fillText(label, lp.x, lp.y - 13);
            ctx.textAlign = "left";
        }
    });
}

function updatePersonsStats(persons) {
    const container = $("#poseFeatures");
    if (!persons || persons.length === 0) {
        container.innerHTML = "";
        return;
    }

    container.innerHTML = persons.map((person, pIdx) => {
        const pid = person.person_id != null ? person.person_id : pIdx + 1;
        const colorIdx = person.person_id != null ? (person.person_id - 1) : pIdx;
        const colors = PERSON_COLORS[colorIdx % PERSON_COLORS.length];
        const featureChips = (person.features || []).map((val, i) =>
            `<span style="font-size:10px; color:var(--text-secondary)">${FEATURE_NAMES[i] || "F" + (i+1)}: <strong style="color:${colors.label}">${formatFeature(i, val)}</strong></span>`
        ).join(" &middot; ");

        const identity = person.identity;
        const isMatched = identity && identity.matched;
        const idLabel = isMatched
            ? `<span style="color:var(--success)">${identity.name}</span> <span style="font-size:10px; color:var(--text-muted)">(${(identity.similarity * 100).toFixed(0)}%)</span>`
            : `<span style="color:var(--text-muted)">Unknown</span>`;

        // Matched: show "Fix" button to correct wrong identity
        // Unknown: show "Enroll" button to add to gallery
        const actionBtn = isMatched
            ? `<button class="btn btn--secondary btn-fix-identity" data-person-index="${pIdx}" data-wrong-name="${identity.name}" style="font-size:10px; padding:2px 8px; margin-left:auto; color:var(--warning)">Fix</button>`
            : `<button class="btn btn--secondary btn-enroll-person" data-person-index="${pIdx}" style="font-size:10px; padding:2px 8px; margin-left:auto">Enroll</button>`;

        return `
            <div data-person-card style="flex:1; min-width:280px; padding:10px 14px; background:var(--bg-secondary); border:1px solid var(--border); border-left:3px solid ${colors.label}; border-radius:var(--radius)">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px">
                    <strong style="color:${colors.label}">#${pid}</strong> ${idLabel}
                    ${actionBtn}
                    <span style="font-size:12px; font-weight:700; margin-left:auto">${
                        (person.activities && person.activities.length > 0 ? person.activities : [person.activity])
                            .map(a => `<span style="color:${COLORS[a] || COLORS.idle}; margin-left:4px">${a}</span>`)
                            .join("")
                    }</span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:4px 10px">${featureChips}</div>
            </div>
        `;
    }).join("");
}


function updatePoseBadge(pred) {
    const badge = $("#poseBadge");
    const activity = $("#poseBadgeActivity");
    const conf = $("#poseBadgeConf");

    if (!pred.pose_detected) {
        badge.style.display = "flex";
        activity.textContent = "No person";
        activity.style.color = COLORS.no_person;
        conf.textContent = "";
        return;
    }

    badge.style.display = "flex";
    const count = pred.person_count || 1;
    activity.textContent = count > 1 ? `${count} persons` : pred.activity;
    activity.style.color = count > 1 ? "var(--accent)" : (COLORS[pred.activity] || COLORS.idle);
    conf.textContent = count > 1 ? "" : `${(pred.confidence * 100).toFixed(0)}%`;
}

// Training reads the clip on the server (data/demo/), so an uploaded file is
// sent there first and then selected like any demo video — a local blob URL
// could be previewed and labeled, but never trained on.
async function uploadTrainingVideos(files) {
    if (!files || files.length === 0) return;
    let last = null;
    for (const file of files) {
        showToast(`Uploading ${file.name}...`);
        const form = new FormData();
        form.append("file", file);
        try {
            const res = await fetch(`${API_BASE}/api/demo/upload`, { method: "POST", body: form });
            const data = await res.json();
            if (!res.ok) {
                showToast(`Upload failed: ${data.error || res.status}`);
                continue;
            }
            last = data;
        } catch (err) {
            showToast(`Upload failed: ${err.message}`);
        }
    }
    if (!last) return;

    await Promise.all([loadTrainingDemoVideos(), loadDemoVideos()]);
    document.querySelector('.upload-tab[data-source="current"]').click();
    const select = $("#trainingDemoSelect");
    select.value = last.url;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    showToast(`Uploaded ${files.length > 1 ? files.length + " videos" : last.name} — ready to label`);
}

trainingInput.addEventListener("change", () => {
    uploadTrainingVideos([...trainingInput.files]);
    trainingInput.value = "";  // picking the same file again should upload again
});

uploadZone.addEventListener("drop", (e) => uploadTrainingVideos([...e.dataTransfer.files]));


// ===== Activity Labeling Timeline =====

const activityLabels = [];  // [{start: frame, end: frame, activity: "sorting", person: "all"|"#1", id: uuid}, ...]
let labelStart = null;
let labelEnd = null;

const ACTIVITY_COLORS_TIMELINE = {
    sorting: "#4f8cff", carrying: "#a78bfa", loading: "#f59e0b",
    idle: "#6b7280", walking: "#34d399",
};

function getTimelineColor(activity) {
    return ACTIVITY_COLORS_TIMELINE[activity] || "#60a5fa";
}

// Mark start
document.addEventListener("click", (e) => {
    if (e.target.id === "btnMarkStart") {
        const slider = document.getElementById("frameSlider");
        if (!slider) return;
        labelStart = parseInt(slider.value);
        updateLabelRange();
    }
});

// Mark end
document.addEventListener("click", (e) => {
    if (e.target.id === "btnMarkEnd") {
        const slider = document.getElementById("frameSlider");
        if (!slider) return;
        labelEnd = parseInt(slider.value);
        if (labelStart !== null && labelEnd < labelStart) {
            [labelStart, labelEnd] = [labelEnd, labelStart];
        }
        updateLabelRange();
    }
});

// Keyboard shortcuts: [ = start, ] = end
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
    const slider = document.getElementById("frameSlider");
    if (!slider) return;

    if (e.key === "[") {
        labelStart = parseInt(slider.value);
        updateLabelRange();
    } else if (e.key === "]") {
        labelEnd = parseInt(slider.value);
        if (labelStart !== null && labelEnd < labelStart) {
            [labelStart, labelEnd] = [labelEnd, labelStart];
        }
        updateLabelRange();
    }
});

function updateLabelRange() {
    const el = document.getElementById("labelRange");
    if (!el) return;
    const video = document.getElementById("previewVideo");
    const slider = document.getElementById("frameSlider");
    if (!video || !slider || !video.duration) {
        el.textContent = "--:-- → --:--";
        return;
    }

    const maxFrame = parseInt(slider.max);
    const toTime = (frame) => {
        const t = (frame / maxFrame) * video.duration;
        const m = Math.floor(t / 60);
        const s = Math.floor(t % 60);
        return `${m}:${s.toString().padStart(2, "0")}`;
    };

    const sText = labelStart !== null ? `${toTime(labelStart)} (f${labelStart})` : "--:--";
    const eText = labelEnd !== null ? `${toTime(labelEnd)} (f${labelEnd})` : "--:--";
    el.textContent = `${sText} → ${eText}`;
}

// Add label with auto-split stability check
document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnAddLabel") return;
    if (labelStart === null || labelEnd === null) {
        showToast("Mark start and end frames first (use [ and ] keys or buttons)");
        return;
    }

    const checked = [...document.querySelectorAll(".label-cb:checked")].map(cb => cb.value);
    const customAct = document.getElementById("labelCustomActivity").value.trim();
    if (customAct) checked.push(customAct);

    if (checked.length === 0) {
        showToast("Select at least one activity");
        return;
    }

    const person = document.getElementById("labelPerson").value;
    const video = document.getElementById("previewVideo");
    const slider = document.getElementById("frameSlider");
    const maxFrame = slider ? parseInt(slider.max) || 1 : 1;
    const duration = video ? video.duration || 1 : 1;

    const startTime = (labelStart / maxFrame) * duration;
    const endTime = (labelEnd / maxFrame) * duration;
    const btn = e.target;

    // Analyze stability (only for ranges > 10 frames)
    let splits = null;
    if (labelEnd - labelStart > 10 && video && video.src) {
        btn.textContent = "Analyzing...";
        btn.disabled = true;
        try {
            const res = await fetch(`${API_BASE}/api/train/analyze-label-stability`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    video_url: video.src,
                    start_time: startTime,
                    end_time: endTime,
                }),
            });
            const data = await res.json();
            if (!data.stable && data.splits && data.splits.length > 1) {
                splits = data.splits;
            }
        } catch (err) {
            console.log("Stability check failed, adding as-is:", err);
        }
        btn.textContent = "Add";
        btn.disabled = false;
    }

    // Add custom activity checkbox if new
    if (customAct) {
        const container = document.getElementById("labelCheckboxes");
        if (![...container.querySelectorAll(".label-cb")].some(cb => cb.value === customAct)) {
            const hint = container.querySelector("span");
            if (hint && hint.textContent.includes("No activities")) hint.remove();
            const lbl = document.createElement("label");
            lbl.className = "label-cb-item";
            lbl.innerHTML = `<input type="checkbox" value="${customAct}" class="label-cb"> ${customAct}`;
            container.appendChild(lbl);
        }
        document.getElementById("labelCustomActivity").value = "";
    }

    if (splits && splits.length > 1) {
        // Auto-split: add one label per stable sub-range
        splits.forEach(s => {
            const sf = Math.round((s.start_time / duration) * maxFrame);
            const ef = Math.round((s.end_time / duration) * maxFrame);
            activityLabels.push({
                start: sf,
                end: ef,
                activities: [...checked],
                activity: checked.join("+"),
                person: person,
                id: Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
            });
        });
        showToast(`Auto-split into ${splits.length} stable ranges (person changes detected)`);
    } else {
        // Stable — add as single label
        activityLabels.push({
            start: labelStart,
            end: labelEnd,
            activities: checked,
            activity: checked.join("+"),
            person: person,
            id: Date.now().toString(36),
        });
        showToast(`Added: ${checked.join("+")} for ${person} (frames ${labelStart}-${labelEnd})`);
    }

    document.querySelectorAll(".label-cb").forEach(cb => cb.checked = false);
    renderLabelTimeline();
});

// Delete label
document.addEventListener("click", (e) => {
    const btn = e.target.closest(".label-segment__delete");
    if (!btn) return;
    const id = btn.dataset.labelId;
    const idx = activityLabels.findIndex(l => l.id === id);
    if (idx >= 0) {
        activityLabels.splice(idx, 1);
        renderLabelTimeline();
    }
});

// Clear all
document.addEventListener("click", (e) => {
    if (e.target.id !== "btnClearLabels") return;
    activityLabels.length = 0;
    labelStart = null;
    labelEnd = null;
    updateLabelRange();
    renderLabelTimeline();
});

function renderLabelTimeline() {
    const container = document.getElementById("labelTimeline");
    const statsEl = document.getElementById("labelStats");
    const trainBtn = document.getElementById("btnTrainFromLabels");
    if (!container) return;

    if (activityLabels.length === 0) {
        container.innerHTML = '<div class="activity-labeler__empty">Mark start/end on the slider, select activity, and click Add. Keyboard: [ = start, ] = end</div>';
        if (statsEl) statsEl.textContent = "0 labels";
        if (trainBtn) trainBtn.disabled = true;
        return;
    }

    const slider = document.getElementById("frameSlider");
    const maxFrame = slider ? parseInt(slider.max) : 100;
    const currentFrame = slider ? parseInt(slider.value) : 0;

    // Group labels by person
    const personGroups = {};
    activityLabels.forEach(l => {
        const key = l.person;
        if (!personGroups[key]) personGroups[key] = [];
        personGroups[key].push(l);
    });

    const activities = [...new Set(activityLabels.map(l => l.activity))];

    container.innerHTML = Object.entries(personGroups).map(([person, labels]) => {
        const segments = labels.map(l => {
            const left = (l.start / maxFrame * 100).toFixed(2);
            const width = ((l.end - l.start) / maxFrame * 100).toFixed(2);
            const color = getTimelineColor(l.activity);
            return `
                <div class="label-segment" style="left:${left}%; width:${width}%; background:${color}" title="${l.activity} (${l.start}-${l.end})">
                    ${l.activity}
                    <button class="label-segment__delete" data-label-id="${l.id}">&#10005;</button>
                </div>
            `;
        }).join("");

        const cursorLeft = (currentFrame / maxFrame * 100).toFixed(2);

        return `
            <div class="label-timeline-row">
                <span class="label-timeline-row__name">${person}</span>
                <div class="label-timeline-row__track">
                    ${segments}
                    <div class="label-timeline-cursor" style="left:${cursorLeft}%"></div>
                </div>
            </div>
        `;
    }).join("");

    if (statsEl) {
        statsEl.textContent = `${activityLabels.length} labels, ${activities.length} activities: ${activities.join(", ")}`;
    }
    if (trainBtn) trainBtn.disabled = false;
}

// Train from labels with progress polling
let trainProgressInterval = null;

document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnTrainFromLabels") return;
    if (activityLabels.length === 0) return;

    const btn = e.target;
    const statsEl = document.getElementById("labelStats");
    btn.disabled = true;

    // Auto-save labels to project before training
    await saveCurrentLabels();

    try {
        const video = document.getElementById("previewVideo");
        if (!video || !video.src) {
            showToast("No video loaded");
            btn.disabled = false;
            return;
        }

        const slider = document.getElementById("frameSlider");
        const maxFrame = parseInt(slider.max);
        const fps = maxFrame / video.duration;

        const payload = {
            video_url: video.src,
            append_to_existing: true,
            labels: activityLabels.map(l => ({
                start_frame: l.start,
                end_frame: l.end,
                start_time: (l.start / maxFrame) * video.duration,
                end_time: (l.end / maxFrame) * video.duration,
                activities: l.activities || [l.activity],
                activity: l.activity,
                person: l.person,
            })),
            fps: fps,
            duration: video.duration,
        };

        // Start progress polling
        trainProgressInterval = setInterval(async () => {
            try {
                const pRes = await fetch(`${API_BASE}/api/train/progress`);
                const p = await pRes.json();
                if (p.status === "extracting") {
                    const pct = p.percent || 0;
                    btn.textContent = `Extracting ${pct}%`;
                    if (statsEl) statsEl.textContent = `Label ${p.label_current}/${p.label_total} | ${p.features_extracted} features | ${p.current_activity || ""}`;
                } else if (p.status === "training" || p.status === "fitting") {
                    btn.textContent = `Fitting model...`;
                    if (statsEl) statsEl.textContent = `${p.features_extracted || "?"} features → fitting classifier`;
                } else if (p.status === "done") {
                    btn.textContent = "Done!";
                    clearInterval(trainProgressInterval);
                }
            } catch (_) {}
        }, 500);

        const res = await fetch(`${API_BASE}/api/train/from-labels`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        clearInterval(trainProgressInterval);
        trainProgressInterval = null;

        if (data.status === "success") {
            showToast(`Trained v${data.version_id}: ${data.samples} features from ${activityLabels.length} labels`);
            loadTrainingStatus();
        } else {
            showToast("Training failed: " + (data.message || "unknown error"));
        }
    } catch (err) {
        clearInterval(trainProgressInterval);
        showToast("Training failed: " + err.message);
    }

    btn.textContent = "Add & Train";
    btn.disabled = false;
});

// Update person dropdown when persons are detected
function updateLabelPersonDropdown(persons) {
    const sel = document.getElementById("labelPerson");
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="all">All persons</option>';
    (persons || []).forEach(p => {
        const pid = p.person_id || 0;
        const name = p.identity && p.identity.matched ? p.identity.name : `#${pid}`;
        const opt = document.createElement("option");
        opt.value = `#${pid}`;
        opt.textContent = name;
        sel.appendChild(opt);
    });
    if ([...sel.options].some(o => o.value === current)) sel.value = current;
}


// ===== Projects & Collections =====

async function loadProjects() {
    try {
        const res = await fetch(`${API_BASE}/api/projects`);
        const data = await res.json();
        const select = document.getElementById("projectSelect");
        if (!select) return;

        // Preserve current selection
        const current = select.value;
        select.innerHTML = '<option value="">-- No Project (Global) --</option>';
        (data.projects || []).forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.id;
            opt.textContent = `${p.name} (${p.dataset_samples} samples, ${p.video_count} videos)`;
            select.appendChild(opt);
        });
        // Restore selection
        if (current && [...select.options].some(o => o.value === current)) {
            select.value = current;
        }
    } catch (err) {
        console.log("Could not load projects");
    }
}

async function activateProject(projectId) {
    if (projectId) {
        // Save current labels before switching
        await saveCurrentLabels();

        try {
            const res = await fetch(`${API_BASE}/api/projects/${projectId}/activate`, { method: "POST" });
            const data = await res.json();
            state.activeProjectId = projectId;
            document.getElementById("btnDeleteProject").style.display = "inline-block";

            // Show pose mode
            const pmEl = document.getElementById("projectPoseMode");
            if (pmEl) pmEl.textContent = data.pose_mode ? `Pose: ${data.pose_mode} (locked)` : "Pose: not locked yet";

            // Collection dropdown mirrors the project's linked collection
            const linkEl = document.getElementById("projectCollectionLink");
            if (linkEl) linkEl.value = data.collection_id || "";

            showToast(`Project: ${data.name}`);
        } catch (err) {
            showToast("Failed to activate project: " + err.message);
            return;
        }
    } else {
        // Deactivate — return to global
        await saveCurrentLabels();
        try {
            await fetch(`${API_BASE}/api/projects/deactivate`, { method: "POST" });
        } catch (_) {}
        state.activeProjectId = null;
        document.getElementById("btnDeleteProject").style.display = "none";
        document.getElementById("projectPoseMode").textContent = "";
        const linkEl = document.getElementById("projectCollectionLink");
        if (linkEl) linkEl.value = "";
    }

    // Reload everything for the new context
    loadTrainingStatus();
    loadDatasetInfo();
    loadGallery();

    // Load labels for current video in new project
    await loadLabelsForCurrentVideo();
}

// Project dropdown change
document.addEventListener("change", (e) => {
    if (e.target.id === "projectSelect") {
        activateProject(e.target.value || null);
    }
});

// New project
document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnNewProject") return;
    const name = prompt("Project name:");
    if (!name || !name.trim()) return;

    try {
        const formData = new FormData();
        formData.append("name", name.trim());
        const res = await fetch(`${API_BASE}/api/projects`, { method: "POST", body: formData });
        const data = await res.json();
        showToast(`Created project: ${data.name}`);

        await loadProjects();
        // Auto-activate the new project
        document.getElementById("projectSelect").value = data.id;
        await activateProject(data.id);
    } catch (err) {
        showToast("Failed to create project: " + err.message);
    }
});

// Delete project
document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnDeleteProject") return;
    if (!state.activeProjectId) return;
    if (!confirm("Delete this project and all its data?")) return;

    try {
        await fetch(`${API_BASE}/api/projects/${state.activeProjectId}`, { method: "DELETE" });
        showToast("Project deleted");
        state.activeProjectId = null;
        document.getElementById("projectSelect").value = "";
        document.getElementById("btnDeleteProject").style.display = "none";
        await loadProjects();
        await activateProject(null);
    } catch (err) {
        showToast("Delete failed: " + err.message);
    }
});

// Link collection to project
document.addEventListener("change", async (e) => {
    if (e.target.id !== "projectCollectionLink") return;
    if (!state.activeProjectId) {
        showToast("Select a project first");
        e.target.value = "";
        return;
    }
    const collectionId = e.target.value || null;
    try {
        await fetch(`${API_BASE}/api/projects/${state.activeProjectId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ collection_id: collectionId }),
        });
        // Re-activate to load the collection's ReID
        await activateProject(state.activeProjectId);
        showToast(collectionId ? "Collection linked to project" : "Collection unlinked");
    } catch (err) {
        showToast("Failed to link collection");
    }
});

// ===== Workers Collections =====

async function loadCollections() {
    try {
        const res = await fetch(`${API_BASE}/api/workers/collections`);
        const data = await res.json();
        const collections = data.collections || [];

        // Populate collection dropdown in Workers panel
        const colSelect = document.getElementById("collectionSelect");
        if (colSelect) {
            const current = colSelect.value;
            colSelect.innerHTML = '<option value="">Global</option>';
            collections.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.id;
                opt.textContent = `${c.name} (${c.gallery_size})`;
                colSelect.appendChild(opt);
            });
            if (current) colSelect.value = current;
        }

        // Populate project-collection link dropdown
        const linkSelect = document.getElementById("projectCollectionLink");
        if (linkSelect) {
            const current = linkSelect.value;
            linkSelect.innerHTML = '<option value="">None</option>';
            collections.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.id;
                opt.textContent = c.name;
                linkSelect.appendChild(opt);
            });
            if (current) linkSelect.value = current;
        }
    } catch (err) {
        console.log("Could not load collections");
    }
}

// New collection
document.addEventListener("click", async (e) => {
    if (e.target.id !== "btnNewCollection") return;
    const name = prompt("Collection name (e.g. 'Warehouse A Day Shift'):");
    if (!name || !name.trim()) return;

    try {
        const formData = new FormData();
        formData.append("name", name.trim());
        const res = await fetch(`${API_BASE}/api/workers/collections`, { method: "POST", body: formData });
        const data = await res.json();
        showToast(`Created collection: ${data.name}`);
        await loadCollections();
        // Auto-select the new collection
        const colSelect = document.getElementById("collectionSelect");
        if (colSelect) colSelect.value = data.id;
        state.activeCollectionId = data.id;
        loadGallery();
    } catch (err) {
        showToast("Failed to create collection: " + err.message);
    }
});

// Switch collection in Workers panel
document.addEventListener("change", (e) => {
    if (e.target.id !== "collectionSelect") return;
    state.activeCollectionId = e.target.value || null;
    loadGallery();
});

// ===== Label Persistence =====

async function saveCurrentLabels() {
    if (!state.activeProjectId || !state.currentVideoName || activityLabels.length === 0) return;
    try {
        const labelsToSave = activityLabels.map(l => ({
            start: l.start,
            end: l.end,
            activities: l.activities || [l.activity],
            activity: l.activity,
            person: l.person,
        }));
        await fetch(`${API_BASE}/api/projects/${state.activeProjectId}/labels/${encodeURIComponent(state.currentVideoName)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ labels: labelsToSave }),
        });
    } catch (err) {
        console.log("Could not save labels:", err);
    }
}

async function loadLabelsForCurrentVideo() {
    if (!state.activeProjectId || !state.currentVideoName) return;
    try {
        const res = await fetch(`${API_BASE}/api/projects/${state.activeProjectId}/labels/${encodeURIComponent(state.currentVideoName)}`);
        const data = await res.json();

        activityLabels.length = 0;
        (data.labels || []).forEach(l => {
            const activities = l.activities || [l.activity || "unknown"];
            activityLabels.push({
                start: l.start || 0,
                end: l.end || 0,
                activities: activities,
                activity: activities.join("+"),
                person: l.person || "all",
                id: Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
            });

            // Add activity checkboxes if missing
            activities.forEach(act => {
                const container = document.getElementById("labelCheckboxes");
                if (container && ![...container.querySelectorAll(".label-cb")].some(cb => cb.value === act)) {
                    const hint = container.querySelector("span");
                    if (hint && hint.textContent.includes("No activities")) hint.remove();
                    const lbl = document.createElement("label");
                    lbl.className = "label-cb-item";
                    lbl.innerHTML = `<input type="checkbox" value="${act}" class="label-cb"> ${act}`;
                    container.appendChild(lbl);
                }
            });
        });

        renderLabelTimeline();
        if (data.labels && data.labels.length > 0) {
            showToast(`Loaded ${data.labels.length} labels for ${state.currentVideoName}`);
        }
    } catch (err) {
        console.log("Could not load labels:", err);
    }
}


// ===== Init =====
document.addEventListener("DOMContentLoaded", () => {
    loadProjects();
    loadCollections();
    loadTrainingStatus();
    loadEvents();
    loadDemoVideos();
    loadTrainingDemoVideos();
    loadGallery();

    // Sync active project from server status
    fetch(`${API_BASE}/api/train/status`).then(r => r.json()).then(data => {
        if (data.active_project) {
            state.activeProjectId = data.active_project;
            const sel = document.getElementById("projectSelect");
            if (sel) sel.value = data.active_project;
            document.getElementById("btnDeleteProject").style.display = "inline-block";
        }
    }).catch(() => {});

    // Refresh events periodically
    setInterval(loadEvents, 10000);

    // Try connecting dashboard WS
    connectDashboardWS();
});
