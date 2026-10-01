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
    frameInterval: null,
    currentPrediction: null,
    timelineData: [],
    distributionData: {},
};

// ===== Activity colors =====
const COLORS = {
    sorting: "#4f8cff",
    carrying: "#a78bfa",
    loading: "#f59e0b",
    idle: "#6b7280",
    walking: "#34d399",
    no_person: "#374151",
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
let trainingSource = "file"; // "file" | "current"

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
$("#trainingDemoSelect").addEventListener("change", (e) => {
    const url = e.target.value;
    const statusEl = $("#currentVideoStatus");
    if (!url) {
        statusEl.textContent = "No video selected";
        statusEl.classList.remove("has-video");
        return;
    }
    const name = url.split("/").pop().replace(/\.mp4$/, "").replace(/_/g, " ");
    statusEl.textContent = `Selected: ${name}`;
    statusEl.classList.add("has-video");
    showPosePreview(url);
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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        trainingInput.files = files;
    }
});

$("#btnUploadTraining").addEventListener("click", async () => {
    const label = $("#customLabel").value.trim() || $("#trainingLabel").value;

    if (trainingSource === "current") {
        // Use the demo video selected in the training tab dropdown
        const demoUrl = $("#trainingDemoSelect").value;
        if (!demoUrl) {
            showToast("Please select a demo video from the dropdown first.");
            return;
        }

        const btn = $("#btnUploadTraining");
        btn.textContent = "Sending...";
        btn.disabled = true;

        try {
            const res = await fetch(demoUrl);
            const blob = await res.blob();
            const filename = demoUrl.split("/").pop() || "demo_video.mp4";

            const formData = new FormData();
            formData.append("file", blob, filename);
            formData.append("label", label);

            const uploadRes = await fetch(`${API_BASE}/api/train/upload`, { method: "POST", body: formData });
            const data = await uploadRes.json();
            console.log("Uploaded current video:", data);
            showToast(`Uploaded current video as "${label}" training sample`);
        } catch (err) {
            showToast("Upload failed: " + err.message);
        }

        btn.textContent = "Upload Sample";
        btn.disabled = false;
        loadTrainingStatus();
        return;
    }

    // File upload mode
    const files = trainingInput.files;
    if (!files || files.length === 0) {
        showToast("Please select a video file first.");
        return;
    }

    for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("label", label);

        try {
            const res = await fetch(`${API_BASE}/api/train/upload`, { method: "POST", body: formData });
            const data = await res.json();
            console.log("Uploaded:", data);
        } catch (err) {
            showToast("Upload failed: " + err.message);
        }
    }

    showToast(`Uploaded ${files.length} sample(s) with label "${label}"`);
    loadTrainingStatus();
});

$("#btnTrain").addEventListener("click", async () => {
    const btn = $("#btnTrain");
    btn.textContent = "Training...";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/train/start`, { method: "POST" });
        const data = await res.json();
        showToast(`Training ${data.status}: ${data.samples || 0} samples, classes: ${(data.classes || []).join(", ")}`);
    } catch (err) {
        showToast("Training failed: " + err.message);
    }

    btn.textContent = "Train Model";
    btn.disabled = false;
    loadTrainingStatus();
});

async function loadTrainingStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/train/status`);
        const data = await res.json();

        $("#modelType").textContent = data.model_type;
        $("#modelTrained").textContent = data.is_trained ? "Yes" : "No";
        $("#modelSamples").textContent = data.training_samples;

        $("#modelClasses").innerHTML = (data.classes || [])
            .map((c) => `<span class="class-tag" style="border-color:${COLORS[c] || '#555'}">${c}</span>`)
            .join("");

        $("#modelFeatures").innerHTML = (data.features || [])
            .map((f) => `<span class="feature-tag">${f}</span>`)
            .join("");
    } catch (err) {
        console.log("Could not load training status");
    }
}

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
    $("#btnStartCamera").textContent = "Start Camera";
    startStreaming();
});

// ===== Pose Preview (Training Tab) =====
const POSE_CONNECTIONS = [
    [11,12],[11,13],[13,15],[12,14],[14,16],  // arms
    [11,23],[12,24],[23,24],                    // torso
    [23,25],[25,27],[24,26],[26,28],            // legs
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8], // face
    [9,10],[11,12],                             // extra
    [15,17],[15,19],[15,21],[16,18],[16,20],[16,22], // hands
    [27,29],[27,31],[28,30],[28,32],            // feet
];

const FEATURE_NAMES = [
    "Torso Angle", "Arm Spread", "Vert. Move",
    "Hand Height", "Stride Len.", "Body Speed", "Symmetry"
];

let poseAnalyzeTimeout = null;

function showPosePreview(videoSrc) {
    const panel = $("#posePreviewPanel");
    const video = $("#previewVideo");
    const slider = $("#frameSlider");

    // Create a fresh video element to avoid stacked listeners
    const newVideo = document.createElement("video");
    newVideo.id = "previewVideo";
    newVideo.muted = true;
    newVideo.playsInline = true;
    newVideo.preload = "auto";
    video.parentNode.replaceChild(newVideo, video);

    // For server-hosted videos, load as blob to avoid Range/seek issues
    if (videoSrc.startsWith("/") || videoSrc.startsWith("http")) {
        fetch(videoSrc)
            .then((r) => r.blob())
            .then((blob) => {
                newVideo.src = URL.createObjectURL(blob);
                newVideo.load();
            });
    } else {
        newVideo.src = videoSrc;
        newVideo.load();
    }

    panel.style.display = "block";

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

// ===== FPS Counter =====
let fpsLastTime = 0;
let fpsFrameCount = 0;
let fpsValue = 0;

function updateFps() {
    const now = performance.now();
    fpsFrameCount++;

    if (fpsLastTime === 0) {
        fpsLastTime = now;
        return;
    }

    const elapsed = now - fpsLastTime;
    if (elapsed >= 1000) {
        fpsValue = Math.round((fpsFrameCount / elapsed) * 1000 * 10) / 10;
        fpsFrameCount = 0;
        fpsLastTime = now;
    }

    const el = $("#poseFps");
    if (!el) return;
    el.textContent = `${fpsValue} FPS`;
    el.className = "pose-fps" + (fpsValue < 1 ? " very-slow" : fpsValue < 5 ? " slow" : "");
}

function resetFps() {
    fpsLastTime = 0;
    fpsFrameCount = 0;
    fpsValue = 0;
    const el = $("#poseFps");
    if (el) el.textContent = "";
}

// ===== Frame-by-Frame Playback =====
let posePlaybackActive = false;

document.addEventListener("click", (e) => {
    if (e.target.id === "btnPlayFrames") {
        togglePosePlayback();
    }
});

function togglePosePlayback() {
    if (posePlaybackActive) {
        stopPosePlayback();
    } else {
        startPosePlayback();
    }
}

function startPosePlayback() {
    posePlaybackActive = true;
    resetFps();
    const btn = $("#btnPlayFrames");
    btn.textContent = "\u25A0 Stop";
    btn.classList.add("playing");
    playNextFrame();
}

function stopPosePlayback() {
    posePlaybackActive = false;
    const btn = $("#btnPlayFrames");
    btn.textContent = "\u25B6 Play";
    btn.classList.remove("playing");
}

async function playNextFrame() {
    if (!posePlaybackActive) return;

    const slider = $("#frameSlider");
    const video = document.getElementById("previewVideo");
    if (!video || !video.duration) { stopPosePlayback(); return; }

    const currentVal = parseInt(slider.value);
    const maxVal = parseInt(slider.max);

    if (currentVal >= maxVal) {
        stopPosePlayback();
        return;
    }

    // Advance one frame
    slider.value = currentVal + 1;
    const t = (slider.value / maxVal) * video.duration;
    video.currentTime = t;
    $("#previewCurrentTime").textContent = formatTime(t);
    $("#poseFrameInfo").textContent = `Frame ${slider.value} / ${maxVal}`;

    // Wait for seek to complete
    await new Promise((resolve) => {
        const onSeeked = () => { video.removeEventListener("seeked", onSeeked); resolve(); };
        video.addEventListener("seeked", onSeeked);
    });

    // Analyze frame (wait for it to complete)
    await analyzeCurrentFrame();

    if (!posePlaybackActive) return;

    // Apply delay
    const delay = parseInt($("#poseFrameDelay").value) || 0;
    if (delay > 0) {
        await new Promise((r) => setTimeout(r, delay));
    }

    if (!posePlaybackActive) return;

    // Continue to next frame
    playNextFrame();
}

// Manual analyze button
document.addEventListener("click", (e) => {
    if (e.target.id === "btnAnalyzeFrame") {
        analyzeCurrentFrame();
    }
});

async function analyzeCurrentFrame() {
    const video = document.getElementById("previewVideo");
    if (!video || !video.duration || video.readyState < 2) return;

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

    try {
        const res = await fetch(`${API_BASE}/api/predict/landmarks`, { method: "POST", body: formData });
        const data = await res.json();
        drawPoseSkeleton(data.landmarks || [], video.videoWidth, video.videoHeight);
        updatePoseBadge(data);
        updatePoseFeatures(data.features || []);
        updateFps();
    } catch (err) {
        console.error("Pose analysis failed:", err);
    }
}

function drawPoseSkeleton(landmarks, videoW, videoH) {
    const canvas = $("#poseCanvas");

    // Match canvas resolution to the displayed size
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!landmarks || landmarks.length === 0) return;

    // Calculate scaling to match object-fit: contain
    const displayW = canvas.width;
    const displayH = canvas.height;
    const videoAspect = videoW / videoH;
    const displayAspect = displayW / displayH;

    let scaleW, scaleH, offsetX, offsetY;
    if (videoAspect > displayAspect) {
        scaleW = displayW;
        scaleH = displayW / videoAspect;
        offsetX = 0;
        offsetY = (displayH - scaleH) / 2;
    } else {
        scaleH = displayH;
        scaleW = displayH * videoAspect;
        offsetX = (displayW - scaleW) / 2;
        offsetY = 0;
    }

    function toPixel(lm) {
        return {
            x: offsetX + lm.x * scaleW,
            y: offsetY + lm.y * scaleH,
        };
    }

    // Draw connections
    ctx.lineWidth = 2.5;
    POSE_CONNECTIONS.forEach(([i, j]) => {
        if (i >= landmarks.length || j >= landmarks.length) return;
        const a = landmarks[i];
        const b = landmarks[j];
        if (a.visibility < 0.4 || b.visibility < 0.4) return;

        const pa = toPixel(a);
        const pb = toPixel(b);

        ctx.strokeStyle = `rgba(79, 140, 255, ${Math.min(a.visibility, b.visibility)})`;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
    });

    // Draw keypoints
    landmarks.forEach((lm, idx) => {
        if (lm.visibility < 0.4) return;
        const p = toPixel(lm);

        // Larger dots for key joints
        const isKey = [11,12,13,14,15,16,23,24,25,26,27,28].includes(idx);
        const radius = isKey ? 5 : 3;

        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isKey
            ? `rgba(52, 211, 153, ${lm.visibility})`
            : `rgba(167, 139, 250, ${lm.visibility})`;
        ctx.fill();

        // White outline for key joints
        if (isKey) {
            ctx.strokeStyle = `rgba(255, 255, 255, ${lm.visibility * 0.6})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    });
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
    activity.textContent = pred.activity;
    activity.style.color = COLORS[pred.activity] || COLORS.idle;
    conf.textContent = `${(pred.confidence * 100).toFixed(0)}%`;
}

function updatePoseFeatures(features) {
    const container = $("#poseFeatures");
    if (!features || features.length === 0) {
        container.innerHTML = "";
        return;
    }

    container.innerHTML = features.map((val, i) => `
        <div class="pose-feature-chip">
            <span class="pose-feature-chip__name">${FEATURE_NAMES[i] || `F${i+1}`}</span>
            <span class="pose-feature-chip__value">${(val * 100).toFixed(0)}%</span>
        </div>
    `).join("");
}

// Hook into file selection — show preview when video is chosen
trainingInput.addEventListener("change", () => {
    const files = trainingInput.files;
    if (files && files.length > 0) {
        const url = URL.createObjectURL(files[0]);
        showPosePreview(url);
    }
});

// Hook into drag-and-drop — show preview
uploadZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
        const url = URL.createObjectURL(files[0]);
        showPosePreview(url);
    }
});


// ===== Pose Model Selector =====

async function loadPoseModels() {
    try {
        const res = await fetch(`${API_BASE}/api/models`);
        const data = await res.json();
        const select = $("#poseModelSelect");
        select.innerHTML = "";

        data.models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.filename;
            const label = m.variant !== "custom"
                ? `${m.variant.charAt(0).toUpperCase() + m.variant.slice(1)} (${m.size_mb} MB, ${m.speed})`
                : `${m.filename} (${m.size_mb} MB)`;
            opt.textContent = label;
            if (m.active) {
                opt.selected = true;
                updateFooterModel(m);
            }
            select.appendChild(opt);
        });
    } catch (e) {
        console.log("Could not load pose models");
    }
}

function updateFooterModel(m) {
    const el = $("#footerPoseModel");
    if (!el) return;
    const name = m.variant !== "custom"
        ? `MediaPipe PoseLandmarker ${m.variant.charAt(0).toUpperCase() + m.variant.slice(1)}`
        : m.filename;
    el.textContent = `${name} (${m.size_mb} MB)`;
}

$("#poseModelSelect").addEventListener("change", async (e) => {
    const filename = e.target.value;
    if (!filename) return;

    const select = $("#poseModelSelect");
    select.disabled = true;

    try {
        const form = new FormData();
        form.append("filename", filename);
        const res = await fetch(`${API_BASE}/api/models/switch`, { method: "POST", body: form });
        const data = await res.json();

        if (data.status === "switched") {
            showToast(`Pose model switched to ${filename}`, "success");
            loadPoseModels(); // refresh footer + dropdown
        } else if (data.status === "already_active") {
            showToast("This model is already active");
        }
    } catch (e) {
        showToast("Failed to switch model: " + e.message, "error");
        loadPoseModels(); // revert dropdown
    }

    select.disabled = false;
});

$("#btnUploadModel").addEventListener("click", () => {
    $("#modelFileInput").click();
});

$("#modelFileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/api/models/upload`, { method: "POST", body: form });
        const data = await res.json();

        if (data.status === "uploaded") {
            showToast(`Model "${data.filename}" uploaded (${data.size_mb} MB)`, "success");
            await loadPoseModels();
        } else {
            showToast(data.error || "Upload failed", "error");
        }
    } catch (e) {
        showToast("Upload failed: " + e.message, "error");
    }

    e.target.value = "";
});

// ===== Init =====
document.addEventListener("DOMContentLoaded", () => {
    loadPoseModels();
    loadTrainingStatus();
    loadEvents();
    loadDemoVideos();
    loadTrainingDemoVideos();

    // Refresh events periodically
    setInterval(loadEvents, 10000);

    // Try connecting dashboard WS
    connectDashboardWS();
});
