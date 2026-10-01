/**
 * Keypoint Annotation Tool
 * Drag-to-correct pose landmarks and export them as COCO keypoints
 * (MediaPipe itself cannot be fine-tuned; see docs/design-decisions.md).
 */

(function () {
    "use strict";

    const API = window.location.origin;
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    // MediaPipe 33-landmark names
    const LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ];

    // Skeleton connections (same as backend)
    const SKELETON = [
        [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
        [9,10],[11,12],[11,13],[13,15],[12,14],[14,16],
        [11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],
        [15,17],[15,19],[15,21],[16,18],[16,20],[16,22],
        [27,29],[27,31],[28,30],[28,32],
    ];

    // Body-part color groups
    const KP_GROUPS = {
        face: [0,1,2,3,4,5,6,7,8,9,10],
        torso: [11,12,23,24],
        left_arm: [11,13,15,17,19,21],
        right_arm: [12,14,16,18,20,22],
        left_leg: [23,25,27,29,31],
        right_leg: [24,26,28,30,32],
    };

    const GROUP_COLORS = {
        face:      "#fbbf24",
        torso:     "#60a5fa",
        left_arm:  "#34d399",
        right_arm: "#a78bfa",
        left_leg:  "#f87171",
        right_leg: "#fb923c",
    };

    function groupOf(idx) {
        for (const [g, indices] of Object.entries(KP_GROUPS)) {
            if (indices.includes(idx)) return g;
        }
        return "face";
    }

    // ===== State =====
    const annotState = {
        projectId: null,
        projectMeta: null,
        currentFrame: 0,
        totalFrames: 0,
        keypoints: [],      // [{x, y, visibility, name}, ...] normalized 0..1
        dragging: -1,       // index of keypoint being dragged
        hovered: -1,        // index of keypoint under cursor
        showSkeleton: true,
        savedFrames: new Set(),
        dirty: false,       // unsaved changes on current frame
    };

    // ===== Init =====
    document.addEventListener("DOMContentLoaded", () => {
        loadAnnotationProjects();
        loadAnnotDemoVideos();
        setupAnnotEventListeners();
    });

    // ===== Project management =====

    async function loadAnnotationProjects() {
        try {
            const res = await fetch(`${API}/api/annotations/projects`);
            const data = await res.json();
            const select = $("#annotProjectSelect");
            // Keep the default option, clear rest
            select.innerHTML = '<option value="">-- Select Project --</option>';
            data.projects.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.textContent = `${p.name} (${p.annotated_frames} annotated / ${p.total_frames} frames)`;
                select.appendChild(opt);
            });
        } catch (e) {
            console.log("Could not load annotation projects");
        }
    }

    async function loadAnnotDemoVideos() {
        try {
            const res = await fetch(`${API}/api/demo/videos`);
            const data = await res.json();
            const select = $("#annotDemoSelect");
            data.videos.forEach((v) => {
                const opt = document.createElement("option");
                opt.value = v.url;
                opt.textContent = v.name.replace(/\.mp4$/, "").replace(/_/g, " ");
                select.appendChild(opt);
            });
        } catch (e) {
            console.log("Could not load demo videos for annotate tab");
        }
    }

    async function openProject(projectId) {
        try {
            const res = await fetch(`${API}/api/annotations/projects/${projectId}`);
            const project = await res.json();
            annotState.projectId = projectId;
            annotState.projectMeta = project;
            annotState.totalFrames = project.total_frames;
            annotState.currentFrame = 0;
            annotState.keypoints = [];
            annotState.savedFrames.clear();
            annotState.dirty = false;

            $("#annotProjectName").textContent = project.name;
            $("#annotCountTotal").textContent = project.total_frames;
            $("#annotFrameSlider").max = Math.max(0, project.total_frames - 1);
            $("#annotFrameSlider").value = 0;
            $("#annotWorkspace").style.display = "block";
            $("#btnDeleteProject").style.display = "inline-block";

            // Load list of already-annotated frames
            const framesRes = await fetch(`${API}/api/annotations/projects/${projectId}/frames`);
            const framesData = await framesRes.json();
            framesData.frames.forEach((f) => annotState.savedFrames.add(f.frame_index));
            updateAnnotStats();
            renderAnnotStrip();

            await loadFrame(0);
        } catch (e) {
            showToast("Failed to open project: " + e.message);
        }
    }

    // ===== Frame loading =====

    async function loadFrame(frameIndex) {
        if (!annotState.projectId) return;
        const clamped = Math.max(0, Math.min(frameIndex, annotState.totalFrames - 1));
        annotState.currentFrame = clamped;
        annotState.dirty = false;

        $("#annotFrameSlider").value = clamped;
        $("#annotFrameInfo").textContent = `Frame ${clamped} / ${annotState.totalFrames - 1}`;

        try {
            const res = await fetch(`${API}/api/annotations/projects/${annotState.projectId}/frame/${clamped}`);
            const data = await res.json();

            // Show frame image
            const img = $("#annotFrameImg");
            img.src = data.image;
            img.onload = () => {
                resizeCanvas();
                if (data.keypoints) {
                    annotState.keypoints = data.keypoints;
                } else {
                    annotState.keypoints = [];
                }
                renderKeypoints();
                renderKpList();
            };
        } catch (e) {
            console.error("Failed to load frame:", e);
        }
    }

    // ===== Canvas rendering =====

    function resizeCanvas() {
        const wrap = $("#annotCanvasWrap");
        const canvas = $("#annotCanvas");
        const img = $("#annotFrameImg");

        canvas.width = wrap.clientWidth;
        canvas.height = wrap.clientHeight;
    }

    function imgToCanvas(nx, ny) {
        const wrap = $("#annotCanvasWrap");
        const img = $("#annotFrameImg");
        const cw = wrap.clientWidth;
        const ch = wrap.clientHeight;

        // object-fit: contain math
        const imgAspect = img.naturalWidth / img.naturalHeight;
        const wrapAspect = cw / ch;

        let renderW, renderH, offsetX, offsetY;
        if (imgAspect > wrapAspect) {
            renderW = cw;
            renderH = cw / imgAspect;
            offsetX = 0;
            offsetY = (ch - renderH) / 2;
        } else {
            renderH = ch;
            renderW = ch * imgAspect;
            offsetX = (cw - renderW) / 2;
            offsetY = 0;
        }

        return {
            x: offsetX + nx * renderW,
            y: offsetY + ny * renderH,
            renderW, renderH, offsetX, offsetY,
        };
    }

    function canvasToNorm(cx, cy) {
        const wrap = $("#annotCanvasWrap");
        const img = $("#annotFrameImg");
        const cw = wrap.clientWidth;
        const ch = wrap.clientHeight;

        const imgAspect = img.naturalWidth / img.naturalHeight;
        const wrapAspect = cw / ch;

        let renderW, renderH, offsetX, offsetY;
        if (imgAspect > wrapAspect) {
            renderW = cw;
            renderH = cw / imgAspect;
            offsetX = 0;
            offsetY = (ch - renderH) / 2;
        } else {
            renderH = ch;
            renderW = ch * imgAspect;
            offsetX = (cw - renderW) / 2;
            offsetY = 0;
        }

        return {
            x: Math.max(0, Math.min(1, (cx - offsetX) / renderW)),
            y: Math.max(0, Math.min(1, (cy - offsetY) / renderH)),
        };
    }

    function renderKeypoints() {
        const canvas = $("#annotCanvas");
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const kps = annotState.keypoints;
        if (!kps || kps.length === 0) return;

        // Draw skeleton connections
        if (annotState.showSkeleton) {
            SKELETON.forEach(([i, j]) => {
                if (i >= kps.length || j >= kps.length) return;
                const a = kps[i];
                const b = kps[j];
                if (a.visibility < 0.2 && b.visibility < 0.2) return;

                const pa = imgToCanvas(a.x, a.y);
                const pb = imgToCanvas(b.x, b.y);
                const alpha = Math.min(a.visibility, b.visibility, 1) * 0.6 + 0.15;

                ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(pa.x, pa.y);
                ctx.lineTo(pb.x, pb.y);
                ctx.stroke();
            });
        }

        // Draw keypoints
        kps.forEach((kp, idx) => {
            const p = imgToCanvas(kp.x, kp.y);
            const group = groupOf(idx);
            const color = GROUP_COLORS[group];
            const isKey = [11,12,13,14,15,16,23,24,25,26,27,28].includes(idx);
            const isHovered = idx === annotState.hovered;
            const isDragging = idx === annotState.dragging;

            let radius = isKey ? 6 : 4;
            if (isHovered || isDragging) radius += 2;

            const alpha = kp.visibility > 0.3 ? 1.0 : 0.35;

            // Outer ring for hovered/dragging
            if (isHovered || isDragging) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, radius + 4, 0, Math.PI * 2);
                ctx.strokeStyle = `rgba(255, 255, 255, 0.5)`;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Filled circle
            ctx.beginPath();
            ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
            ctx.fillStyle = isDragging ? "#fff" : color;
            ctx.globalAlpha = alpha;
            ctx.fill();
            ctx.globalAlpha = 1;

            // White outline
            ctx.strokeStyle = `rgba(0, 0, 0, 0.6)`;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Label on hover
            if (isHovered && !isDragging) {
                const name = kp.name || LANDMARK_NAMES[idx] || `#${idx}`;
                ctx.font = "11px Inter, sans-serif";
                const tw = ctx.measureText(name).width;
                const lx = p.x + radius + 6;
                const ly = p.y - 6;

                ctx.fillStyle = "rgba(0,0,0,0.8)";
                ctx.beginPath();
                ctx.roundRect(lx - 3, ly - 11, tw + 6, 16, 3);
                ctx.fill();

                ctx.fillStyle = "#fff";
                ctx.fillText(name, lx, ly + 2);
            }
        });

        // Dirty indicator
        if (annotState.dirty) {
            ctx.fillStyle = "#fbbf24";
            ctx.beginPath();
            ctx.arc(canvas.width - 14, 14, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.font = "bold 9px Inter, sans-serif";
            ctx.fillStyle = "#000";
            ctx.textAlign = "center";
            ctx.fillText("!", canvas.width - 14, 17);
            ctx.textAlign = "left";
        }
    }

    // ===== Keypoint list sidebar =====

    function renderKpList() {
        const container = $("#annotKpList");
        const kps = annotState.keypoints;

        if (!kps || kps.length === 0) {
            container.innerHTML = '<div class="annot-kp-empty">Auto-detect or load a frame to see keypoints</div>';
            return;
        }

        container.innerHTML = kps.map((kp, idx) => {
            const group = groupOf(idx);
            const color = GROUP_COLORS[group];
            const vis = kp.visibility > 0.3;
            return `
                <div class="annot-kp-item ${vis ? "" : "annot-kp-item--hidden"}"
                     data-idx="${idx}"
                     style="border-left: 3px solid ${color}">
                    <span class="annot-kp-item__idx">${idx}</span>
                    <span class="annot-kp-item__name">${kp.name || LANDMARK_NAMES[idx]}</span>
                    <span class="annot-kp-item__coords">(${kp.x.toFixed(3)}, ${kp.y.toFixed(3)})</span>
                    <button class="annot-kp-vis-btn" data-idx="${idx}" title="Toggle visibility">
                        ${vis ? "&#128065;" : "&#128683;"}
                    </button>
                </div>
            `;
        }).join("");
    }

    // ===== Annotated frames strip =====

    function renderAnnotStrip() {
        const strip = $("#annotStrip");
        const total = annotState.totalFrames;

        if (total === 0) {
            strip.innerHTML = "";
            return;
        }

        // Show markers for annotated frames on a mini timeline
        const sorted = [...annotState.savedFrames].sort((a, b) => a - b);
        strip.innerHTML = sorted.map((fi) => {
            const pct = ((fi / Math.max(1, total - 1)) * 100).toFixed(2);
            return `<div class="annot-strip__marker" data-frame="${fi}" style="left:${pct}%" title="Frame ${fi}"></div>`;
        }).join("") + `<div class="annot-strip__current" id="annotStripCurrent"></div>`;

        updateStripCurrent();
    }

    function updateStripCurrent() {
        const el = $("#annotStripCurrent");
        if (!el) return;
        const pct = ((annotState.currentFrame / Math.max(1, annotState.totalFrames - 1)) * 100).toFixed(2);
        el.style.left = pct + "%";
    }

    function updateAnnotStats() {
        $("#annotCountSaved").textContent = annotState.savedFrames.size;
        $("#annotCountTotal").textContent = annotState.totalFrames;
    }

    // ===== Mouse interactions on canvas =====

    function getCanvasCoords(e) {
        const canvas = $("#annotCanvas");
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    function findNearestKp(cx, cy, threshold) {
        const kps = annotState.keypoints;
        let nearest = -1;
        let minDist = threshold;

        kps.forEach((kp, idx) => {
            const p = imgToCanvas(kp.x, kp.y);
            const dist = Math.hypot(p.x - cx, p.y - cy);
            if (dist < minDist) {
                minDist = dist;
                nearest = idx;
            }
        });

        return nearest;
    }

    function setupAnnotEventListeners() {
        const canvas = $("#annotCanvas");

        // --- Canvas drag interactions ---
        canvas.addEventListener("mousedown", (e) => {
            if (annotState.keypoints.length === 0) return;
            const { x, y } = getCanvasCoords(e);
            const idx = findNearestKp(x, y, 18);
            if (idx >= 0) {
                annotState.dragging = idx;
                canvas.style.cursor = "grabbing";
                e.preventDefault();
            }
        });

        canvas.addEventListener("mousemove", (e) => {
            const { x, y } = getCanvasCoords(e);

            if (annotState.dragging >= 0) {
                const norm = canvasToNorm(x, y);
                annotState.keypoints[annotState.dragging].x = norm.x;
                annotState.keypoints[annotState.dragging].y = norm.y;
                // If it was hidden, make it visible when dragged
                if (annotState.keypoints[annotState.dragging].visibility < 0.3) {
                    annotState.keypoints[annotState.dragging].visibility = 1.0;
                }
                annotState.dirty = true;
                renderKeypoints();
                return;
            }

            // Hover detection
            const idx = findNearestKp(x, y, 18);
            if (idx !== annotState.hovered) {
                annotState.hovered = idx;
                canvas.style.cursor = idx >= 0 ? "grab" : "default";
                renderKeypoints();
            }
        });

        canvas.addEventListener("mouseup", () => {
            if (annotState.dragging >= 0) {
                annotState.dragging = -1;
                canvas.style.cursor = annotState.hovered >= 0 ? "grab" : "default";
                renderKeypoints();
                renderKpList();
            }
        });

        canvas.addEventListener("mouseleave", () => {
            annotState.dragging = -1;
            annotState.hovered = -1;
            canvas.style.cursor = "default";
            renderKeypoints();
        });

        // --- Window resize ---
        window.addEventListener("resize", () => {
            if (annotState.projectId) {
                resizeCanvas();
                renderKeypoints();
            }
        });

        // --- Project select ---
        $("#annotProjectSelect").addEventListener("change", (e) => {
            if (e.target.value) openProject(e.target.value);
        });

        // --- New project from demo ---
        $("#btnNewProjectDemo").addEventListener("click", async () => {
            const url = $("#annotDemoSelect").value;
            if (!url) { showToast("Select a demo video first."); return; }

            const btn = $("#btnNewProjectDemo");
            btn.textContent = "Creating...";
            btn.disabled = true;

            try {
                const form = new FormData();
                form.append("video_url", url);
                form.append("name", "");
                const res = await fetch(`${API}/api/annotations/projects/demo`, { method: "POST", body: form });
                const project = await res.json();
                await loadAnnotationProjects();
                await openProject(project.id);
                $("#annotProjectSelect").value = project.id;
                showToast(`Project "${project.name}" created`);
            } catch (e) {
                showToast("Failed to create project: " + e.message);
            }

            btn.textContent = "New from Demo";
            btn.disabled = false;
        });

        // --- New project from upload ---
        $("#btnNewProjectUpload").addEventListener("click", () => {
            $("#annotVideoInput").click();
        });

        $("#annotVideoInput").addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            try {
                const form = new FormData();
                form.append("file", file);
                form.append("name", file.name.replace(/\.[^.]+$/, ""));
                const res = await fetch(`${API}/api/annotations/projects`, { method: "POST", body: form });
                const project = await res.json();
                await loadAnnotationProjects();
                await openProject(project.id);
                $("#annotProjectSelect").value = project.id;
                showToast(`Project "${project.name}" created`);
            } catch (e) {
                showToast("Upload failed: " + e.message);
            }
        });

        // --- Delete project ---
        $("#btnDeleteProject").addEventListener("click", async () => {
            if (!annotState.projectId) return;
            if (!confirm("Delete this annotation project and all its saved frames?")) return;

            await fetch(`${API}/api/annotations/projects/${annotState.projectId}`, { method: "DELETE" });
            annotState.projectId = null;
            $("#annotWorkspace").style.display = "none";
            $("#btnDeleteProject").style.display = "none";
            await loadAnnotationProjects();
            showToast("Project deleted");
        });

        // --- Frame navigation ---
        $("#annotFrameSlider").addEventListener("input", (e) => {
            loadFrame(parseInt(e.target.value));
            updateStripCurrent();
        });

        $("#btnPrevFrame").addEventListener("click", () => {
            loadFrame(annotState.currentFrame - 1);
        });

        $("#btnNextFrame").addEventListener("click", () => {
            loadFrame(annotState.currentFrame + 1);
        });

        $("#btnFrameJump").addEventListener("click", () => {
            const val = parseInt($("#annotFrameJump").value);
            if (!isNaN(val)) loadFrame(val);
        });

        $("#annotFrameJump").addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) loadFrame(val);
            }
        });

        // --- Auto-detect ---
        $("#btnAutoDetect").addEventListener("click", async () => {
            if (!annotState.projectId) return;
            const btn = $("#btnAutoDetect");
            btn.textContent = "Detecting...";
            btn.disabled = true;

            try {
                const res = await fetch(
                    `${API}/api/annotations/projects/${annotState.projectId}/detect/${annotState.currentFrame}`,
                    { method: "POST" }
                );
                const data = await res.json();

                if (data.detected) {
                    annotState.keypoints = data.keypoints;
                    annotState.dirty = true;
                    renderKeypoints();
                    renderKpList();
                    showToast("Pose detected — adjust keypoints and save.");
                } else {
                    showToast("No person detected in this frame.");
                }
            } catch (e) {
                showToast("Detection failed: " + e.message);
            }

            btn.textContent = "Auto-Detect";
            btn.disabled = false;
        });

        // --- Save frame ---
        $("#btnSaveFrame").addEventListener("click", async () => {
            if (!annotState.projectId || annotState.keypoints.length === 0) {
                showToast("No keypoints to save. Run Auto-Detect first.");
                return;
            }

            try {
                const form = new FormData();
                form.append("frame_index", annotState.currentFrame);
                form.append("keypoints", JSON.stringify(annotState.keypoints));
                await fetch(`${API}/api/annotations/projects/${annotState.projectId}/save`, {
                    method: "POST",
                    body: form,
                });
                annotState.dirty = false;
                annotState.savedFrames.add(annotState.currentFrame);
                updateAnnotStats();
                renderAnnotStrip();
                renderKeypoints();
                showToast(`Frame ${annotState.currentFrame} saved.`);
            } catch (e) {
                showToast("Save failed: " + e.message);
            }
        });

        // --- Skeleton toggle ---
        $("#annotShowSkeleton").addEventListener("change", (e) => {
            annotState.showSkeleton = e.target.checked;
            renderKeypoints();
        });

        // --- Keypoint list clicks (visibility toggle + highlight) ---
        $("#annotKpList").addEventListener("click", (e) => {
            const visBtn = e.target.closest(".annot-kp-vis-btn");
            if (visBtn) {
                const idx = parseInt(visBtn.dataset.idx);
                const kp = annotState.keypoints[idx];
                kp.visibility = kp.visibility > 0.3 ? 0.0 : 1.0;
                annotState.dirty = true;
                renderKeypoints();
                renderKpList();
                return;
            }

            const item = e.target.closest(".annot-kp-item");
            if (item) {
                const idx = parseInt(item.dataset.idx);
                annotState.hovered = idx;
                renderKeypoints();
                // Scroll keypoint into view on canvas (flash effect)
                setTimeout(() => {
                    annotState.hovered = -1;
                    renderKeypoints();
                }, 1200);
            }
        });

        // --- Annotated frame strip clicks ---
        $("#annotStrip").addEventListener("click", (e) => {
            const marker = e.target.closest(".annot-strip__marker");
            if (marker) {
                loadFrame(parseInt(marker.dataset.frame));
            }
        });

        // --- Export COCO ---
        $("#btnExportCOCO").addEventListener("click", async () => {
            if (!annotState.projectId) return;

            try {
                const res = await fetch(`${API}/api/annotations/projects/${annotState.projectId}/export`);
                const data = await res.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `coco_keypoints_${annotState.projectId}.json`;
                a.click();
                URL.revokeObjectURL(url);
                showToast(`Exported ${data.annotations.length} annotations as COCO JSON`);
            } catch (e) {
                showToast("Export failed: " + e.message);
            }
        });

        // --- Export frames ---
        $("#btnExportFrames").addEventListener("click", async () => {
            if (!annotState.projectId) return;

            const btn = $("#btnExportFrames");
            btn.textContent = "Exporting...";
            btn.disabled = true;

            try {
                const res = await fetch(
                    `${API}/api/annotations/projects/${annotState.projectId}/export-frames`,
                    { method: "POST" }
                );
                const data = await res.json();
                showToast(`Exported ${data.frames_exported} frames to ${data.export_dir}`);
            } catch (e) {
                showToast("Export failed: " + e.message);
            }

            btn.textContent = "Export Frames";
            btn.disabled = false;
        });

        // --- Keyboard shortcuts ---
        document.addEventListener("keydown", (e) => {
            // Only handle if annotate tab is active
            if (!$("#tab-annotate").classList.contains("active")) return;
            if (!annotState.projectId) return;
            // Don't capture when typing in inputs
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

            switch (e.key) {
                case "ArrowLeft":
                case "a":
                    e.preventDefault();
                    loadFrame(annotState.currentFrame - 1);
                    break;
                case "ArrowRight":
                case "d":
                    e.preventDefault();
                    loadFrame(annotState.currentFrame + 1);
                    break;
                case "s":
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        $("#btnSaveFrame").click();
                    }
                    break;
                case " ":
                    e.preventDefault();
                    $("#btnAutoDetect").click();
                    break;
            }
        });
    }
})();
