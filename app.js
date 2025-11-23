// DETECT-BEES App.js — Final version (UI + AI logic)

let session = null;
let modelLoaded = false;
let modelInputName = null;

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const themeBtn = document.getElementById("themeToggle");
const body = document.body;

const imgPreview = document.getElementById("imagePreview");
const videoPreview = document.getElementById("videoPreview");
const statusEl = document.getElementById("status");
const scoreFill = document.getElementById("scoreFill");
const scoreText = document.getElementById("scoreText");
const labelEl = document.getElementById("labelText");
const aiEl = document.getElementById("aiInsight");
const previewPlaceholder = document.getElementById("previewPlaceholder");

const workCanvas = document.getElementById("workCanvas");
const ctx = workCanvas.getContext("2d");

// 🌓 Theme toggle
themeBtn.addEventListener("click", () => {
  const isDark = body.classList.contains("dark");
  body.classList.toggle("dark", !isDark);
  body.classList.toggle("light", isDark);
  themeBtn.textContent = isDark ? "☀️" : "🌙";
});

// 📦 Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// 📂 File selection
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  handleFile(file);
});

function handleFile(file) {
  if (!file) return;

  if (file.type.startsWith("image/")) {
    handleImageFile(file);
  } else if (file.type.startsWith("video/")) {
    handleVideoFile(file);
  } else {
    alert("Please upload an image or a video file.");
  }
}

// 🧠 Model init
async function initModel() {
  try {
    statusEl.textContent = "Loading ONNX model…";
    const modelUrl = "models/deepfake_light.onnx";

    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm", "webgl"],
      graphOptimizationLevel: "all",
    });

    modelInputName = session.inputNames?.[0] || "input";
    modelLoaded = true;
    statusEl.textContent =
      "Model loaded · Real-time deepfake scoring active.";
  } catch (err) {
    console.warn("[DETECT-BEES] Model load failed:", err);
    modelLoaded = false;
    statusEl.textContent =
      "Model not found · Using mock scores for demo.";
  }
}

// 🔧 Preprocess pixel data
function preprocessImageData(imageData) {
  const target = 224;
  const channels = 3;
  const data = new Float32Array(1 * channels * target * target);

  let idx = 0;
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < target; y++) {
      for (let x = 0; x < target; x++) {
        const srcIndex = (y * target + x) * 4 + c;
        data[idx++] = imageData[srcIndex] / 127.5 - 1.0;
      }
    }
  }

  return new ort.Tensor("float32", data, [1, channels, target, target]);
}

// 🎯 Map raw probability → UI risk
function calibrateRisk(p) {
  const min = 0.1;
  const max = 0.6;
  let x = (p - min) / (max - min);
  if (x < 0) x = 0;
  if (x > 1) x = 1;
  return x;
}

// 🧠 Run model (or mock) on canvas
async function runModelFromCanvas() {
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const inputTensor = preprocessImageData(imageData.data);

  if (!modelLoaded || !session) {
    return mockRiskScore();
  }

  const feeds = {};
  feeds[modelInputName] = inputTensor;
  const results = await session.run(feeds);
  const out = results[Object.keys(results)[0]].data;

  if (out.length === 2) {
    const l0 = out[0]; // Deepfake
    const l1 = out[1]; // Real
    const maxLogit = Math.max(l0, l1);
    const e0 = Math.exp(l0 - maxLogit);
    const e1 = Math.exp(l1 - maxLogit);
    const sum = e0 + e1;
    const pFake = e0 / sum;
    return calibrateRisk(pFake);
  }

  return calibrateRisk(out[0]);
}

// 🎭 Mock risk if model missing
function mockRiskScore() {
  const r = Math.random();
  if (r < 0.7) return Math.random() * 0.3;
  if (r < 0.9) return 0.3 + Math.random() * 0.4;
  return 0.7 + Math.random() * 0.3;
}

// 🖼 Handle image file
async function handleImageFile(file) {
  clearPreview();
  const url = URL.createObjectURL(file);
  imgPreview.src = url;
  imgPreview.style.display = "block";
  previewPlaceholder.style.display = "none";

  await imgPreview.decode();

  ctx.clearRect(0, 0, workCanvas.width, workCanvas.height);
  ctx.drawImage(imgPreview, 0, 0, 224, 224);

  statusEl.textContent = "Analyzing image…";
  const risk = await runModelFromCanvas();
  setScore(risk);
  statusEl.textContent = "Analysis complete.";
}

// 🎥 Handle video file (single frame in middle)
async function handleVideoFile(file) {
  clearPreview();
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoPreview.style.display = "block";
  previewPlaceholder.style.display = "none";

  await new Promise((res, rej) => {
    videoPreview.onloadeddata = res;
    videoPreview.onerror = rej;
  });

  const duration = videoPreview.duration || 0;
  videoPreview.currentTime = duration ? duration / 2 : 0;

  await new Promise((res) => {
    videoPreview.onseeked = res;
  });

  ctx.clearRect(0, 0, workCanvas.width, workCanvas.height);
  ctx.drawImage(videoPreview, 0, 0, 224, 224);

  statusEl.textContent = "Analyzing video frame…";
  const risk = await runModelFromCanvas();
  setScore(risk);
  statusEl.textContent = "Analysis complete (single-frame estimate).";
}

// 📊 Update score + labels + insight
function setScore(risk) {
  const pct = Math.round((risk || 0) * 100);
  scoreText.textContent = pct + "%";

  scoreFill.className = "score-fill";
  if (risk >= 0.8) {
    scoreFill.classList.add("high");
  } else if (risk >= 0.4) {
    scoreFill.classList.add("moderate");
  } else {
    scoreFill.classList.add("safe");
  }
  scoreFill.style.width = pct + "%";

  if (risk >= 0.8) {
    labelEl.textContent = "⚠️ Likely Deepfake";
    labelEl.style.color = "#e74c3c";
    aiEl.textContent =
      "The model is highly confident this content matches patterns of AI-generated or manipulated faces. Treat as suspicious and verify via trusted channels.";
  } else if (risk >= 0.4) {
    labelEl.textContent = "🟡 Moderate — Needs Review";
    labelEl.style.color = "#f1c40f";
    aiEl.textContent =
      "The model detects some anomalies that may indicate manipulation. Review carefully and cross-check this media before using it for important decisions.";
  } else {
    labelEl.textContent = "🟢 Likely Real";
    labelEl.style.color = "#2ecc71";
    aiEl.textContent =
      "The model does not detect strong signs of deepfake manipulation. Still, no detector is perfect — stay cautious for high-impact or sensitive content.";
  }
}

// 🧹 Reset UI preview
function clearPreview() {
  imgPreview.style.display = "none";
  videoPreview.style.display = "none";
  previewPlaceholder.style.display = "block";
  scoreText.textContent = "--%";
  scoreFill.style.width = "0%";
  scoreFill.className = "score-fill";
  labelEl.textContent = "Awaiting input…";
  labelEl.style.color = "#a0a6c3";
  aiEl.textContent =
    "Upload an image or video to get a model-based explanation of whether the face appears synthetic or manipulated.";
}

// 🚀 Initialize app
window.addEventListener("DOMContentLoaded", () => {
  workCanvas.width = 224;
  workCanvas.height = 224;
  clearPreview();
  initModel();
});
