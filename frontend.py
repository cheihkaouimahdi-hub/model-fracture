UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>X-ray Classifier Tester</title>
  <style>
    :root {
      --bg: #f4f7f9;
      --card: #ffffff;
      --text: #13222e;
      --muted: #4c6478;
      --line: #d8e1e8;
      --accent: #0b7285;
      --accent-2: #0f9f80;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 0% 0%, #e6f7ff 0, transparent 36%),
        radial-gradient(circle at 100% 100%, #ddf9ef 0, transparent 30%),
        var(--bg);
      min-height: 100vh;
      padding: 24px;
    }
    .wrap {
      max-width: 1024px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 20px rgba(17, 24, 39, 0.05);
    }
    h1 { margin: 0 0 8px; font-size: 1.5rem; }
    p { margin: 0; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .field {
      display: grid;
      gap: 8px;
    }
    label { font-weight: 600; font-size: 0.92rem; }
    input[type="file"], input[type="range"], button {
      width: 100%;
    }
    .checks {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 6px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .btn {
      border: 0;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 700;
      padding: 11px 14px;
      cursor: pointer;
      transition: transform 0.15s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .status {
      margin-top: 10px;
      font-size: 0.92rem;
      color: var(--muted);
      min-height: 20px;
    }
    .error { color: var(--danger); }
    .result-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 320px));
      justify-content: start;
      gap: 12px;
    }
    .chip {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
      margin-right: 8px;
      margin-bottom: 8px;
      font-size: 0.88rem;
      background: #f8fafc;
    }
    .img-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fcfdff;
    }
    .img-card strong {
      display: block;
      margin-bottom: 6px;
      font-size: 0.9rem;
    }
    .img-card img {
      width: 75%;
      max-width: 304px;
      height: 220px;
      border-radius: 6px;
      display: block;
      margin: 0 auto;
      object-fit: contain;
      border: 1px solid #ebf0f4;
      background: #f2f7fb;
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="card">
      <h1>X-ray Classifier Tester</h1>
     

      <form id="predictForm" class="grid">
        <div class="field">
          <label for="file">Image file</label>
          <input id="file" name="file" type="file" accept="image/*" required />
        </div>
      </form>

      <div class="checks">
        <label class="check"><input id="showImage" type="checkbox" checked /> Include uploaded image</label>
        <label class="check"><input id="showCam" type="checkbox" checked /> Include Grad-CAM</label>
      </div>

      <div style="margin-top: 12px;">
        <button id="runBtn" class="btn" type="button">Run Prediction</button>
      </div>
      <div id="status" class="status"></div>
    </section>

    <section class="card">
      <div id="chips"></div>
      <div id="images" class="result-grid"></div>
    </section>
  </main>

  <script>
    const form = document.getElementById("predictForm");
    const fileInput = document.getElementById("file");
    const showImage = document.getElementById("showImage");
    const showCam = document.getElementById("showCam");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const chips = document.getElementById("chips");
    const images = document.getElementById("images");

    function setStatus(msg, isError = false) {
      statusEl.textContent = msg;
      statusEl.className = isError ? "status error" : "status";
    }

    function addImageCard(title, src) {
      if (!src) return;
      const card = document.createElement("div");
      card.className = "img-card";
      const h = document.createElement("strong");
      h.textContent = title;
      const img = document.createElement("img");
      img.src = src;
      img.alt = title;
      card.appendChild(h);
      card.appendChild(img);
      images.appendChild(card);
    }

    runBtn.addEventListener("click", async () => {
      if (!fileInput.files || fileInput.files.length === 0) {
        setStatus("Select an image first.", true);
        return;
      }

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      const params = new URLSearchParams({
        show_image: String(showImage.checked),
        show_cam: String(showCam.checked)
      });

      runBtn.disabled = true;
      setStatus("Running prediction...");
      chips.innerHTML = "";
      images.innerHTML = "";

      try {
        const res = await fetch(`/predict?${params.toString()}`, {
          method: "POST",
          body: formData
        });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || "Request failed");
        }

        const p = Number(data.probability);
        const pPct = Number.isFinite(p) ? `${(p * 100).toFixed(2)}%` : "-";
        const camChips = [];
        if (data.cam_thr != null) {
          camChips.push(`<span class="chip"><strong>cam_thr:</strong> ${data.cam_thr}</span>`);
        }
        if (data.mask_area_pct != null) {
          camChips.push(
            `<span class="chip"><strong>mask_area_pct:</strong> ${Number(data.mask_area_pct).toFixed(2)}%</span>`
          );
        }
        chips.innerHTML = `
          <span class="chip"><strong>Label:</strong> ${data.label ?? "-"}</span>
          <span class="chip"><strong>Probability:</strong> ${pPct}</span>
          ${camChips.join("")}
        `;

        addImageCard("Uploaded image", data.image_data_url);
        addImageCard("Input 224", data.input_224_data_url);
        addImageCard("Grad-CAM heatmap", data.cam_heatmap_data_url);
        addImageCard("Overlay", data.overlay_data_url);
        addImageCard("Overlay + Box", data.overlay_box_data_url);

        setStatus("Done.");
      } catch (err) {
        setStatus(err.message || "Unexpected error", true);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
