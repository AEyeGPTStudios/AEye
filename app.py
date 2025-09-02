# app.py
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, jsonify
import os, json, requests, traceback
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import torch, torch.nn.functional as F
from torchvision import models, transforms

# -------------------------
# Configuration
# -------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
CUSTOM_FILE = os.path.join(APP_ROOT, "custom_labels.json")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(CUSTOM_FILE):
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

# -------------------------
# Model & transforms
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = requests.get(labels_url).text.strip().split("\n")

base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
base_model.eval(); feature_extractor.eval()
base_model = base_model.to(device); feature_extractor = feature_extractor.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------
# Helpers
# -------------------------
def load_json():
    with open(CUSTOM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(d):
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

def is_image_content_type(ct):
    return ct and ct.split(";")[0].strip().startswith("image/")

def is_valid_image_url(url):
    try:
        r = requests.get(url, timeout=8, stream=True)
        if r.status_code != 200:
            return False
        ct = r.headers.get("content-type","")
        if not is_image_content_type(ct):
            return False
        # quick try to open
        Image.open(BytesIO(r.content)).verify()
        return True
    except Exception:
        return False

def load_image_from_source(source):
    # source: http(s) URL or local path (relative to APP_ROOT)
    try:
        if source.startswith("http://") or source.startswith("https://"):
            r = requests.get(source, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        else:
            path = os.path.join(APP_ROOT, source)
            img = Image.open(path).convert("RGB")
            return img
    except Exception as e:
        raise

def get_embedding(img):
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = feature_extractor(t)
    return f.squeeze()

def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -------------------------
# Compute embeddings & valid counts
# -------------------------
custom_labels = load_json()
custom_embeddings = {}
valid_counts = {}

def compute_custom_embeddings():
    global custom_embeddings, valid_counts
    labels = load_json()
    embeddings = {}
    counts = {}
    for cls, urls in labels.items():
        embs = []
        valid_list = []
        for u in urls:
            try:
                img = load_image_from_source(u)
                embs.append(get_embedding(img))
                valid_list.append(u)
            except Exception:
                # skip broken example
                continue
        if embs:
            embeddings[cls] = torch.stack(embs).mean(dim=0)
        counts[cls] = len(valid_list)
    custom_embeddings = embeddings
    valid_counts = counts
    return embeddings

compute_custom_embeddings()

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder=None)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# -------------------------
# HTML template (single file)
# -------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Finished AEye AI</title>
  <style>
    body{font-family:Arial;padding:18px;max-width:980px;margin:auto}
    .row{display:flex;gap:12px;align-items:center;margin-bottom:12px}
    input[type=text]{padding:8px;width:60%}
    select, textarea{padding:8px}
    .thumb{height:80px;margin:6px;border:1px solid #ddd;padding:4px}
    .class-row{margin-bottom:8px}
    #drop{border:2px dashed #bbb;padding:12px;text-align:center}
    .small{font-size:13px;color:#555}
  </style>
</head>
<body>
  <h1>Finished AEye AI</h1>

  <section>
    <h3>Predict (URL)</h3>
    <form method="post" action="/predict">
      <input name="url" placeholder="Paste image URL (must be direct image link)" required>
      <button>Predict</button>
    </form>
    {% if pred_img %}
      <div class="row"><img src="{{ pred_img }}" class="thumb"><div><b>Prediction:</b> {{ prediction }}</div></div>
    {% elif prediction %}
      <p><b>Prediction:</b> {{ prediction }}</p>
    {% endif %}
  </section>

  <hr>

  <section>
    <h3>Predict (paste image or drag & drop)</h3>
    <div id="drop" ondragover="event.preventDefault()" ondrop="handleDrop(event)">
      Drop image here or paste (Ctrl+V). Preview will show and prediction will run.
    </div>
    <div id="preview"></div>
  </section>

  <hr>

  <section>
    <h3>Add class by URL(s)</h3>
    <form method="post" action="/add">
      <input name="name" placeholder="Class name" required>
      <input name="urls" placeholder="Image URLs, comma separated" style="width:60%">
      <button>Add (URLs)</button>
    </form>
    <p class="small">Only valid image URLs are saved. Duplicates removed.</p>
  </section>

  <section>
    <h3>Add class by uploading files</h3>
    <form id="add-files-form" method="post" action="/add_files" enctype="multipart/form-data">
      <input name="name" placeholder="Class name" required>
      <input name="files" type="file" multiple accept="image/*">
      <button type="submit">Upload & Add</button>
    </form>
    <p class="small">Files saved inside server and added as examples.</p>
  </section>

  <hr>

  <section>
    <h3>Edit class (add/remove URLs)</h3>
    <form method="post" action="/edit">
      <select name="name">
        {% for c in classes %}<option value="{{c}}">{{c}}</option>{% endfor %}
      </select>
      <input name="add_urls" placeholder="Add URLs (comma separated)" style="width:50%">
      <input name="remove_urls" placeholder="Remove URLs (comma separated)" style="width:50%">
      <button>Update</button>
    </form>
  </section>

  <section>
    <h3>Delete class</h3>
    <form method="post" action="/delete">
      <select name="name">
        {% for c in classes %}<option value="{{c}}">{{c}}</option>{% endfor %}
      </select>
      <button>Delete</button>
    </form>
  </section>

  <hr>

  <section>
    <h3>Custom classes</h3>
    <button onclick="fetch('/clean',{method:'POST'}).then(()=>location.reload())">Clean broken links</button>
    <div>
      {% for cls, urls in labels.items() %}
        <div class="class-row">
          <b>{{cls}}</b> — stored: {{ urls|length }}, valid: {{ valid_counts.get(cls,0) }}
          <div>
            {% for u in urls %}
              <a href="{{ u }}" target="_blank"><img src="{{ u }}" class="thumb" onerror="this.style.opacity=0.35"></a>
            {% endfor %}
          </div>
        </div>
      {% endfor %}
    </div>
  </section>

<script>
async function postFileForPredict(file){
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch('/predict_file', { method:'POST', body: fd });
  return res.json();
}

function handleDrop(e){
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(f){ previewAndPredict(f); }
}

document.addEventListener('paste', (ev)=>{
  const items = ev.clipboardData && ev.clipboardData.items;
  if(!items) return;
  for(let i=0;i<items.length;i++){
    const it = items[i];
    if(it.kind === 'file'){
      const file = it.getAsFile();
      if(file) { previewAndPredict(file); return; }
    }
  }
});

async function previewAndPredict(file){
  const preview = document.getElementById('preview');
  preview.innerHTML = '';
  const img = document.createElement('img');
  img.className='thumb';
  img.src = URL.createObjectURL(file);
  preview.appendChild(img);
  const p = document.createElement('div');
  p.textContent = 'Predicting...';
  preview.appendChild(p);
  try{
    const json = await postFileForPredict(file);
    p.innerHTML = '<b>Prediction:</b> ' + (json.prediction || json.error);
  }catch(err){
    p.textContent = 'Error';
  }
}

document.getElementById('add-files-form').addEventListener('submit', async function(e){
  // allow normal submit; server handles it.
});

</script>
</body>
</html>
"""

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    labels = load_json()
    return render_template_string(HTML, labels=labels, classes=list(labels.keys()), valid_counts=valid_counts)

@app.route("/predict", methods=["POST"])
def route_predict():
    url = request.form.get("url","").strip()
    if not url:
        return redirect(url_for("home"))
    try:
        img = load_image_from_source(url)  # will raise on fail
        pred = _predict_from_image(img)
        # show image in template by using the URL directly (works for public URLs)
        return render_template_string(HTML, prediction=pred, pred_img=url, labels=load_json(), classes=list(load_json().keys()), valid_counts=valid_counts)
    except Exception as e:
        return render_template_string(HTML, prediction=f"Error: could not load image ({e})", labels=load_json(), classes=list(load_json().keys()), valid_counts=valid_counts)

@app.route("/predict_file", methods=["POST"])
def route_predict_file():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"no file"}), 400
    try:
        img = Image.open(f.stream).convert("RGB")
        pred = _predict_from_image(img)
        return jsonify({"prediction": pred})
    except UnidentifiedImageError:
        return jsonify({"error":"not an image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add", methods=["POST"])
def route_add():
    name = request.form.get("name","").strip()
    urls_raw = request.form.get("urls","").strip()
    urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
    if not name:
        return redirect(url_for("home"))
    labels = load_json()
    labels.setdefault(name, [])
    # validate URLs and keep only valid images
    kept = []
    for u in urls:
        try:
            if u.startswith("http://") or u.startswith("https://"):
                if is_valid_image_url(u):
                    kept.append(u)
            else:
                # reject non-http entries here
                continue
        except Exception:
            continue
    # add and dedupe
    labels[name].extend(kept)
    labels[name] = dedupe_preserve_order(labels[name])
    save_json(labels)
    compute_custom_embeddings()
    return redirect(url_for("home"))

@app.route("/add_files", methods=["POST"])
def route_add_files():
    name = request.form.get("name","").strip()
    files = request.files.getlist("files")
    if not name or not files:
        return redirect(url_for("home"))
    labels = load_json()
    labels.setdefault(name, [])
    class_dir = os.path.join(UPLOAD_FOLDER, secure_filename(name))
    os.makedirs(class_dir, exist_ok=True)
    for f in files:
        filename = secure_filename(f.filename)
        if not filename:
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            # try to allow by detecting content type
            pass
        save_path = os.path.join(class_dir, filename)
        # avoid overwrite by appending number
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(save_path):
            filename = f"{base}_{counter}{ext}"
            save_path = os.path.join(class_dir, filename)
            counter += 1
        f.save(save_path)
        rel = os.path.relpath(save_path, APP_ROOT).replace("\\","/")
        labels[name].append(rel)
    labels[name] = dedupe_preserve_order(labels[name])
    save_json(labels)
    compute_custom_embeddings()
    return redirect(url_for("home"))

@app.route("/edit", methods=["POST"])
def route_edit():
    name = request.form.get("name","").strip()
    add_raw = request.form.get("add_urls","").strip()
    rem_raw = request.form.get("remove_urls","").strip()
    labels = load_json()
    if name not in labels:
        return redirect(url_for("home"))
    add_list = [u.strip() for u in add_raw.split(",") if u.strip()]
    rem_list = [u.strip() for u in rem_raw.split(",") if u.strip()]
    # validate add_list URLs
    kept = []
    for u in add_list:
        try:
            if u.startswith("http://") or u.startswith("https://"):
                if is_valid_image_url(u):
                    kept.append(u)
            else:
                # not http -> ignore unless it's local upload path
                continue
        except Exception:
            continue
    labels[name].extend(kept)
    # remove exact matches
    if rem_list:
        labels[name] = [u for u in labels[name] if u not in rem_list]
    labels[name] = dedupe_preserve_order(labels[name])
    save_json(labels)
    compute_custom_embeddings()
    return redirect(url_for("home"))

@app.route("/delete", methods=["POST"])
def route_delete():
    name = request.form.get("name","").strip()
    labels = load_json()
    if name in labels:
        # optionally remove uploaded files (keep simple: do not delete files)
        del labels[name]
        save_json(labels)
        compute_custom_embeddings()
    return redirect(url_for("home"))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # serve uploaded files
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/clean", methods=["POST"])
def route_clean():
    # re-validate all saved urls and remove broken remote urls
    labels = load_json()
    changed = False
    for cls, urls in list(labels.items()):
        newlist = []
        for u in urls:
            if u.startswith("http://") or u.startswith("https://"):
                if is_valid_image_url(u):
                    newlist.append(u)
                else:
                    changed = True
            else:
                # local upload path, check file exists
                if os.path.exists(os.path.join(APP_ROOT, u)):
                    newlist.append(u)
                else:
                    changed = True
        labels[cls] = dedupe_preserve_order(newlist)
    if changed:
        save_json(labels)
    compute_custom_embeddings()
    return ("",204)

# -------------------------
# Prediction helper
# -------------------------
def _predict_from_image(img, custom_threshold=0.6):
    # img is PIL RGB
    emb = get_embedding(img)
    best_class, best_score = None, -1
    for cls, ref_emb in custom_embeddings.items():
        try:
            sim = F.cosine_similarity(emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            if sim > best_score:
                best_score, best_class = sim, cls
        except Exception:
            continue
    if best_score >= custom_threshold:
        return f"{best_class} (custom, sim={best_score:.2f})"
    # fallback ImageNet
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = base_model(t)
        _, p = out.max(1)
    return imagenet_labels[p.item()]

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    compute_custom_embeddings()
    app.run(debug=True)
