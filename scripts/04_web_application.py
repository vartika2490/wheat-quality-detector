import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template_string, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import base64
import datetime

# ── CONFIG ──────────────────────────────────────────────────
MODEL_PATH = r"E:\Wheat Detection Model\models\final_wheat_model.pth"
UPLOAD_DIR = r"E:\Wheat Detection Model\uploads"
REPORT_DIR = r"E:\Wheat Detection Model\reports"
IMG_SIZE   = 224
# ─────────────────────────────────────────────────────────────

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

app    = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INFO = {
    "healthy": {
        "grade":       "Grade A — Premium Quality",
        "color":       "#1D9E75",
        "bg":          "#E1F5EE",
        "border":      "#5DCAA5",
        "text":        "#085041",
        "decision":    "ACCEPT",
        "icon":        "circle-check",
        "description": "Plump, full-sized grain with uniform golden-amber color. No visible damage, cracks, mold, or discoloration. Meets premium quality standards.",
        "advice":      "Suitable for direct sale. Offer standard credit terms to buyer."
    },
    "broken": {
        "grade":       "Grade C — Broken Grain",
        "color":       "#BA7517",
        "bg":          "#FAEEDA",
        "border":      "#EF9F27",
        "text":        "#633806",
        "decision":    "CONDITIONAL",
        "icon":        "alert-triangle",
        "description": "Kernel is cracked or split into pieces. Starchy interior is exposed. Reduces flour yield and overall batch quality.",
        "advice":      "Accept at reduced price (15-25% discount). Not suitable for premium market."
    },
    "fungus_damage": {
        "grade":       "Grade D — Fungus Damaged",
        "color":       "#D85A30",
        "bg":          "#FAECE7",
        "border":      "#F0997B",
        "text":        "#4A1B0C",
        "decision":    "REJECT",
        "icon":        "virus",
        "description": "White or grey fuzzy patches on surface indicating fungal or mold growth. Can produce harmful mycotoxins.",
        "advice":      "Reject this batch. Do not offer credit. Risk of health hazard."
    },
    "insect_damage": {
        "grade":       "Grade D — Insect Damaged",
        "color":       "#D85A30",
        "bg":          "#FAECE7",
        "border":      "#F0997B",
        "text":        "#4A1B0C",
        "decision":    "REJECT",
        "icon":        "bug",
        "description": "Visible holes or tunnels on kernel surface caused by weevils or grain moths. Reduced nutritional value.",
        "advice":      "Reject this batch. Insect infestation may spread to stored grain."
    },
    "black_tip": {
        "grade":       "Grade C — Black Tip",
        "color":       "#BA7517",
        "bg":          "#FAEEDA",
        "border":      "#EF9F27",
        "text":        "#633806",
        "decision":    "CONDITIONAL",
        "icon":        "alert-circle",
        "description": "Dark discoloration at the grain tip caused by environmental stress or disease during growth.",
        "advice":      "Accept at reduced price (10-20% discount). Test flour quality before full acceptance."
    },
    "shriveled": {
        "grade":       "Grade C — Shriveled Grain",
        "color":       "#BA7517",
        "bg":          "#FAEEDA",
        "border":      "#EF9F27",
        "text":        "#633806",
        "decision":    "CONDITIONAL",
        "icon":        "arrow-autofit-down",
        "description": "Thin, wrinkled, lightweight kernels caused by drought or early frost. Low test weight and poor flour yield.",
        "advice":      "Accept at reduced price (20-30% discount). Low flour extraction rate expected."
    },
}

# ── LOAD MODEL ───────────────────────────────────────────────
print("Loading model...")
checkpoint  = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()
print(f"Model loaded! Classes: {class_names}")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    img  = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > h * 1.5:
        img = img.crop((0, 0, w // 2, h))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        conf, pred = probs.max(0)
    label     = class_names[pred.item()]
    all_probs = {class_names[i]: round(probs[i].item() * 100, 2)
                 for i in range(num_classes)}
    return label, round(conf.item() * 100, 2), all_probs

def generate_pdf(label, confidence, all_probs):
    info     = CLASS_INFO.get(label, CLASS_INFO["healthy"])
    filename = f"wheat_quality_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORT_DIR, filename)

    doc   = SimpleDocTemplate(filepath, pagesize=A4,
                              topMargin=0.6*inch, bottomMargin=0.6*inch,
                              leftMargin=0.8*inch, rightMargin=0.8*inch)
    story = []

    title_style = ParagraphStyle("title", fontSize=24, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#1a1a1a"),
                                 spaceAfter=4, alignment=0)
    sub_style   = ParagraphStyle("sub", fontSize=11,
                                 textColor=colors.HexColor("#888888"),
                                 spaceAfter=20, alignment=0)
    head_style  = ParagraphStyle("head", fontSize=12, fontName="Helvetica-Bold",
                                 textColor=colors.HexColor("#1a1a1a"),
                                 spaceBefore=16, spaceAfter=6)
    body_style  = ParagraphStyle("body", fontSize=11, leading=17,
                                 textColor=colors.HexColor("#444444"), spaceAfter=8)

    story.append(Paragraph("Wheat Quality Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%d %B %Y at %H:%M')}",
        sub_style))

    dec_colors = {
        "ACCEPT":      "#1D9E75",
        "CONDITIONAL": "#BA7517",
        "REJECT":      "#D85A30"
    }
    grade_color = colors.HexColor(dec_colors.get(info["decision"], "#888888"))

    grade_data  = [
        ["CLASSIFICATION", "DECISION", "CONFIDENCE"],
        [info["grade"], info["decision"], f"{confidence:.1f}%"]
    ]
    grade_table = Table(grade_data, colWidths=[2.8*inch, 2*inch, 1.5*inch])
    grade_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 10),
        ("BACKGROUND",  (0, 1), (-1, 1), grade_color),
        ("TEXTCOLOR",   (0, 1), (-1, 1), colors.white),
        ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, 1), 12),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 1, colors.white),
        ("ROWHEIGHT",   (0, 0), (-1, -1), 0.42*inch),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Physical Description", head_style))
    story.append(Paragraph(info["description"], body_style))
    story.append(Paragraph("Recommendation", head_style))
    story.append(Paragraph(info["advice"], body_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Confidence Breakdown", head_style))
    prob_data = [["Category", "Score"]]
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        prob_data.append([cls.replace("_", " ").title(), f"{prob:.1f}%"])

    prob_table = Table(prob_data, colWidths=[3.5*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),  colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ALIGN",      (1, 0), (1, -1),  "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("ROWHEIGHT",  (0, 0), (-1, -1), 0.3*inch),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8F8F8")),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "Wheat Quality Detection System — AI powered grain analysis",
        ParagraphStyle("footer", fontSize=9,
                       textColor=colors.HexColor("#AAAAAA"), alignment=1)
    ))

    doc.build(story)
    return filepath, filename

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wheat Quality Detector</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
           background: #FAFAF8; color: #1a1a1a; min-height: 100vh; }

    .nav { background: #1a1a1a; padding: 0 40px; height: 60px;
           display: flex; align-items: center; justify-content: space-between; }
    .nav-brand { font-size: 16px; font-weight: 500; color: #ffffff; letter-spacing: 0.3px; }
    .nav-tag { font-size: 11px; color: #888; border: 0.5px solid #444;
               padding: 3px 10px; border-radius: 20px; }

    .hero { background: #1a1a1a; padding: 60px 40px 50px; text-align: center; }
    .hero h1 { font-size: 42px; font-weight: 500; color: #ffffff;
               letter-spacing: -0.5px; margin-bottom: 12px; line-height: 1.2; }
    .hero p { font-size: 16px; color: #888888; max-width: 480px;
              margin: 0 auto; line-height: 1.6; }

    .main { max-width: 780px; margin: 0 auto; padding: 40px 24px 60px; }

    .upload-card { background: #ffffff; border: 0.5px solid #E8E8E0;
                   border-radius: 16px; overflow: hidden; margin-bottom: 24px; }
    .upload-inner { padding: 36px; }
    .drop-zone { border: 1.5px dashed #D8D8D0; border-radius: 12px;
                 padding: 48px 24px; text-align: center; cursor: pointer;
                 transition: all 0.2s; margin-bottom: 16px; }
    .drop-zone:hover { border-color: #1a1a1a; background: #FAFAF8; }
    .drop-icon { font-size: 36px; color: #BBBBBB; display: block; margin-bottom: 14px; }
    .drop-title { font-size: 16px; font-weight: 500; color: #1a1a1a; margin-bottom: 6px; }
    .drop-sub { font-size: 13px; color: #999; margin-bottom: 20px; }
    .file-name { font-size: 13px; color: #555; margin-bottom: 14px;
                 display: none; padding: 8px 14px; background: #F5F5F0;
                 border-radius: 8px; text-align: left; }
    .file-name i { font-size: 14px; vertical-align: -2px; margin-right: 6px; }
    .btn-browse { background: #ffffff; color: #1a1a1a; border: 1px solid #D8D8D0;
                  padding: 9px 22px; border-radius: 8px; font-size: 14px;
                  cursor: pointer; display: inline-flex; align-items: center; gap: 7px; }
    .btn-browse:hover { background: #F5F5F0; }
    .btn-analyze { width: 100%; background: #1a1a1a; color: #ffffff;
                   border: none; padding: 14px; border-radius: 10px;
                   font-size: 15px; font-weight: 500; cursor: pointer;
                   display: flex; align-items: center; justify-content: center; gap: 8px;
                   transition: background 0.2s; }
    .btn-analyze:hover { background: #333333; }

    .result-card { background: #ffffff; border: 0.5px solid #E8E8E0;
                   border-radius: 16px; overflow: hidden; margin-bottom: 24px; }
    .result-top { padding: 28px 28px 0; }
    .result-label { font-size: 11px; font-weight: 500; color: #999;
                    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .result-name { font-size: 28px; font-weight: 500; color: #1a1a1a;
                   margin-bottom: 4px; }
    .result-grade { font-size: 14px; color: #666; margin-bottom: 20px; }

    .result-meta { display: grid; grid-template-columns: 1fr 1fr 1fr;
                   gap: 1px; background: #E8E8E0; border-top: 0.5px solid #E8E8E0; }
    .meta-item { background: #ffffff; padding: 18px 20px; }
    .meta-label { font-size: 11px; color: #999; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 4px; }
    .meta-value { font-size: 20px; font-weight: 500; }

    .result-body { padding: 24px 28px; display: grid;
                   grid-template-columns: 1fr 1fr; gap: 24px; }
    .section-title { font-size: 11px; font-weight: 500; color: #999;
                     text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .section-text { font-size: 14px; color: #555; line-height: 1.7; }

    .advice-box { padding: 14px 16px; border-radius: 10px;
                  display: flex; align-items: flex-start; gap: 10px; }
    .advice-box i { font-size: 16px; margin-top: 2px; flex-shrink: 0; }

    .prob-item { display: flex; align-items: center; gap: 10px;
                 margin-bottom: 10px; }
    .prob-name { font-size: 12px; color: #666; width: 110px; flex-shrink: 0; }
    .prob-bar-wrap { flex: 1; background: #F0F0E8; border-radius: 4px; height: 5px; }
    .prob-bar { border-radius: 4px; height: 5px; background: #1a1a1a; }
    .prob-pct { font-size: 12px; font-weight: 500; color: #1a1a1a;
                width: 40px; text-align: right; flex-shrink: 0; }

    .img-wrap { padding: 0 28px 0; margin-bottom: 0; }
    .img-preview { width: 100%; max-height: 180px; object-fit: contain;
                   border-radius: 10px; border: 0.5px solid #E8E8E0;
                   background: #F8F8F8; }

    .pdf-section { padding: 20px 28px 24px; border-top: 0.5px solid #F0F0E8; }
    .btn-pdf { display: inline-flex; align-items: center; gap: 8px;
               background: #1a1a1a; color: #ffffff; text-decoration: none;
               padding: 11px 22px; border-radius: 9px; font-size: 14px; font-weight: 500; }
    .btn-pdf:hover { background: #333; }

    .footer { text-align: center; padding: 32px 24px;
              border-top: 0.5px solid #E8E8E0; }
    .footer p { font-size: 12px; color: #BBBBBB; }
  </style>
</head>
<body>

<nav class="nav">
  <span class="nav-brand">Wheat Quality Detector</span>
  <span class="nav-tag">AI Powered</span>
</nav>

<div class="hero">
  <h1>Grain quality,<br>instantly detected.</h1>
  <p>Upload a wheat grain image and get an instant AI-powered quality report with business recommendation.</p>
</div>

<div class="main">
  <div class="upload-card">
    <div class="upload-inner">
      <form method="POST" enctype="multipart/form-data" action="/predict" id="uploadForm">
        <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
          <i class="ti ti-cloud-upload drop-icon" aria-hidden="true"></i>
          <p class="drop-title">Upload grain image</p>
          <p class="drop-sub">JPG, PNG or BMP — scanner images recommended</p>
          <button type="button" class="btn-browse">
            <i class="ti ti-folder" aria-hidden="true"></i>
            Choose file
          </button>
        </div>
        <input type="file" name="image" accept="image/*" required
               id="fileInput" style="display:none"
               onchange="showFileName(this)">
        <div class="file-name" id="fileNameBox">
          <i class="ti ti-file-filled"></i>
          <span id="fileNameText"></span>
        </div>
        <button type="submit" class="btn-analyze">
          <i class="ti ti-sparkles" aria-hidden="true"></i>
          Analyze quality
        </button>
      </form>
    </div>
  </div>

  {% if result %}
  <div class="result-card">
    {% if img_data %}
    <div class="img-wrap" style="padding-top: 24px;">
      <img src="data:image/jpeg;base64,{{ img_data }}" class="img-preview" alt="Uploaded grain">
    </div>
    {% endif %}

    <div class="result-top">
      <p class="result-label">Detection result</p>
      <p class="result-name">{{ result.label.replace('_',' ').title() }}</p>
      <p class="result-grade">{{ result.grade }}</p>
    </div>

    <div class="result-meta">
      <div class="meta-item">
        <div class="meta-label">Confidence</div>
        <div class="meta-value" style="color: {{ result.color }};">{{ result.confidence }}%</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Decision</div>
        <div class="meta-value" style="font-size:15px; color: {{ result.color }};">{{ result.decision }}</div>
      </div>
      <div class="meta-item">
        <div class="meta-label">Scanned</div>
        <div class="meta-value" style="font-size:13px; color:#999;">{{ result.time }}</div>
      </div>
    </div>

    <div class="result-body">
      <div>
        <p class="section-title">Description</p>
        <p class="section-text">{{ result.description }}</p>

        <div class="advice-box" style="background: {{ result.bg }}; margin-top: 14px;">
          <i class="ti ti-info-circle" style="color: {{ result.color }};" aria-hidden="true"></i>
          <p style="font-size: 13px; color: {{ result.text }}; line-height: 1.6;">{{ result.advice }}</p>
        </div>
      </div>

      <div>
        <p class="section-title">Confidence breakdown</p>
        {% for cls, prob in result.all_probs.items()|sort(attribute='1', reverse=True) %}
        <div class="prob-item">
          <span class="prob-name">{{ cls.replace('_',' ').title() }}</span>
          <div class="prob-bar-wrap">
            <div class="prob-bar" style="width: {{ [prob, 100]|min }}%;
                 background: {% if prob > 50 %}{{ result.color }}{% else %}#D8D8D0{% endif %};"></div>
          </div>
          <span class="prob-pct">{{ prob }}%</span>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="pdf-section">
      <a href="/download/{{ result.pdf_name }}" class="btn-pdf">
        <i class="ti ti-file-download" aria-hidden="true"></i>
        Download quality report
      </a>
    </div>
  </div>
  {% endif %}
</div>

<div class="footer">
  <p>Wheat Quality Detector &nbsp;·&nbsp; AI powered grain analysis</p>
</div>

<script>
function showFileName(input) {
  if (input.files && input.files[0]) {
    document.getElementById('fileNameText').textContent = input.files[0].name;
    document.getElementById('fileNameBox').style.display = 'block';
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, result=None, img_data=None)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return "No image uploaded", 400

    file     = request.files["image"]
    img_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(img_path)

    label, confidence, all_probs = predict(img_path)
    info = CLASS_INFO.get(label, CLASS_INFO["healthy"])

    pdf_path, pdf_name = generate_pdf(label, confidence, all_probs)

    with open(img_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    result = {
        "label":       label,
        "confidence":  confidence,
        "grade":       info["grade"],
        "color":       info["color"],
        "bg":          info["bg"],
        "border":      info["border"],
        "text":        info["text"],
        "decision":    info["decision"],
        "description": info["description"],
        "advice":      info["advice"],
        "all_probs":   all_probs,
        "pdf_name":    pdf_name,
        "time":        datetime.datetime.now().strftime("%H:%M"),
    }
    return render_template_string(HTML, result=result, img_data=img_data)

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(REPORT_DIR, filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Wheat Quality Detector")
    print("="*50)
    print("  Open browser at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)