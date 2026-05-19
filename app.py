import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io

st.set_page_config(
    page_title="GrainScan AI — Wheat Quality Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F5F2EC;
    color: #1C1C1C;
}

footer, #MainMenu, header, .stDeployButton { display: none !important; }

/* ── NAVBAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    background: #FFFFFF;
    border-bottom: 1px solid #E8E4DC;
    position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1C1C1C;
    letter-spacing: -0.3px;
}
.nav-logo span { color: #2D7A4F; }
.nav-links {
    display: flex; gap: 32px;
    list-style: none;
}
.nav-links li a {
    font-size: 14px; font-weight: 500;
    color: #6B6B6B; text-decoration: none;
    letter-spacing: 0.1px;
}
.nav-badge {
    background: #2D7A4F;
    color: white; font-size: 12px; font-weight: 500;
    padding: 7px 18px; border-radius: 100px;
    letter-spacing: 0.2px;
}

/* ── HERO ── */
.hero-section {
    max-width: 1100px;
    margin: 72px auto 56px;
    padding: 0 48px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 64px;
    align-items: center;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 7px;
    background: #EEF7F1;
    border: 1px solid #C2DECE;
    color: #2D7A4F; font-size: 12px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    padding: 6px 14px; border-radius: 100px;
    margin-bottom: 22px;
}
.hero-tag::before { content: '●'; font-size: 8px; }
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 52px; line-height: 1.1;
    letter-spacing: -1px; color: #1C1C1C;
    margin-bottom: 18px;
}
.hero-title em { font-style: italic; color: #2D7A4F; }
.hero-desc {
    font-size: 16px; color: #6B6B6B; line-height: 1.75;
    font-weight: 300; max-width: 420px; margin-bottom: 32px;
}
.hero-stats {
    display: flex; gap: 32px;
}
.hero-stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 28px; color: #1C1C1C; display: block;
}
.hero-stat-label {
    font-size: 12px; color: #9B9B9B; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.hero-visual {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 4px 40px rgba(0,0,0,0.06);
}
.hero-visual-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; color: #9B9B9B; margin-bottom: 16px;
}
.grade-pills {
    display: flex; flex-wrap: wrap; gap: 8px;
}
.grade-pill {
    font-size: 13px; font-weight: 500;
    padding: 8px 16px; border-radius: 100px;
    display: inline-flex; align-items: center; gap: 6px;
}
.pill-a { background: #EEF7F1; color: #2D7A4F; border: 1px solid #C2DECE; }
.pill-b { background: #FFF8EE; color: #C97B00; border: 1px solid #F0D490; }
.pill-c { background: #FEF0F0; color: #C0392B; border: 1px solid #F5C6C6; }

/* ── UPLOAD SECTION ── */
.upload-section {
    max-width: 1100px;
    margin: 0 auto 80px;
    padding: 0 48px;
}
.section-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 2px;
    color: #2D7A4F; margin-bottom: 10px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 34px; color: #1C1C1C;
    letter-spacing: -0.5px; margin-bottom: 8px;
}
.section-sub {
    font-size: 15px; color: #6B6B6B; font-weight: 300;
    margin-bottom: 36px; line-height: 1.6;
}
.upload-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    align-items: start;
}
.upload-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}
.upload-card-title {
    font-size: 16px; font-weight: 600;
    color: #1C1C1C; margin-bottom: 6px;
}
.upload-card-sub {
    font-size: 13px; color: #9B9B9B;
    margin-bottom: 20px; line-height: 1.5;
}
.info-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}
.info-card-header {
    font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #9B9B9B; margin-bottom: 20px;
}
.quality-row {
    display: flex; align-items: flex-start;
    gap: 14px; padding: 14px 0;
    border-bottom: 1px solid #F0EDE7;
}
.quality-row:last-child { border-bottom: none; }
.q-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.q-icon-green { background: #EEF7F1; }
.q-icon-amber { background: #FFF8EE; }
.q-icon-red   { background: #FEF0F0; }
.q-name { font-size: 14px; font-weight: 600; color: #1C1C1C; }
.q-desc { font-size: 12px; color: #9B9B9B; margin-top: 2px; }

/* ── RESULT SECTION ── */
.result-wrapper {
    max-width: 1100px;
    margin: 0 auto 80px;
    padding: 0 48px;
}
.result-header {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 2px;
    color: #2D7A4F; margin-bottom: 24px;
}
.result-grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 24px;
}
.result-verdict-card {
    border-radius: 20px;
    padding: 32px;
    color: white;
}
.verdict-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 2px;
    opacity: 0.65; margin-bottom: 10px;
}
.verdict-name {
    font-family: 'DM Serif Display', serif;
    font-size: 34px; line-height: 1.1;
    margin-bottom: 6px;
}
.verdict-grade {
    font-size: 13px; opacity: 0.7;
    margin-bottom: 28px;
}
.verdict-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    padding: 8px 18px; border-radius: 100px;
    font-size: 14px; font-weight: 600;
}
.verdict-meta {
    margin-top: 24px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.15);
    display: flex; gap: 24px;
}
.vmeta-num {
    font-family: 'DM Serif Display', serif;
    font-size: 26px; display: block;
}
.vmeta-lbl {
    font-size: 11px; opacity: 0.55;
    text-transform: uppercase; letter-spacing: 1px;
}
.result-details-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}
.detail-section-title {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    color: #9B9B9B; margin-bottom: 14px;
}
.detail-text {
    font-size: 14px; color: #4A4A4A; line-height: 1.75;
    margin-bottom: 24px; padding-bottom: 24px;
    border-bottom: 1px solid #F0EDE7;
}
.recommendation-box {
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 24px;
}
.rec-title {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.rec-text { font-size: 14px; line-height: 1.65; }
.conf-section { margin-top: 4px; }
.conf-item { margin-bottom: 12px; }
.conf-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
.conf-name { font-size: 13px; color: #4A4A4A; }
.conf-pct  { font-size: 13px; font-weight: 600; color: #1C1C1C;
             font-family: 'JetBrains Mono', monospace; }
.conf-bar-bg {
    background: #F0EDE7; border-radius: 4px;
    height: 5px; overflow: hidden;
}
.conf-bar-fill { height: 5px; border-radius: 4px; }

/* ── FOOTER ── */
.site-footer {
    background: #1C1C1C;
    color: rgba(255,255,255,0.35);
    text-align: center;
    padding: 32px 48px;
    font-size: 13px;
    letter-spacing: 0.2px;
}
.site-footer span { color: rgba(255,255,255,0.6); }

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: #F9F7F4 !important;
    border: 2px dashed #D8D3CA !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] label { color: #9B9B9B !important; }
[data-testid="stFileUploader"] section { padding: 20px !important; }

.stButton > button {
    background: #2D7A4F !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.1px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #235f3d !important; }

.stDownloadButton > button {
    background: #1C1C1C !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 12px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stDownloadButton > button:hover { background: #333 !important; }

[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid #E8E4DC !important;
}

div[data-testid="column"] { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── CLASS INFO ──────────────────────────────────────────────
CLASS_INFO = {
    "healthy": {
        "grade": "Grade A — Premium Quality",
        "color": "#2D7A4F",
        "hero_bg": "linear-gradient(145deg, #1a4d30, #2D7A4F)",
        "box_bg": "#EEF7F1", "box_border": "#C2DECE", "box_text": "#1a4d30",
        "rec_bg": "#EEF7F1", "rec_border": "#C2DECE", "rec_text": "#1a4d30",
        "decision": "✅ ACCEPT", "bar_color": "#2D7A4F",
        "description": "Plump, full-sized grain with uniform golden-amber color. No visible damage, cracks, mold, or discoloration. Meets premium quality standards.",
        "advice": "Suitable for direct sale at market rate. Offer standard credit terms to buyer."
    },
    "broken": {
        "grade": "Grade C — Broken Grain",
        "color": "#C97B00",
        "hero_bg": "linear-gradient(145deg, #6b3d00, #C97B00)",
        "box_bg": "#FFF8EE", "box_border": "#F0D490", "box_text": "#6b3d00",
        "rec_bg": "#FFF8EE", "rec_border": "#F0D490", "rec_text": "#6b3d00",
        "decision": "⚠️ CONDITIONAL", "bar_color": "#C97B00",
        "description": "Kernel is cracked or split into pieces. Starchy interior is exposed, reducing flour yield and overall batch quality.",
        "advice": "Accept at reduced price (15–25% discount). Not suitable for premium market."
    },
    "fungus_damage": {
        "grade": "Grade D — Fungus Damaged",
        "color": "#C0392B",
        "hero_bg": "linear-gradient(145deg, #6b0f0f, #C0392B)",
        "box_bg": "#FEF0F0", "box_border": "#F5C6C6", "box_text": "#6b0f0f",
        "rec_bg": "#FEF0F0", "rec_border": "#F5C6C6", "rec_text": "#6b0f0f",
        "decision": "❌ REJECT", "bar_color": "#C0392B",
        "description": "White or grey fuzzy patches indicating fungal or mold growth. Can produce harmful mycotoxins dangerous to health.",
        "advice": "Reject this batch immediately. Do not offer credit. Risk of serious health hazard."
    },
    "insect_damage": {
        "grade": "Grade D — Insect Damaged",
        "color": "#C0392B",
        "hero_bg": "linear-gradient(145deg, #6b0f0f, #C0392B)",
        "box_bg": "#FEF0F0", "box_border": "#F5C6C6", "box_text": "#6b0f0f",
        "rec_bg": "#FEF0F0", "rec_border": "#F5C6C6", "rec_text": "#6b0f0f",
        "decision": "❌ REJECT", "bar_color": "#C0392B",
        "description": "Visible holes or tunnels on kernel surface caused by weevils or grain moths. Reduced nutritional value.",
        "advice": "Reject this batch. Insect infestation may spread to all stored grain."
    },
    "black_tip": {
        "grade": "Grade C — Black Tip",
        "color": "#C97B00",
        "hero_bg": "linear-gradient(145deg, #6b3d00, #C97B00)",
        "box_bg": "#FFF8EE", "box_border": "#F0D490", "box_text": "#6b3d00",
        "rec_bg": "#FFF8EE", "rec_border": "#F0D490", "rec_text": "#6b3d00",
        "decision": "⚠️ CONDITIONAL", "bar_color": "#C97B00",
        "description": "Dark discoloration at the grain tip caused by environmental stress or disease during growth.",
        "advice": "Accept at reduced price (10–20% discount). Test flour quality before full acceptance."
    },
    "shriveled": {
        "grade": "Grade C — Shriveled Grain",
        "color": "#C97B00",
        "hero_bg": "linear-gradient(145deg, #6b3d00, #C97B00)",
        "box_bg": "#FFF8EE", "box_border": "#F0D490", "box_text": "#6b3d00",
        "rec_bg": "#FFF8EE", "rec_border": "#F0D490", "rec_text": "#6b3d00",
        "decision": "⚠️ CONDITIONAL", "bar_color": "#C97B00",
        "description": "Thin, wrinkled, lightweight kernels caused by drought or early frost. Low test weight and poor flour yield.",
        "advice": "Accept at reduced price (20–30% discount). Low flour extraction rate expected."
    },
}

# ── MODEL ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import os
    import gdown
    MODEL_PATH = "final_wheat_model.pth"
    if not os.path.exists(MODEL_PATH):
        file_id = "1i5rOIpBNjeXc_GW3C2sTroh8g5eJBbKJ"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.fc.in_features, 512),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, class_names, device

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(img, model, class_names, device):
    w, h = img.size
    if w > h * 1.5:
        img = img.crop((0, 0, w // 2, h))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        conf, pred = probs.max(0)
    label     = class_names[pred.item()]
    all_probs = {class_names[i]: round(probs[i].item() * 100, 2) for i in range(len(class_names))}
    return label, round(conf.item() * 100, 2), all_probs

def build_conf_bars(all_probs, info):
    max_prob  = max(all_probs.values())
    bar_main  = info["bar_color"]
    bars = ""
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar_color = bar_main if prob == max_prob else "#D8D3CA"
        cls_name  = cls.replace('_', ' ').title()
        width     = min(prob, 100)
        bars += f"""
        <div class="conf-item">
            <div class="conf-row">
                <span class="conf-name">{cls_name}</span>
                <span class="conf-pct">{prob}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{width}%; background:{bar_color};"></div>
            </div>
        </div>"""
    return bars

def generate_pdf(label, confidence, all_probs):
    info   = CLASS_INFO.get(label, CLASS_INFO["healthy"])
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               topMargin=0.7*inch, bottomMargin=0.7*inch,
                               leftMargin=0.9*inch, rightMargin=0.9*inch)
    story  = []
    t_style = ParagraphStyle("t", fontSize=26, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1a1a1a"), spaceAfter=16)
    s_style = ParagraphStyle("s", fontSize=11, textColor=colors.HexColor("#888888"), spaceAfter=24)
    h_style = ParagraphStyle("h", fontSize=12, fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#1a1a1a"), spaceBefore=18, spaceAfter=6)
    b_style = ParagraphStyle("b", fontSize=11, leading=18,
                              textColor=colors.HexColor("#444444"), spaceAfter=8)
    story.append(Paragraph("Wheat Quality Report", t_style))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%d %B %Y at %H:%M')}", s_style))
    dec_map     = {"✅ ACCEPT": "#2D7A4F", "⚠️ CONDITIONAL": "#C97B00", "❌ REJECT": "#C0392B"}
    grade_color = colors.HexColor(dec_map.get(info["decision"], "#888888"))
    clean_dec   = info["decision"].replace("✅ ","").replace("⚠️ ","").replace("❌ ","")
    gt = Table([["CLASSIFICATION","DECISION","CONFIDENCE"],[info["grade"],clean_dec,f"{confidence:.1f}%"]],
               colWidths=[2.8*inch,2*inch,1.5*inch])
    gt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),10),
        ("BACKGROUND",(0,1),(-1,1),grade_color),
        ("TEXTCOLOR",(0,1),(-1,1),colors.white),
        ("FONTNAME",(0,1),(-1,1),"Helvetica-Bold"),
        ("FONTSIZE",(0,1),(-1,1),13),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("GRID",(0,0),(-1,-1),1,colors.white),
        ("ROWHEIGHT",(0,0),(-1,-1),0.44*inch),
    ]))
    story.append(gt)
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph("Physical Description", h_style))
    story.append(Paragraph(info["description"], b_style))
    story.append(Paragraph("Business Recommendation", h_style))
    story.append(Paragraph(info["advice"], b_style))
    story.append(Spacer(1,0.2*inch))
    story.append(Paragraph("Confidence Breakdown", h_style))
    pd_data = [["Category","Score"]]
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        pd_data.append([cls.replace("_"," ").title(), f"{prob:.1f}%"])
    pt = Table(pd_data, colWidths=[3.5*inch,1.5*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ALIGN",(1,0),(1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#E0E0E0")),
        ("ROWHEIGHT",(0,0),(-1,-1),0.3*inch),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#F8F8F8")),
    ]))
    story.append(pt)
    story.append(Spacer(1,0.5*inch))
    story.append(Paragraph("GrainScan AI — Wheat Quality Detection System",
                            ParagraphStyle("f", fontSize=9, textColor=colors.HexColor("#AAAAAA"), alignment=1)))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ── MAIN ────────────────────────────────────────────────────
def main():

    # NAVBAR
    st.markdown("""
    <div class="navbar">
        <div class="nav-logo">Grain<span>Scan</span> AI</div>
        <ul class="nav-links">
            <li><a href="#">How it works</a></li>
            <li><a href="#">Grades</a></li>
            <li><a href="#">About</a></li>
        </ul>
        <div class="nav-badge">🌾 Powered by ResNet50</div>
    </div>
    """, unsafe_allow_html=True)

    # HERO
    st.markdown("""
    <div class="hero-section">
        <div>
            <div class="hero-tag">AI Quality Detection</div>
            <h1 class="hero-title">Instant wheat<br>quality <em>analysis</em></h1>
            <p class="hero-desc">
                Upload a grain image and receive a precise AI-powered quality classification
                with a business recommendation in under 2 seconds.
            </p>
            <div class="hero-stats">
                <div>
                    <span class="hero-stat-num">99%</span>
                    <span class="hero-stat-label">Accuracy</span>
                </div>
                <div>
                    <span class="hero-stat-num">6</span>
                    <span class="hero-stat-label">Categories</span>
                </div>
                <div>
                    <span class="hero-stat-num">&lt;2s</span>
                    <span class="hero-stat-label">Analysis</span>
                </div>
            </div>
        </div>
        <div class="hero-visual">
            <p class="hero-visual-label">Detectable quality grades</p>
            <div class="grade-pills">
                <span class="grade-pill pill-a">✓ Healthy</span>
                <span class="grade-pill pill-b">⚠ Broken</span>
                <span class="grade-pill pill-b">⚠ Black Tip</span>
                <span class="grade-pill pill-b">⚠ Shriveled</span>
                <span class="grade-pill pill-c">✕ Fungus Damage</span>
                <span class="grade-pill pill-c">✕ Insect Damage</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model silently
    with st.spinner(""):
        model, class_names, device = load_model()

    # UPLOAD SECTION
    st.markdown("""
    <div class="upload-section">
        <p class="section-label">Step 1 — Upload</p>
        <h2 class="section-title">Analyze your grain sample</h2>
        <p class="section-sub">Upload a clear, close-up photo of wheat kernels. JPG, PNG, or BMP supported.</p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 22, 1])
    with center:
        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            st.markdown("""
            <div class="upload-card">
                <p class="upload-card-title">Upload grain image</p>
                <p class="upload-card-sub">Drag and drop or click to browse from your device</p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader("", type=["jpg","jpeg","png","bmp"], label_visibility="collapsed")

            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🔍   Analyze grain quality"):
                    with st.spinner("Running AI analysis..."):
                        label, confidence, all_probs = predict(img, model, class_names, device)
                    st.session_state["result"] = (label, confidence, all_probs)

        with right_col:
            st.markdown("""
            <div class="info-card">
                <p class="info-card-header">Quality categories</p>
                <div class="quality-row">
                    <div class="q-icon q-icon-green">✓</div>
                    <div>
                        <p class="q-name">Healthy</p>
                        <p class="q-desc">Premium, full-sized, undamaged grain</p>
                    </div>
                </div>
                <div class="quality-row">
                    <div class="q-icon q-icon-amber">⚠</div>
                    <div>
                        <p class="q-name">Broken</p>
                        <p class="q-desc">Cracked or split kernel, reduced yield</p>
                    </div>
                </div>
                <div class="quality-row">
                    <div class="q-icon q-icon-amber">⚠</div>
                    <div>
                        <p class="q-name">Black Tip</p>
                        <p class="q-desc">Dark discoloration at kernel tip</p>
                    </div>
                </div>
                <div class="quality-row">
                    <div class="q-icon q-icon-amber">⚠</div>
                    <div>
                        <p class="q-name">Shriveled</p>
                        <p class="q-desc">Thin, wrinkled, low test weight</p>
                    </div>
                </div>
                <div class="quality-row">
                    <div class="q-icon q-icon-red">✕</div>
                    <div>
                        <p class="q-name">Fungus Damage</p>
                        <p class="q-desc">Mold growth, mycotoxin risk</p>
                    </div>
                </div>
                <div class="quality-row">
                    <div class="q-icon q-icon-red">✕</div>
                    <div>
                        <p class="q-name">Insect Damage</p>
                        <p class="q-desc">Weevil or grain moth infestation</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # RESULTS
    if "result" in st.session_state:
        label, confidence, all_probs = st.session_state["result"]
        info = CLASS_INFO.get(label, CLASS_INFO["healthy"])
        conf_bars = build_conf_bars(all_probs, info)
        clean_label = label.replace('_', ' ').title()
        scan_time   = datetime.datetime.now().strftime('%H:%M')

        st.markdown(f"""
        <div class="result-wrapper">
            <p class="result-header">Step 2 — Results</p>
            <div class="result-grid">
                <div class="result-verdict-card" style="background:{info['hero_bg']};">
                    <p class="verdict-label">Detection result</p>
                    <p class="verdict-name">{clean_label}</p>
                    <p class="verdict-grade">{info['grade']}</p>
                    <span class="verdict-badge">{info['decision']}</span>
                    <div class="verdict-meta">
                        <div>
                            <span class="vmeta-num">{confidence}%</span>
                            <span class="vmeta-lbl">Confidence</span>
                        </div>
                        <div>
                            <span class="vmeta-num">{scan_time}</span>
                            <span class="vmeta-lbl">Scanned at</span>
                        </div>
                    </div>
                </div>
                <div class="result-details-card">
                    <p class="detail-section-title">Physical description</p>
                    <p class="detail-text">{info['description']}</p>
                    <p class="detail-section-title">Business recommendation</p>
                    <div class="recommendation-box" style="background:{info['rec_bg']};border:1px solid {info['rec_border']};">
                        <p class="rec-title" style="color:{info['color']};">Action required</p>
                        <p class="rec-text" style="color:{info['rec_text']};">{info['advice']}</p>
                    </div>
                    <div class="conf-section">
                        <p class="detail-section-title">Confidence breakdown</p>
                        {conf_bars}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, center2, _ = st.columns([1, 22, 1])
        with center2:
            _, dl_col, _ = st.columns([1, 2, 1])
            with dl_col:
                pdf_buffer = generate_pdf(label, confidence, all_probs)
                st.download_button(
                    label="📄   Download quality report (PDF)",
                    data=pdf_buffer,
                    file_name=f"grainscan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

    # FOOTER
    st.markdown("""
    <div class="site-footer">
        <span>GrainScan AI</span> &nbsp;·&nbsp; Wheat Quality Detection System
        &nbsp;·&nbsp; Built with ResNet50 Transfer Learning &nbsp;·&nbsp;
        Richsoil Agri-Tech
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()