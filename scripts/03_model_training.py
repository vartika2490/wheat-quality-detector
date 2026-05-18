import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── CONFIG ──────────────────────────────────────────────────
FULL_CSV = r"E:\Wheat Detection Model\correct_labels.csv"
MODEL_DIR  = r"E:\Wheat Detection Model\models"
REPORT_DIR = r"E:\Wheat Detection Model\reports"
 # per class

FULL_CSV   = r"E:\Wheat Detection Model\correct_labels.csv"
SAMPLES    = 1000
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 0.0001
IMG_SIZE   = 224
# ─────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}")
print(f"  Wheat Grain — Fast Retraining")
print(f"{'='*50}")
print(f"  Device : {device}")

# ── LOAD AND SAMPLE CSV ──────────────────────────────────────
print(f"\n  Loading CSV...")
df = pd.read_csv(FULL_CSV)
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {df.columns.tolist()}")


print(f"\n  Sampled: {len(df):,} images")
print(df["label"].value_counts().to_string())

# ── ENCODE LABELS ────────────────────────────────────────────
le             = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
class_names    = list(le.classes_)
num_classes    = len(class_names)

print(f"\n  Classes:")
for i, name in enumerate(class_names):
    print(f"    {i} → {name}")

# ── SPLIT ────────────────────────────────────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_id"], random_state=42)
val_df, test_df   = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

print(f"\n  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}\n")

# ── TRANSFORMS ───────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── DATASET ──────────────────────────────────────────────────
class WheatDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["filepath"]).convert("RGB")
        except:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return self.transform(image), int(row["label_id"])

train_loader = DataLoader(WheatDataset(train_df, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(WheatDataset(val_df,   val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(WheatDataset(test_df,  val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── MODEL ────────────────────────────────────────────────────
print("  Loading pretrained ResNet50...")
model = models.resnet50(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model = model.to(device)
print(f"  Model ready — {num_classes} classes\n")

# ── TRAIN ────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0.0
print(f"{'='*50}")
print(f"  Training {EPOCHS} epochs on {len(train_df)} images")
print(f"  Expected: 20-30 minutes total")
print(f"{'='*50}\n")

for epoch in range(EPOCHS):
    model.train()
    t_loss = t_correct = t_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item()
        t_correct += (out.argmax(1) == labels).sum().item()
        t_total   += labels.size(0)

    model.eval()
    v_correct = v_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out        = model(images)
            v_correct += (out.argmax(1) == labels).sum().item()
            v_total   += labels.size(0)

    train_acc = t_correct / t_total
    val_acc   = v_correct / v_total
    scheduler.step()

    print(f"  Epoch {epoch+1:02d}/{EPOCHS} "
          f"| Train Loss: {t_loss/len(train_loader):.4f} "
          f"| Train Acc: {train_acc:.2%} "
          f"| Val Acc: {val_acc:.2%}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "class_names": class_names,
            "label_map":   {i: n for i, n in enumerate(class_names)}
        }, os.path.join(MODEL_DIR, "final_wheat_model.pth"))
        print(f"  ✓ Saved — Val Acc: {best_val_acc:.2%}\n")

# ── TEST ─────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Test Set Evaluation")
print(f"{'='*50}\n")

ck = torch.load(os.path.join(MODEL_DIR, "final_wheat_model.pth"),
                map_location=device)
model.load_state_dict(ck["model_state"])
model.eval()

preds, actuals = [], []
with torch.no_grad():
    for images, labels in test_loader:
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        actuals.extend(labels.numpy())

print(classification_report(actuals, preds, target_names=class_names))

print(f"\n{'='*50}")
print(f"  Done! Best Val Accuracy: {best_val_acc:.2%}")
print(f"  Model: {MODEL_DIR}\\final_wheat_model.pth")
print(f"{'='*50}\n")