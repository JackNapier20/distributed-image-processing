#!/usr/bin/env python3
# Here we train the simple 2‑class CNN defined in cnn.py and report accuracy and precision.

import argparse, os, time, random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score   

from cnn import CNN  

# 1.  custom dataset to infer the label from the file‑name prefix (cat or dog)
# images in jpg or jpeg format
class CatsDogsFiles(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.paths = sorted([p for p in self.root.glob("*.jp*g")])   # jpg or jpeg
        if len(self.paths) == 0:
            raise RuntimeError(f"No *.jpg images found in {root_dir}")
        self.transform = transform

    def __len__(self):  return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # here 0 = cat, 1 = dog
        label = 0 if path.name.startswith("cat") else 1              
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# 2. Training/evaluation helper functions
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        pred = torch.argmax(outputs, 1).cpu()
        preds.extend(pred.tolist())
        gts.extend(labels.tolist())

    acc  = accuracy_score(gts, preds)
    prec = precision_score(gts, preds, pos_label=1, average="binary")
    return acc, prec

# 3. Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Folder with the cat and dog Kaggle training images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of images separated for validation")
    parser.add_argument("--lr", type=float, default=1e-3)  
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="cnn.pth",
                        help="Where to save the trained weights")
    parser.add_argument("--resume", default=None,
                        help="Path to an existing .pth file to resume from")
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    # Transforms to same resolution and normalization as required by the inference code 
    tfm_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    full_ds = CatsDogsFiles(args.data_dir, transform=tfm_train)
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(args.seed))
    # validation uses its own transform 
    val_ds.dataset.transform = tfm_val

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training: {train_len} images for training, {val_len} for validation")
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_prec = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"loss={train_loss:.4f}  val_accuracy={val_acc:.4f}  val_precision={val_prec:.4f}  "
              f"time={elapsed:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"New best model saved to {args.output}")

    print("Training finished – best val accuracy "
          f"{best_acc*100:.2f}% | weights saved to {args.output}")

if __name__ == "__main__":
    main()
