# fine tuning a pretrained ResNet‑50 on Cats vs Dogs dataset, training only the final FC layer
import argparse, os, time, random
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score

class CatsDogsFiles(Dataset):
    def __init__(self, root_dir: str, transform=None, ext: str = "jpg|jpeg"):
        self.root = Path(root_dir)
        self.transform = transform
        self.paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower().lstrip(".") in ext.lower().split("|")
        )
        if not self.paths:
            raise RuntimeError(f"No *.{ext} images found in {root_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name_l = path.name.lower()
        parent_l = path.parent.name.lower()
        if name_l.startswith("cat") or "cat" in parent_l:
            label = 0
        elif name_l.startswith("dog") or "dog" in parent_l:
            label = 1
        else:
            raise RuntimeError(f"Cannot infer label for {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def trainOneEpoch(model, loader, criterion, optimizer, device):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Folder with cat.* / dog.* images (or sub‑folders)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for fine‑tuning the FC layer")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="resnet50_ft.pth",
                        help="Where to save the best weights")
    parser.add_argument("--ext", default="jpg|jpeg",
                        help="Image extensions (pipe‑separated)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tfm_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    # splitting dataset
    full_ds = CatsDogsFiles(args.data_dir, transform=tfm_train, ext=args.ext)
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(
        full_ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_ds.dataset.transform = tfm_val

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    #replacing final FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    # putting only FC params in optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    print(f"Fine‑tuning ResNet‑50 on {train_len} imgs, validating on {val_len} imgs")
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss = trainOneEpoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_prec = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"loss={train_loss:.4f}  val_accuracy={val_acc:.4f}  val_precision={val_prec:.4f}  "
              f"time={elapsed:.1f}s")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model to {args.output}")

    print(f"Best val accuracy {best_acc*100:.2f}% | weights in {args.output}")

if __name__ == "__main__":
    main()
