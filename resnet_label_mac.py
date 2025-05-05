# Load your fine‑tuned ResNet‑50, run inference on a folder or a .tar,
# and write out a CSV of (filename, pred_label).
import argparse, csv
from pathlib import Path
from io import BytesIO
import tarfile

import torch
from torchvision import transforms, models
from PIL import Image

def load_model(weights_path, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model

def process_folder(model, device, src, out_csv):
    pre = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "pred"])
        #if src is tar
        if src.suffix in {".tar", ".tgz", ".tar.gz"}:
            data = src.read_bytes()
            with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    buf = tf.extractfile(m).read()
                    img = Image.open(BytesIO(buf)).convert("RGB")
                    x = pre(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p = torch.argmax(model(x), 1).item()
                    writer.writerow([m.name, p])
        else:
            #assuming itis  a directory of .jpg/.jpeg
            for img_path in sorted(src.glob("*.jp*g")):
                img = Image.open(img_path).convert("RGB")
                x = pre(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.argmax(model(x), 1).item()
                writer.writerow([img_path.name, p])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to resnet50_ft.pth")
    parser.add_argument("--src", required=True,
                        help="Folder or .tar of images to label")
    parser.add_argument("--out", default="resnet_labels.csv",
                        help="CSV output path")
    args = parser.parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = load_model(args.weights, device)
    src = Path(args.src)
    process_folder(model, device, src, args.out)
    print(f"Written predictions to {args.out}")

if __name__ == "__main__":
    main()
