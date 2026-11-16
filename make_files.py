"""Backup: train the demo CNN on your own laptop and create cnn_mnist.pt + mnist_samples.npz.
Run once in this folder:   py -m pip install torchvision   then   py make_files.py
It downloads MNIST (about 10 MB) and trains for about 2-3 minutes on CPU."""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
import faults

torch.manual_seed(0)
tr = datasets.MNIST("mnist_data", train=True, download=True)
te = datasets.MNIST("mnist_data", train=False, download=True)
Xtr = tr.data.float().div(255).unsqueeze(1)
ytr = tr.targets
Xte = te.data.float().div(255).unsqueeze(1)
yte = te.targets

model = faults.make_cnn()
opt = torch.optim.Adam(model.parameters(), 1e-3)
for epoch in range(3):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()
        F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    model.eval()
    print(f"epoch {epoch + 1}: test accuracy {faults.accuracy(model, Xte, yte.numpy()):.2f}%")

torch.save(model.state_dict(), "cnn_mnist.pt")
np.savez("mnist_samples.npz", x=te.data[:500].numpy().astype(np.uint8), y=yte[:500].numpy())
print("Saved cnn_mnist.pt and mnist_samples.npz. Now run: py -m streamlit run app.py")
