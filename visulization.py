import numpy as np
from matplotlib import offsetbox, pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import torch
from HashingTwoBranch import collate_fn

from data_processor import TwoBranchDataset
from torch.utils.data import DataLoader

from models import HashingResNet


test_dataset = TwoBranchDataset('test', './datasets/quickdraw', './datasets/png')
select = [np.arange(40) + i * 2500 for i in (0, 2, 5, 12, 17)]  # bear, cat, dog, lion, pig
select = np.concatenate(select)
images = 255 - test_dataset.X_png[select]
label = np.concatenate([[c] * 40 for c in 'BCDLP'])
test_dataset.X_raw = test_dataset.X_raw[select]
test_dataset.X_png = test_dataset.X_png[select]
test_dataset.y = test_dataset.y[select]
test_loader = DataLoader(test_dataset, 200, collate_fn=collate_fn)

def plot_embedding(X):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for i, cls in enumerate('BCDLP'):
        ax.scatter(
            *X[label == cls].T,
            marker=f"${cls}$",
            s=80,
            color=plt.cm.Dark2(i),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 1.5e-2:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=0.8), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.axis("off")

model = HashingResNet(3, (1, 28, 28), 25)
path = 'saved_models/HashingResNet.bin'
with torch.no_grad():
    model.load_state_dict(torch.load(path))
    model.eval()
    y_true, y_pred = [], []
    for i, batch in enumerate(test_loader):
        batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
        code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
    code = code.data.numpy()

tsne = TSNE(
    n_components=2,
    init="pca",
    learning_rate="auto",
    n_iter=500,
    n_iter_without_progress=150,
    n_jobs=2,
    random_state=42,
)
projection = tsne.fit_transform(code)
plot_embedding(projection)
plt.show()
