"""Dataset classes."""
import os

import numpy as np
from PIL import Image
import scipy
import scipy.cluster
import torch as th
from torchvision.transforms.functional import to_tensor, crop


class AnimationDataset(th.utils.data.Dataset):
    def __init__(self, data, canvas_size=128):
        self.canvas_size = canvas_size
        self.files = [os.path.join(data, f)
                      for f in os.listdir(data)
                      if f.endswith('png') or f.endswith('jpg')
                      or f.endswith('jpeg')]
        self.files = sorted(
            self.files,
            key=lambda f: int(''.join(x for x in os.path.basename(f)
                                      if x.isdigit())))

        colors = []
        for f in np.random.choice(self.files, size=min(len(self.files), 100),
                                  replace=False):
            im = Image.open(f).convert('RGB')
            w, h = im.size
            a = min(w, h) / self.canvas_size
            im = im.resize((int(w/a), int(h/a)), Image.LANCZOS)
            colors.append(np.asarray(im).reshape(-1, 3).astype(float))

        colors = np.vstack(colors)
        codes, _ = scipy.cluster.vq.kmeans(colors, 5)
        vecs, _ = scipy.cluster.vq.vq(colors, codes)
        counts, _ = scipy.histogram(vecs, len(codes))
        self.bg = codes[scipy.argmax(counts)].astype(np.float32) / 255

    def __repr__(self):
        return "AnimationDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        w, h = im.size
        a = min(w, h) / self.canvas_size
        im = im.resize((int(w/a), int(h/a)), Image.LANCZOS)
        w, h = im.size
        im = im.crop(((w-self.canvas_size) // 2, (h-self.canvas_size) // 2,
                      (w-self.canvas_size) // 2 + self.canvas_size,
                      (h-self.canvas_size) // 2 + self.canvas_size))
        im = to_tensor(im)

        return {
            'im': im,
            'fname': self.files[idx]
        }


class RandomCropDataset(th.utils.data.Dataset):
    def __init__(self, data, canvas_size, grid=False):
        self.canvas_size = canvas_size
        self.files = [os.path.join(data, f)
                      for f in os.listdir(data)
                      if f.endswith('png') or f.endswith('jpg')]

        colors = []
        for f in np.random.choice(self.files, size=min(len(self.files), 100),
                                  replace=False):
            im = Image.open(f).convert('RGB')
            colors.append(np.asarray(im).reshape(-1, 3).astype(float))

        colors = np.vstack(colors)
        codes, _ = scipy.cluster.vq.kmeans(colors, 5)
        vecs, _ = scipy.cluster.vq.vq(colors, codes)
        counts, _ = scipy.histogram(vecs, len(codes))
        self.bg = codes[scipy.argmax(counts)].astype(np.float32) / 255

        if grid:
            self.patches = []
            for f in self.files:
                im = to_tensor(Image.open(f).convert('RGB'))
                _, h, w = im.shape
                h = h // self.canvas_size
                w = w // self.canvas_size
                im = im[:, :h*self.canvas_size, :w *
                        self.canvas_size].view(-1, h, self.canvas_size, w,
                                               self.canvas_size)
                im = im.permute(1, 3, 0, 2, 4)
                self.patches.append(im.flatten(0, 1))
            self.patches = th.cat(self.patches, dim=0)
        else:
            self.patches = None

    def __repr__(self):
        return "RandomCropDataset"

    def __len__(self):
        return 2000 if self.patches is None else self.patches.shape[0]

    def __getitem__(self, idx):
        if self.patches is not None:
            return {'im': self.patches[idx]}

        im = Image.open(np.random.choice(self.files)).convert('RGB')
        w, h = im.size
        x = np.random.randint(0, w-self.canvas_size)
        y = np.random.randint(0, h-self.canvas_size)

        im1 = crop(im, y, x, self.canvas_size, self.canvas_size)

        return {
            'im': to_tensor(im1)
        }


class SpriteDataset(th.utils.data.Dataset):
    def __init__(self, data, canvas_size):
        self.canvas_size = canvas_size
        files = [os.path.join(data, f)
                 for f in os.listdir(data)
                 if f.endswith('png') or f.endswith('jpg')]

        self.sprites = []
        for f in files:
            sprite = Image.open(f).convert('RGBA')
            w, h = sprite.size
            size = max(w, h)
            if size > 16:
                a = 16 / size
                sprite = sprite.resize((int(a*w), int(a*h)), Image.LANCZOS)
            self.sprites.append(sprite)

        self.bg = [1., 1., 1.]

    def __repr__(self):
        return "SpriteDataset"

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        im = Image.new(
            'RGB', (self.canvas_size, self.canvas_size), (255, 255, 255))
        idxs = np.random.choice(
            len(self.sprites), size=np.random.randint(5, 16))
        for i in idxs:
            sprite = self.sprites[i]
            w, h = sprite.size
            x = np.random.randint(0, self.canvas_size-w)
            y = np.random.randint(0, self.canvas_size-h)
            im.paste(sprite, (x, y), sprite)

        return {
            'im': to_tensor(im)
        }
