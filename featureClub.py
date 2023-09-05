#!/usr/bin/env python

import os

if 'G' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['G']

import gc
import sys
from pathlib import Path
from random import shuffle

import numpy as np
import timm
import torch
import openslide
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tiler_durden import get_tile_coordinates
from torchvision import transforms as T

class WSI_Tile_Dataset(Dataset):

    def __init__(self, slidePath, tileSize=384, transform=None):
        ''' Upon initialization, finds thumbnail and gets mask coordinates '''

        self.tileSize = tileSize
        self.transform = transform
        self.slide = openslide.OpenSlide(str(slidePath))

        # get the coordinates of each tile in the grid - shape is (tileRows, tileCols, 2)
        self.tile_coordinates = get_tile_coordinates(self.slide, tileSize)
        nRows, nCols, _ = self.tile_coordinates.shape

        # get a thumbnail, wherein each pixel represents a tile
        thumbPath = Path('/nfs/tcga/thumbs') / (slidePath.stem+'.png')
        if thumbPath.exists():
            print('Found cached thumb')
            thumb = Image.open(str(thumbPath))
        else:
            thumb = self.slide.get_thumbnail((nCols, nRows))
            thumb.save(str(thumbPath))
        print(self.slide.dimensions, thumb.size)

        thr = 255 // 2          # seems like a reasonable threshold
        #thr = 100
        thumbArr = np.asarray(thumb.convert('RGB'))  # .copy()
        foreGroundTiles = (thumbArr < [thr, thr, thr]).all(axis=2)
        tileCoords = np.where(foreGroundTiles)
        rows, cols = tileCoords
        self.foreground_tile_coordinates = list(zip(rows, cols))

        print(f'Found {len(self)} foreground tiles out of {nRows*nCols}, in a grid of {nRows}x{nCols}')
        print(self.foreground_tile_coordinates)

    def __len__(self):
        return len(self.foreground_tile_coordinates)

    def __getitem__(self, item):
        row, col = self.foreground_tile_coordinates[item]
        xPos, yPos = self.tile_coordinates[row][col]
        tile = self.slide.read_region((xPos, yPos), 0, (self.tileSize, self.tileSize)).convert('RGB')
        if self.transform:
            tile = self.transform(tile)
        return tile


ENCODER = 'resnet50'
TensorsPath = Path(f'/nfs/tcga/features/{ENCODER}/')
TensorsPath.mkdir(exist_ok=True,parents=True)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

model = timm.create_model(ENCODER, pretrained=True, num_classes=0)
model.to(dev)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean = model.pretrained_cfg['mean'],
        std = model.pretrained_cfg['std']
    ),
])




def featurize_slide(slidePath):

    tensorPath = TensorsPath / (slidePath.stem+'.pt')
    if tensorPath.exists():
        print(f'Already done')
        return

    tiles = WSI_Tile_Dataset(slidePath, transform=transform)
    if len(tiles) == 0:
        print(f'No foreground tiles found in {str(slidePath)}')
        return

    dataloader = DataLoader(tiles, num_workers=4, batch_size=8)

    #for n, tile in enumerate(tqdm(tiles)):
    features = []
    for n, tensors in enumerate(tqdm(dataloader)):
        feats = model(tensors.to(dev))
        features.append(feats.detach().cpu())

    features = torch.concat(features)
    print(features.shape)
    torch.save(features, str(tensorPath))

def testDataset(slidePath):

    tiles = WSI_Tile_Dataset(slidePath)

    for n, tile in enumerate(tqdm(tiles)):
        tile.save(f'tmp2/{n}.png')



def just_make_thumbs():
    for p in Path('/nfs/tcga/svs_methyl/').glob('*/*.svs'):
        print(p)
        tiles = WSI_Tile_Dataset(p)
        del tiles
        gc.collect()



if __name__=='__main__':
    #just_make_thumbs()
    #sys.exit()
    #for p in Path('/nfs/tcga/svs_methyl/').glob('*/*.svs'):
    #    print(p)
    #    break

    matches = list(Path('/nfs/tcga/svs_methyl/').glob('*/*DX*.svs'))
    shuffle(matches)
    #slidePath = p
    #slidePath = matches[3]
    #print(slidePath)
    for slidePath in tqdm(matches):
        featurize_slide(slidePath)

