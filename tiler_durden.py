from pathlib import Path
from random import shuffle

import numpy as np
import openslide
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange

svs_in = '/srv/dev-disk-by-uuid-ceacc299-7d18-4dfd-8416-889c76bcb7ca/tcga'
#png_out = Path('/nfs/tcga/png/')
png_out = Path('/srv/dev-disk-by-uuid-dc5580b8-bfa8-43e4-a2b8-e111e58bdc38/tcga/png/')   # skip NFS for faster writes

csv_out = Path('/nfs/tcga/tile_csv/')

csv_out.mkdir(exist_ok=True, parents=True)



def parse_tcga_code(slidePath):
    slideFN = slidePath.stem
    tcga_code, uuid = slideFN.split('.')
    tcga_cols = tcga_code.split('-')
    case_id = '-'.join(tcga_cols[:3])
    slide_type = tcga_cols[-1]

    return case_id, slide_type

def fraction_white_pixels(pilImg, thr=200, verbose=False):
    #pil.setflags(write=1)
    img = np.asarray(pilImg.convert('RGB'))#.copy()
    #img.sum(axis=2).max()
    #img.setflags(write=True)\
    #img[np.where((img > [thr, thr, thr]).all(axis=2))] = [0, 0, 0]
    #img[img>200]=0
    #Image.fromarray(img)
    whitePixels = (img > [thr, thr, thr]).all(axis=2)
    h,w,_ = img.shape
    whiteFrac = whitePixels.sum() / whitePixels.size
    if verbose:
        print(f'White pixels | nonWhite : {whitePixels.size - whitePixels.sum()}  count : {whitePixels.sum()}   total pix : {whitePixels.size}  frac : {whiteFrac}')
    return whiteFrac



def get_tile_coordinates(slide, tileSize):
    slideWidth, slideHeight = slide.dimensions
    tilesX, tilesY = slideWidth//tileSize, slideHeight//tileSize     # grid size
    xCoords = np.linspace(0, (tilesX-1)*tileSize, tilesX).astype(int)
    yCoords = np.linspace(0, (tilesY-1)*tileSize, tilesY).astype(int)

    tile_coordinates = np.dstack(np.meshgrid(xCoords, yCoords))      # shape is (tileRows, tileCols, 2)
    print(tile_coordinates.shape)
    return tile_coordinates

def tile_WSI(slidePath, tileSize=384):

    slideFN = slidePath.stem
    #dataUUID = slidePath.parent.stem
    #outPath = png_out / dataUUID / slideFN
    outPath = png_out / slideFN
    csvF = csv_out / f'{slideFN}.csv'

    if csvF.exists():
        print('already done')
        return

    '''    
    if outPath.exists():
        
        if len(list(outPath.glob('*.png')))>0:
            print('already done')
            return
    '''

    outPath.mkdir(exist_ok=True, parents=True)

    slide = openslide.OpenSlide(str(slidePath))

    tile_coordinates = get_tile_coordinates(slide, tileSize)

    def read_save_tile(job):
        slide = openslide.OpenSlide(str(slidePath))
        row, cols = job
        records = []
        for col in cols:
            xPos, yPos = tile_coordinates[row][col]
            tile = slide.read_region((xPos, yPos), 0, (tileSize, tileSize)).convert('RGB')
            wf = fraction_white_pixels(tile)
            if wf > 0.1: continue
            R = dict(col=col, row=row, xPos=xPos, yPos=yPos)
            records.append(R)
            tile.save(str(outPath / f'{col}_{row}.png'))

        return records

    '''
    kept_tile_coords = []
    for row in trange(tile_coordinates.shape[0]):
        for col in range(tile_coordinates.shape[1]):
            R = read_save_tile(row, col)
            kept_tile_coords.append(R)
    '''

    jobs = []
    for row in range(tile_coordinates.shape[0]):
        cols = [col for col in range(tile_coordinates.shape[1])]
        jobs.append((row, cols))
        #for col in range(tile_coordinates.shape[1]):
        #    jobs.append((row, col))

    records_lists = Parallel(n_jobs=40)(delayed(read_save_tile)(job) for job in jobs)
    dfs = [pd.DataFrame(l) for l in records_lists if len(l)]
    print(records_lists)
    #kept_tile_coords = sum(records_lists)
    kept_tile_records = pd.concat(dfs)

    #kept_tile_coords = [r for r in kept_tile_coords if r is not None]
    #pd.DataFrame(kept_tile_coords)\
    kept_tile_records.sort_values(['col','row']).to_csv(str(csvF), index=False)


def test(slidePath):
    tile_WSI(slidePath)

def thumb(slidePath):
    slide = openslide.OpenSlide(slidePath)

    slide = openslide.OpenSlide(str(slidePath))
    tile_coordinates = get_tile_coordinates(slide, 384)
    print(tile_coordinates.shape)
    nRows, nCols, _ = tile_coordinates.shape

    thumb = slide.get_thumbnail((nCols, nRows))
    print(slide.dimensions, thumb.size)

    fraction_white_pixels(thumb, verbose=True, thr=255//2)

    thr = 255//2
    thumbArr = np.asarray(thumb.convert('RGB'))  # .copy()
    foreGroundTiles = (thumbArr < [thr, thr, thr]).all(axis=2)
    print(foreGroundTiles.shape)
    tileCoords = np.where(foreGroundTiles)
    print(tileCoords)
    rows, cols = tileCoords
    for row, col in zip(rows, cols):
        yPos = row*384
        xPos = col*384
        print(row, col)
        tile = slide.read_region((xPos, yPos), 0, (384, 384))
        print(np.asarray(tile).mean())
        #tile.save(f'tmp/{row}_{col}.png')


def tile_slides():
    paths = list(Path('/nfs/tcga/svs_methyl/').glob('*/*.svs'))
    shuffle(paths)    # allows cheap parallelization, since this is a quick and dirty project
    for slidePath in paths:
        print(str(slidePath))
        case_id, slide_type = parse_tcga_code(slidePath)
        if slide_type != 'DX1':
            #print(f'Skipping {slide_type}')
            continue

        tile_WSI(slidePath)


def svs_file_mapping():
    rows = []
    for slidePath in Path('/nfs/tcga/svs_methyl/').glob('*/*.svs'):
        case_id, slide_type = parse_tcga_code(slidePath)
        rows.append(dict(case_id=case_id, svs=slidePath, slide_type=slide_type))

    df = pd.DataFrame(rows)
    print(df)
    print(df.slide_type.value_counts())
    df.to_csv('tcga_cases_files.csv', sep='\t', index=False)

if __name__ == '__main__':
    #tile_slides()
    #svs_file_mapping()

    for p in Path('/nfs/tcga/svs_methyl/').glob('*/*.svs'):
        print(p)
        break
    slidePath = p
    test(slidePath)
    thumb(slidePath)

