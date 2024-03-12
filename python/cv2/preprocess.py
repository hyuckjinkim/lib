from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import cv2
from PIL import Image

class ImagePreprocess:
    def __init__(self, data, image_id, image_path, image_label=None):
        self.data = data
        self.image_id = image_id
        self.image_path = image_path
        self.image_label = image_label
        
    def label_encoding(self):
        le = LabelEncoder()
        self.data[self.image_label] = le.fit_transform(self.data[self.image_label])
        self.classes_ = le.classes_

    def _get_cropped_images(self, file_path, keep_values, keep_columns, th_area):
        image = Image.open(file_path)

        # Aspect ratio
        as_ratio = image.size[0] / image.size[1]

        sxs, exs, sys, eys = [],[],[],[]
        if as_ratio >= 1.5:
            # Crop
            mask = np.max( np.array(image) > 0, axis=-1 ).astype(np.uint8)
            retval, labels = cv2.connectedComponents(mask)
            if retval >= as_ratio:
                x, y = np.meshgrid( np.arange(image.size[0]), np.arange(image.size[1]) )
                for label in range(1, retval):
                    area = np.sum(labels == label)
                    if area < th_area:
                        continue
                    xs, ys= x[ labels == label ], y[ labels == label ]
                    sx, ex = np.min(xs), np.max(xs)
                    cx = (sx + ex) // 2
                    crop_size = image.size[1]
                    sx = max(0, cx-crop_size//2)
                    ex = min(sx + crop_size - 1, image.size[0]-1)
                    sx = ex - crop_size + 1
                    sy, ey = 0, image.size[1]-1
                    sxs.append(sx)
                    exs.append(ex)
                    sys.append(sy)
                    eys.append(ey)
            else:
                crop_size = image.size[1]
                for i in range(int(as_ratio)):
                    sxs.append( i * crop_size )
                    exs.append( (i+1) * crop_size - 1 )
                    sys.append( 0 )
                    eys.append( crop_size - 1 )
        else:
            # Not Crop (entire image)
            sxs, exs, sys, eys = [0,],[image.size[0]-1],[0,],[image.size[1]-1]

        df_crop = pd.DataFrame()
        df_crop["sx"] = sxs
        df_crop["ex"] = exs
        df_crop["sy"] = sys
        df_crop["ey"] = eys
        df_crop[keep_columns] = keep_values
        df_crop = df_crop[keep_columns+['sx','ex','sy','ey']]
        
        return df_crop
    
    def crop(self, th_area=1000, drop_duplicates=True):
        keep_columns = self.data.columns.tolist()
        crop_data = []
        for i in trange(len(self.data),desc='crop'):
            file_path = self.data[self.image_path].values[i]
            keep_values = self.data[keep_columns].values[i]
            cropped = self._get_cropped_images(file_path, keep_values, keep_columns, th_area)
            crop_data.append(cropped)
        
        crop_data = pd.concat(crop_data)
        
        if drop_duplicates:
            crop_data = crop_data\
                .drop_duplicates(subset=[self.image_id, "sx", "ex", "sy", "ey"])\
                .reset_index(drop=True)
            
        self.data = crop_data