from typing import Union

from PIL import Image as PillowImage, ImageDraw, ImageFont

class Image():
    
    def __init__(self, origin: PillowImage.Image|str):

        self.data = None
        self._load_data(origin=origin)

    
    def _load_data(self, origin: PillowImage.Image|str):
        if isinstance(origin, PillowImage.Image):
            self.data = origin
        elif isinstance(origin, str):
            self.data = PillowImage.open(origin)
        else:
            raise ValueError('Unsupported "origin" type.')
        

    def save(self, file_path: str):
        self.data.save(file_path)

        return