from segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import os, time

class SAMDetector:
    def __init__(self) -> None:
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = 'cuda' # cpu
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.model = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            output_mode = 'coco_rle'
        )
    
    def detect(self, image):
        origin_size = image.shape[:2]
        height, width = origin_size
        do_resize = True
        target_len = 1024
        if origin_size[0] <= target_len and origin_size[1] <= target_len:
            do_resize = False
        else:
            # 计算将最大的一条边缩放到1024时，另一条边的大小保持比例
            if width > height:
                new_width = target_len
                new_height = int(height * (target_len / width))
            else:
                new_width = int(width * (target_len / height))
                new_height = target_len
            downsample_size = (new_height, new_width)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if do_resize:
            image = cv2.resize(image, dsize=downsample_size)

        st = time.time()
        masks = self.model.generate(image)
        et = time.time()

        print(f'predict cost {et-st:.3f} seconds')