from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import os, time

def show_anns(img, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = img.copy()
    polygons = []
    color = []
    mask = np.zeros_like(img)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.uint8(np.random.random((1, 3))*255).tolist()[0]
        mask[m] = color_mask
    img = cv2.addWeighted(img, 0.7, mask, 0.3, 1)
    return img

models = {
    # 'vit_h': "sam_vit_h.pth",
    # 'vit_l': "sam_vit_l.pth",
    'vit_b': "sam_vit_b_01ec64.pth",
}
folder_path = 'test_imgs'
device = "cuda"
if not os.path.exists('result_imgs'):
    os.mkdir('result_imgs')
all_imgs = os.listdir(folder_path)
# all_imgs = [os.path.join(folder_path, i) for i in all_imgs]

for model_type, sam_checkpoint in models.items():
    print(f'########## change to model {model_type} ##########')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        output_mode = 'coco_rle'
    )
    for img_filename in all_imgs:
        origin_path = os.path.join(folder_path, img_filename)
        print(f'predicting {origin_path}...')
        image = cv2.imread(origin_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(1024, 1024))

        st = time.time()
        masks = mask_generator.generate(image)
        et = time.time()
        # print(len(masks))
        # print(masks[0].keys())

        # img_with_color = show_anns(image, masks)
        # save_path = os.path.join('result_imgs', os.path.split(img_filename)[0]+f'_{model_type}.jpg')
        # cv2.imwrite(save_path, img_with_color)
        print(f'predict {origin_path} cost {et-st:.3f}/{time.time()-st:.3f} seconds')
