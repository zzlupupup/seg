import numpy as np
import nibabel as nib
import torch

from networks.hn_un import HN
from pathlib import Path
from monai import transforms
from utils.test_3d_patch import test_single_case_HN

def post_trans_demo(image_list, model, post, save_dir, r):
    imagLoader = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1, a_max=10, b_min=0, b_max=1, clip=True
            ),
        transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=5
            ),
        transforms.ToTensord(keys=['image', 'label'], track_meta=False)
    ])

    for image_path in image_list:
        imageName = image_path.name
        sample = {
            "image": image_path/(imageName + ".nii.gz"),
            "label": image_path/(imageName + "_label.nii.gz")
        }
        sample = imagLoader(sample)
        image = sample['image'].squeeze(0).numpy()
        label = sample['label'].squeeze(0).numpy()

        prediction, _ = test_single_case_HN(model, image, 32, 32, (64, 64, 64), 3)

        if post is not None:
            prediction = torch.as_tensor(prediction).cuda().unsqueeze(0)
            prediction = post(prediction)
            prediction = torch.argmax(prediction, dim=0)

        case_save_dir = save_dir / imageName

        if not case_save_dir.exists():
            case_save_dir.mkdir()

        nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), case_save_dir/(f'pred_{r}.nii.gz'))
        nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), case_save_dir/(f'img_{r}.nii.gz'))
        nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), case_save_dir/(f'label_{r}.nii.gz'))

        break
    
    return None
    

if __name__ == '__main__':

    r = 30
    threshold = 4/3 * np.pi * r ** 3

    remover = transforms.RemoveSmallObjects(min_size=threshold, independent_channels=True, by_measure=True, pixdim=[1.0, 1.0, 1.0])
    def apply_remove_to_last_channel(img):
        channels_to_keep = img[:-1]
        last_channel = img[-1:]
        transformed_last_channel = remover(last_channel)
        
        return torch.cat((channels_to_keep, transformed_last_channel), dim=0)

    remove_trans = transforms.Lambda(apply_remove_to_last_channel)

    post_trans = transforms.Compose([
        transforms.AsDiscrete(to_onehot=3),
        remove_trans
    ])

    image_list = [dir for dir in (Path('data\\LUNG') / 'test').iterdir() if dir.is_dir()]
    save_dir = Path('C:\\Users\\zhangzilong\\Desktop\\post_trans_demo')

    model = HN().cuda()
    model.load_state_dict(torch.load(Path('final_model\\HN_Fusion_un') / 'best_model.pth'))
    model.eval()    

    with torch.no_grad():
        post_trans_demo(image_list, model, post_trans, save_dir, r)


    

    

