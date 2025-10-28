from monai import transforms

def get_transform():
    label_transform = transforms.Compose(
        [
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1, a_max=10, b_min=0, b_max=1, clip=True
            ),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                margin=5
            ),
            transforms.RandCropByLabelClassesd(
                keys=["image","label"],
                label_key="label",
                spatial_size=[64, 64, 64],
                ratios=[1, 2, 3],
                num_classes=3,
                num_samples=1,
                image_key="image",
                image_threshold=0
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            transforms.ToTensord(keys=["image", "label"], track_meta=False),
        ]
    )

    unlabel_transform = transforms.Compose([
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS", labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1, a_max=10, b_min=0, b_max=1, clip=True
            ),
            transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[64, 64, 64], random_size=False),
            transforms.ToTensord(keys=["image", "label"], track_meta=False)
                          ])

    return label_transform, unlabel_transform