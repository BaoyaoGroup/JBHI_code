import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import importlib


from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt


def read_image_mask(path):
    img = Image.open(path)
    img_array = np.array(img)
    # if len(img_array.shape) == 3 and img_array.shape[2] == 3:
    #     img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    img_array = (img_array > 0).astype(np.int8)  # 转换为二值图像
    return img_array


def visualize_prediction_mask(prediction_mask_path, ground_truth_mask_path, args):
    prediction_mask = read_image_mask(prediction_mask_path)
    ground_truth_mask = read_image_mask(ground_truth_mask_path)

    plt.figure(figsize=(5, 5))

    ax = plt.subplot(1, 1, 1)
    # 预测分割结果
    ax.imshow(prediction_mask, cmap="YlOrBr", alpha=0.5)
    # 真实标注
    # ground_truth_mask[ground_truth_mask < 1e-6] = 0
    ax.imshow(ground_truth_mask, cmap="Greens", alpha=0.5)
    ax.set_title("QGD-Net")
    ax.axis("off")

    # 添加图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#D5C5BB", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, color=plt.cm.Greens(0.5), alpha=0.5),
    ]
    labels = ["Prediction Segmentation", "Ground Truth"]
    plt.legend(handles, labels, loc="upper left")
    plt.savefig(
        prediction_mask_path.split(".")[0] + f"_group{args.num_groups}_visualize.png"
    )
    plt.close()


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    zeros_tensor = torch.zeros(img.shape[0], 1, img.shape[2], img.shape[3])
    img = torch.cat((zeros_tensor, img), dim=1)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model_path",
        "-p",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="MODEL.pth",
        help="Model name",
    )
    parser.add_argument(
        "--num_groups",
        "-g",
        type=int,
        default=0,
        help="Num layer per group",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="INPUT",
        nargs="+",
        help="Filenames of input images",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT",
        nargs="+",
        help="Filenames of output images",
    )
    parser.add_argument(
        "--viz",
        "-v",
        action="store_true",
        help="Visualize the images as they are processed",
    )
    parser.add_argument(
        "--no-save", "-n", action="store_true", help="Do not save the output masks"
    )
    parser.add_argument(
        "--mask-threshold",
        "-t",
        type=float,
        default=0.5,
        help="Minimum probability value to consider a mask pixel white",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Scale factor for the input images",
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_OUT.png"

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros(
            (mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8
        )
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = args.input
    out_files = get_output_filenames(args)

    net = getattr(importlib.import_module(args.model), "Model")(
        n_channels=4,
        n_classes=args.classes,
        bilinear=args.bilinear,
        num_groups=args.num_groups,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model {args.model}, Path: {args.model_path}")
    logging.info(f"Using device {device}")

    net = net.to(device=device)
    state_dict = torch.load(args.model_path, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    for i, filename in enumerate(in_files):
        real_mask_path = filename.replace("imgs", "masks")
        if "isic2016" in args.model_path:
            real_mask_path = os.path.join(
                "/".join(real_mask_path.split("/")[:-1]),
                real_mask_path.split("/")[-1].split(".")[0] + "_Segmentation.png",
            )
        elif "isic2017" in args.model_path:
            real_mask_path = os.path.join(
                "/".join(real_mask_path.split("/")[:-1]),
                real_mask_path.split("/")[-1].split(".")[0] + "_segmentation.png",
            )
        else:
            real_mask_path = os.path.join(
                "/".join(real_mask_path.split("/")[:-1]),
                real_mask_path.split("/")[-1].split(".")[0] + "_mask.gif",
            )
            logging.info(f"Predicting image {filename} ...")
        img = Image.open(filename)

        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f"Mask saved to {out_filename}")
            visualize_prediction_mask(out_files[i], real_mask_path, args)
            logging.info(
                f"Visualization saved to {out_files[i].split(".")[0] + "_visualization.png"}"
            )

        if args.viz:
            logging.info(
                f"Visualizing results for image {filename}, close to continue..."
            )
            plot_img_and_mask(img, mask)
