import argparse
from pathlib import Path

import cv2

from research.autoencoder.model.mae.mae_infer_class import AutoencoderClass


def get_args_parser():
    parser = argparse.ArgumentParser('MAE inference')

    # parser.add_argument('--source_img', default='test.jpg', type=str,
    parser.add_argument('--source_img', default='../data/raw/PolypGen2021_MultiCenterData_v3/data_C3/images_C3/C3_EndoCV2021_001.jpg', type=str,
                        help='Path to the image to be processed.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--model', default='MaskedAutoencoderViT', type=str, metavar='MODEL',
                        help='Name of model to inference')
    # parser.add_argument('--checkpoint_path', default='./output_dir/checkpoint.pth', type=str,
    parser.add_argument('--checkpoint_path', default='./model/mae/checkpoint/checkpoint(0).pth', type=str,
                        help='Path to checkpoint file.')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save the output image. If None, the output image will not be saved.')

    return parser


def main():
    args = get_args_parser().parse_args()

    if not Path(args.source_img).exists():
        raise FileNotFoundError(f'Image file not found: {args.source_img}')

    model = AutoencoderClass(model_name=args.model, checkpoint_path=args.checkpoint_path)

    # Load image
    image = cv2.imread(args.source_img)

    # Inference
    output = model(image, mask_ratio=args.mask_ratio)

    # Save output image
    if args.output_dir is not None:
        cv2.imwrite(args.output_dir, output)

    # Show output image
    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()
