import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import functools


def scale_format(scale, x, pos):
    return f"{round(scale * x, 3)}mm"


def get_scale_formatters(scales):
    return [functools.partial(scale_format, float(scale)) for scale in scales]


def crop(dat):
    # https://stackoverflow.com/a/39466129
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = dat[
        top_left[0] : bottom_right[0] + 1,  # plus 1 because slice isn't
        top_left[1] : bottom_right[1] + 1,
        top_left[2] : bottom_right[2] + 1,
    ]  # inclusive
    return out


def scale_shape(shape, scale_factors):
    scaled_dimensions = []
    max_range = max([(dimension[0] - dimension[1]) * 2.65 for dimension in shape])
    # Set minimum scale to 1.0
    scale_correction_factor = min(scale_factors)
    for dimension, factor in zip(shape, scale_factors):
        dim_range = dimension[1] - dimension[0]
        anchor_point = dimension[0] + dim_range / 2
        scaled_range = max_range / (factor * scale_correction_factor)
        scaled_dimensions.append(
            [anchor_point - scaled_range / 2, anchor_point + scaled_range / 2]
        )
    return scaled_dimensions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("annotation")
    ap.add_argument(
        "--sum-axis", "-s", help="For 2D projection, which axis to drop.", default=2
    )
    args = ap.parse_args()

    anno = nib.load(args.annotation)

    fig3d = plt.figure("3D Annotation")
    fig2d = plt.figure("2D Annotation")
    ax3d = fig3d.add_subplot(projection="3d")
    ax2d = fig2d.add_subplot()

    img_data = anno.get_fdata()
    img_data = crop(img_data)
    dimensions_original = [[0, dim] for dim in img_data.shape]
    dimensions_scaled = scale_shape(dimensions_original, anno.header.get_zooms())

    ax3d.set_xlim(dimensions_scaled[0])
    ax3d.set_ylim(dimensions_scaled[1])
    ax3d.set_zlim(dimensions_scaled[2])

    formatters = get_scale_formatters(anno.header.get_zooms())

    ax3d.xaxis.set_major_formatter(formatters[0])
    ax3d.yaxis.set_major_formatter(formatters[1])
    ax3d.zaxis.set_major_formatter(formatters[2])

    ax3d.tick_params(axis="both", which="major", pad=7.5)

    img_data_2d = np.sum(img_data, axis=args.sum_axis)
    plot2d = ax2d.imshow(img_data_2d)
    fig2d.colorbar(plot2d)

    # Maybe match colors of 2D plot?
    ax3d.voxels(img_data, color="red", edgecolor="maroon", linewidth=0.5)

    fig3d.savefig(
        args.annotation + ".3d.png", pad_inches=0.025, dpi=300, bbox_inches="tight"
    )
    fig2d.savefig(
        args.annotation + ".2d.png", pad_inches=0.025, dpi=300, bbox_inches="tight"
    )
