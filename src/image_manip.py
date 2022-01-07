import nibabel as nib
from nibabel.spatialimages import Header

# import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from numpy.core.numeric import indices
from scipy.ndimage.interpolation import zoom
from skimage.segmentation import watershed
import skimage.morphology as smorph
import skimage.io as sio
import skimage.filters as sfilters
import skimage.util as sutil
import scipy.ndimage as scimage
from math import ceil, sqrt
import matplotlib.pyplot as plt
from typing import Any

from PIL import Image


def dimensions_used(image: nib.nifti1.Nifti1Image):
    # Get the sum of values in each slice for each dimension.
    # If in any dimension only one slice is not 0, this dimension
    # could be scrapped.

    img = image.get_fdata()

    used_dimensions = 0  # How many dimensions have >1
    for dimension, slice_count in enumerate(img.shape):
        slice_array = [slice(None) for slice_index in range(img.ndim)]
        used_slices = 0
        for slice_index in range(slice_count):
            slice_array[dimension] = slice(slice_index, slice_index + 1, None)
            slice_tuple = tuple(slice_array)

            reduced_img = img[slice_tuple]
            if np.count_nonzero(reduced_img):  # Is any value in this slice non-zero?
                used_slices += 1
        if (
            used_slices > 1
        ):  # If not more than one slice is used in a dimension, this dimension could be left out
            used_dimensions += 1
    return used_dimensions


def get_largest_area(image: nib.nifti1.Nifti1Image):
    """
    Finds the slice with the most non-zero values.

    Returns index of axis and slice.
    """

    img = image.get_fdata()
    zooms = image.header.get_zooms()

    max_area: float = 0
    most_area_index: tuple = ()
    most_area_slice: tuple = ()
    for dimension, slice_count in enumerate(img.shape):
        # Area of 2D pixel
        pixel_area = np.prod(zooms[:dimension] + zooms[dimension + 1 :])
        slice_array = [slice(None) for slice_index in range(img.ndim)]
        used_slices = 0
        for slice_index in range(slice_count):
            slice_array[dimension] = slice(slice_index, slice_index + 1, None)
            slice_tuple = tuple(slice_array)

            reduced_img = img[slice_tuple]

            # Count pixels/voxels and account for resolution by multiplying by pixel area
            slice_area: float = np.count_nonzero(reduced_img) * pixel_area
            if slice_area > max_area:
                max_area = slice_area
                most_area_slice = slice_tuple
                most_area_index = (dimension, slice_index)
            # print(f"Dimension {dimension}, Slice {slice_index}: ", end="")
            # print(np.count_nonzero(reduced_img))
    return most_area_index, most_area_slice


# def reduce
def _close_enough(a_coords, b_coords, distance):
    # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    voxel_distance: float = np.linalg.norm(a_coords - b_coords)
    return voxel_distance <= distance


def inverse_distance_transform_alt(
    annotation: nib.nifti1.Nifti1Image,
) -> nib.nifti1.Nifti1Image:
    def _get_cube(axes_scales: list[float], distance: float) -> list[int]:
        """
        Get cube of influence (actually sphere, but cube is easier)
        A value n means, that a pixel n layers away is influenced.
        A n of 0 means the current layer is still affected.
        The multiplication and addition blows up the cube to 3D
        """
        # print(", ".join([f"{distance} // {x} * 2 + 1 == {int(distance // x) * 2 + 1}" for x in axes_scales]))
        return [int(distance // axis_scale) * 2 + 1 for axis_scale in axes_scales]

    # Determine base layer
    base_layer_index, base_layer_slice = get_largest_area(annotation)
    image_layer = annotation.get_fdata()[
        base_layer_slice
    ].squeeze()  # Drop 1-dimensional axes
    # Axis scaling for base layer
    axes_scales_2d = (
        annotation.header.get_zooms()[: base_layer_index[0]]
        + annotation.header.get_zooms()[base_layer_index[0] + 1 :]
    )
    axes_scales = annotation.header.get_zooms()

    distances = scimage.distance_transform_edt(image_layer, sampling=axes_scales_2d)

    # image_layer_img = image_layer
    # distances_img = distances
    # Image.fromarray((image_layer_img * (255.0 / image_layer.max())).astype(np.uint8), mode="L").save(f"img_base.jpg")
    # Image.fromarray((distances_img * (255.0 / distances.max())).astype(np.uint8), mode="L").save(f"img_dist.jpg")

    # Create empty new annotation
    new_anno = np.zeros(annotation.shape)

    # Idea for optimisation:
    # For each distance voxel calculate block
    # of influence, only check those pixels
    current_pixel_count = 1
    total_pixels = np.count_nonzero(distances)
    print("Calculating inverse_distance_transform_alt...")
    for coords_reference in np.argwhere(distances):
        coords_3d_reference = np.insert(coords_reference, *base_layer_index)
        distance = distances[coords_reference[0]][coords_reference[1]]
        influence_cube: Any = _get_cube(axes_scales, distance)
        if not current_pixel_count % 1000:
            print(f"{current_pixel_count}/{total_pixels}")
        current_pixel_count += 1
        for cube_index in np.ndindex(*influence_cube):
            # Center cube
            cube_index_centered = [
                coords[0] - coords[1] // 2 for coords in zip(cube_index, influence_cube)
            ]
            # Index in 3D annotation, centered around current voxel
            image_index = [
                sum(x) for x in zip(coords_3d_reference, cube_index_centered)
            ]
            # TODO: What the hell is this syntax
            try:
                if new_anno[image_index[0]][image_index[1], image_index[2]]:
                    continue
            except IndexError:
                # Index out of bounds, do not check this voxel
                continue
            if _close_enough(image_index, coords_3d_reference, distance):
                new_anno[image_index[0]][image_index[1], image_index[2]] = 1.0
            else:
                # print(f"Not close enough: {image_index} - {coords_3d_reference} - {distance}")
                pass
    new_annotation_object = nib.nifti1.Nifti1Image(
        new_anno, annotation.affine, header=annotation.header
    )
    return new_annotation_object


def inverse_distance_transform(
    annotation: nib.nifti1.Nifti1Image,
) -> nib.nifti1.Nifti1Image:
    # Get skeleton and distances from original
    # Generate ellipsoids along skeleton, use distance for radius (and axis scales for
    # distortion). Combine with original with logical OR.

    # Alternative:
    # Generate skeleton and distance info.
    # For each pixel in image, check for each
    # part of skeleton if this pixel is closer
    # than the skeleton's distance info. If so,
    # mark pixel as annotated.

    # Determine base layer
    base_layer_index, base_layer_slice = get_largest_area(annotation)
    image_layer = annotation.get_fdata()[
        base_layer_slice
    ].squeeze()  # Drop 1-dimensional axes
    # Axis scaling for base layer
    axes_scales_2d = (
        annotation.header.get_zooms()[: base_layer_index[0]]
        + annotation.header.get_zooms()[base_layer_index[0] + 1 :]
    )

    distances = scimage.distance_transform_edt(image_layer, sampling=axes_scales_2d)

    # Show image for debugging
    # distances = distances * (255.0/distances.max())
    # Image.fromarray(image_layer > 0).show()
    # Image.fromarray(distances).show()

    # print(distances)
    # for index in np.argwhere(distances):
    #     print(f"distances{index} == {distances[index[0]][index[1]]}")
    #     print(f"distances{index} == {distances[tuple(index)]}")

    # Create empty new annotation
    new_anno = np.zeros(annotation.shape)
    for a_coords, _ in np.ndenumerate(new_anno):
        print(a_coords)
        voxel_true: bool = False
        for b_coords in np.argwhere(distances):
            b_3d_coords = np.insert(b_coords, *base_layer_index)

            voxel_true = _close_enough(
                a_coords, b_3d_coords, distances[tuple(b_coords)]
            )
            if voxel_true:
                new_anno[tuple(a_coords)] = voxel_true
                break
    new_annotation_object = nib.nifti1.Nifti1Image(
        new_anno, annotation.affine, header=annotation.header
    )
    return new_annotation_object


def max_radius_erosion(annotation: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    """
    Erode each layer by a fraction of the maximum
    radius of the annotation.
    """

    if len(annotation.shape) != 3:
        raise ValueError("This function can only work on 3D annotations")

    # Determine base layer
    base_layer_index, base_layer_slice = get_largest_area(annotation)

    axes_scales = (
        annotation.header.get_zooms()[: base_layer_index[0]]
        + annotation.header.get_zooms()[base_layer_index[0] + 1 :]
    )

    # Determine largest distance in annotation
    ## Get distances and indices of distance-to points, axis-scale corrected via "sampling"
    image_layer = annotation.get_fdata()[
        base_layer_slice
    ].squeeze()  # Drop 1-dimensional axes
    # breakpoint()
    distances, indices = scimage.distance_transform_edt(
        image_layer, sampling=axes_scales, return_indices=True
    )
    ## Get maximum distance index
    max_index = np.unravel_index(np.argmax(distances), distances.shape)
    distance_to = [indices[axis][max_index] for axis in range(2)]

    radius = sqrt(
        ((max_index[0] - distance_to[0]) * axes_scales[0]) ** 2
        + ((max_index[1] - distance_to[1]) * axes_scales[1]) ** 2
    )

    # Determine number of layers to iterate through
    ## Get distance between layers
    axis_scale = annotation.header.get_zooms()[base_layer_index[0]]

    ## Output for sanity check
    print(
        f"For a radius of {radius}mm and an axis scale of {axis_scale}mm, {int(radius // axis_scale)} layers will be changed."
    )

    # Generate new annotation
    ## New empty annotation
    new_anno = np.zeros(annotation.shape, dtype=annotation.get_data_dtype())

    ## Copy base layer
    ### Un-squeeze by adding "empty" dimensions
    tmp_image_layer = image_layer
    for index, dimension in enumerate(new_anno[base_layer_slice].shape):
        if dimension <= 1:
            # Is this deprecated?
            # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims
            print(f"Shape is {tmp_image_layer.shape}")
            print(f"Expanding index {index}")
            tmp_image_layer = np.expand_dims(tmp_image_layer, index)
            print(f"Shape is now {tmp_image_layer.shape}")
    new_anno[base_layer_slice] = tmp_image_layer
    # new_anno[base_layer_slice] = image_layer[..., np.newaxis] # Change shape from [...] to [..., 1], which was lost to squishing

    print(f"Center layer {base_layer_index[1]}")

    ## Iterate through layers offset from base layer (symmetrical along base layer)
    for layer_offset in range(1, int(radius // axis_scale) + 1):
        # Calculate radius for erosion. [mm].
        distance_from_base = axis_scale * layer_offset
        ## Flipped circular dome. Radius increases with distance from base layer.
        ## If distance exceeds radius, nothing of annotation is left
        ##
        ## Plug radius of circle into half-circle/dome formula. Use distance to base
        ## layer as x, to get the diameter of a circle, that fits between the dome
        ## the y-axis.
        erosion_diameter = radius - sqrt((radius ** 2) - (distance_from_base ** 2))
        # Take axis scaling into consideration
        round_function = ceil  # Rather erode more than less
        erosion_structure_dimensions = [
            round_function(erosion_diameter / axis_scale) for axis_scale in axes_scales
        ]
        erosion_structure = np.ones(erosion_structure_dimensions)
        # breakpoint()
        # Create new layer
        new_layer = scimage.binary_erosion(image_layer, structure=erosion_structure)

        for sign in [1, -1]:
            # Generate slice list to extract layers above/below the base layer
            slice_list = [slice(None) for _ in range(len(annotation.shape))]
            slice_list[base_layer_index[0]] = slice(
                base_layer_index[1] + layer_offset * sign,
                base_layer_index[1] + layer_offset * sign + 1,
            )
            print(f"Now modifying layer {base_layer_index[1] + layer_offset * sign}")
            layer_slice = tuple(slice_list)
            tmp_new_layer = new_layer
            for index, dimension in enumerate(new_anno[layer_slice].shape):
                if dimension <= 1:
                    # Is this deprecated?
                    # https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims
                    print(f"Shape is {tmp_new_layer.shape}")
                    print(f"Expanding index {index}")
                    tmp_new_layer = np.expand_dims(tmp_new_layer, index)
                    print(f"Shape is now {tmp_new_layer.shape}")
            new_anno[layer_slice] = tmp_new_layer
            # new_anno[layer_slice] = new_layer[..., np.newaxis] # Change shape from [...] to [..., 1], which was lost to squishing

    # Add header to create an actual annotation
    new_annotation_object = nib.nifti1.Nifti1Image(
        new_anno, annotation.affine, header=annotation.header
    )
    return new_annotation_object


def plot_binary_voxels(img: np.ndarray, name: str, zooms=None):
    # https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
    def crop2(dat, clp=True):
        if clp:
            np.clip(dat, 0, 1, out=dat)
        for i in range(dat.ndim):
            dat = np.swapaxes(dat, 0, i)  # send i-th axis to front
            while np.all(dat[0] == 0):
                dat = dat[1:]
            while np.all(dat[-1] == 0):
                dat = dat[:-1]
            dat = np.swapaxes(dat, 0, i)  # send i-th axis to its original position
        return dat

    # https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection="3d")
    cropped = crop2(img)
    ax.voxels(cropped, facecolors="red", edgecolor="k", linewidth=0.1)
    # TODO: I don't think this scaling works
    if zooms is not None:
        ratios = [axis / zooms[0] for axis in zooms]
        ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] * ratios[0]])
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * ratios[1]])
        ax.set_zlim([ax.get_zlim()[0], ax.get_zlim()[1] * ratios[2]])
    fig.savefig(name)
    print("showing figure")
    # plt.show()
    plt.close(fig)


# Generate 3D annotation from 2D annotation
def twoD23D(annotation: nib.nifti1.Nifti1Image) -> nib.nifti1.Nifti1Image:
    # Check if this needs to be reduced
    if dimensions_used(annotation) != 2:
        # raise NotImplementedError("Non-2D images are not yet supported")
        print("Warning: Non-2D image")

    # return inverse_distance_transform(annotation)
    try:
        return inverse_distance_transform_alt(annotation)
    except Exception as ex:
        print(f"An error occured while 3D-ifying the annotation: {ex}")
        raise ex
    # return max_radius_erosion(annotation)


def watershed_by_annotation(image: nib.nifti1.Nifti1Image, annotation):
    # Original Plan:
    # Invert annotation
    # Skeletonize original annotation
    # Add inverted to skeletonized annotation, to get outside and inside markers

    # WARNING: Doesn't really work

    img = image.get_fdata()
    _, anno_slice = get_largest_area(annotation)

    anno_original = np.copy(annotation)
    anno_inv = smorph.skeletonize_3d(anno_original.max() - anno_original)
    anno_skel = smorph.skeletonize_3d(annotation)
    anno_mask = np.add(anno_inv, anno_skel)
    # anno_mask = np.copy(annotation)

    # Save markers to check

    anno_img = anno_mask[anno_slice]

    sio.imsave("marker_test.png", anno_img)

    return watershed(img, markers=anno_mask)


def img2nd(path):
    try:
        epi_img = nib.load(path)
    except:
        print(f'Could not read "{path}"', file=sys.stderr)
        exit(1)

    return epi_img.get_fdata()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers()
    cdim_p = sp.add_parser(
        "cdim",
        help="Count dimensions with >1 elements. An image could be reduced to this number of dimensions, without losing elements.",
    )
    cdim_p.add_argument(
        "image",
        type=str,
        nargs="+",
        help="Input image",
        default="/media/watson/Dataset_V2/MR100A.nii.gz",
    )
    # cdim_p.add_argument("image", type=str, nargs="+", help="Input image", default="/media/watson/Dataset_V2/MR100A.nii.gz")

    args = ap.parse_args()
    for image in args.image:
        annotation_path = image[:-7] + "A" + image[-7:]
        img: nib.nifti1.Nifti1Image = nib.load(image)
        anno: nib.nifti1.Nifti1Image = nib.load(annotation_path)
        if dimensions_used(anno) != 2:
            print(f"Image is {dimensions_used(anno)}D; skipping...")
        else:
            print(img.header.get_zooms())
            # anno_watershed = watershed_by_annotation(img.get_fdata(), anno.get_fdata())
            # anno_watershed = nib.nifti1.Nifti1Image(anno_watershed, anno.affine, anno.header)
            # anno_watershed.to_filename(annotation_path + "watershed.nii.gz")
        print(f"{image}:\t{dimensions_used(img)}")
        print(f"{image}:\t{get_largest_area(img)}")
        new_anno = max_radius_erosion(anno)
        new_anno.to_filename(annotation_path + "erosion.nii.gz")
        plot_binary_voxels(
            new_anno.get_fdata(),
            annotation_path + "erosion.png",
            zooms=new_anno.header.get_zooms(),
        )
