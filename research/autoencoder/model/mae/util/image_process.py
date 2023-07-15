from math import sqrt

import torch


def patchify(images, patch_size=16):
    """
    Splitting images into patches.

    Args:
        images: Input tensor with size (batch, channels, height, width)
            We can assume that image is square where height == width.
        patch_size: Patch size
    Returns:
        A batch of image patches with size (
          batch, (height / patch_size) * (width / patch_size),
        channels * patch_size * patch_size)
    """
    batch_size, channels, height, width = images.shape

    # Raise error if image is not square or patch_size is not a divisor of height and width
    assert width == height and width % patch_size == 0

    # Count number of patches in height and width
    height_count = width_count = width // patch_size

    # Splitting images into patches
    images = images.reshape((batch_size, channels, height_count, patch_size, width_count, patch_size))
    images = torch.einsum('nchpwq->nhwpqc', images)
    images = images.reshape((batch_size, height_count * width_count, patch_size ** 2 * channels))

    return images


def unpatchify(patches, patch_size=16):
    """
    Combining patches into images.

    Args:
        patches: Input tensor with size (
            batch,
            (height / patch_size) * (width / patch_size),
            channels * patch_size * patch_size
            )
        patch_size: Patch size
    Returns:
        A batch of images with size (batch, channels, height, width)
    """
    batch_size, num_patches, total_patch_size = patches.shape
    channels = total_patch_size // patch_size ** 2

    # Count number of patches in height and width
    height_count = width_count = int(sqrt(num_patches))

    # Calculate height and width of the image
    height = width = height_count * patch_size

    # Raise error if num_patches is not a square number
    assert height_count * width_count == num_patches

    # Unpatching patches into images
    patches = patches.reshape((batch_size, height_count, width_count, patch_size, patch_size, channels))
    patches = torch.einsum('nhwpqc->nchpwq', patches)
    images = patches.reshape((batch_size, channels, height, width))

    return images


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.

    Args:
        x: torch.Tensor, shape [N, L, D]
           The input sequence with embedded vector of elements of samples in batch.
           N: batch size, L: embedded vector length of a sample (number of elements), D: encoder embedding dimension

        mask_ratio: float
           The ratio of elements to be masked (removed) from each sample.

    Returns:
        kept_patches: torch.Tensor, shape [N, L_masked, D]
           The masked sequence with a subset of elements removed for each sample.
           L_masked: masked sequence length after removing elements.

        mask: torch.Tensor, shape [N, L]
           Binary mask indicating which elements are kept (0) or removed (1) from each sequence.

        ids_restore: torch.Tensor, shape [N, L]
           Indices used to restore the original order of the elements after shuffling.

    """
    batch_size, num_patches, dimension = x.shape
    keep_count = int(num_patches * (1 - mask_ratio))

    noise = torch.rand(batch_size, num_patches, device=x.device)  # noise in [0, 1]

    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :keep_count]
    kept_patches = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dimension))

    # Generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([batch_size, num_patches], device=x.device)
    mask[:, :keep_count] = 0
    # Un-shuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return kept_patches, mask, ids_restore


def restore_masked(kept_x, masked_x, ids_restore):
    """
    Restore masked patches

    Args:
        kept_x: unmasked patches
        masked_x: masked patches
        ids_restore: indices to restore x
    Returns:
        restored patches
    """
    x = torch.cat((kept_x, masked_x), dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

    return x
