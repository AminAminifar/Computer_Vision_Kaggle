# pip install scikit-image
from skimage import draw, transform
import numpy as np
import tensorflow as tf


# from visiontools
def circle(size, val=None, r_shrink=0):
    circle = np.zeros([size[0]+1, size[1]+1])
    rr, cc = draw.circle_perimeter(
        size[0]//2, size[1]//2,
        radius=size[0]//2 - r_shrink,
        shape=[size[0]+1, size[1]+1],
    )
    if val is None:
        circle[rr, cc] = np.random.uniform(size=circle.shape)[rr, cc]
    else:
        circle[rr, cc] = val
    circle = transform.resize(circle, size, order=0)
    return circle

def random_transform(image, jitter=0, rotate=0, scale=1, **kwargs):
    jx = tf.random.uniform([], -jitter, jitter)
    jy = tf.random.uniform([], -jitter, jitter)
    r = tf.random.uniform([], -rotate, rotate)
    s = tf.random.uniform([], 1.0, scale)
    image = apply_affine_transform(
        image,
        theta=r,
        tx=jx, ty=jy,
        zx=s, zy=s,
        **kwargs,
    )
    return image

def random_transform(image, jitter=0, rotate=0, scale=1, **kwargs):
    jx = tf.random.uniform([], -jitter, jitter)
    jy = tf.random.uniform([], -jitter, jitter)
    r = tf.random.uniform([], -rotate, rotate)
    s = tf.random.uniform([], 1.0, scale)
    image = apply_affine_transform(
        image,
        theta=r,
        tx=jx, ty=jy,
        zx=s, zy=s,
        **kwargs,
    )
    return image

def apply_affine_transform(x,
                           theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_method='reflect', cval=0.,
                           interpolation_method='nearest'):
    """ Apply an affine transformation to an image x. """

    theta = tf.convert_to_tensor(theta, dtype=tf.float32)
    tx = tf.convert_to_tensor(tx, dtype=tf.float32)
    ty = tf.convert_to_tensor(ty, dtype=tf.float32)
    shear = tf.convert_to_tensor(shear, dtype=tf.float32)
    zx = tf.convert_to_tensor(zx, dtype=tf.float32)
    zy = tf.convert_to_tensor(zy, dtype=tf.float32)

    transform_matrix = _get_inverse_affine_transform(
        theta,
        tx, ty,
        shear,
        zx, zy,
    )

    x = _apply_inverse_affine_transform(
        x,
        transform_matrix,
        fill_method=fill_method,
        interpolation_method=interpolation_method,
    )

    return x

def _get_inverse_affine_transform(theta, tx, ty, shear, zx, zy):
    """ Construct the inverse of the affine transformation matrix with the given transformations. 
    
    The transformation is taken with respect to the usual right-handed coordinate system."""

    transform_matrix = tf.eye(3, dtype=tf.float32)

    if theta != 0:
        theta = theta * math.pi / 180 # convert degrees to radians
        # this is 
        rotation_matrix = tf.convert_to_tensor(
            [[tf.math.cos(theta), tf.math.sin(theta), 0],
             [-tf.math.sin(theta), tf.math.cos(theta), 0],
             [0, 0, 1]],
            dtype=tf.float32)
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = tf.convert_to_tensor(
            [[1, 0, -tx],
             [0, 1, -ty],
             [0, 0, 1]],
            dtype=tf.float32)
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = tf.matmul(transform_matrix, shift_matrix)

    if shear != 0:
        shear = shear * math.pi / 180 # convert degrees to radians
        shear_matrix = tf.convert_to_tensor(
            [[1, tf.math.sin(shear), 0],
             [0, tf.math.cos(shear), 0],
             [0, 0, 1]],
            dtype=tf.float32)
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = tf.matmul(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        # need to assert !=0
        zoom_matrix = tf.convert_to_tensor(
            [[1/zx, 0, 0],
             [0, 1/zy, 0],
             [0, 0, 1]],
            dtype=tf.float32)
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = tf.matmul(transform_matrix, zoom_matrix)
            
    return transform_matrix

def _apply_inverse_affine_transform(A, Ti, fill_method, interpolation_method):
    """Perform an affine transformation of the image A defined by a
transform whose inverse is Ti. The matrix Ti is assumed to be in
homogeneous coordinate form.

    Available fill methods are "replicate" and "reflect" (default).
    Available interpolation method is "nearest".

    """
    nrows, ncols, _ = A.shape

    # Create centered coordinate grid
    x = tf.range(ncols*nrows) % ncols
    x = tf.cast(x, dtype=tf.float32) - ((ncols-1)/2) # center
    y = tf.range(ncols*nrows) // ncols
    y = tf.cast(y, dtype=tf.float32) - ((nrows-1)/2) # center
    y = -y # left-handed to right-handed coordinates
    z = tf.ones([ncols*nrows], dtype=tf.float32)
    grid = tf.stack([x, y, z])

    # apply transformation
    # x, y, _ = tf.matmul(Ti, grid)
    xy = tf.matmul(Ti, grid)
    x = xy[0, :]
    y = xy[1, :]
    
    # convert coordinates to (approximate) indices
    i = -y + ((nrows-1)/2)
    j = x + ((ncols-1)/2)

    # replicate: 111|1234|444
    if fill_method is 'replicate':
        i = tf.clip_by_value(i, 0.0, nrows-1)
        j = tf.clip_by_value(j, 0.0, ncols-1)
    # reflect: 432|1234|321
    elif fill_method is 'reflect':
        i = _reflect_index(i, nrows-1)
        j = _reflect_index(j, ncols-1)
        
    # nearest neighbor interpolation
    grid = tf.stack([i, j])
    grid = tf.round(grid)
    grid = tf.cast(grid, dtype=tf.int32)
    B = tf.gather_nd(A, tf.transpose(grid))
    B = tf.reshape(B, A.shape)

    return B