#!/usr/bin/env python3

import math

from PIL import Image

def get_pixel(image, row, col, out_of_bounds="zero"):
    width = image["width"]  # true width, remember to subtract 1
    height = image["height"]  # true height, remember to subtract 1

    if 0 <= row < height and 0 <= col < width:
        return image["pixels"][row * width + col]

    if out_of_bounds == "zero":
        return 0
    elif out_of_bounds == "extend":
        return image["pixels"][
            max(min(row, height - 1), 0) * width + max(min(col, width - 1), 0)
        ]
    elif out_of_bounds == "wrap":
        return image["pixels"][row % height * width + col % width]
    else:
        return 0


def set_pixel(image, row, col, color):
    width = image["width"]
    image["pixels"][row * width + col] = color


def apply_per_pixel(image, func):
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"]),  # initialize to zeros
    }
    for row in range(image["height"]):
        for col in range(image["width"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    return apply_per_pixel(image, lambda color: 255 - color)


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    """
    if boundary_behavior not in ["zero", "extend", "wrap"]:
        return None

    height = image["height"]
    width = image["width"]
    result = {
        "height": height,
        "width": width,
        "pixels": [0] * (height * width),
    }

    kernel_height = len(kernel)
    kernel_width = len(kernel[0])
    kh_offset = kernel_height // 2  # how far center is from top of the kernel
    kw_offset = kernel_width // 2  # how far center is from each side of the kernel

    for row_num in range(height):
        for col_num in range(width):
            sum_val = 0
            for kernel_row in range(kernel_height):
                for kernel_col in range(kernel_width):
                    pixel_value = get_pixel(
                        image,
                        row_num + kernel_row - kh_offset,
                        boundary_behavior,
                    )
                    sum_val += pixel_value * kernel[kernel_row][kernel_col]
            result["pixels"][row_num * width + col_num] = sum_val
    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].
    """
    for idx, pixel in enumerate(image["pixels"]):
        rounded_pixel = round(pixel)
        image["pixels"][idx] = max(0, min(255, rounded_pixel))
    return image


def box_kernel(n):
    """Generate an n x n box kernel
    where each element of the kernel is 1/n^2"""
    value = 1.0 / (n * n)
    return [[value] * n for _ in range(n)]


# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    kernel = box_kernel(kernel_size)
    # print(kernel)
    blurred_image = correlate(image, kernel, "extend")
    output_image = round_and_clip_image(blurred_image)
    return output_image


def sharpened(image, n):
    kernel = [[-1 / (n * n) for _ in range(n)] for _ in range(n)]
    kernel[n // 2][
        n // 2
    ] += 2  # 2 times the center pixel to multiply each pixel times 2

    sharpened_image = correlate(image, kernel, "extend")
    output_image = round_and_clip_image(sharpened_image)
    return output_image


def edges(image):
    """
    Return a new image emphasizing edges using the given Sobel operator.
    Kernels k1 and k2 are applied to the image in O1 and O2,
    with the final image consisting of pixels that are the square root of
    the pixels in O1^2 plus O2^2
    """
    k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    k2 = [[-1, 0, 1], 
          [-2, 0, 2], 
          [-1, 0, 1]]

    O1 = correlate(image, k1, "extend")
    O2 = correlate(image, k2, "extend")

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"]),
    }

    for i in range(image["height"]):
        for j in range(image["width"]):
            val1 = get_pixel(O1, i, j)
            val2 = get_pixel(O2, i, j)
            final_value = math.sqrt(val1**2 + val2**2)
            set_pixel(result, i, j, final_value)

    return round_and_clip_image(result)
  

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the "mode" parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    print("Start testing here:  ")

    ## testing greyscale
    # img = load_greyscale_image("test_images/bluegill.png")
    # inverted_img = inverted(img)
    # save_greyscale_image(inverted_img, "test_images/inverted_bluegill.png")

    ## testing correlate
    # kernel_size = 13
    # kernel = []

    # for i in range(kernel_size):
    #     row = []
    #     for j in range(kernel_size):
    #         if i == 2 and j == 0:  # 3rd row and 1st column
    #             row.append(1)
    #         else:
    #             row.append(0)
    #     kernel.append(row)

    # img = load_greyscale_image("test_images/pigbird.png")
    # zero_img = correlate(img, kernel, "zero")
    # extended_img = correlate(img, kernel, "extend")
    # wrapped_img = correlate(img, kernel, "wrap")

    # save_greyscale_image(zero_img, "test_images/zero_pigbird.png")
    # save_greyscale_image(extended_img, "test_images/extended_pigbird.png")
    # save_greyscale_image(wrapped_img, "test_images/wrapped_pigbird.png")

    ## testing blurred
    # img = load_greyscale_image("test_images/cat.png")
    # blurr_img = blurred(img, 13)
    # save_greyscale_image(blurr_img, "test_images/blurred_cat.png")

    ## testing sharpened
    # img = load_greyscale_image("test_images/python.png")
    # sharp_python = sharpened(img,11)
    # save_greyscale_image(sharp_python, "test_images/sharp_python.png")

    ## testing edges
    # image = load_greyscale_image("test_images/construct.png")
    # edge_img = edges(image)
    # save_greyscale_image(edge_img, "test_images/edge_construct.png")
