import math
from PIL import Image

def get_pixel(image, row, col, out_of_bounds="zero"):
    """
    Retrieve a pixel's value at a specified row and column from an image.
    """
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
    """
    Set the value of a pixel at a specified row and column in an image.
    """
    width = image["width"]
    image["pixels"][row * width + col] = color


def apply_per_pixel(image, func):
    """
    Apply a given function to every pixel in an image.
    """
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
                        col_num + kernel_col - kw_offset,
                        boundary_behavior,
                    )
                    sum_val += pixel_value * kernel[kernel_row][kernel_col]
            result["pixels"][row_num * width + col_num] = sum_val
    return result


def round_and_clip_image(image):
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
    k1 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    k2 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    kernel1 = correlate(image, k1, "extend")
    kernel2 = correlate(image, k2, "extend")

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [0] * (image["height"] * image["width"]),
    }

    for i in range(image["height"]):
        for j in range(image["width"]):
            val1 = get_pixel(kernel1, i, j)
            val2 = get_pixel(kernel2, i, j)
            final_value = math.sqrt(val1**2 + val2**2)
            set_pixel(result, i, j, final_value)

    return round_and_clip_image(result)


# VARIOUS FILTERS
def reshape_pixels_to_rgb(pixels):
    """Converts a flat list [r, g, b, r, g, b, ...] into a list of (r, g, b) tuples."""
    return [(pixels[i], pixels[i + 1], pixels[i + 2]) for i in range(0, len(pixels), 3)]


def split_image_into_colors(image):
    """Splits an image into its three red, green, and blue respective pixels"""
    # for pixel in image["pixels"]:
    #     if not isinstance(pixel, tuple) or len(pixel) != 3:
    #         print(f"Unexpected pixel data: {pixel}")

    width, height = image["width"], image["height"]
    template = {"height": height, "width": width, "pixels": []}

    red = template.copy()
    green = template.copy()
    blue = template.copy()

    if isinstance(image["pixels"][0], int):
        image["pixels"] = [
            (image["pixels"][i], image["pixels"][i + 1], image["pixels"][i + 2])
            for i in range(0, len(image["pixels"]), 3)
        ]

    # red['pixels'] = [pixel[0] for pixel in image['pixels']]
    red["pixels"], green["pixels"], blue["pixels"] = zip(*image["pixels"])

    return red, green, blue


def combine_image(red, green, blue):
    colored_image = {"height": red["height"], "width": red["width"], "pixels": []}

    for r, g, b in zip(red["pixels"], green["pixels"], blue["pixels"]):
        colored_image["pixels"].append((r, g, b))

    return colored_image


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def color_filter(image):
        red_comp, green_comp, blue_comp = split_image_into_colors(image)

        red_filtered = filt(red_comp)
        green_filtered = filt(green_comp)
        blue_filtered = filt(blue_comp)

        return combine_image(red_filtered, green_filtered, blue_filtered)

    return color_filter


def make_blur_filter(kernel_size):
    def blur_filter(img):
        return blurred(img, kernel_size)

    return blur_filter
    # return color_filter_from_greyscale_filter(blur_filter)


def make_sharpen_filter(kernel_size):
    def sharpen_filter(img):
        return sharpened(img, kernel_size)

    return sharpen_filter
    # return color_filter_from_greyscale_filter(sharpen_filter)


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def cascaded_filter(image):
        output_image = image
        for filt in filters:
            output_image = filt(output_image)
        return output_image

    return cascaded_filter


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    for _ in range(ncols):
        grey_img = greyscale_image_from_color_image(image)
        edges_img = compute_energy(grey_img)
        cem = cumulative_energy_map(edges_img)
        minimum_energy = minimum_energy_seam(cem)
        # print(minimum_energy)
        final_img = image_without_seam(image, minimum_energy)
        image = final_img
        # print(image)
    return final_img


# Optional Helper Functions for Seam Carving
def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    grayscale_pixels = [
        round(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in image["pixels"]
    ]
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": grayscale_pixels,
    }


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    width, height = energy["width"], energy["height"]
    cem_pixels = list(energy["pixels"])
    for y in range(1, height):
        for x in range(width):
            if x == 0:
                min_adjacent = min(
                    cem_pixels[(y - 1) * width], cem_pixels[(y - 1) * width + 1]
                )
            elif x == width - 1:
                min_adjacent = min(
                    cem_pixels[(y - 1) * width + x - 1], cem_pixels[(y - 1) * width + x]
                )
            else:
                min_adjacent = min(
                    cem_pixels[(y - 1) * width + x - 1],
                    cem_pixels[(y - 1) * width + x],
                    cem_pixels[(y - 1) * width + x + 1],
                )

            cem_pixels[y * width + x] += min_adjacent
    return {"height": height, "width": width, "pixels": cem_pixels}


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    # print(cem)

    width, height = cem["width"], cem["height"]
    seam = [0] * height
    seam[-1] = min(range(width), key=lambda x: cem["pixels"][(height - 1) * width + x])

    for y in range(height - 2, -1, -1):
        x = seam[y + 1]
        if x == 0:
            seam[y] = min(x, x + 1, key=lambda col, y=y: cem["pixels"][y * width + col])
        elif x == width - 1:
            seam[y] = min(x - 1, x, key=lambda col, y=y: cem["pixels"][y * width + col])
        else:
            seam[y] = min(
                x - 1, x, x + 1, key=lambda col, y=y: cem["pixels"][y * width + col]
            )
    return [y * width + x for y, x in enumerate(seam)]


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    width, height = image["width"], image["height"]
    new_pixels = [pixel for i, pixel in enumerate(image["pixels"]) if i not in seam]

    return {"height": height, "width": width - 1, "pixels": new_pixels}


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def custom_feature(image, intensity=1.5):
    """
    Apply a vignette effect to an image.

    :param intensity: strength of the vignette effect. Default is 1.5.
    """
    # Open the image
    width, height = image["width"], image["height"]
    pixels = image["pixels"].copy()

    # Calculate the center of the image
    center_x, center_y = width / 2, height / 2

    # Apply vignette effect
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[y * width + x]

            # Calculate the distance to the center of the image
            dist_to_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            # Calculate the scaling factor based on the distance
            scale = 1 - intensity * dist_to_center / max(center_x, center_y)
            r = int(r * scale)
            g = int(g * scale)
            b = int(b * scale)

            # Set the new pixel value
            pixels[y * width + x] = (r, g, b)

    return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
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
    by the 'mode' parameter.
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
    print("Testing starts here ")
    # # Testing color_filter_from_greyscale_filter
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # inverted_color_cat = color_inverted(load_color_image('test_images/cat.png'))
    # save_color_image(inverted_color_cat, "inverted_cat.png", mode="PNG")

    # # Testing blurred with color_filter_from_greyscale_filter
    # blur_filter = make_blur_filter(9)
    # blurry2 = blur_filter(load_color_image('test_images/python.png'))
    # save_color_image(blurry2, "blurred_python.png", mode="PNG")

    # Testing sharpened with color_filter_from_greyscale_filter
    # sharpened_filter = make_sharpen_filter(7)
    # sharppy2 = sharpened_filter(load_color_image('test_images/sparrowchick.png'))
    # save_color_image(sharppy2, "sharppy_sparrowchick.png", mode="PNG")

    # # Testing cascade filters
    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = make_blur_filter(5)
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # final = filt(load_color_image('test_images/frog.png'))
    # save_color_image(final, "cascade_frog.png", mode="PNG")

    ## Testing the seam functionalities
    # pattern = load_color_image("test_images/centered_pixel.png")
    # final = seam_carving(pattern, 100)
    # save_color_image(final, "seam.png", mode="PNG")
    # print(pattern)

    # # Testing custom feature
    # tree = load_color_image("test_images/tree.png")
    # vignette_tree = custom_feature(tree, 1.5)
    # save_color_image(vignette_tree, "test.png", mode="PNG")
