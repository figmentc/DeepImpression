from PIL import Image, ImageEnhance
image = Image.open('bieber.jpg')

# An enhancement factor of 0.0 gives a black and white image.
# A factor of 1.0 gives the original image.
colour = ImageEnhance.Color(image)
colour.enhance(0.0).show()

# contrast: An enhancement factor of 0.0 gives a solid grey image.
# A factor of 1.0 gives the original image.
contrast = ImageEnhance.Contrast(image)
contrast.enhance(2).show()

# Maximum sharpness
contrast = ImageEnhance.Sharpness(image)
contrast.enhance(2).show()

# minimum sharpness
contrast = ImageEnhance.Sharpness(image)
contrast.enhance(0).show()

# An enhancement factor of 0.0 gives a black image.
# A factor of 1.0 gives the original image.
contrast = ImageEnhance.Brightness(image)
contrast.enhance(0.5).show()
