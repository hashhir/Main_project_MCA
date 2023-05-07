from PIL import Image

input_dims = (299,299)
image = Image.open("./testimages/ISIC_0024326.jpg")
print(image.size)
image = image.resize(input_dims)
image.save("resized.png")