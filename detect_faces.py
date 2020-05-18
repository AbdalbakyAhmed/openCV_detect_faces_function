# The OpenCV library contains mechanisms to do face detection on
# images. The technique used is based on Haar cascades, which is a machine learning approach.
# OpenCV comes with trained models for detecting faces, eyes, and smiles which we'll be using.
# You can train models for detecting other things - like hot dogs or flutes - and if you're
# interested in that I'd recommend you check out the Open CV docs on how to train a cascade
# classifier: https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html
import cv2 as cv
from PIL import Image,ImageDraw
from IPython.display import display
# def display(image_to_show):
#     image_to_show.show()

# load opencv and the XML-based classifiers
face_cascade = cv.CascadeClassifier('data_dir/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('data_dir/haarcascade_eye.xml')

img = cv.imread('data_dir/floyd.jpg')
# convert it to grayscale using the cvtColor image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# use the detectMultiScale() function. This function returns
# a list of objects as rectangles. The first parameter is an ndarray of the image.
faces = face_cascade.detectMultiScale(gray)
# OpenCV is return the coordinates as (x,y,w,h), while PIL.ImageDraw is looking (x1,y1,x2,y2). 
pil_img=Image.fromarray(gray,mode="L")
# Setup our drawing context
drawing=ImageDraw.Draw(pil_img)
rec = faces.tolist()[0]
# And draw the new box
drawing.rectangle((rec[0],rec[1],rec[0]+rec[2],rec[1]+rec[3]), outline="white")
display(pil_img)

##
# Lets try this on something a bit more complex, lets read in our MSI recruitment image
##
# It turns out that the root of this error is that OpenCV can't work with Gif images. that we could
# just open this in PIL and then save it as a png.
pil_img=Image.open('data_dir/msi_recruitment.gif')
# Lets convert it to RGB mode
pil_img = pil_img.convert("RGB")
drawing=ImageDraw.Draw(pil_img)
# And iterate through the faces sequence, tuple 
for x,y,w,h in faces:
    drawing.rectangle((x,y,x+w,y+h), outline="white")
display(pil_img)


# There are a few ways we could try and improve this, and really, it requires a lot of 
# experimentation to find good values for a given image. First, lets create a function
# which will plot rectanges for us over the image
def show_rects(faces):
    #Lets read in our gif and convert it
    pil_img=Image.open('data_dir/msi_recruitment.gif').convert("RGB")
    # Set our drawing context
    drawing=ImageDraw.Draw(pil_img)
    # And plot all of the rectangles in faces
    for x,y,w,h in faces:
        drawing.rectangle((x,y,x+w,y+h), outline="white")
    #Finally lets display this
    display(pil_img)

# The detectMultiScale() function from OpenCV also has a couple of parameters. The first of
# these is the scale factor. The scale factor changes the size of rectangles which are
# considered against the model, that is, the haarcascades XML file.
pil_img=Image.open('data_dir/msi_recruitment.gif')
open_cv_version=pil_img.convert("L")
# now lets just write that to a file
open_cv_version.save("msi_recruitment.png")
cv_img=cv.imread('msi_recruitment.png')

# Lets experiment with the scale factor. Usually it's a small value, lets try 1.05
faces = face_cascade.detectMultiScale(cv_img,1.05)
# Show those results
show_rects(faces)
# Now lets also try 1.15
faces = face_cascade.detectMultiScale(cv_img,1.15)
# Show those results
show_rects(faces)
# Finally lets also try 1.25
faces = face_cascade.detectMultiScale(cv_img,1.25)
# Show those results
show_rects(faces)
