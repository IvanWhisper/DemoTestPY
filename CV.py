#import library - MUST use cv2 if using opencv_traincascade
import cv2
# rectangle color and stroke
color = (0,0,255)	   # reverse of RGB (B,G,R) - weird
strokeWeight = 1		# thickness of outline
# set window name
windowName = "Object Detection"
# load an image to search for faces
img = cv2.imread(r".\testpic\test8.jpg")
# load detection file (various files for different views and uses)
cascade = cv2.CascadeClassifier(r".\data\haarcascades_cuda\haarcascade_frontalface_alt.xml")
# preprocessing, as suggested by: http://www.bytefish.de/wiki/opencv/object_detection
# img_copy = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
# gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)
# detect objects, return as list
rects = cascade.detectMultiScale(img)
print(rects)
# display until escape key is hit
while True:
	# get a list of rectangles
	for x,y, width,height in rects:
		#print(x,y, width,height)
		cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
	# display!
	cv2.imshow(windowName, img)
	cv2.imwrite(r".\DonePic\img.jpg",img)


	# escape key (ASCII 27) closes window
	if cv2.waitKey(20) == 27:
		break
# if esc key is hit, quit!
exit()