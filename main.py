import cv2
import face_recognition

img=cv2.imread("ronaldo.jpg")
rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding=face_recognition.face_encodings(rgb_img)[0]


img2=cv2.imread("ronaldo1.jpeg")
rgb_img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img_encoding2=face_recognition.face_encodings(rgb_img2)[0]

# Use face_recognition to detect faces
face_locations = face_recognition.face_locations(rgb_img)
print("Found {} face(s) in the image.".format(len(face_locations)))

result=face_recognition.compare_faces([img_encoding2],img_encoding)
print("Result: ", result)

# cv2.imshow("RGB Image 1", rgb_img)
# cv2.imshow("RGB Image 2", rgb_img2)

# cv2.imshow("Img",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()