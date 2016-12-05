import csv
import cv2
import numpy as np

bboxes = []
with open('1stbb5.csv','rb') as csvFile:
    reader = csv.reader(csvFile,delimiter=' ',quotechar='|')
    for row in reader:
        row = ''.join(r for r in str(row) if r not in "['']")
        img = cv2.imread('../dataset/MPI/images/cropped1/'+row)
        print "Appending image: {}".format(row)
        bboxes.append(img)

np.save("bboxes5",bboxes)
