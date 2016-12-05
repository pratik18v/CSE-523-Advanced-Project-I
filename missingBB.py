import csv
import shutil

bb_names = []
with open('1stbb1.csv','rb') as csvFile:
    reader = csv.reader(csvFile,delimiter=' ',quotechar='|')
    for row in reader:
        row = ''.join(r for r in str(row) if r not in "['']")
        bb_names.append(row);
        #print row
print len(bb_names)

im_names = []
with open('images.csv','rb') as csvFile:
    reader = csv.reader(csvFile,delimiter=' ',quotechar='|')
    for row in reader:
        row = ''.join(r for r in str(row) if r not in "['']")
        im_names.append(row);
        #print row
print len(im_names)

missing_names = []
for i in im_names:
    #print i.split(".")[0] + "_0.jpg"
    if i.split(".")[0] + "_0.jpg" not in bb_names:
        missing_names.append(i)
        #print i

print len(missing_names)

for i in missing_names:
    shutil.copy('../dataset/MPI/images/'+i,'../dataset/MPI/images/cropped1/'+i.split(".")[0] + "_0.jpg")
    print "Copied {} successfully".format(i)
