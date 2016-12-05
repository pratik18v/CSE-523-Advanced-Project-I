import scipy.io as sio
import numpy as np
import json
import lmdb
import caffe
import os.path
import struct
import csv
import cv2


def writeLMDB(datasets, lmdb_path, validation):
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    data = []
    im_names = []
    with open('output.csv','rb') as csvFile:
        reader = csv.reader(csvFile,delimiter=' ',quotechar='|')
        for row in reader:
            row = ''.join(r for r in str(row) if r not in "['\n']")
            im_names.append(row);

    for d in range(len(datasets)):
        print datasets[d]
        with open('json/MPI_annotations.json') as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            data = data + data_this
        numSample = len(data)
        print numSample
        #print data[0]['img_paths']

    if numSample!=len(im_names):
        print "----Error!!----"


    isValidationArray = [data[i]['isValidation'] for i in range(numSample)];
    if(validation == 1):
        totalWriteCount = isValidationArray.count(0.0);
    else:
        totalWriteCount = len(data)
    print 'going to write %d images..' % totalWriteCount;

    writeCount = 0
    json_names = []
    for jname in range(numSample):
        json_names.append(data[jname]['img_paths'])

    #print json_names.index('066294927.jpg')

    for count in range(numSample):
        im_name = im_names[count].split('_')[0] + '.jpg'
        #print im_name

        idx = json_names.index(im_name)
        #print idx

        if (data[idx]['isValidation'] != 0 and validation == 1):
            print '%d/%d skipped' % (count,idx)
            continue
        #print idx

        path_header = '../dataset/MPI/images/cropped1/'

        img = cv2.imread(os.path.join(path_header, im_names[count]))
        #print os.path.join(data[idx]['img_paths'])

        height = img.shape[0]
        width = img.shape[1]
        #print 'Image: {}, BB: {}'.format(img.shape,bb_img.shape)

        if(width < 64):
            img = cv2.copyMakeBorder(img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
            #bb_img = cv2.copyMakeBorder(bb_img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
            print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            cv2.imwrite('padded_img.jpg', img)
            #cv2.imwrite('padded_bb_img.jpg',bb_img)
            width = 64
            # no modify on width, because we want to keep information

        meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
        #print 'image', type(img), img.shape
        #print 'metadata', type(meta_data), meta_data.shape
        clidx = 0 # current line index
        # dataset name (string)
        for i in range(len(data[idx]['dataset'])):
            meta_data[clidx][i] = ord(data[idx]['dataset'][i])
        clidx = clidx + 1

        # image height, image width
        height_binary = float2bytes(data[idx]['img_height'])
        for i in range(len(height_binary)):
            meta_data[clidx][i] = ord(height_binary[i])
        width_binary = float2bytes(data[idx]['img_width'])
        for i in range(len(width_binary)):
            meta_data[clidx][4+i] = ord(width_binary[i])
        clidx = clidx + 1
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
        meta_data[clidx][0] = data[idx]['isValidation']
        meta_data[clidx][1] = data[idx]['numOtherPeople']
        meta_data[clidx][2] = data[idx]['people_index']
        annolist_index_binary = float2bytes(data[idx]['annolist_index'])
        for i in range(len(annolist_index_binary)): # 3,4,5,6
            meta_data[clidx][3+i] = ord(annolist_index_binary[i])
        count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count!
        for i in range(len(count_binary)):
            meta_data[clidx][7+i] = ord(count_binary[i])
        totalWriteCount_binary = float2bytes(float(totalWriteCount))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11+i] = ord(totalWriteCount_binary[i])
        nop = int(data[idx]['numOtherPeople'])
        clidx = clidx + 1
        # (b) objpos_x (float), objpos_y (float)
        objpos_binary = float2bytes(data[idx]['objpos'])
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = ord(objpos_binary[i])
        clidx = clidx + 1
        # (c) scale_provided (float)
        scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = ord(scale_provided_binary[i])
        clidx = clidx + 1

        # (d) joint_self (3*16) or (3*22) (float) (3 line)
        joints = np.asarray(data[idx]['joint_self']).T.tolist() # transpose to 3*16
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = ord(row_binary[j])
            clidx = clidx + 1
        # (e) check nop, prepare arrays
        if(nop!=0):
            if(nop==1):
                joint_other = [data[idx]['joint_others']]
                objpos_other = [data[idx]['objpos_other']]
                scale_provided_other = [data[idx]['scale_provided_other']]
            else:
                joint_other = data[idx]['joint_others']
                objpos_other = data[idx]['objpos_other']
                scale_provided_other = data[idx]['scale_provided_other']
            # (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
            for i in range(nop):
                objpos_binary = float2bytes(objpos_other[i])
                for j in range(len(objpos_binary)):
                    meta_data[clidx][j] = ord(objpos_binary[j])
                clidx = clidx + 1
            # (g) scale_provided_other (nop floats in 1 line)
            scale_provided_other_binary = float2bytes(scale_provided_other)
            for j in range(len(scale_provided_other_binary)):
                meta_data[clidx][j] = ord(scale_provided_other_binary[j])
            clidx = clidx + 1
            # (h) joint_others (3*16) (float) (nop*3 lines)
            for n in range(nop):
                joints = np.asarray(joint_other[n]).T.tolist() # transpose to 3*16
                for i in range(len(joints)):
                    row_binary = float2bytes(joints[i])
                    for j in range(len(row_binary)):
                        meta_data[clidx][j] = ord(row_binary[j])
                    clidx = clidx + 1

        # print meta_data[0:12,0:48]
        # total 7+4*nop lines

        #print img.shape
        #img_comb = np.concatenate((img,bb_img), axis=2)
        #print img_comb.shape
        img4ch = np.concatenate((img, meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        #print img4ch.shape
        datum = caffe.io.array_to_datum(img4ch, label=0)
        key = '%07d' % writeCount
        txn.put(key, datum.SerializeToString())
        if(writeCount % 1000 == 0):
            txn.commit()
            txn = env.begin(write=True)
        print 'count(count): %d/ write count(writeCount): %d/ randomizedi(idx): %d/ all(totalWriteCount): %d' % (count,writeCount,idx,totalWriteCount)

        #print 'count: %d/ write count: %d/ randomized: %d/ all: %d' % (count,writeCount,idx,totalWriteCount)
        writeCount = writeCount + 1

    txn.commit()
    env.close()


def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats)

if __name__ == "__main__":

	#writeLMDB(['MPI'], 'lmdb/MPI_train_split', 1) # only include split training data (validation data is held out)
	writeLMDB(['MPI'], 'lmdb/MPI_bb', 0)
	#writeLMDB(['LEEDS'], 'lmdb/LEEDS_PC', 0)
	#writeLMDB(['FLIC'], 'lmdb/FLIC', 0)

	#writeLMDB(['MPI', 'LEEDS'], 'lmdb/MPI_LEEDS_alltrain', 0) # joint dataset
