########################################
##### 기본적으로 필요한 라이브러리 import #####
########################################
from tqdm import tqdm
# import matplotlib.pyplot as plt
import os
import cv2
import skimage
import numpy as np
from glob import glob

########################################
##### 머신러닝 관련 라이브러리 import #####
########################################
import sklearn
import scipy
from skimage.transform import resize
import keras.models as km
import keras.layers as kl

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical

train_dir = "../chest_xray/train/"
test_dir =  "../chest_xray/test/"

# 데이터 크기 (150 x 150)
_N_ROW = 150  # 세로 150
_N_COL = 150  # 가로 150
_N_PIXEL = _N_ROW * _N_COL

def get_data(folder):
    X = []
    y = []

    for folderName in os.listdir(folder):
        print('foldername = ' + folderName)

        if folderName.startswith('.'):
            continue

        if folderName == 'NORMAL':
            label = 0
        elif folderName == 'PNEUMONIA':
            label = 1
        else:
            print("Err! I don't know this data...")
            return

        # out of memory 때문에 커널이 자꾸 죽기 때문에, 이미지 개수를 제한한다.
        count_num = 0
        for image_filename in tqdm(os.listdir(folder + folderName)):
            count_num += 1
            # if count_num > 100:
            #     break
            img_file = cv2.imread(folder + folderName + '/' + image_filename)

            if img_file is not None:
                img_file = skimage.transform.resize(img_file, (_N_ROW, _N_COL, 3))
                img_arr = np.asarray(img_file)
                X.append(img_arr)
                y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)

y_trainHot = to_categorical(y_train, num_classes = 2)
y_testHot = to_categorical(y_test, num_classes = 2)

def confirmErr(realAnswerList, testAnswerList):
    answerNum = 0
    for i, eachAns in enumerate(realAnswerList):
        if testAnswerList[i] == eachAns:
            answerNum += 1
    return float(len(realAnswerList)-answerNum)/float(len(realAnswerList))*100


def study(trDataList, trLabelList, save_h5_name, log_txt_name):
    trDataList = X_train
    trLabelList = y_trainHot

    # 모델 설정
    model = km.Sequential()
    model.add(kl.Conv2D(input_shape=(_N_ROW, _N_COL, 3), filters=32,
                        kernel_size=(5, 5), strides=1))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(filters=10,
                        kernel_size=(5, 5), strides=1))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Activation('relu'))
    model.add(kl.Flatten())
    model.add(kl.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 출력
    print(model.summary())
    print('\n\n\n')

    # 학습 데이터를 추출하여 저장하고 출력한다
    history_callback = model.fit(trDataList, trLabelList, epochs=15, batch_size=200, verbose=2)
    loss_history = history_callback.history
    train_log = ''
    for i in range(len(loss_history['acc'])):
        train_log += "%d epoch _ acc : %.4f, loss : %.4f\n" % (i + 1, loss_history['acc'][i], loss_history['loss'][i])

    # 로그 기록
    fd = open(log_txt_name, 'w')
    fd.write(train_log)
    fd.close()

    # 모델 저장
    km.save_model(model, save_h5_name)

    # 실제 에러율 확인
    print('err is  %f%%' % (confirmErr(np.argmax(trLabelList, axis=1),
                                       np.argmax(model.predict(trDataList), axis=1))))

    return model



_model = study(trDataList = X_test, trLabelList = y_testHot, save_h5_name= 'model.h5', log_txt_name='train_log.txt')

trDataList = X_test
trLabelList = y_testHot

# 실제 에러율 확인
print ('err is  %f%%' % (confirmErr(np.argmax(trLabelList, axis=1),
                                    np.argmax(_model.predict(trDataList), axis=1))) )