from  wsj_loader import *
import os
DATA_FOLDER_PATH = '../data'
import numpy as np
wsj = WSJ()

# train wsj.train,
# test, wsj.test,
dev = wsj.dev

def flatten_data_optimizer(data=(), padding=20, freq_band=40, filename="", Limit=0):
    curr_idx = 0

    IS_TEST_FILE = True if data[1] is None else False  # File has only features no labels
    total_frames = sum([i.shape[0] for i in data[0]])
    num_padded_rows = total_frames + len(data[0] ) * 2 *padding # num_frames_utterance + 2*k*num_utterance

    X = np.zeros((num_padded_rows, freq_band))
    Y = np.zeros((total_frames))
    Lookup = np.zeros((total_frames))
    padding_frame = np.zeros((padding ,freq_band))
    X_idx, Y_idx, LU_idx = 0, 0, 0  # not using LU_idx as Y_idx and LU_idx point to same
    for idx in range(data[0].shape[0] if Limit == 0 else  Limit):
        #         print("Working on Frame number ",idx)
        curr_idx += padding
        frame = data[0][idx]

        X_idx = X_idx + padding_frame.shape[0]
        X[X_idx: X_idx+ frame.shape[0]] = frame
        idxes_range = np.array(range(X_idx, X_idx + frame.shape[0]))
        Lookup[Y_idx: Y_idx + frame.shape[0]] = idxes_range

        X_idx = X_idx + padding_frame.shape[0]

        if IS_TEST_FILE == False:
            labels = data[1][idx]
            Y[Y_idx: Y_idx + frame.shape[0]] = labels

        Y_idx = Y_idx + frame.shape[0]

    # np.save(os.path.join(DATA_FOLDER_PATH, 'flattened', filename + "Flattened_X"), arr=X)
    print("*****************Saved X file **********************")
    print("\tSize : ", X.shape)
    #     np.save(os.path.join(DATA_FOLDER_PATH, 'flattened', filename + "Flattened_Lookup"), arr=Lookup)
    print("*****************Saved Lookup file **********************")
    print("\tSize : ", Lookup.shape)
    if IS_TEST_FILE == False:
        #         np.save(os.path.join(DATA_FOLDER_PATH, 'flattened', filename + "Flattened_Y"), arr=Y)
        print("*****************Saved Labels file **********************")
        print("\tSize : ", Y.shape)
    # del X, Lookup, Y
    return X, Lookup, Y

X, L, Y = flatten_data_optimizer(data=dev, filename='dev')