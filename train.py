import tensorflow.python.keras as K
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import cv2
from scipy import integrate
from scipy import interpolate
import matplotlib.colors as mcolors
import pandas as pd

from model import ResNet_base_decoder
from cmap import rgb_cmap, rgb_cmap_inverse

def integrating(y_pred, vmin, vmax, step, point):
    num = 256
    we = (2*step)/(num - 1)

    three_axis_pred = []

    for t in range(y_pred.shape[0]):
        z = rgb_cmap_inverse(y_pred[t], vmin, vmax)
        z = cv2.resize(z, (256, 256)) 
        z = cv2.GaussianBlur(z, (9, 9), 1.5) 
        
        # x_axis discretely integrate
        x_s = []
        for i in range(z.shape[0]):
            x_z = integrate.simpson(z[:, i])*we
            x_s.append(x_z) 
        x_s = np.array(x_s)   
        
        # y_axis discretely integrate
        y_s = []
        for i in range(z.shape[1]):
            y_z = integrate.simpson(z[i, :])*we
            y_s.append(y_z) 
        y_s = np.array(y_s)  
        
        # u_axis discretely integrate
        z_s = []
        n1 = [i for i in range(1, num+1, 2)]
        n2 = [i for i in range(num-1, -1, -2)]
      
        z_s = []
        for k in n1:
            t_per1 = []
            for j in range(k):
                t_per1.append(np.flip(z, 0)[(num-1) - (k - 1 -j), j])
            t_per1 = np.array(t_per1)
            t = integrate.simpson(t_per1)*we*np.sqrt(2)
            z_s.append(t)
            
        for k in n2:
            t_per2 = []
            for j in range(k):
                t_per2.append(np.flip(z, 0)[j, (num-1) - (k - 1 -j)])
            t_per2 = np.array(t_per2)
            t = integrate.simpson(t_per2)*we*np.sqrt(2)
            z_s.append(t)
            
        z_s = np.array(z_s)
        mlin = np.linspace(-step, step, 256)
        mlin2 = np.linspace(-step, step, point)       
        mfx = interpolate.interp1d(mlin, x_s, kind = 'linear')
        mfy = interpolate.interp1d(mlin, y_s, kind = 'linear')
        mfu = interpolate.interp1d(mlin, z_s, kind = 'linear')
        mx_new = mfx(mlin2)
        my_new = mfy(mlin2)
        mu_new = mfu(mlin2)
        
        m = np.concatenate((mx_new, my_new, mu_new), axis = 0)
        three_axis_pred.extend([m])
        
    three_axis_pred = np.array(three_axis_pred)
    three_axis_pred = np.reshape(three_axis_pred, (three_axis_pred.shape[0], three_axis_pred.shape[1], 1))
    
    return three_axis_pred

def training(dataset_filename, model_name, epochs = 250, batch_size = 32, data_patch = 4):
    # set save path
    osd = os.getcwd()
    front_path = os.path.abspath(os.path.join(osd, os.path.pardir))
    path_dataset = os.path.join(front_path, dataset_filename)
     
    x_train = []
    y_train = []
    
    for i in range(data_patch):
        x_train += list(np.load(os.path.join(path_dataset, 'x_train%d.npy'%(i+1)), allow_pickle=True))
        y_train += list(np.load(os.path.join(path_dataset, 'y_train%d.npy'%(i+1)), allow_pickle=True))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # load model
    inputs_shape = K.Input(shape = x_train[0].shape)
    decoder_model = ResNet_base_decoder(inputs_shape)
    decoder_model.summary()
    
    # model compile
    decoder_model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), metrics = ['accuracy'], )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    history = decoder_model.fit(x_train, y_train, validation_split = 0.4, epochs = epochs, batch_size = batch_size, verbose = 1, shuffle = True, callbacks = reduce_lr)
    
    # model save
    decoder_model.save('%s.h5'%model_name)
    print("Decoder_model.h5 save in {osd}".format(osd = osd))
    
    # plot historyplt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')    
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
       
    
def test_comparison(train_filename, model_name, image_num = 5, vmin = -0.01, vmax = 0.045, point = 721, step = 4.5):
    # load decoder model h5 file
    decoder_model = load_model('%s.h5'%model_name)

    front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    path_combined_dataset = os.path.join(front_path, train_filename)
    
    x_test = np.load(os.path.join(path_combined_dataset, 'x_test.npy'))
    y_test = np.load(os.path.join(path_combined_dataset, 'y_test.npy'))
    x_train = np.load(os.path.join(path_combined_dataset, 'x_train1.npy'))
    y_train = np.load(os.path.join(path_combined_dataset, 'y_train1.npy'))
    
    y_pred = decoder_model.predict(x_test)
    y_pred_train = decoder_model.predict(x_train)
    y_pre = y_pred
    x_test = x_test
    y_test = y_test

    three_axis_pred = integrating(y_pred, vmin, vmax, step = step, point = point)[:,:,0]
   
    for t in range(image_num):      
        fig , ax = plt.subplots()
        pic1 = t*6 + 1
        pic2 = t*6 + 6
        pictrue_num = pic2 - pic1 + 1 
        for i in range(pictrue_num-1):
            plt.subplot(5, pictrue_num, 1)
            plt.text(0.5, 0.15, 'Ground\ntruth\ndistribution', fontsize=8, ha='center')
            plt.axis('off')
            plt.subplot(5, pictrue_num, i+2)
            plt.imshow(np.flip(y_test[i+pic1-1], 0))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis('auto')
        for i in range(pictrue_num-1):
            plt.subplot(5, pictrue_num, pictrue_num + 1)
            plt.text(0.5, 0.15, 'Predicted\njoint\ndistribution', fontsize=8, ha='center')
            plt.axis('off')
            plt.subplot(5, pictrue_num, i+pictrue_num+2)
            plt.imshow(np.flip(y_pred[i+pic1-1], 0))
            diff = np.sum(abs(y_test[i+pic1-1] - y_pred[i+pic1-1])**2)
            plt.text(245, 10, str(round(diff, 1)), fontsize=6, ha='right', va = 'top')
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis('auto')
        for i in range(pictrue_num-1):
            plt.subplot(5, pictrue_num, 2*pictrue_num + 1)
            plt.text(0.5, 0.15, 'x1-marginal\ndistributions\ncomparison', fontsize=8, ha='center')
            plt.axis('off')
            plt.subplot(5, pictrue_num, i+2*pictrue_num+2)
            cmap = ['black', 'red']
            plt.plot(np.arange(x_test.shape[1])[:int((x_test.shape[1])/3+1)], x_test[i+pic1-1][:int((x_test.shape[1])/3+1)], cmap[0], linewidth = 0.5)
            plt.plot(np.arange(three_axis_pred.shape[1])[:int((three_axis_pred.shape[1])/3+1)], three_axis_pred[i+pic1-1][:int((three_axis_pred.shape[1])/3+1)], cmap[1], linewidth = 0.5)
            diffx = np.sum(abs(x_test[i+pic1-1][:int((x_test.shape[1])/3+1)] - three_axis_pred[i+pic1-1][:int((three_axis_pred.shape[1])/3+1)]))
            plt.text(point, np.max(three_axis_pred[i+pic1-1][:int((three_axis_pred.shape[1])/3+1)]), str(round(diffx, 2)), fontsize=6, ha='right', va = 'top')
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis('auto')
        for i in range(pictrue_num-1):
            plt.subplot(5, pictrue_num, 3*pictrue_num + 1)
            plt.text(0.5, 0.15, 'x13-marginal\ndistributions\ncomparison', fontsize=8, ha='center')
            plt.axis('off')
            plt.subplot(5, pictrue_num, i+3*pictrue_num+2)
            cmap = ['black', 'red']
            plt.plot(np.arange(x_test.shape[1])[int((x_test.shape[1])/3):int((x_test.shape[1])*2/3+1)], x_test[i+pic1-1][int((x_test.shape[1])/3):int((x_test.shape[1])*2/3+1)], cmap[0], linewidth = 0.5)
            plt.plot(np.arange(three_axis_pred.shape[1])[int((three_axis_pred.shape[1])/3):int((three_axis_pred.shape[1])*2/3+1)], three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])/3):int((three_axis_pred.shape[1])*2/3+1)], cmap[1], linewidth = 0.5)            
            diffy = np.sum(abs(x_test[i+pic1-1][int((x_test.shape[1])/3):int((x_test.shape[1])*2/3+1)] - three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])/3):int((three_axis_pred.shape[1])*2/3+1)]))
            plt.text(2*point, np.max(three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])/3):int((three_axis_pred.shape[1])*2/3+1)]), str(round(diffy, 2)), fontsize=6, ha='right', va = 'top')
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis('auto')          
        for i in range(pictrue_num-1):
            plt.subplot(5, pictrue_num, 4*pictrue_num + 1)
            plt.text(0.5, 0.15, 'u-marginal\ndistributions\ncomparison', fontsize=8, ha='center')
            plt.axis('off')
            plt.subplot(5, pictrue_num, i+4*pictrue_num+2)
            cmap = ['black', 'red']
            plt.plot(np.arange(x_test.shape[1])[int((x_test.shape[1])*2/3):-1], x_test[i+pic1-1][int((x_test.shape[1])*2/3):-1], cmap[0], linewidth = 0.5)
            plt.plot(np.arange(three_axis_pred.shape[1])[int((three_axis_pred.shape[1])*2/3):-1], three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])*2/3):-1], cmap[1], linewidth = 0.5)
            diffu = np.sum(abs(x_test[i+pic1-1][int((x_test.shape[1])*2/3):-1] - three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])*2/3):-1]))
            plt.text(3*point, np.max(three_axis_pred[i+pic1-1][int((three_axis_pred.shape[1])*2/3):-1]), str(round(diffu, 2)), fontsize=6, ha='right', va = 'top')
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis('auto')
        
        fig.tight_layout()
        prediction_filename = 'test_visualization_Wigner'
        front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
        path_prediction = os.path.join(front_path, prediction_filename)
        if os.path.exists(path_prediction) == False:
           os.mkdir(path_prediction)
        
        fig.savefig(os.path.join(path_prediction, 'test_comparison_%d.png'%t), dpi=2000)