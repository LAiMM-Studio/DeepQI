import numpy as np
import random
import os
from tqdm import trange
import cv2
import pandas as pd 
from scipy.special import laguerre,hermite
import math
import matplotlib.pyplot as plt

def build_data(dataset_filename, types, num, resolution, length):
    
    # set save path of dataset
    osd = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    
    if types == 'coherent' :
        data_type_filename = "distribution_coherent"
    if types == 'cat' :
        data_type_filename = "distribution_cat"
    
    main_distribution_npy_filename = 'main_data'
    x_distribution_filename = 'x_data'
    y_distribution_filename = 'y_data'
    u_distribution_filename = 'u_data'
    
    path =  os.path.join(osd, dataset_filename, data_type_filename)
    path_main_data_npy = os.path.join(osd, dataset_filename, data_type_filename, main_distribution_npy_filename)
    path_x_data = os.path.join(osd, dataset_filename, data_type_filename, x_distribution_filename)
    path_y_data = os.path.join(osd, dataset_filename, data_type_filename, y_distribution_filename)
    path_u_data = os.path.join(osd, dataset_filename, data_type_filename, u_distribution_filename)
    
    print("Data save in {path}".format(path = path))
    
    path_list = [os.path.join(osd, dataset_filename), path, path_main_data_npy, path_x_data, path_y_data, path_u_data]
    
    for path in path_list:
        if os.path.exists(path) == False:
           os.mkdir(path)
    kx = np.arange(-length,length+ resolution,resolution)
    ky = np.arange(-length, length + resolution, resolution)
    ku = np.arange(-length*np.sqrt(2), (length + resolution)*np.sqrt(2), resolution*np.sqrt(2))
    x, y = np.meshgrid(kx,ky)    
    for e in trange(num):
           
        if types == 'coherent' :
             r= random.randint(50,100)/100
             n= random.randint(0,200)/100
             v= (1-(abs(r))**2)*n
             cons = (1+2*v)
             c=random.randint(-200, 200)/100
             d=random.randint(-200, 200)/100
             a= complex(c,d)
         
             W=1/np.pi/cons*np.exp(-((x-np.sqrt(2)*r*a.real)**2/cons)-((y-np.sqrt(2)*r*a.imag)**2/cons))
             W=cv2.resize(W, (256, 256))
             W1=1/np.sqrt(np.pi*cons)*np.exp(-((kx-np.sqrt(2)*r*a.real)**2)/cons)
             W2=1/np.sqrt(np.pi*cons)*np.exp(-((ky-np.sqrt(2)*r*a.imag)**2)/cons)
             W3=1/np.sqrt(np.pi*cons)*np.exp(-((ku-np.sqrt(2)*r*(a*np.exp(-1j*np.pi/4)).real)**2)/cons)
             np.save(os.path.join(path_x_data, '{}p1.npy').format(e), W1)
             np.save(os.path.join(path_y_data, '{}p2.npy').format(e), W2)
             np.save(os.path.join(path_u_data, '{}p3.npy').format(e), W3)
             np.save(os.path.join(path_main_data_npy, '{}p0.npy').format(e), W)           
                
        if types == 'cat' :
            r= random.randint(50,100)/100
            n= random.randint(0,200)/100
            v= (1-(abs(r))**2)*n
            cons = (1+2*v)
            c=random.randint(-200, 200)/100
            d=random.randint(-200, 200)/100
            a= complex(c,d)
            ca=random.uniform(0, 2*np.pi)
            C=1/np.pi/cons*1/(2+2*math.cos(ca)*np.exp(-2*abs(a)**2))*(np.exp(-((x-np.sqrt(2)*r*a.real)**2+(y-np.sqrt(2)*r*a.imag)**2)/cons)+np.exp(-((x-np.sqrt(2)*r*(-a).real)**2+(y-np.sqrt(2)*r*(-a).imag)**2)/cons))
            C3=1/np.pi/cons*1/(1+math.cos(ca)*np.exp(-2*abs(a)**2))*np.exp((-x**2-y**2)/cons)
            C2=np.cos((2*x*r*np.sqrt(2)*a.imag-2*y*r*np.sqrt(2)*a.real)/cons-ca)
            C=C+C2*C3
            C=cv2.resize(C, (256, 256))
            
            c1=1/np.sqrt(np.pi*cons)*1/(2+2*math.cos(ca)*np.exp(-2*abs(a)**2))*(np.exp(-(kx-np.sqrt(2)*r*a.real)**2/cons)+np.exp(-(kx-np.sqrt(2)*r*(-a).real)**2/cons)+2*np.cos((2*kx*r*np.sqrt(2)*a.imag)/cons-ca)*np.exp((-kx**2-2*r**2*(a.real)**2)/cons))
            c2=1/np.sqrt(np.pi*cons)*1/(2+2*math.cos(ca)*np.exp(-2*abs(a)**2))*(np.exp(-(ky-np.sqrt(2)*r*a.imag)**2/cons)+np.exp(-(ky-np.sqrt(2)*r*(-a).imag)**2/cons)+2*np.cos((-2*ky*r*np.sqrt(2)*a.real)/cons-ca)*np.exp((-ky**2-2*r**2*(a.imag)**2)/cons))
            a1=a*np.exp(-1j*np.pi/4)
            c3=1/np.sqrt(np.pi*cons)*1/(2+2*math.cos(ca)*np.exp(-2*abs(a)**2))*(np.exp(-(ku-np.sqrt(2)*r*a1.real)**2/cons)+np.exp(-(ku-np.sqrt(2)*r*(-a1).real)**2/cons)+2*np.cos((2*ku*r*np.sqrt(2)*a1.imag)/cons-ca)*np.exp((-ku**2-2*r**2*(a1.real)**2)/cons))
            np.save(os.path.join(path_x_data, '{}p1.npy').format(e), c1)
            np.save(os.path.join(path_y_data, '{}p2.npy').format(e), c2)
            np.save(os.path.join(path_u_data, '{}p3.npy').format(e),c3)
            np.save(os.path.join(path_main_data_npy, '{}p0.npy').format(e), C) 
