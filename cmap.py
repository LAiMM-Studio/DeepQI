import cv2
import numpy as np
import matplotlib.colors as mcolors

def rgb_cmap(z, vmin, vmax):   
    z = np.clip(z, vmin, vmax)
    h = (z - vmin)/(vmax - vmin)
    zero = (0 - vmin)/(vmax - vmin)
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    rgb = np.clip(cv2.merge([r, g, b]), -1, 1)
    rgb = np.array(rgb)
    
    return rgb

def rgb_cmap_inverse(rgb, vmin, vmax):
    e = pow(10, -4)
    rgb = np.clip(rgb, -1+e, 1-e)
    zero = (0 - vmin)/(vmax - vmin)
    r, g, b = cv2.split(rgb)
    g1_r = np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    g1_l = -np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    
    rk_r = (1.148/(np.exp(-25*(g1_r - (zero - 0.12))) + 1)/(np.exp(5*(g1_r - (zero + 0.45))) + 1))*2 - 1
    rk_l = (1.148/(np.exp(-25*(g1_l - (zero - 0.12))) + 1)/(np.exp(5*(g1_l - (zero + 0.45))) + 1))*2 - 1
    
    right = abs(rk_r - r).flatten()
    left = abs(rk_l - r).flatten()
    
    z = []
    for i in range(np.prod(r.shape)):
    
        if right[i] > left[i] :
            z.append(g1_l.flatten()[i])
        elif right[i] <= left[i] :
            z.append(g1_r.flatten()[i])
            
    z = np.array(z).reshape((256, 256))     
    denormal = z*(vmax - vmin) + vmin   
    
    return denormal