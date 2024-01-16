import numpy as np
from sklearn.utils import shuffle
import os
from cmap import rgb_cmap

def build_combined_data(dataset_filename, train_filename, coherent_num, cat_num, test_data_num = 100, split_num = 8, vmin = -0.01, vmax = 0.045):
    # reading definition
    def read_type_data(distribution_num, type_filename):      
        x_train = list([])
        y_train1 = list([])
        y_train2 = list([])
        y_train3 = list([])
        
        for i in range(distribution_num//split_num*k, distribution_num//split_num*(k+1)):
            m0 = np.load(os.path.join(osd, dataset_filename, type_filename, main_distribution_filename, '{}p0.npy').format(i), allow_pickle=True)
            m1 = np.load(os.path.join(osd, dataset_filename, type_filename, x_distribution_filename, '{}p1.npy').format(i), allow_pickle=True)
            m2 = np.load(os.path.join(osd, dataset_filename, type_filename, y_distribution_filename, '{}p2.npy').format(i), allow_pickle=True)
            m3 = np.load(os.path.join(osd, dataset_filename, type_filename, u_distribution_filename, '{}p3.npy').format(i), allow_pickle=True)
            m0 = rgb_cmap(m0, vmin, vmax)
            x_train.extend([m0])
            y_train1.extend([m1])
            y_train2.extend([m2])
            y_train3.extend([m3])
            
        del m0, m1, m2, m3
        
        x_train = np.array(x_train)
        
        y_train1 = np.array(y_train1)
        y_train2 = np.array(y_train2)
        y_train3 = np.array(y_train3)
        
        y_train_per = np.concatenate((y_train1, y_train2, y_train3), axis = 1)
        
        return x_train, y_train_per
    

    # set load path of dataset
    osd = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))   
    coherent_filename = "distribution_coherent"
    cat_filename = "distribution_cat"
    
    main_distribution_filename = 'main_data'
    x_distribution_filename = 'x_data'
    y_distribution_filename = 'y_data'
    u_distribution_filename = 'u_data'
    
    # get the front path
    front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    
    path_combined_dataset = os.path.join(front_path, train_filename)
    
    print("Combined data save in {path_combined_dataset}".format(path_combined_dataset = path_combined_dataset))
    
    if os.path.exists(path_combined_dataset) == False:     
        os.mkdir(path_combined_dataset)

    print("coherent data number : ", coherent_num)
    print("cat data number : ", cat_num)

    x_data_num = 0
    for k in range(split_num):
        xn1_train, yn1_train = read_type_data(coherent_num, coherent_filename) 
        xn2_train, yn2_train = read_type_data(cat_num, cat_filename) 
        x_train = np.concatenate((xn2_train,xn1_train), axis = 0)
        y_train = np.concatenate((yn2_train,yn1_train), axis = 0)     
              
        x_train = x_train.astype('float64')
        y_train = y_train.astype('float64')
        X_train, Y_train = shuffle(x_train, y_train)
        
        del x_train, y_train
        
        x_train = X_train
        y_train = Y_train
                  
        # set test set
        if k == split_num - 1:
            
            x_test = X_train[:test_data_num]
            x_train = X_train[test_data_num:]
            
            y_test = Y_train[:test_data_num]
            y_train = Y_train[test_data_num:]
            np.save(os.path.join(path_combined_dataset, 'x_test.npy'), y_test)
            np.save(os.path.join(path_combined_dataset, 'y_test.npy'), x_test)
        
        np.save(os.path.join(path_combined_dataset, 'x_train%d.npy'%(k+1)), y_train)
        np.save(os.path.join(path_combined_dataset, 'y_train%d.npy'%(k+1)), x_train)
        x_data_num += y_train.shape[0]