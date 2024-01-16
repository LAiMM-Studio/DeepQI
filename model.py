import tensorflow.python.keras as K

def identity_block(short, filters, deconv , block):
    
    filters1, filters2, filters3 = filters
    name = 'Deconv_' + str(deconv) + '_Block_' + str(block)
    
    x = K.layers.Conv2DTranspose(filters1, (1, 1), strides=(1, 1), name = str(name) +'_deconv1')(short)
    x = K.layers.BatchNormalization(name = str(name) +'_BN1')(x)
    x = K.layers.Activation('relu', name = str(name) + '_relu1')(x)
    x = K.layers.Conv2DTranspose(filters2, (3, 3), padding = 'same', strides=(1, 1), name = str(name) + '_deconv2')(x)
    x = K.layers.BatchNormalization(name = str(name) +'_BN2')(x)
    x = K.layers.Activation('relu', name = str(name) + '_relu2')(x)
    x = K.layers.Conv2DTranspose(filters3, (1, 1), strides=(1, 1), name = str(name) + '_deconv3')(x)
    x = K.layers.BatchNormalization(name = str(name) +'_BN3')(x)
    x = K.layers.add([x, short], name = str(name) + '_add')
    x = K.layers.Activation('relu', name = str(name) + '_relu3')(x)

    return x

def deconv_block(short, filters, stride, deconv, block):

    if stride == 1:
        strides=(1, 1)
    elif stride ==2:
        strides=(2, 2)
        
    filters1, filters2, filters3 = filters 
    name = 'Deconv_' + str(deconv) + '_Block_' + str(block)

    x = K.layers.Conv2DTranspose(filters1, (1, 1), strides=(1, 1), name = str(name) +'_deconv1')(short)
    x = K.layers.BatchNormalization(name = str(name) +'_BN1')(x)
    x = K.layers.Activation('relu', name = str(name) + '_relu1')(x)
    x = K.layers.Conv2DTranspose(filters2, (3, 3), padding = 'same', strides=(1, 1), name = str(name) +'_deconv2')(x)
    x = K.layers.BatchNormalization(name = str(name) +'_BN2')(x)
    x = K.layers.Activation('relu', name = str(name) + '_relu2')(x)
    x = K.layers.Conv2DTranspose(filters3, (1, 1), strides, name = str(name) +'_deconv3')(x)
    x = K.layers.BatchNormalization(name = str(name) +'_BN3')(x)
    
    short = K.layers.Conv2DTranspose(filters3, (1, 1), strides, name = str(name) +'_deconv4')(short)
    short = K.layers.BatchNormalization(name = str(name) +'_BN4')(short)
    x = K.layers.add([x, short], name = str(name) + '_add')
    x = K.layers.Activation('relu', name = str(name) + '_relu3')(x) 
    
    return x

# define resnet decoder
def ResNet_base_decoder(inputs_shape):
    x = K.layers.Dense(8*8*2048, activation = 'relu', name= 'Dense_inputs')(inputs_shape)
    x = K.layers.Reshape((8, 8, 2048), name = 'Reshape_inputs')(x)
    
    x = identity_block(x, [512, 512, 2048], deconv = 1, block = 1)
    x = identity_block(x, [512, 512, 2048], deconv = 1, block = 2)
    x = deconv_block(x, [512, 512, 1024], stride = 2, deconv = 1, block = 3)
    
    x = identity_block(x, [256, 256, 1024], deconv = 2, block = 1)
    x = identity_block(x, [256, 256, 1024], deconv = 2, block = 2)
    x = identity_block(x, [256, 256, 1024], deconv = 2, block = 3)
    x = identity_block(x, [256, 256, 1024], deconv = 2, block = 4)
    x = identity_block(x, [256, 256, 1024], deconv = 2, block = 5)
    x = deconv_block(x, [256, 256, 512], stride = 2, deconv = 2, block = 6)
    
    x = identity_block(x, [128, 128, 512], deconv = 3, block = 1)
    x = identity_block(x, [128, 128, 512], deconv = 3, block = 2)
    x = identity_block(x, [128, 128, 512], deconv = 3, block = 3)
    x = identity_block(x, [128, 128, 512], deconv = 3, block = 4)
    x = identity_block(x, [128, 128, 512], deconv = 3, block = 5)
    x = deconv_block(x, [128, 128, 256], stride = 2, deconv = 3, block = 6)
    
    x = identity_block(x, [64, 64, 256], deconv = 4, block = 1)
    x = identity_block(x, [64, 64, 256], deconv = 4, block = 2)
    x = identity_block(x, [64, 64, 256], deconv = 4, block = 3)
    x = deconv_block(x, [64, 64, 128], stride = 2, deconv = 4, block = 4)
    
    x = identity_block(x, [32, 32, 128], deconv = 5, block = 1)
    x = identity_block(x, [32, 32, 128], deconv = 5, block = 2)
    x = identity_block(x, [32, 32, 128], deconv = 5, block = 3)
    x = deconv_block(x, [32, 32, 64], stride = 2, deconv = 5, block = 4)
    
    x = identity_block(x, [16, 16, 64], deconv = 6, block = 1)
    x = identity_block(x, [16, 16, 64], deconv = 6, block = 2)
    x = deconv_block(x, [8, 8, 32], stride = 1, deconv = 6, block = 3)
    
    x = K.layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), name = 'Deconv_7')(x)
    
    decoder_model = K.Model(inputs_shape, x)
    
    return decoder_model