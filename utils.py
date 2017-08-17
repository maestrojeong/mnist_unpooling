import matplotlib.pyplot as plt
import numpy as np

def show_gray_image(ax, image):
    '''
    Print the image with grayscale
    
    Arg: 
        ax - <matplotlib.axes._subplots> 
            generated by 
                    fig = plt.figure()
                    ax = fig.add_subplot()
        image - 2D or 3D image
    '''
    gray_image = np.average(image, axis=2) if len(image.shape) == 3 else image
    ax.imshow(gray_image, cmap = 'gray')
    ax.set_axis_off()

def show_gray_image_with_channel(gr_img_3d, col = 5, figsize = (5, 5)):
    '''
    Args:
        3D_gr_img - 3D array [height, width, nchannel]
    '''
    height, width, nchannel = gr_img_3d.shape
    fig = plt.figure(figsize = figsize, edgecolor = 'k')
    row = np.ceil(nchannel/col)
    for nchannel_ in range(nchannel):
        show_gray_image(fig.add_subplot(row, col, nchannel_+1), gr_img_3d[:,:,nchannel_])
    plt.show()

class struct:
    def __str__ (self):
        return "struct{\n    "+"\n   ".join(["{} : {}".format(key, vars(self)[key]) for key in vars(self).keys()]) + "\n}"
