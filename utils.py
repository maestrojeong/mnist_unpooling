import matplotlib.pyplot as plt
import numpy as np

class struct:
    def __str__ (self):
        return "struct{\n    "+"\n   ".join(["{} : {}".format(key, vars(self)[key]) for key in vars(self).keys()]) + "\n}"

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

def show_gray_image_with_channel(gr_img_3d, col = 5, figsize = (5, 5) ):
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

def dataset_split(dataset, nsplits = 5):
    '''
    Sort the dataset and 
    '''
    length = len(dataset)
    index = np.arange(length)
    np.random.shuffle(index)
    dataset = dataset[index]
    split_size = int(length/nsplits)
    return [dataset[split_size*i:split_size*(i+1)] for i in range(nsplits)]

def sample_img(dataset, batch_size):
    return dataset[np.random.randint(0, dataset.shape[0], size= batch_size)]

def normalize_gray_image(gray_img):
    """Normalize image to be [0, 1]"""
    gray_img = np.squeeze(gray_img)
    min_ = np.min(gray_img)
    max_ = np.max(gray_img)
    height, width = gray_img.shape
    if max_ == min_:
        return np.zeros((height, width))
    else:
        return (gray_img-min_)/(max_-min_)
