# coding: utf-8

# In[77]:


import numpy as np

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def activate(x):
    if x<0:
        return 0
    else:
        return x

f = np.vectorize(activate)


def normalize(w):
    w = (255.0*(w - w.min()))/(1.0* (w.max()-w.min()))
    w = w.astype('int')
    return w
def convolve(image, cfilter, base):
    cfilter_x, cfilter_y, cfilter_z = cfilter.shape
    im_x, im_y, im_z = image.shape
    #print cfilter.shape, image.shape
    r1 = []
    dot_sum = 0
    for i in range(im_x - cfilter_x + 1):
        r2 = []
        for j in range(im_y - cfilter_y + 1):
            im = image[i:i+cfilter_x,j:j+cfilter_y,:]
            dot_sum = np.dot(im.flatten(), cfilter.flatten()) + base
            r2.append(dot_sum)
        r1.append(r2)
    w = np.array(r1)
    w = (255.0*(w - w.min()))/(1.0* (w.max()-w.min()))
    w = w.astype('int')
    #print w
    ans = w.reshape(w.shape[0],w.shape[1],1)
    return ans

def convolutional_layer(image, filters):
    output = []
    im_x = image.shape[0]
    im_y = image.shape[1]
    im_z = image.shape[2]
    for cfilter in filters:
        cfilter_x, cfilter_y, cfilter_z = cfilter.shape
        #print image.shape, cfilter.shape
        conv = convolve(image, cfilter, 0)
        p1 = im_x - cfilter_x + 1
        p2 = im_y - cfilter_y + 1
        p3 = im_z - cfilter_z + 1
        #output.append(f(conv[:p1, :p2, :p3]))
        #print conv
        output.append(f(conv))
    return output        
        

import cv2
im = cv2.imread("mnist_output_0.png")
r = 32
dim = (32, 32) 
# perform the actual resizing of the image and show it
resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
resized.shape
im = resized




def get_filters(x,y,z,n):
    output = []
    for i in range(n):
        r=np.random.rand(x,y,z)*2-1
        output.append(np.around(r).astype('int'))
    return output



filters = get_filters(5,5,im.shape[2],6)



t = convolutional_layer(im,filters)


#convolutional layer 1 output
for r in t:
    print show(r.reshape(r.shape[0],r.shape[0]))


def calculate(array):
    return np.amax(array)

def subsampling(imset, layer_dim, stride):
    output = []
    l, h = layer_dim
    for r in imset:
        x = r.reshape(r.shape[0],r.shape[0])
        out = []
        for i in xrange(0,x.shape[0]-l+1,stride):
            row = []
            for j in xrange(0,x.shape[1]-h+1,stride):
                row.append(calculate(x[i:i+l, j:j+h]))
            out.append(row)
        output.append(np.asarray(out))
    return np.asarray(output), np.dstack(output)

list_out2, out2 = subsampling(t,(2,2),2)
#subsampling layer 1 output
for r in list_out2:
    print show(r)

filters = get_filters(5,5,out2.shape[2],16)
t2 = convolutional_layer(out2,filters)
#convolutional layer 2 output
for r in t2:
    print show(r.reshape(r.shape[0],r.shape[0]))


# In[94]:


# subsampling 2
list_out3, out3 = subsampling(t2,(2,2),2)


# In[95]:


for r in list_out3:
    print show(r)


# In[96]:


def relu(x):
    return np.maximum(0,x)

def softmax(x):
    x = normalize(x)
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# In[97]:


#fully connected 1
flat_mat = out3.flatten()
w = np.asmatrix(np.random.rand(120,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1))
valf = relu(np.asarray(np.matmul(w,x.T)))


# In[98]:


#fully connected 2
flat_mat = valf.flatten()
w = np.asmatrix(np.random.rand(84,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1))
valf2 = relu(np.asarray(np.matmul(w,x.T)))


# In[99]:


#fully connected 3
flat_mat = valf2.flatten()
w = np.asmatrix(np.random.rand(10,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1))
valf3 = softmax(np.asarray(np.matmul(w,x.T)))
print(valf3)
