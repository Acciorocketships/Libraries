import sys
import os 
try:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
	pass
import cv2 
import numpy as np 


class Stream: 

    imgnum = 0
 
    def __init__(self,mode="webcam",src=""): 
 
      if ('pic' in mode) or ('img' in mode) or ('image' in mode): 
        self.mode = 'pic' 
      elif ('vid' in mode) or ('movie' in mode): 
        self.mode = 'vid'
      elif ('pi' in mode):
        self.mode = 'pi'
      else: 
        self.mode = 'cam' 
 
      self.src = src 
      self.isfolder = (not '.' in src) 
      if self.isfolder and self.mode != 'cam': 
        self.files = [] 
        self.filenum = -1 
        self.getfiles(src) 
 
      if self.mode == 'vid': 
        if self.isfolder: 
          self.stream = cv2.VideoCapture(self.nextfile()) 
        else: 
          self.stream = cv2.VideoCapture(src) 
      elif self.mode == 'cam': 
        self.stream = cv2.VideoCapture(0) 
      elif self.mode == 'pi':
        import picamera
        self.stream = picamera.PiCamera()
 
 
    # 'Private' Functions 
 
    def __iter__(self): 
      return self 

    def next(self):
      return self.__next__()
 
    def __next__(self): 
      img = self.get() 
      if img is None: 
        raise StopIteration 
      else: 
        return img 
 
    def getfiles(self,folder): 
      self.files = os.listdir(os.path.join(os.getcwd(),folder)) 
 
    def nextfile(self): 
      while self.filenum < len(self.files)-1: 
        self.filenum += 1 
        if self.mode == 'vid': 
          if self.files[self.filenum].endswith('.m4v') or \
             self.files[self.filenum].endswith('.mp4') or \
             self.files[self.filenum].endswith('.mov'): 
            return self.files[self.filenum] 
        elif self.mode == 'pic': 
          if self.files[self.filenum].endswith('.jpg') or \
             self.files[self.filenum].endswith('.JPG') or \
             self.files[self.filenum].endswith('.png') or \
             self.files[self.filenum].endswith('.bmp'): 
            return self.files[self.filenum] 
      return None 
 
    def pic(self): 
      try: 
        if self.isfolder: 
          currdir = os.getcwd() 
          os.chdir(self.src) 
          image = cv2.imread(self.nextfile()) 
          os.chdir(currdir) 
        else: 
          image = cv2.imread(self.src) 
      except Exception as err: 
        return None 
      return np.array(image) 
 
    def vid(self): 
      try: 
        success,image = self.stream.read() 
        if not success and self.isfolder: 
          currdir = os.getcwd() 
          os.chdir(self.src) 
          self.stream = cv2.VideoCapture(self.nextfile()) 
          os.chdir(currdir) 
          _,image = self.stream.read() 
      except Exception as err: 
        return None 
      return np.array(image) 
 
    def cam(self): 
      success,image = self.stream.read() 
      if success: 
        return np.array(image) 
      else: 
        return None 

    def pi(self):
      try:
        image = picamera.array.PiRGBArray(self.stream)
        self.stream.capture(image, 'rgb')
        return np.array(image)
      except:
        return None
 
 
    # Public Functions 
 
    # Returns the next frame as a numpy array 
    def get(self): 
      if self.mode == 'pic': 
        image = self.pic() 
      elif self.mode == 'vid': 
        image = self.vid() 
      elif self.mode == 'cam': 
        image = self.cam() 
      elif self.mode == 'pi':
        image = self.pi()
      if image.dtype is np.dtype('object'): 
        return None 
      return image 
 
    # type(size)==int: percent of current size 
    # type(size)==float: fraction of current size 
    # type(size)==tuple: new size
    @staticmethod
    def resize(image,shape): 
      from skimage.transform import resize as skresize 
      return skresize(image,shape,preserve_range=True,mode='constant').astype(np.uint8) 
 
    # returns grayscale image 
    @staticmethod
    def im2gray(image): 
      return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
 
    # if pause=True, then the program will wait on the current frame until 
    # you press a key (hold a key to run in real time) 
    # if pause=False, then the program will run in real time until q is pressed 
    # give 2-tuple shape = (height,width) to resize 
    @staticmethod
    def show(image,name=None,pause=False,shape=None): 
      if name is None: 
        name = str(Stream.imgnum) 
        Stream.imgnum += 1 
      if image is None: 
        sys.exit() 
      if shape is not None: 
        image = Stream.resize(image,shape) 
      cv2.imshow(name,image) 
      if pause: 
        if cv2.waitKey(0) & 0xFF == ord('q'): 
          sys.exit() 
      else: 
        if cv2.waitKey(2) & 0xFF == ord('q'): 
          sys.exit() 
 
    # TODO: Add argmax and labels
    @staticmethod
    def mask(mask,image=None,alpha=0.3,argmax=False,labels=[]): 
      colors = [[1.,0.5,0.5],[0.5,0.5,1.],[0.5,0.1,0.5],[0.8,0.8,0.5],[0.8,0.5,0.8],[0.5,0.8,0.8],[0.2,0.7,0.4]] 
      maskout = np.zeros([3,mask.shape[0],mask.shape[1]]) 
      if len(mask.shape)==2: 
        mask = np.expand_dims(mask,axis=2)
      if (mask.shape[2]==1) and (image is None):
        colors[0] = [1,1,1]
      if image is None: 
        image = np.zeros((mask.shape[0],mask.shape[1],3)) 
        alpha = 0 
      elif image.shape[:2] != mask.shape[:2]: 
        image = Stream.resize(image,mask.shape[:2]) 
      for i in range(mask.shape[2]):
        if i < len(labels) and (labels[i]==None or labels[i]=='Nothing'):
          color = np.zeros((3,1,1))
          colors.insert(i,[0,0,0])
        elif i < len(colors):
          color = np.transpose(np.array([[colors[i]]])) 
        else: 
          color = np.random.rand(3,1,1)
          colors.append(list(color[:,0,0]))
        if argmax:
          mask[:,:,i] = (mask[:,:,i] >= np.amax(mask,axis=2)).astype(int)
        maskout += mask[:,:,i]*color 
      maskout = np.moveaxis(maskout,0,2)
      h = np.max(maskout,axis=2)
      h = (h>0.1)*h + (h<0.1)*0.1*np.ones(h.shape) 
      maskout = (h>1)[:,:,np.newaxis]*maskout/h[:,:,np.newaxis] + (h<=1)[:,:,np.newaxis]*maskout # normalizes pixels with channel sum >1 
      output = alpha*image + (1-alpha)*255*maskout
      for i,label in enumerate(labels):
        output = Stream.mark(output, marks=[(15,i*20+15,4)], size=8,
                             color=(128*colors[i][0],128*colors[i][1],128*colors[i][2]))
        output = Stream.mark(output, marks=[(20,i*20+20,label)], size=1,
                             color=(72*colors[i][0],72*colors[i][1],72*colors[i][2]))
      return output.astype(np.uint8) 
 
    # softmax of numpy array along 
    @staticmethod
    def softmax(X, axis=None):
      theta = 1.0
      y = np.atleast_2d(X) # make X at least 2d
      if axis is None: # find axis
          axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
      y = y * float(theta) # multiply y against the theta parameter
      y = y - np.expand_dims(np.max(y, axis = axis), axis) # subtract the max for numerical stability
      y = np.exp(y) # exponentiate y
      ax_sum = np.expand_dims(np.sum(y, axis = axis), axis) # take the sum along the specified axis
      p = y / ax_sum # finally: divide elementwise
      if len(X.shape) == 1: p = p.flatten() # flatten if X was 1D
      return p
 
    # type(mark) == 2-tuple: (x,y) draws points 
    # type(mark) == 2-tuple of 2-tuples: ((x1,y1), (x2,y2)) draws lines
    # type(mark) == 3-tuple: (x,y,radius) draws circles 
    # type(mark) == 4-tuple: (x,y,length,height) draws rectangles 
    # type(mark) == string: 'text' places text in corner 
    # type(mark) == (int,int,string): (x,y,'text') places text at position 
    # if xyaxis=True then it will draw in xy coordinates, not image coordinates 
    # copy=False: updates original. copy=True: does not edit original, returns new img
    @staticmethod
    def mark(image,marks,color=(0,0,255),xyaxis=False,size=4,copy=False): 
 
      if copy: 
        image = np.copy(image) 
      if type(marks) != list: 
        marks = [marks] 
 
      for mark in marks: 
        if type(mark) == str or (len(mark)==3 and type(mark[2]) == str): 
          if type(mark) == str: 
            mark = [0,0,mark] 
            if xyaxis: 
              pos = (5,image.shape[0]-10) 
            else: 
              pos = (5,16*size+5)
          else: 
            if xyaxis: 
              pos = (int(mark[0]),int(image.shape[0]-mark[1])) 
            else: 
              pos = (int(mark[0]),int(mark[1])) 
          cv2.putText(image,mark[2],pos,cv2.FONT_HERSHEY_COMPLEX_SMALL,size,color,size) 
        elif (len(mark) == 2) and (isinstance(mark[0], tuple) or isinstance(mark[0], list)):
          mark = (tuple(mark[0]), tuple(mark[1]))
          if xyaxis: 
            mark = ((mark[0][0], image.shape[0]-mark[0][1]), (mark[1][0], image.shape[0]-mark[1][1]))
          cv2.line(image,mark[0],mark[1],color,size)
        elif len(mark) == 2:
          if xyaxis: 
            mark = (mark[0],image.shape[0]-mark[1]) 
          cv2.circle(image,(int(mark[0]),int(mark[1])),1,color,size) 
        elif len(mark) == 3:
          if xyaxis: 
            mark = (mark[0],image.shape[0]-mark[1],mark[2]) 
          cv2.circle(image,(int(mark[0]),int(mark[1])),int(mark[2]),color,size) 
        elif len(mark) == 4: 
          if xyaxis: 
            mark = (mark[0],image.shape[0]-mark[1],mark[2],mark[3]) 
          cv2.rectangle(image,(int(mark[0]-mark[2]/2),int(mark[1]-mark[3]/2)), 
                      (int(mark[0]+mark[2]/2),int(mark[1]+mark[3]/2)),color,size) 
      return image 
 
 
 
#Example Usage 
 
#import imgstream 
#stream = imgstream.Stream(mode='image',src='imgfolder') 
#while True: 
#  img = stream.get() 
#  img = stream.mark(img,[('output stream',40,20),(300,300),(200,250,30)],size=3,xyaxis=True) 
#  stream.show(img,'Output') 
 
from random import random 
 
if __name__ == '__main__': 
  stream = Stream(mode='webcam') 
  for img in stream: 
    img = stream.mark(img,['Testing'],size=2) 
    stream.show(img,'Output',pause=False,resize=True)

    