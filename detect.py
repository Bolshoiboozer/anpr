for repeat in range(1):
    import random
    import time
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    import matplotlib.patches as patches
    from matplotlib.path import Path
    import tensorflow as tf
    from pylab import rcParams
    import os
    import json
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage import exposure
    from IPython import get_ipython
    
    import cv2
    from skimage.filters import threshold_otsu


    def ResetImg():
        dir_to_clean = 'C:\\Users\\AS-Bolshoi\\Desktop\\train\\train\\supervisely-tutorials-master\\anpr\\data\\artificial\\img'
        l = os.listdir(dir_to_clean)
        for n in range(5000):
            target = dir_to_clean + '/' + str(l[n])
            if os.path.isfile(target):
                os.unlink(target)
    
    def ResetJson():
        dir_to_clean = 'C:\\Users\\AS-Bolshoi\\Desktop\\train\\train\\supervisely-tutorials-master\\anpr\\data\\artificial\\ann'
        l = os.listdir(dir_to_clean) 
        for n in range(5000):
            target = dir_to_clean + '/' + str(l[n])
            if os.path.isfile(target):
                os.unlink(target)
                
    MODEL_PATH = "C:\\Users\\AS-Bolshoi\\Desktop\\train\\train\\supervisely-tutorials-master\\anpr\\data\\model_artif"
    SAMPLES_PATHS =["C:\\Users\\AS-Bolshoi\\Desktop\\train\\train\\supervisely-tutorials-master\\anpr\\data\\artificial"]
    
    EPOCH = 10
    
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    def LoadImage(fname):
        myimage = io.imread(fname, as_gray=True)
        resized_image = resize(myimage, (200,200), anti_aliasing=True)
        gamma_corrected = exposure.adjust_gamma(resized_image, 2)
        logarithmic_corrected = exposure.adjust_log(gamma_corrected, 4)
        
        """img = cv2.imread(fname)
        print(fname)
        print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))
        thresh = threshold_otsu(img)
        log = img > thresh"""

        return logarithmic_corrected
    
    def LoadAnnotation(fname):
        with open(fname) as data_file:    
            data = json.load(data_file)
            x1 = data["objects"][0]["points"]["exterior"][0][0]
            y1 = data["objects"][0]["points"]["exterior"][0][1]
            x2 = data["objects"][0]["points"]["exterior"][1][0]
            y2 = data["objects"][0]["points"]["exterior"][1][1]
            x3 = data["objects"][0]["points"]["exterior"][2][0]
            y3 = data["objects"][0]["points"]["exterior"][2][1]
            x4 = data["objects"][0]["points"]["exterior"][3][0]
            y4 = data["objects"][0]["points"]["exterior"][3][1]
        return [x1, y1, x2, y2, x3, y3, x4, y4]
    
    def ReadDirFiles(dname):
        amount = 0
        paths = []
        for file in os.listdir(os.path.join(dname, "img")):
            bname = os.path.basename(file).split(".")[0]     
            img_name = os.path.join(dname, "img", file)
            ann_name = os.path.join(dname, "ann", bname + ".json")
            paths.append((img_name, ann_name))
            amount = amount + 1
            if(amount >= 5000):
                break
        return paths
    
    def ReadPaths(paths):
        all_paths = []
        amount =0
        for path in paths:
            temp_paths = ReadDirFiles(path)
            all_paths.extend(temp_paths)
            amount = amount + 1
            if(amount >= 5000):
                break
        return all_paths
    
    def get_tags(fname):
        with open(fname) as data_file:
            data = json.load(data_file)
        tags = data["tags"]
        return tags
    
    def train_test_split(paths, train_tag="train", test_tag="test"):
        train_paths = []
        test_paths = []
        for path in paths:
            img_path, ann_path = path
            tags = get_tags(ann_path)
            if train_tag in tags:
                train_paths.append(path)
            if test_tag in tags:
                test_paths.append(path)
        return train_paths, test_paths
    
    def LoadData(paths):
        xs = []
        ys = []
        for ex_paths in paths:
            img_path = ex_paths[0]
            ann_path = ex_paths[1]
            xs.append(LoadImage(img_path))
            ys.append(LoadAnnotation(ann_path))
        return np.array(xs), np.array(ys)
    
    
    def show_image(image, labels):
        vertices = [(labels[0], labels[1]), (labels[2], labels[3]),(labels[4], labels[5]), (labels[6], labels[7])]
        # print(labels[0] + " " + labels[1]+ " " + labels[2] + " " + labels[3] + " " + labels[4] + " " + labels[5] + " " + labels[6] + " " + labels[7])
        for i in range(8):
            print(labels[i])
        verts = [
        (labels[4],labels[5]), #sol üst / sol alt
        (labels[2],labels[3]), #sag ust / sag alt
        (labels[6],labels[7]), #sag alt / sag ust
        (labels[0],labels[1]), #sol alt / sol ust
        (0., 0.),
        ]
    
        codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    
        myPath = Path(verts, codes)
    
        patch = patches.PathPatch(myPath, alpha = 0.5, lw=2)
        # poly = Polygon(vertices)
        # # poly = Polygon.convex_hull(vert)
        # #rect = Rectangle((labels[0], labels[1]), labels[2]-labels[0], labels[3]-labels[1], edgecolor='r', fill=False)
        plt.imshow(image)
        gca = plt.gca()
        gca.add_patch(patch)
    
    def plot_images(images, labels):
        rcParams['figure.figsize'] = 14, 8
        plt.gray()
        fig = plt.figure()
        for i in range(min(2, images.shape[0])):
            fig.add_subplot(3, 3, i+1)
            show_image(images[i], labels[i])
        plt.show()    
    
    """all_paths = ReadPaths(SAMPLES_PATHS)
    tr_paths, te_paths = train_test_split(all_paths)
    print(len(tr_paths))
    print(len(te_paths))"""
    
    """for anana in range(10):
        print(tr_paths[anana])
    
    
    xs = [random.randint(0, X_train.shape[0]-1) for _ in range(2)]                   
    plot_images(X_train[xs], Y_train[xs]) """
    
    """for anan in range(1000):
        xs=[anan, anan]
        plot_images(X_train[xs], Y_train[xs])"""
    
    class Dataset:
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
            self._epochs_completed = 0
            self._index_in_epoch = 0
            self._num_examples = X.shape[0]
        def next_batch(self, batch_size=20):
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                self._epochs_completed += 1
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.X = self.X[perm]
                self.Y = self.Y[perm]
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self.X[start:end], self.Y[start:end]
        def epoch_completed(self):
            return self._epochs_completed
    
    def mse(expected, predicted):
        se = tf.square(expected - predicted)
        return tf.reduce_mean(se)
    
    def weight_variable(name, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.get_variable(name, initializer=initial)
    
    def bias_variable(name, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.get_variable(name, initializer=initial)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    
    
    
    # Create placeholders for image data and expected point positions
    
    class Model(object):
        xxx = 0
    
    # Build neural network
    def build_model():
        x_placeholder = tf.placeholder(tf.float32, shape=[None, PIXEL_COUNT])
        y_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])
        x_image = tf.reshape(x_placeholder, [-1, 200, 200, 1])
        W_conv1 = weight_variable("w1", [3, 3, 1, 32])
        b_conv1 = bias_variable("b1", [32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        W_conv2 = weight_variable("w2", [2, 2, 32, 64])
        b_conv2 = bias_variable("b2", [64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        W_conv3 = weight_variable("w3", [2, 2, 64, 128])
        b_conv3 = bias_variable("b3", [128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 25*25*128])
        W_fc1 = weight_variable("w4", [25*25*128, 500])
        b_fc1 = bias_variable("b4", [500])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)   
        W_fc2 = weight_variable("w5", [500, 500])
        b_fc2 = bias_variable("b5", [500])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  
        W_out = weight_variable("w6", [500, LABEL_COUNT])
        b_out = bias_variable("b6", [LABEL_COUNT])
        output = tf.matmul(h_fc2, W_out) + b_out
        model = Model()
        model.x_placeholder = x_placeholder
        model.y_placeholder = y_placeholder
        model.output = output
        return model
    
    g = tf.Graph()

#for repeat in range(5):
    # Get the images
    all_paths = ReadPaths(SAMPLES_PATHS)
    tr_paths, te_paths = train_test_split(all_paths)
    print(len(tr_paths))
    print(len(te_paths))
    
    X_train, Y_train = LoadData(tr_paths)
    X_test, Y_test = LoadData(te_paths)
    print("check shapes:")
    print("X_train - ", X_train.shape)
    print("Y_train - ", Y_train.shape)
    print("X_test - ", X_test.shape)
    print("Y_test - ", Y_test.shape)
    
    for anan in range(100):
            xs=[anan, anan]
            plot_images(X_train[xs], Y_train[xs])
    
    PIXEL_COUNT = X_train.shape[1] * X_train.shape[2]
    LABEL_COUNT = Y_train.shape[1]
    print(LABEL_COUNT)
    
    X2_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    print(X_train.shape[0])
    print(X_train.shape[1])
    print(X_train.shape[2])
    Y2_train = Y_train / (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0) - 1.0
    X2_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    Y2_test = Y_test / (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0) - 1
    print (Y_test)
    dataset = Dataset(X2_train, Y2_train)
    
    # training starts
    """with g.as_default():
         session = tf.InteractiveSession()
         model = build_model()
         loss = mse(model.y_placeholder, model.output)
         saver = tf.train.Saver()
         start_time = time.time()
         best_score = 1
         train_step = tf.train.AdamOptimizer().minimize(loss)
         session = tf.InteractiveSession()
         session.run(tf.global_variables_initializer())
         if(repeat>=1):
             print("Session is Loaded")
             saver.restore(session, os.path.join(MODEL_PATH, "model"))
         last_epoch = -1
         while dataset.epoch_completed() < EPOCH:
             (batch_x, batch_y) = dataset.next_batch(20)
             train_step.run(feed_dict={model.x_placeholder: batch_x, model.y_placeholder: batch_y})
             if dataset.epoch_completed() > last_epoch:
                 last_epoch = dataset.epoch_completed()
                 score_test = loss.eval(feed_dict={model.x_placeholder: X2_test, model.y_placeholder: Y2_test})
                 if score_test < best_score:
                     best_score = score_test
                     saver.save(session, os.path.join(MODEL_PATH, "model"))
                 if dataset.epoch_completed() % 1 == 0:
                     epm = 60 * dataset.epoch_completed() / (time.time()-start_time)
                     print('Epoch: %d, Score: %f, Epoch per minute: %f' % (dataset.epoch_completed(), score_test, epm))
         print('Finished in %f seconds.' % (time.time()-start_time)) 
         session.close()
    ResetImg()
    ResetJson()
    get_ipython().magic('reset -sf')"""
# =============================================================================
    """for ak in range(100):
        xs=[ak, ak]
        plot_images(X_train[xs], Y_train[xs])"""
    
    g = tf.Graph()
    with g.as_default():
        session = tf.InteractiveSession()
        model = build_model()
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(MODEL_PATH, "model"))
        ids = [random.randint(0, X_test.shape[0]-1) for _ in range(1)]
        predictions = model.output.eval(session=session, feed_dict={model.x_placeholder: X2_test[ids]})
        myarr =(predictions+1) * (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
        
        topMinY=0
        botMaxY=0
        leftMinX=0
        rightMaxX=0
        if(myarr[0][1] < myarr[0][7]):
            topMinY=myarr[0][1]
            myarr[0][7]=topMinY
        else:
            topMinY=myarr[0][7]
            myarr[0][1]=topMinY
        if(myarr[0][3] > myarr[0][5]):
            botMaxY=myarr[0][3]
            myarr[0][5]=botMaxY
        else:
            botMaxY=myarr[0][5]
            myarr[0][3]=botMaxY
        if(myarr[0][0]<myarr[0][4]):
            leftMinX=myarr[0][0]
            myarr[0][4]=leftMinX
        else:
            leftMinX=myarr[0][4]
            myarr[0][0]=leftMinX
        if(myarr[0][2]>myarr[0][6]):
            rightMaxX=myarr[0][2]
            myarr[0][6]=rightMaxX
        else:
            rightMaxX=myarr[0][6]
            myarr[0][2]=rightMaxX
            
        myarr[0][0]=myarr[0][0]-10
        myarr[0][1]=myarr[0][1]-5
        myarr[0][2]=myarr[0][2]+10
        myarr[0][3]=myarr[0][3]+5
        myarr[0][4]=myarr[0][4]-10
        myarr[0][5]=myarr[0][5]+5
        myarr[0][6]=myarr[0][6]+10
        myarr[0][7]=myarr[0][7]-5
        
        #plot_images(X_test[ids], (predictions+1) * (100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0))
        plot_images(X_test[ids], myarr)
        print(myarr[0][0])
        session.close()
        
        print(myarr)

# =============================================================================
