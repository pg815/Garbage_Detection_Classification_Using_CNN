import operator
import cv2 as cv
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import model_from_json

class DetectGarbage:
    def classify(self,testImageFile):

        # read the json model
        file = open('GarbageDetection.json', 'r')
        data = file.read()
        #print(data)

        file.close()

        # classifier will load the model from the data
        # data -> contents of the my_model.json file
        classifier = model_from_json(data)

        # load waits
        classifier.load_weights('GarbageDetection.h5')

        # load the test image
        from keras.preprocessing import image

        test_image = image.load_img(testImageFile, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        result = classifier.predict(test_image)
        if result[0][0] == 1:
            prediction = 'Garbage'
        else:
            prediction = 'NoGarbage'

        return (prediction)



if __name__ == "__main__":
    Obj = DetectGarbage()
    Obj.classify("Upload_image/gar3.jpg")
