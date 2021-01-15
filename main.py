import pickle
import numpy as np
from parts.vision import *
from parts.mobility import *
from time import sleep

class Brain():

    def __init__(self):
        self.die = False
        self.model_path = "terrain_model"
        self.probability = None
        self.classified = None
        self.model = self.load_vision_intelligence()

        self.mobility = Mobility()
        self.mobility.initialize()

        self.vision = Vision()
        self.vision.start()

        th_predictor = Thread(target=self.prediction_process, name="Predictor")
        th_predictor.daemon=True
        th_predictor.start()
        sleep(1)

        self.process() # Main Process => Blocking Thread

    def load_vision_intelligence(self):
        return pickle.load(open(self.model_path, 'rb'))

    def process(self):
        while not self.die:
            try:
                print("Class: "+str("Gravel" if self.classified[0]==1 else "Smooth") + " " + str(self.probability[0][self.classified[0]]*100) + "%")
            except: pass

    def prediction_process(self):
        while not self.die:
            try:
                ml_input = self.vision.get_input()
                self.probability = self.model.predict_proba([ml_input])
                self.classified = self.model.predict([ml_input])
            except Exception as error: print("Predictor: " + str(error))
if __name__ == "__main__":
    Brain()