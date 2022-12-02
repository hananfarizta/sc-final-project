import base64
import numpy as np
from model import Model
from preprocessing import Preprocessing
from inference import Inference
import torch


class Main:
    def __init__(self, img_input):
        self.img_input = img_input
        self.inference()
        self.postpreprocessing()

    def get_model(self):
        model = Model()
        print(model)
        model = torch.load("/Users/hananfarizta/Desktop/Final-Project-SC/sc-final-project/models/model.pt", map_location=torch.device('cpu')) #torch
        model.eval()
        return model

    def prepare_input(self):
        prep = Preprocessing(img_name=self.img_input)
        prep.load_image()
        prep = prep.get_image_
        return prep

    def inference(self):
        model = self.get_model()
        prep = self.prepare_input()
        print(model)
        model = Inference(model, prep)
        self.result = model.infer()

    def postpreprocessing(self):
        dict_map = {
            0:"T-shirt/Top",
            1:"Trouser",
            2:"Pullover",
            3:"Dress",
            4:"Coat",
            5:"Sandal",
            6:"Shirt",
            7:"Sneaker",
            8:"Bag",
            9:"Ankle Boot"
        }
        label = dict_map[self.result.tolist()[0]]
        self._label = label

    @property
    def get_results(self):
        return self._label



if __name__ == "__main__":
    print("welcome to classifier\nplease input the model and image path")
    img_path = input("Image Path : ")


    binary_fc       = open(img_path, 'rb').read()
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')
    print(f"{img_path}, base64: {True}")

    model_img = Main(img_input=base64_utf8_str)
    print(model_img.get_results)
