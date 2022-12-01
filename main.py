import numpy as np
from model import Model
from preprocessing import Preprocessing
from inference import Inference


class Main:
    def __init__(self, img_input:str):
        self.img_input = img_input
        self.inference()

    def get_model(self):
        model = Model()
        model = model.get_model
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
        results = model.infer()
        return results

    @property
    def get_results(self):
        return self.res



if __name__ == "__main__":
    print("welcome to classifier\nplease input the model and image path")
    img_path = input("Image Path : ")

    model_img = Main(img_input=img_path)