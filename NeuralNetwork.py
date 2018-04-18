import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, shape):
        self.activate_function = lambda x: numpy.tanh(x)
        # self.activate_function = lambda x: scipy.special.expit(x)
        self.shape = shape
        pass

    def query(self, input, weight):
        pre_layer = input.T
        for i in range(len(self.shape) - 1):
            pre_layer = numpy.insert(pre_layer, 0, 1, axis=0)
            layer_input = numpy.dot(weight[i], pre_layer)
            layer_output = self.activate_function(layer_input)
            pre_layer = layer_output
        return pre_layer