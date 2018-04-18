import numpy
import NeuralNetwork


class PSO:
    def __init__(self, swarm_size, num_feature, Y, data, shape):
        self.Y = Y
        self.data = data
        self.shape = shape
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.m = num_feature
        self.n = swarm_size
        self.maxite = 100
        self.LB = -1
        self.UB = 1

        self.X = numpy.random.uniform(-1, 1, (self.n, self.m))
        self.V = numpy.random.uniform(-0.1, 0.1, (self.n, self.m))
        self.Cost = self.eval_cost(self.X)
        self.Pbest = self.X.copy()
        self.Gbest = self.Pbest[self.Cost.argmin()]
        self.best_cost = self.Cost.min()
        pass

    def MSE(self, Y_hat, Y):
        m = Y.shape[0]
        return 1 / m * numpy.sum((Y_hat - Y) ** 2)

    def vector_to_weights(self, vector, shape):
        weight = []
        index = 0
        for i in range(len(shape) - 1):
            row = shape[i + 1]
            col = shape[i] + 1
            id_min = index
            id_max = index + row * col
            weight.append(vector[id_min:id_max].reshape(row, col))
        return weight

    def eval_cost(self, position):
        Cost = []
        neuralNetwork = NeuralNetwork.NeuralNetwork(self.shape)
        for x in position:
            weight = self.vector_to_weights(x, self.shape)
            Y_predict = neuralNetwork.query(self.data, weight)
            Cost.append(self.MSE(Y_predict, self.Y))
        return numpy.asarray(Cost)

    def update(self):
        ite = 0
        while ite < self.maxite and self.best_cost > 10 ** -12:
            R1 = numpy.random.uniform(0, 1, (self.n, self.m))
            R2 = numpy.random.uniform(0, 1, (self.n, self.m))

            w = 0.729
            self.V = w * self.V + self.c1 * R1 * (self.Pbest - self.X) + self.c2 * R2 * (self.Gbest - self.X)
            self.X = self.X + self.V

            x_lower_than_lb = self.X < self.LB
            x_higher_than_ub = self.X > self.UB
            self.X[x_lower_than_lb] = self.LB
            self.X[x_higher_than_ub] = self.UB

            v_lower_than_lb = self.V < (self.LB * 0.1)
            v_higher_than_ub = self.V > (self.UB * 0.1)
            self.V[v_lower_than_lb] = self.LB * 0.1
            self.V[v_higher_than_ub] = self.UB * 0.1

            current_cost = self.eval_cost(self.X)
            better_cost = current_cost < self.Cost
            self.Cost[better_cost] = current_cost[better_cost]
            self.Pbest[better_cost] = self.X[better_cost]
            self.Gbest = self.Pbest[self.Cost.argmin()]
            self.best_cost = self.Cost.min()
            ite += 1
        return self.Gbest, self.best_cost
