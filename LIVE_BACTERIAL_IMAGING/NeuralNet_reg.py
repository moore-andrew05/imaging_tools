import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_inputs, hidden_layers, class_names):
        
        self.n_inputs = n_inputs
        
        self.classes = np.array(class_names).reshape(-1,1)
        self.n_outputs = len(class_names)

        self.hiddens = hidden_layers
        self.n_hidden_layers = len(self.hiddens)
        
        self.weights = []
        self._weights_wrapper()
        
        self.iv = None
        self.H1 = None
        self.H2 = None


        self.layers_out = []
        self.mse_trace = []
        self.percent_correct_trace = []

        self.X_means = None
        self.X_stds = None
        
    def _weights_wrapper(self):
        self.weights.append(self._make_W(self.n_inputs, self.hiddens[0]))
        for i in range(self.n_hidden_layers - 1):
            self.weights.append(self._make_W(self.hiddens[i], self.hiddens[i+1]))
        self.weights.append(self._make_W(self.hiddens[-1],self.n_outputs))

    def _make_W(self, ni, nu):
        return np.random.uniform(-1, 1, size=(ni + 1, nu)) / np.sqrt(ni + 1)
    
    def train(self, X, T, n_epochs, learning_rate):
        X = self._standardize(X)
        self.iv = self._make_indicator_vars(T)

        for epoch in range(n_epochs):
            #Forward Prop
            Y_classes, Y_softmax = self._fprop(X)
            #Backward Prop
            self._bprop(X, Y_softmax, learning_rate)
            
            self.mse_trace.append(np.mean((self.iv - Y_softmax) ** 2))
            self.percent_correct_trace.append(self.percent_correct(T, Y_classes))

    def use(self, X, standardized=False):
        if not standardized:
            X = self._standardize(X)
        Y_classes, Y_softmax = self._fprop(X)
        self.H1 = self.layers_out[0]
        self.H2 = self.layers_out[1]
        return Y_classes, Y_softmax


    def _fprop(self, X):
        self.layers_out = []
        self.layers_out.append(self._f(self._add_ones(X) @ self.weights[0]))

        for i in range(1,len(self.weights)-1):
            self.layers_out.append(self._f(self._add_ones(self.layers_out[-1]) @ self.weights[i]))

        Y = self._add_ones(self.layers_out[-1]) @ self.weights[-1]
        Y_softmax = self._softmax(Y)
        Y_classes = self.classes[np.argmax(Y_softmax, axis=1)]
        return Y_classes, Y_softmax
    
    def _bprop(self, X, Y_sm, learning_rate):
        delta = -2 * (self.iv - Y_sm)    
            
        for i in range(len(self.weights)-1, 0, -1):
            self.weights[i] -= learning_rate / X.shape[0] * self._add_ones(self.layers_out[i-1]).T @ delta
            delta = delta @ self.weights[i][1:, :].T * self._df(self.layers_out[i-1])

        self.weights[0] -= learning_rate / X.shape[0] * self._add_ones(X).T @ delta

    def _standardize(self, X):        
        if self.X_means is None:
            self.X_means = np.mean(X, axis=0)
            self.X_stds = np.std(X, axis=0)
            self.X_stds[self.X_stds == 0] = 1
        return (X - self.X_means) / self.X_stds
        
    
    def _add_ones(self, M):
        return np.insert(M, 0, 1, 1)
    
    def _make_indicator_vars(self, T):
        return (T == np.unique(T)).astype(int)
    
    def _softmax(self, Y):
        fs = np.exp(Y)  # N x K
        denom = np.sum(fs, axis=1).reshape((-1, 1))
        return fs / denom
    
    def _f(self, S):
        return np.tanh(S)

    def _df(self, fS):
        return (1 - fS ** 2)
    
    def percent_correct(self, T, Y_classes):
        return 100 * np.mean(T == Y_classes)
    
    def plot_mse_trace(self):
        if len(self.mse_trace) == 0:
            print("Train Model Before Attempting to Plot!")
            return None

        plt.plot(self.mse_trace)
        plt.title("MSE Trace")
        plt.xlabel("Epoch #")
        plt.ylabel("MSE")

    def plot_percent_correct_trace(self):
        if len(self.percent_correct_trace) == 0:
            print("Train Model Before Attempting to Plot!")
            return None

        plt.plot(self.percent_correct_trace)
        plt.title("% Correct Trace")
        plt.xlabel("Epoch #")
        plt.ylabel("% Correct")