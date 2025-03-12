"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt
        
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}

        # First moment estimates
        self.m = {}
        # Second moment estimates
        self.v = {}

        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            
            # TODO: (Extra Credit) You may set parameters for Adam optimizer here
            self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
            self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])
            self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
            self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        # print(f"W shape: {W.shape}, X shape: {X.shape}, b shape: {b.shape}")

        return X @ W + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix                  (m input, n output)
            X: the input data                     (B batch_size, m input)
            de_dz: the gradient of loss           (B, m)
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        # TODO: implement me
        de_dw = X.T @ de_dz                     # shape (m, n)
        de_db = np.sum(de_dz, axis=0)           # shape (n,)
        de_dx = de_dz @ W.T                     # shape (B, m)

        return de_dw, de_db, de_dx
  

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(0, X)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
         # TODO: implement me
        return (X > 0).astype(int)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        # return 1/(1 + np.exp(-x))

        # Create boolean masks
        pos = x >= 0   # mask for x >= 0
        neg = x < 0    # mask for x < 0

        # Create output array of same shape as x
        result = np.zeros_like(x, dtype=float)

        # Compute sigmoid for x >= 0
        result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

        # Compute sigmoid for x < 0
        #   rewriting the formula as exp(x) / (1 + exp(x))
        result[neg] = np.exp(x[neg]) / (1.0 + np.exp(x[neg]))

        return result

    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # TODO implement this
        # return self.sigmoid(X) * (1 - self.sigmoid(X))
        # X: the output of the sigmoid
        return X * (1 - X)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.mean((y-p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return (2 * (p - y)) / y.size
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this                    
        return self.mse_grad(y, p) * self.sigmoid_grad(p)   

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {"A0": X}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.


        for i in range(1, self.num_layers + 1):
          W = self.params["W" + str(i)]
          b = self.params["b" + str(i)]
          linear_output = self.linear(W, self.outputs["A" + str(i-1)], b)
          self.outputs["linear" + str(i)] = linear_output

          # Apply activation function
          if i < self.num_layers:  # Hidden layers use ReLU
              A = self.relu(linear_output)
              # A = np.where(linear_output > 0, linear_output, 0.01 * linear_output)

          else:  # Last layer uses sigmoid
              A = self.sigmoid(linear_output)
          
          self.outputs["A" + str(i)] = A
          # print(self.outputs["A" + str(i)].shape)

        return self.outputs["A" + str(self.num_layers)]


        # self.outputs = {'act_0': X}
    
        # for i in range(1, self.num_layers + 1):
        #     X = X @ self.params[f'W{i}'] + self.params[f'b{i}']
        #     self.outputs[f'linear_{i}'] = X

        #     if i != self.num_layers:
        #         X = np.where(X > 0, X, 0.01 * X)  
        #     else:
        #         X = self.sigmoid(X)

        #     self.outputs[f'act_{i}'] = X
        
        # return X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        
        y_pred = self.outputs["A" + str(self.num_layers)]
        mse_l = self.mse(y=y, p=y_pred)
        
        # First Upstream gradient
        de_dx = self.mse_sigmoid_grad(y=y, p=y_pred)

        for i in range(self.num_layers, 0, -1):
          linear_output = self.outputs["linear" + str(i)]     # Z
          input_prev = self.outputs["A" + str(i-1)]           # A
          W = self.params["W" + str(i)]


          dA_dZ = 1 if i == self.num_layers else self.relu_grad(linear_output)
          de_dz = de_dx * dA_dZ

          de_dw, de_db, de_dx = self.linear_grad(W, input_prev, de_dz)

          # Store the gradients for the updates below
          self.gradients["W" + str(i)] = de_dw
          self.gradients["b" + str(i)] = de_db

 
        return mse_l



        # loss = self.mse(y, self.outputs[f'act_{self.num_layers}'])
        # backprop_signal = -2 * (y - self.outputs[f'act_{self.num_layers}']) / 3

        # for layer_idx in range(self.num_layers, 0, -1):
        #     linear_key = f'linear_{layer_idx}'
        #     activation_key = f'act_{layer_idx - 1}'
        #     weight_key = f'W{layer_idx}'
        #     bias_key = f'b{layer_idx}'

        #     if layer_idx != self.num_layers:
        #         activation_grad = np.where(self.outputs[linear_key] > 0, 1, 0.01)  
        #     else:
        #         activation_grad = self.sigmoid_grad(self.outputs[linear_key])  
            
        #     val = backprop_signal * activation_grad
        #     self.gradients[bias_key] = np.mean(val, axis=0)
        #     self.gradients[weight_key] = np.dot(self.outputs[activation_key].T, val) / self.batch_size
        #     backprop_signal = np.dot(val, self.params[weight_key].T)
        
        # return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if self.opt == 'SGD':
            # TODO: implement SGD optimizer here
            for i in range(1, self.num_layers + 1):
                self.params['W' + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params['b' + str(i)] -= lr * self.gradients["b" + str(i)]
              
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            # Initialize Adam-specific parameters if not already done
            # pass

            for i in range(1, self.num_layers + 1):
                key_W = "W" + str(i)
                key_b = "b" + str(i)

                g_W = self.gradients[key_W]
                g_b = self.gradients[key_b]

                # Update first and second moment estimates
                self.m[key_W] = b1 * self.m[key_W] + (1 - b1) * g_W
                self.m[key_b] = b1 * self.m[key_b] + (1 - b1) * g_b

                self.v[key_W] = b2 * self.v[key_W] + (1 - b2) * (g_W ** 2)
                self.v[key_b] = b2 * self.v[key_b] + (1 - b2) * (g_b ** 2)

                # # Bias correction
                # m_hat_W = self.m[key_W] / (1 - b1 ** self.t)
                # m_hat_b = self.m[key_b] / (1 - b1 ** self.t)

                # v_hat_W = self.v[key_W] / (1 - b2 ** self.t)
                # v_hat_b = self.v[key_b] / (1 - b2 ** self.t)

                # Update parameters
                self.params[key_W] -= lr * self.m[key_W] / (np.sqrt(self.v[key_W]) + eps)
                self.params[key_b] -= lr * self.m[key_b] / (np.sqrt(self.v[key_b]) + eps)

        else:
            raise NotImplementedError
        