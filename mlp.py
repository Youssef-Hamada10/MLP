import numpy as np

class MLP:
  def __init__(self, hidden_layers, learning_rate, epochs, use_bias=False, activation_type='sigmoid'):
    self.input_layer = 5
    self.output_layer = 3
    self.hidden_layers = hidden_layers
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.use_bias = use_bias
    self.activation_type = activation_type
    self.network_archi = [self.input_layer] + self.hidden_layers + [self.output_layer]
    self.weight_list = []
    self.output_list = []
    self.bias_list = []

    for i in range(len(self.network_archi) - 1):
      #init weights list
      w = np.random.randn(self.network_archi[i], self.network_archi[i + 1]) * 0.1
      self.weight_list.append(w)
      #init bias list
      if self.use_bias:
        b = np.random.randn(1, self.network_archi[i + 1]) * 0.1
      else:
        b = np.zeros((1, self.network_archi[i + 1]))
      self.bias_list.append(b)

  def activation_fun(self, net):
    if self.activation_type.lower() == "sigmoid":
      net = np.clip(net, -500, 500)
      return 1.0 / (1.0 + np.exp(-net))
    elif self.activation_type.lower() == "tanh":
      return np.tanh(net)
    else:
      raise ValueError("Invalid activation function")

  def derivative_activation_fun(self, net):
    if self.activation_type.lower() == 'sigmoid':
      return net * (1 - net)
    elif self.activation_type.lower() == 'tanh':
      return 1 - net**2
    else:
      raise ValueError("Invalid activation function")

  def feed_forward(self, x, y):
    x = np.asarray(x).reshape(1, -1)
    y = np.asarray(y).reshape(1, -1)
    self.output_list = []
    for i in range(len(self.network_archi) - 1):
      if i == 0:
        net = x @ self.weight_list[i] + self.bias_list[i]
      else:
        net = self.output_list[i - 1] @ self.weight_list[i] + self.bias_list[i]
      z = self.activation_fun(net)
      self.output_list.append(z)
    prediction = self.output_list[-1]
    return prediction

  def MSE(self, y, prediction):
    mse = 0.5 * np.sum((prediction - y) ** 2, axis=1)
    return mse

  def back_prop(self, x, y):
    x = np.asarray(x).reshape(1, -1)
    y = np.asarray(y).reshape(1, -1)
    prediction = self.output_list[-1]
    num_layers = len(self.network_archi)
    delta = [None] * (num_layers - 1)
    #calc delta
    #for output layer
    delta[-1] = self.derivative_activation_fun(prediction) * (y - prediction)
    #for hidden layer
    for i in range(num_layers - 3, -1, -1):
      delta[i] = (delta[i + 1] @ self.weight_list[i + 1].T) * self.derivative_activation_fun(self.output_list[i])
    #calc new weight
    for i in range(num_layers - 1):
      if i == 0:
        self.weight_list[i] += self.learning_rate * x.T @ delta[i]
      else:
        self.weight_list[i] += self.learning_rate * self.output_list[i - 1].T @ delta[i]
      #bias
      if self.use_bias:
        self.bias_list[i] += self.learning_rate * np.sum(delta[i], axis=0, keepdims=True)

  def train(self, x_train, y_train):
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    loss_history = []
    for epoch in range(self.epochs):
      epoch_loss = 0
      for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        # Forward propagation
        prediction = self.feed_forward(x, y)
        # Calculate loss
        loss = self.MSE(y, prediction)
        epoch_loss += loss[0]
        # Backpropagation and update weights
        self.back_prop(x, y)
      # Average loss for this epoch
      avg_loss = epoch_loss / len(x_train)
      loss_history.append(avg_loss)
      print(f"Epoch {epoch + 1}/{self.epochs}, Y:, {y}, prediction:, {prediction},Loss: {avg_loss:.6f}")
    return loss_history

  def predict(self, X):
    predictions = []
    for x in X:
      output = self.feed_forward(x, None)
      # Get the index of the neuron with the highest output
      predicted_class = np.argmax(output)
      predictions.append(predicted_class)
    return np.array(predictions)