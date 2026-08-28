import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    L = len(layer_dims) #  L layers(including input layer)  
    parameters ={}
    np.random.seed(3)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b"+ str(l)] = np.zeros((layer_dims[l],1))

    return parameters
        

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A) + b
    cache = (A,W,b)

    return Z,cache

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    cache = Z
    A = 1/(1+np.exp(-Z))

    return A,cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    cache = Z
    A = np.maximum(0,Z)

    return A,cache
    

def linear_activation_forward(A_prev,W,b,activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation=='sigmoid':
        Z ,linear_cache = linear_forward(A_prev,W,b)
        A ,activation_cache = sigmoid(Z)

    elif activation=='relu':
        Z ,linear_cache = linear_forward(A_prev,W,b)
        A ,activation_cache = relu(Z)


    cache = (linear_cache,activation_cache)


    return A ,cache
        

        

def L_model_forward(X , parameters,keep_probs=None):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    keep_probs -- python list of keep_prob values, one per layer (e.g. [1.0, 0.7, 0.8, 1.0])
                  keep_probs[l] corresponds to the dropout applied AFTER layer l's activation
                  Use 1.0 for a layer where you don't want dropout (e.g. often the input/output layers)
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """


    caches = []
    A = X
    L = len(parameters)//2

    if keep_probs is None:
        keep_probs = [1.0] * (L - 1)

    #Hidden Layers 1 to L-1(Relu)
    for l in range(1,L):
        A_prev = A

        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        activation = 'relu'
        A, cache = linear_activation_forward(A_prev ,W ,b ,activation)

        keep_prob = keep_probs[l-1]
        if(keep_prob<1.0):
            D = np.random.rand(A.shape[0],A.shape[1])
            D = (D<keep_prob).astype(int)
            A = A*D
            A = A/keep_prob
        else : D = None
            
        cache  = cache + (D,keep_prob)
        caches.append(cache)


    #Output Layer(L) -sigmoid
    W = parameters['W'+str(L)]
    b = parameters['b'+str(L)]
    activation = 'sigmoid'
    AL ,cache = linear_activation_forward(A, W, b, activation)
    caches.append(cache)


    return AL, caches
     
    
        
def compute_cost (AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- binary-cross-entropy cost
    """

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    return cost

def compute_cost_regularized (AL, Y,parameters,lambd):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- binary-cross-entropy cost
    """

    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)

    L = len(parameters)//2

    reg_cost = 0;

    for l in range(1,L+1):
        reg_cost += np.sum(np.square(parameters['W'+str(l)]))

    reg_cost = (lambd/(2*m))*reg_cost

    cost += reg_cost

    return cost

       
def linear_backward(dZ, cache,lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """


    A_prev,W,b =cache
    m=A_prev.shape[1]


    dW = (1/m)*np.dot(dZ,A_prev.T)
    if(lambd>0):
        dW += (lambd/m)*W
        
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)

    dA_prev =np.dot(W.T,dZ)


    return dA_prev, dW, db


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA*(s*(1-s))


    return dZ
      

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0

    return dZ

def linear_activation_backward(dA, cache, activation,lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)


    dA_prev, dW, db = linear_backward(dZ, linear_cache,lambd)


    return dA_prev, dW, db



def L_model_backward(AL, Y, caches,lambd = 0):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """


    grads = {} 
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # Initializing the backpropagation
    dZL = AL - Y
        
    current_cache = caches[L-1]
        
    linear_cache, activation_cache = current_cache
        
    dA_prev_temp, dW_temp, db_temp = linear_backward(
            dZL, linear_cache,lambd
    )

        
    grads["dA"+str(L-1)] = dA_prev_temp
    grads["dW"+str(L)] = dW_temp
    grads["db"+str(L)] =db_temp


    # 1 to L-1 layers(Hidden) ==> 0 to L-2(caches)
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        linear_cache, activation_cache, D, keep_prob = current_cache
        
        dA = grads["dA"+str(l+1)]
        if keep_prob < 1.0:
            dA = dA*D
            dA = dA/keep_prob
            
        grads["dA"+str(l+1)] = dA
        
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, (linear_cache,activation_cache), activation='relu',lambd=lambd)
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] =db_temp



    return grads
   
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters)//2

    for l in range(L):
        parameters['W'+str(l+1)] -= learning_rate*grads['dW'+str(l+1)] 
        parameters['b'+str(l+1)] -= learning_rate*grads['db'+str(l+1)] 


    return parameters
    
       
    
    
    
    





def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    AL, caches = L_model_forward(X, parameters)
    predictions = (AL>0.5)
    return predictions
