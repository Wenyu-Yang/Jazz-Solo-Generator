#%%
import numpy as np
from music_utils import one_hot
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Reshape, Lambda
from keras.utils import to_categorical

def global_shared_layers(n_a=64, n_values=78):
    
    reshapor = Reshape((1, n_values))
    LSTM_cell = LSTM(n_a, return_state = True)
    densor = Dense(n_values, activation='softmax')      
    
    return reshapor, LSTM_cell, densor

def djmodel(Tx=30, n_a=64, n_values=78, reshapor, LSTM_cell, densor):
    """
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    # Create empty list to append the outputs while you iterate
    outputs = []
    
    for t in range(Tx):
        
        # Select the "t"th time step vector from X. 
        x = Lambda(lambda z: z[:, t, :])(X)
        # Use reshapor to reshape x to be (1, n_values)
        x = reshapor(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Add the output to "outputs"
        outputs.append(out)
        
    # Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model

def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50):
    """
    Uses the trained "LSTM_cell" and "densor" to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    # Create an empty list of "outputs" to later store your predicted values
    outputs = []
    
    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        
        # Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)

        # Append the prediction "out" to "outputs".
        outputs.append(out)
    
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        x = Lambda(one_hot)(out)
        
    # Create model instance with the correct "inputs" and "outputs"
    inference_model = Model(inputs=[x0, a0, c0], output=outputs)
    
    return inference_model

def predict_and_sample(inference_model, n_values=78, n_a=64):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    x_initializer = np.zeros((1, 1, n_values))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))
    
    # Use the inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, axis=2)
    # Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices, num_classes=78)
    
    return results, indices
