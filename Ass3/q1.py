import torch
import numpy as np
from data_rnn import load_imdb


def create_batches(x, y, batch_size, pad_token, isAutoregressiveTask=False, start_token=None, end_token=None):
    """
    Given the input data the function creates batches based on a specific batch_size, pads each batch
    with a special padding token and returns the batches as pytorch tensors. If the task is
    autoregressive then it also adds the special characters 'start' and 'end' into each list in the batch.

    :param x: input lists of data
    :param y: the class labels of each sequence 
    :param batch_size: the size of each batch
    :param pad_token: the token used for padding
    :param isAutoregressiveTask: parameter for declaring if the task is autoregressive
    :param start_token: the '.start' special token
    :param end_token: The '.end' special token
    :returns: torch tensors with the batch of input sequences and the lables
    """

    # sort our data by sequence length in descending order
    sorted_indices = np.argsort([len(seq) for seq in x])[::-1]
    x = [x[i] for i in sorted_indices]
    y = [y[i] for i in sorted_indices]

    # create the batches and apply padding to the x batch
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # if it's an autoregressive task then add the special 'start' and 'end' tokens to each sequence
        if isAutoregressiveTask:
            x_batch = [[start_token] + seq + [end_token] for seq in x_batch]

        # find the length of the longest sequence in this batch
        max_len = max(len(seq) for seq in x_batch)
        
        # we pad all the sequences in this batch with the special pad character to match the length of the longest sequence
        x_batch_padded = [seq + [pad_token]*(max_len - len(seq)) for seq in x_batch]
        
        # converting batches to torch tensor entities
        x_batch_tensor = torch.tensor(x_batch_padded, dtype=torch.long)
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.long)

        yield x_batch_tensor, y_batch_tensor


if __name__ == '__main__':

    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

    batch_size = 32
    pad_token = w2i['.pad']
    start_token = w2i['.start']
    end_token = w2i['.end']
    isAutoregressiveTask = False

    x_y_batches = create_batches(x_train, y_train, batch_size, pad_token, isAutoregressiveTask, start_token, end_token)

    for x_batch, y_batch in x_y_batches:
        pass