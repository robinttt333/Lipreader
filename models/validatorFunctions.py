import torch


def temporalCNNValidator(outputs, labels):
    """ Here we get a batchSize * total labels(500) outputs shape and batchSize shaped labels.eg - 
        Consider the word afternoon [ 10 * 500 ] -----> [1]
        It maps from a 10 * 500 tensor(matrix) to a single label.  
    """
    """Calculate the max for each batch out of the 500 possible outputs. In return we get the 
    index of the word along with its actual probability which is of little use to us"""
    maxvalues, maxindices = torch.max(outputs, 1)
    count = 0
    for i in range(0, labels.size(0)):
        if maxindices[i] == labels[i]:
            count += 1

    return count  # return the number of correct predictions in the batch


def lstmValidator(outputs, labels):
    """The input is of the form batchsize * frames * total labels. So we need to get the correct
    label corresponding to each frame.We first take avg across all 29 frames in a video giving us back a vector with
    dimensions batchSize * total labels and then the process is similar to what we did in the function above."""

    # same as taking avg as sum/29 for all labels
    outputsTransformed = torch.sum(outputs, 1)
    return temporalCNNValidator(outputsTransformed, labels)
