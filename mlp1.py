import numpy as np
import loglinear as ll

STUDENT={'name': 'bandele_passove1',
         'ID': '308130038_305930265'}

def classifier_output(x, params):
    W, b, U, b_tag = params
    probs = ll.classifier_output(np.tanh(ll.linear_output(x, (W,b))), (U, b_tag))
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """

    W, b, U, b_tag = params
    lo = ll.linear_output(x, (W, b))
    x_tag = np.tanh(lo)

    loss, (gU, gb_tag) = ll.loss_and_gradients(x_tag, y, (U, b_tag))
    gb = np.dot(gb_tag, U.T) * (1 - (np.tanh(lo) ** 2))
    gW = np.dot(np.atleast_2d(x).T, np.atleast_2d(gb))
    return loss,[gW, gb, gU, gb_tag]

def xavier_initialization(in_dim, out_dim=None):
    if out_dim is None:
        return np.random.randn(in_dim) / np.sqrt(in_dim/2)
    return np.random.randn(in_dim, out_dim) / np.sqrt(in_dim/2)


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    init = xavier_initialization
    W = init(in_dim, hid_dim)
    b = init(hid_dim)
    U = init(hid_dim, out_dim)
    b_tag = init(out_dim)
    params = [W, b, U, b_tag]
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3,4,3)

    def _loss_and_W_grad(W):
        global b
        global U
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global W
        global U
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[3]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)