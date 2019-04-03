import numpy as np
from dataset import mnist
from nn import SimpleNN
import pickle


# get mnist dataset
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(one_hot_label=True)

print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)

# set hyper parameter
iternum = 10000
train_size = x_train.shape[1]
batch_size = 100
learning_rate = 0.1
epoch = max(x_train.shape[0] / batch_size, 1)

# result lists
train_loss_history = []
train_acc_history = []
test_acc_history = []

# define simple NN       in : 784    hidden node num -> 50  out:10
network = SimpleNN(train_size, 50, 10)


# start learning
for i in range(iternum):

    # make batch data
    batch_idx = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_idx]
    t_batch = t_train[batch_idx]

    grad = network.back_propagation(x_batch, t_batch)
    network.pram_update(grad, learning_rate)

    loss = network.loss(x_batch, t_batch)
    train_loss_history.append(loss)

    if i % epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print("Training Accuracy : {}%, Test Accuracy : {}%  @Step {}".format(train_acc*100, test_acc*100, i))

# save result
with open("simplenet_results.pkl", "wb") as f:
    print("Save Results")
    pickle.dump([train_loss_history, train_acc_history, test_acc_history], f)

# save model
with open("simple_net_model.pkl", "wb") as f:
    print("Save Model")
    pickle.dump(network, f)



