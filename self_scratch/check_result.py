import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("simplenet_results.pkl", "rb") as f:
    # results = pd.DataFrame(pickle.load(f), columns=[train_loss, train_acc_history, test_acc_history])
    train_loss_history, train_acc_history, test_acc_history  = pickle.load(f)

print(train_acc_history)


# with open("simple_net_model.pkl", "rb") as f:
#     # results = pd.DataFrame(pickle.load(f), columns=[train_loss, train_acc_history, test_acc_history])
#     trained_net = pickle.load(f)
#
# print(trained_net.params["W1"])


x = [i for i in range(0,10000, 120)]
print(x)
idx = np.argmax(np.asarray(train_acc_history))
print(idx)
print(train_acc_history[idx], test_acc_history[idx])

plt.plot(x,train_acc_history,label="Train Accuracy")
plt.plot(x,test_acc_history,label="Test Accuracy")
plt.legend()
plt.show()