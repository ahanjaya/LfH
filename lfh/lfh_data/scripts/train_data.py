#!/usr/bin/python3
import os

import torch
import torch.nn as nn
import torch.optim as optim

import rospy
import rospkg
import numpy as np
import matplotlib.pyplot as plt

from lfh_utils import utils
from lfh_utils.model import LfHModel


class LfHTrain:
    def __init__(self) -> None:
        """ """
        self._rospack = rospkg.RosPack()
        self._data_dir = os.path.join(self._rospack.get_path("lfh_data"), "data")
        self._weight_dir = os.path.join(self._rospack.get_path("lfh_config"), "weights")

        self.config()
        np.set_printoptions(suppress=True)
        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)

        rospy.loginfo("LfH convert data node ready.")

    def config(self) -> None:
        """TODO: Make a global config"""
        self._max_lin_x = rospy.get_param("/LfH/max_v")
        x_train_file = f"x_train_{self._max_lin_x:.1f}_m.npy"
        y_train_file = f"y_train_{self._max_lin_x:.1f}_m.npy"
        self._x_train_fname = os.path.join(self._data_dir, x_train_file)
        self._y_train_fname = os.path.join(self._data_dir, y_train_file)

        self._num_of_raycast = rospy.get_param("/LfH/num_of_raycast")
        self._num_of_vel = rospy.get_param("/LfH/num_of_vel")
        self._learning_rate = rospy.get_param("/LfH/learning_rate")

        self._frame_w = rospy.get_param("/LfH/frame_width")
        self._frame_h = rospy.get_param("/LfH/frame_height")

        self._device = rospy.get_param("/LfH/device")
        self._num_epoch = rospy.get_param("/LfH/num_epoch")
        self._batch_size = rospy.get_param("/LfH/batch_size")
        self._epoch_train_loss = []
        self._epoch_test_loss = []

        # LfH weights.
        self._ckpt_file = f"lfh_weights_{self._max_lin_x}_m.pt"
        self._fig_file = f"lfh_loss_{self._max_lin_x}_m.svg"
        self._ckpt_path = os.path.join(self._weight_dir, self._ckpt_file)
        self._fig_path = os.path.join(self._weight_dir, self._fig_file)

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.train()
        train_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                rospy.loginfo(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss /= num_batches

        return train_loss

    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self._device), y.to(self._device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        # correct /= size
        # rospy.loginfo(
        #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        # )
        rospy.loginfo(
            f"Test Error: Avg loss: {test_loss:>8f} \n"
        )
        return test_loss


    def run(self) -> None:
        """TODO:"""
        # 1. Load data.
        x_npy = np.load(self._x_train_fname)
        y_npy = np.load(self._y_train_fname)

        rospy.loginfo(f"X_train: {x_npy.shape}")
        rospy.loginfo(f"Y_train: {y_npy.shape}")

        # 2. Prepare dataset.
        input_tensor = torch.from_numpy(x_npy).type(torch.float32)
        target_tensor = torch.from_numpy(y_npy).type(torch.float32)
        data_set = torch.utils.data.TensorDataset(input_tensor, target_tensor)

        train_size = int(len(data_set) * 0.9)
        test_size = len(data_set) - train_size
        train_set, test_set = torch.utils.data.random_split(
            data_set, [train_size, test_size]
        )
        rospy.loginfo(f"Train set: {len(train_set)}, Test set: {len(test_set)}")

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self._batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._batch_size, shuffle=True
        )

        # 3. Create model.
        model = LfHModel(input_size=self._num_of_raycast, output_size=self._num_of_vel)
        model.to(self._device)
        rospy.loginfo(model)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)
        n_iterations = len(train_loader)

        # 4. Training.
        for epoch in range(self._num_epoch):
            rospy.loginfo(
                f"Epoch: {epoch}/{self._num_epoch}\n-------------------------------"
            )

            train_loss = self.train(train_loader, model, loss_fn, optimizer)
            test_loss = self.test(test_loader, model, loss_fn)

            self._epoch_train_loss.append(train_loss)
            self._epoch_test_loss.append(test_loss)

            # Plotting data
            self._ax.cla()
            self._ax.plot(self._epoch_train_loss, label="Train loss")
            self._ax.plot(self._epoch_test_loss, label="Test loss")
            plt.legend()
            plt.show(block=False)
            plt.pause(0.0001)

        plt.savefig(self._fig_path)

        # 6. Save weight.
        torch.save(model.state_dict(), self._ckpt_path)
        rospy.loginfo(f"Saved weights: {self._ckpt_path}")


if __name__ == "__main__":
    rospy.init_node("lfh_train", anonymous=True)
    node = LfHTrain()
    node.run()
