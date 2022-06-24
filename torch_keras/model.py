import torch
from torchsummary import summary as summary_
import pkbar


class Model():
  def __init__(self, inputs, outputs, device):
    self.input_size = inputs
    self.device = device
    self.model = outputs.to(self.device)

  def parameters(self):
    return self.model.parameters()

  def compile(self, optimizer, loss):
    self.opt = optimizer
    self.criterion = loss

  def summary(self):
    summary_(self.model, self.input_size, device=self.device)
    print("Device Type:", self.device)

  def fit(self, data_x, data_y, epochs):
    self.model.train()

    for epoch in range(epochs):
      print("Epoch {}/{}".format(epoch+1, epochs))
      progress = pkbar.Kbar(target=len(data_x), width=25)
      
      for i, (data, target) in enumerate(zip(data_x, data_y)):
        self.opt.zero_grad()

        train_out = self.model(data.to(self.device))
        loss = self.criterion(train_out, target.to(self.device))
        loss.backward()

        self.opt.step()

        progress.update(i, values=[("train_loss: ", loss.item())])

      progress.add(1)

  def evaluate(self, test_x, test_y):
    self.model.eval()
    correct, loss = 0.0, 0.0

    progress = pkbar.Kbar(target=len(test_x), width=25)

    for i, (data, target) in enumerate(zip(test_x, test_y)):
      out = self.model(data.to(self.device))
      loss += self.criterion(out, target.to(self.device))

      progress.update(i, values=[("eval_loss", loss.item()/len(test_x))])
    progress.add(1)


  def fit_generator(self, generator, epochs):
    self.model.train()

    for epoch in range(epochs):
      print("Epoch {}/{}".format(epoch+1, epochs))
      progress = pkbar.Kbar(target=len(generator), width=25)

      for i, (data, target) in enumerate(generator):
        self.opt.zero_grad()

        train_out = self.model(data.to(self.device))
        loss = self.criterion(train_out.squeeze(), target.to(self.device))
        loss.backward()

        self.opt.step()

        progress.update(i, values=[("train_loss: ", loss.item())])

      progress.add(1)
      

  def evaluate_generator(self, generator):
    self.model.eval()
    correct, loss = 0.0, 0.0

    progress = pkbar.Kbar(target=len(generator), width=25)

    for i, (data, target) in enumerate(generator):
      out = self.model(data.to(self.device))
      loss += self.criterion(out.squeeze(), target.to(self.device))

      progress.update(i, values=[("eval_loss", loss.item()/len(generator))])

    progress.add(1)

  def predict_generator(self, generator):
    self.model.train()
    out = []
    for i, (data, labels) in enumerate(generator):
      out.append(self.model(data.to(self.device)))

    return out