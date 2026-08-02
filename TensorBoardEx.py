"""""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

NUM_WORKERS = 0

writer=SummaryWriter()




def create_dataloaders(
        train_dir : str,
        test_dir : str,
        transform : transforms.Compose,
        batch_size : int,
        num_workers : int = NUM_WORKERS

):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform,
                                     target_transform=None)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    return train_dataloader, test_dataloader, class_names

class DesertClassifier(nn.Module):
    def __init__(self,input_shape: int , hidden_units : int , output_shape: int):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)


        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)


        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units *16 *16,
                      out_features=output_shape)

        )


    def forward(self,x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module , optimizer : torch.optim.Optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch , (X,y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)


    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return  train_loss,train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module ):
    model.eval()

    test_loss = 0
    test_acc  = 0

    with torch.inference_mode():
        for batch , (X,y) in enumerate(dataloader):
            test_pred_logits = model(X)
            loss =  loss_fn(test_pred_logits,y)
            test_loss +=loss.item()


            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return  test_loss ,test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 10,
          experiment_name: str = "experiment"):  # <-- BU SATIR EKLENMELİ

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # TensorBoard writer, gönderdiğimiz experiment_name ile oluşturuluyor
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 64, 64))

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={
                               "train_loss": train_loss,
                               "test_loss": test_loss
                           },
                           global_step=epoch)

        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={
                               "train_acc": train_acc,
                               "test_acc": test_acc
                           },
                           global_step=epoch)

    writer.close()
    return results



if __name__ == "__main__":
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    train_dir = "IntroToComputerVision/data/desert101/train"
    test_dir = "IntroToComputerVision/data/desert101/test"

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5483, 0.4638, 0.3865],
                             std=[0.2173, 0.2279, 0.2263])
    ])

    train_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
    )

    # MODEL 1 BURADA TANIMLANIYOR
    model1 = DesertClassifier(
        input_shape=3,
        hidden_units=10,
        output_shape=len(class_names)
    )

    # MODEL 2 BURADA TANIMLANIYOR
    model2 = DesertClassifier(
        input_shape=3,
        hidden_units=32,
        output_shape=len(class_names)
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print("Training Model 1 (10 hidden units)...")
    # MODEL 1 BURADA EĞİTİLİYOR
    train(model=model1,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          loss_fn=loss_fn,
          optimizer=torch.optim.Adam(model1.parameters(), lr=LEARNING_RATE),
          epochs=NUM_EPOCHS,
          experiment_name="hidden_units_10"
          )

    print("\nTraining Model 2 (32 hidden units)...")
    # MODEL 2 BURADA EĞİTİLİYOR
    train(model=model2,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          loss_fn=loss_fn,
          optimizer=torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE),
          epochs=NUM_EPOCHS,
          experiment_name="hidden_units_32"
          )

# tensorboard --logdir=runs(terminal)

"""