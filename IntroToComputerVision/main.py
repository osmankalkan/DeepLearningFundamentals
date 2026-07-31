import torch
from sympy.codegen.rewriting import optimize
from torchvision import transforms

import setup_data , training_testing_engine , models , utils ,model_cration



def main():
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir = "data/desert101/train"
    test_dir = "data/desert101/train"

    data_transform = transforms.Compose(
        [
            transforms.Resize(size=(64,64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5483, 0.4638, 0.3865],
                                 std=[0.2173, 0.2279, 0.2263])

         ]

    )

    train_dataloader, test_dataloader, class_names =setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
    )

    model = model_cration.DesertClassifier(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters() , lr=LEARNING_RATE)



    training_testing_engine.train(
        model = model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn = loss_fn,
        epochs=NUM_EPOCHS
    )

    utils.save_model(
        model=model,
        target_dir="models",
        model_name="desert_classifier.pth"
    )

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn' , force=True)
    main()


