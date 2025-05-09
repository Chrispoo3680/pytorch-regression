class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        test_dataloader,
        device,
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        pass
