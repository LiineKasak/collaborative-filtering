import torch
import torch.nn as nn
import torch.optim as optim
from .algobase import AlgoBase
from utils.fitting import train
from utils.dataset import DatasetWrapper


class NCF(AlgoBase):

    class Model(nn.Module):
        def __init__(self, gmf, mlp):
            super().__init__()
            self.gmf = gmf
            self.mlp = mlp
            self.alpha = nn.Parameter(torch.rand(1, requires_grad=True))

        def forward(self, users, movies):
            return self.alpha * self.gmf(users, movies) + (1 - self.alpha) * self.mlp(users, movies)

    def __init__(self, gmf, mlp, device, epochs=20, batch_size=128, learning_rate=1e-2):
        super().__init__()
        self.device = device
        self.model = self.Model(gmf, mlp).to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, train_data: DatasetWrapper, test_data: DatasetWrapper = None):
        return train(
            train_data,
            test_data,
            device=self.device,
            model=self.model,
            optimizer=optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1),
            num_epochs=self.epochs,
            batch_size=self.batch_size,
        )

    def predict(self, users, movies):
        return self.model(users, movies)

    def save(self, filename: str):
        torch.save(self.model, filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils.experiment import run_experiment
    import torch


    def parser_setup(parser: ArgumentParser):
        parser.add_argument('--gmf', type=str, help='GMF state dict')
        parser.add_argument('--mlp', type=str, help='MLP state dict')


    def model_factory(args, device):
        gmf = torch.load(args.gmf)
        gmf.eval()

        mlp = torch.load(args.mlp)
        mlp.eval()

        return NCF(gmf, mlp, device, epochs=args.epochs, batch_size=128, learning_rate=args.lr)


    run_experiment(
        'NCF',
        parser_setup,
        model_factory,
    )
