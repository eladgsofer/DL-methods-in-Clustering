import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid

from ptdec.dec_idec import DEC_IDEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST("./data", download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float() * 0.02

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(self._cache[index][1], dtype=torch.long).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)


def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode, clustering_epochs):
    writer = SummaryWriter()  # create the TensorBoard object

    # % cd /Users/elad.sofer/src/Engineering Project/pt-dec/examples/mnist; tensorboard --logdir=runs
    # callback function to call during training, uses writer from the scope
    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars("data/autoencoder", {"lr": lr, "loss": loss, "validation_loss": validation_loss, }, epoch)

    ds_train = CachedMNIST(train=True, cuda=cuda, testing_mode=testing_mode)
    ds_val = CachedMNIST(train=False, cuda=cuda, testing_mode=testing_mode)
    autoencoder = StackedDenoisingAutoEncoder([28 * 28, 500, 500, 2000, 10], final_activation=None)

    if cuda:
        autoencoder.cuda()

    print("Pretraining stage.")
    ae.pretrain(ds_train, autoencoder,
                cuda=cuda, validation=ds_val,
                epochs=pretrain_epochs, batch_size=batch_size,
                optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
                scheduler=lambda x: StepLR(x, 100, gamma=0.1), corruption=0.2, )

    torch.save(autoencoder, 'ae_pretrain.pt')

    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(ds_train, autoencoder,
             cuda=cuda, validation=ds_val,
             epochs=finetune_epochs, batch_size=batch_size,
             optimizer=ae_optimizer, scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
             corruption=0.2, update_callback=training_callback, )
    torch.save(autoencoder, 'ae_train.pt')
    writer.flush()

    print("DEC/IDEC stage.")
    model = DEC_IDEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder, decoder=autoencoder.decoder,
                     mode='IDEC')
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(dataset=ds_train, model=model, epochs=clustering_epochs,
          batch_size=256, optimizer=dec_optimizer, stopping_delta=0.000001, cuda=cuda, )
    torch.save(model, 'dec.pt')

    predicted, actual, embeddings = predict(ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda,
                                            return_embeddings=True)
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(embeddings)
    df = pd.DataFrame()
    df["y"] = actual
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="{0} MNIST data T-SNE projection".format(model.mode))
    plt.show()

    reassignment, accuracy = cluster_accuracy(actual, predicted)

    print("Final DEC accuracy: %s" % accuracy)
    if not testing_mode:
        predicted_reassigned = [reassignment[item] for item in predicted]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis])
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig("confusion_%s.png" % confusion_id)
        plt.show()
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--cuda", help="whether to use CUDA (default False).", type=bool, default=False)
    arg.add_argument("--batch-size", help="training batch size (default 256).", type=int, default=256)
    arg.add_argument("--pretrain-epochs", help="number of pretraining epochs (default 300).", type=int, default=200, )
    arg.add_argument("--finetune-epochs", help="number of finetune epochs (default 500).", type=int, default=300, )
    arg.add_argument("--clustering-epochs", help="number of finetune epochs (default 500).", type=int, default=500, )
    arg.add_argument("--testing-mode", help="whether to run in testing mode (default False).", type=bool, default=False, )
    arg = arg.parse_args()
    main(arg.cuda, arg.batch_size, arg.pretrain_epochs, arg.finetune_epochs, arg.testing_mode, arg.clustering_epochs)
