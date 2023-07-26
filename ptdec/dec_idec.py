import torch
import torch.nn as nn

from ptdec.cluster import ClusterAssignment


class DEC_IDEC(nn.Module):
    def __init__(self, cluster_number: int,
                 hidden_dimension: int,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 alpha: float = 1.0,
                 mode: str = 'DEC',
                 lambda_: float = 0.005):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param decoder: decoder to use used, only in IDEC mode
        :param mode: IDEC/DEC/DCEC mode - influencing which Loss to be used
        :param mode: lambda_ weight for structure loss
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC_IDEC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha)
        self.mode = mode
        self.lambda_ = lambda_

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))

    def get_embeddings(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder(batch)