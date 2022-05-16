import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import trange
from topologylayer.nn import RipsLayer
from torch.nn.functional import relu


class Ensembler(torch.nn.Module):
  def __init__(self, N, style='full', seed=23529, normalize_columns=False):
    super(Ensembler, self).__init__()
    self.style = style
    self.normalize_columns = normalize_columns
    torch.manual_seed(seed)
    if style == 'full':
      self.weights = torch.nn.Parameter(
          torch.randn(N, N),
          requires_grad=True
      )
    elif style == 'line':
      self.weights = torch.nn.Parameter(
          torch.randn(N, 1) * .1 - 5,
          requires_grad=True
      )

  def forward(self, x):
      if self.style == 'full':
        weights = torch.nn.functional.softmax(self.weights, dim=1)
        out = torch.matmul(weights, x)
      elif self.style == 'line':
        weights = torch.nn.functional.sigmoid(self.weights)
        out = weights * x
      return out, weights


def sort_and_split_top_K_torch(x, dim, K):
    h1_lifes = x[0][dim][:, 1] - x[0][dim][:, 0]
    sorted = torch.sort(h1_lifes, descending=True)[0]
    return relu(sorted[:K]), relu(sorted[K:])


def sort_and_split_top_K_rips(x, dim, K):
    h1_lifes = x['dgms'][dim][:, 1] - x['dgms'][dim][:, 0]
    sorted = np.sort(h1_lifes)[::-1]
    return sorted[:K], sorted[K:]


def sort_and_split_top_K_giotto(diagrams, K):
    h1_lifes = diagrams[0][:, :2][:, 1] - diagrams[0][:, :2][:, 0]
    sorted = np.sort(h1_lifes)[::-1]
    return sorted[:K], sorted[K:]


class Teddy:
    def __init__(
            self,
            data_matrix,  # neurons x time steps
            labels=None,
            signature=(1, 2),  # optimize the first {signature[1]} H_{signature[0]}
            batch_size=16,
            max_steps=100000,  # maximum number of iterations
            log_steps=100,
            lr=1e-2,
            seed=58274,
            device="cpu",
            num_worse=5,  # how many epochs loss may get worse before stopping
            save_file=None,
            stop_accuracy=0.95,
            verbose=True,
    ):
        # Parameters
        self.data_matrix = data_matrix.copy()
        self.labels = labels
        self.N, self.T = data_matrix.shape
        self.signature = signature
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.log_steps = log_steps
        self.lr = lr
        self.seed = seed
        self.num_worse = num_worse
        self.save_file = save_file
        self.stop_accuracy = stop_accuracy
        self.verbose = verbose

        # Setup Data, Model, Optimizer
        self.data_torch = torch.tensor(
            self.data_matrix, dtype=torch.float).to(device)
        self.rips = RipsLayer(batch_size, maxdim=signature[0]).to(device)
        self.model = Ensembler(self.N).to(device)
        self.optimizer = torch.optim.Adam([self.model.weights], lr=lr)

    def train(self):
        worse = 0  # counter for early stopping
        running_loss = 0.0
        running_loss_top = 0.0
        running_loss_rest = 0.0
        run_loss, run_bars = [], []
        self.stats = {'iteration': [], 'loss_top': [], 'loss_rest': [], 'loss': []}
        if self.labels is not None:
            self.stats['accuracy'] = []

        with trange(self.max_steps) as pbar:
            #for i in range(self.max_steps):
            for i in pbar:
                # training step
                self.optimizer.zero_grad()
                np.random.seed(self.seed + i)
                batch_ind = np.random.choice(self.T, self.batch_size, replace=False)
                out, weights = self.model(self.data_torch[:, batch_ind])
                layer_out = self.rips(out.T)
                bars_top_k, bars_rest = sort_and_split_top_K_torch(
                    layer_out, dim=self.signature[0], K=self.signature[1])
                # optimal loss from simulations:
                loss_top = torch.sum(bars_top_k)
                loss_rest = torch.sum(bars_rest)
                loss = loss_rest - loss_top
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.lr)
                self.optimizer.step()

                # logging
                running_loss += loss.item()
                running_loss_top += loss_top.item()
                running_loss_rest += loss_rest.item()
                run_loss.append(loss.item())
                run_bars.append(layer_out[0][1].cpu().detach().numpy())

                if i > -1 and not (i % self.log_steps):
                    if self.labels is not None:  # compute accuracy
                        weights = weights.cpu().detach().numpy()
                        if self.model.style == 'full':
                            label = weights.max(0)
                            label = label > np.median(label)
                        else:
                            label = weights > np.median(weights)
                        dominant_class = np.bincount(self.labels[label]).argmax()
                        target = np.zeros_like(self.labels)
                        target[self.labels == dominant_class] = 1
                        predictions = np.zeros_like(self.labels)
                        predictions[label] = 1
                        self.stats['accuracy'].append(np.mean(target == predictions))

                    self.stats['iteration'].append(i)
                    self.stats['loss'].append(running_loss / self.log_steps)
                    self.stats['loss_top'].append(running_loss_top / self.log_steps)
                    self.stats['loss_rest'].append(running_loss_rest / self.log_steps)
                    pbar.set_postfix(  # print latest of each
                        dict([(stat, self.stats[stat][-1]) for stat in self.stats if stat is not 'iteration'])
                    )

                    # stop when supervised
                    if self.labels is not None:
                        if self.stats['accuracy'][-1] >= self.stop_accuracy:
                            if self.save_file is not None:
                                out = {'weights': weights, 'stats': self.stats}
                                with open(self.save_file + '.pickle', 'wb') as handle:
                                    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            exit_message = 'Stop (done) at iteration %s' % i
                            break

                    # early stopping
                    if self.stats['loss'][-1] > np.min(self.stats['loss']):
                        worse += 1
                        if worse > self.num_worse:
                            exit_message = 'Early stopping at iteration %s' % i
                            break
                    else:
                        worse = 0  # reset counter
                        if self.save_file is not None:
                            out = {'weights': weights, 'stats': self.stats}
                            with open(self.save_file + '.pickle', 'wb') as handle:
                                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # if not stop, prepare next training iteration.
                    running_loss = 0.0
                    running_loss_top = 0.0
                    running_loss_rest = 0.0
                    run_loss, run_bars = [], []

        if 'exit_message' not in locals():
            exit_message = "maximum number of iterations reached."
        print('Training finished:', exit_message)
        if self.verbose:
            self.plotting(weights)


    def plotting(self, weights):
        plt.figure(figsize=(15, 4))

        if self.model.style in ['full', 'gumbel']:
            label = weights.max(0)
            label_sum = weights.sum(0)
            plt.subplot(1, 3, 1)  # complete weight matrix
            plt.imshow(weights)
            plt.colorbar()
            plt.title("Learned weight matrix")
        elif self.model.style == 'line':
            label = weights
            label_sum = None

        plt.subplot(1, 3, 2)  # max over columns of weight matrix
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(label, 'g-')
        ax1.set_xlabel('weights')
        ax1.set_ylabel('max over columns', color='g', alpha=.5)
        if label_sum is not None:
            ax2.plot(label_sum, 'b-', alpha=.5)
            ax2.set_ylabel('sum over columns', color='b')
        ax1.hlines(np.median(label), 0, self.N)
        plt.title("Learned weights")

        plt.subplot(1, 3, 3)  # plot running stats
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax1.plot(self.stats['iteration'], self.stats['loss'],
                 '.-', label='loss_total')
        ax1.plot(self.stats['iteration'], self.stats['loss_top'],
                 '.-', label='loss_top')
        ax1.plot(self.stats['iteration'], self.stats['loss_rest'],
                 '.-', label='loss_rest')
        ax2.plot(self.stats['iteration'], self.stats['accuracy'],
                 '.-', color='black', label='accuracy')
        ax1.set_xlabel('Training iteration')
        ax1.set_ylabel('Loss value')
        ax2.set_ylabel('Accuracy')
        plt.title("Training progress")
        ax1.legend()
        plt.tight_layout()
        plt.show()


