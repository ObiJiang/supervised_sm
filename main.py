import numpy as np
import torch, torchvision
from single_layer import SingleLayerDSSM
from sklearn.metrics import accuracy_score

# misc
from misc import AttrDict
import random
from tqdm import tqdm
import argparse

# set random seed
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)

class SupervisedSNN():
    def __init__(self, args):
        config = AttrDict()

        # input
        input_dims = [28, 28, 1]
        input_dims_linear = int(np.prod(input_dims))
        # output
        config.nb_classes = 10
        config.nb_subset_classes = args.nb_test_classes
        config.subset_classes = random.sample(range(config.nb_classes), config.nb_subset_classes)
        print("Chosen classes: {}".format(config.subset_classes))
        config.output_dims = config.nb_subset_classes

        config.nb_units = [30]
        config.nb_units.insert(0, input_dims_linear)
        config.nb_layers = len(config.nb_units)
        config.nb_units.append(config.output_dims)

        # feedback parameter
        config.gamma = 0.10

        # host and device
        config.host = torch.device("cpu")
        if args.gpu:
            config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            config.device = config.host 

        # learning rate and step size
        config.euler_lr = 0.1
        config.lr = 1e-3
        config.tol = 1e-5

        # training params
        config.nb_epochs = args.nb_epochs
        config.batch_size = args.batch_size

        # data dir
        config.data_dir = args.data_dir

        self.layers = {}

        self.config = config

        # mnist train data lodaer
        train_dataset = torchvision.datasets.MNIST(self.config.data_dir, train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ]))
        train_mask = self._get_indices(train_dataset, config.subset_classes)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
          batch_size=self.config.batch_size, shuffle=False, sampler = torch.utils.data.sampler.SubsetRandomSampler(train_mask))

        # mnist test data lodaer
        test_dataset = torchvision.datasets.MNIST(self.config.data_dir, train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ]))
        test_mask = self._get_indices(test_dataset, config.subset_classes)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
          batch_size=self.config.batch_size, shuffle=False, sampler = torch.utils.data.sampler.SubsetRandomSampler(test_mask))

    """ helper func """
    def _label_embedding(self, labels):
        label_emb = torch.nn.functional.one_hot(labels, self.config.nb_classes).index_select(1, torch.tensor(self.config.subset_classes)).float()
        # label_emb = (label_emb * 2 - 1)/2

        label_emb = label_emb/np.sqrt(self.config.nb_units[-1])
        return label_emb 

    def _get_indices(self, dataset, class_names):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] in class_names:
                indices.append(i)
        return indices

    def _preprocessing(self, inp):
        processed = torch.max(torch.min(inp, torch.ones_like(inp, device = self.config.device) * 1/np.sqrt(self.config.nb_units[0])), 
                 torch.zeros_like(inp, device = self.config.device))

        return processed

    """ main func """
    def create_network(self):
        for layer_id in range(self.config.nb_layers):
            layer_config = AttrDict()

            # layer-wise configs
            layer_config.layer_ind = layer_id
            layer_config.input_dims = self.config.nb_units[layer_id]
            layer_config.output_dims = self.config.nb_units[layer_id + 1]

            # network-wise configs
            layer_config.network_config = self.config

            self.layers[layer_id] = SingleLayerDSSM(layer_config)

    def init_layers(self, batch_size):
        for layer_id in range(self.config.nb_layers):
            self.layers[layer_id].initialize(batch_size)

    def run_test(self, inp):
        delta = np.ones(self.config.nb_layers) * np.inf

        self.layers[self.config.nb_layers - 1].clear_z()

        for dynamic_step in range(3000):
            cur_inp = inp
            for layer_id in range(self.config.nb_layers - 1):

                delta[layer_id] = self.layers[layer_id].run_dynamics(cur_inp, self.layers[layer_id+1].feedback, step = dynamic_step)

                cur_inp = self.layers[layer_id].output
            
            delta[self.config.nb_layers - 1] = self.layers[self.config.nb_layers - 1].run_dynamics(cur_inp, step = dynamic_step)

            if delta.mean() < self.config.tol:
                break

        return delta.mean()

    def run_train(self, inp, out, epoch = 0):
        delta = np.ones(self.config.nb_layers-1) * np.inf

        self.layers[self.config.nb_layers - 1].set_z(out)

        for dynamic_step in range(3000):
            cur_inp = inp
            for layer_id in range(self.config.nb_layers - 1):

                delta[layer_id] = self.layers[layer_id].run_dynamics(cur_inp, self.layers[layer_id+1].feedback, step = dynamic_step)
                cur_inp = self.layers[layer_id].output

            if delta.mean() < self.config.tol:
                break

        cur_inp = inp
        for layer_id in range(self.config.nb_layers):
            self.layers[layer_id].update_plasiticity(cur_inp, epoch = epoch)
            cur_inp = self.layers[layer_id].output

        return delta.mean()

    def get_pred(self):
        return np.argmax(self.layers[self.config.nb_layers - 1].output.cpu().data.numpy(), axis=1)
    
    def get_acc(self, label):
        label_pred = self.get_pred()
        label_golden = np.argmax(label.cpu().data.numpy(), axis=1)

        acc = accuracy_score(label_golden, label_pred)
        return acc

    def train(self):
        for epoch in tqdm(range(self.config.nb_epochs)):

            # training loop
            loss = 0
            acc_list = []
            for idx, (image, label) in enumerate(tqdm(self.train_loader)):

                batch_size = image.shape[0]
                image = image.to(self.config.device)
                image = image.view([batch_size, -1])
                label = self._label_embedding(label).to(self.config.device).view([batch_size, -1])
                
                self.init_layers(batch_size)

                loss_per_batch = self.run_train(image, label, epoch)
                loss +=  loss_per_batch

                acc = self.get_acc(label)
                acc_list.append(acc)
            
            # test loop
            test_acc = self.test()

            print("Epoch {:}: loss {:}, test accuracy {:}".format(epoch, loss/idx, test_acc))

    def test(self):
        # test loop
        val_list = []
        for idx, (image, label) in enumerate(tqdm(self.test_loader)):
            batch_size = image.shape[0]
            image = image.to(self.config.device)
            image = image.view([batch_size, -1])
            label = self._label_embedding(label).to(self.config.device).view([batch_size, -1])
            
            self.init_layers(batch_size)

            _ = self.run_test(image)

            val = self.get_acc(label)

            val_list.append(val)
        
        return np.mean(val_list)

if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser()

    # training_parameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument("--nb_epochs", default=2, type=int)
    parser.add_argument("--nb_test_classes", default=2, type=int)

    # train/test
    parser.add_argument('--test', default=False, action='store_true')

    # save/load model
    parser.add_argument('--model_save_dir', default='./save.pickle')
    parser.add_argument('--data_dir', default='./data')

    # sys params
    parser.add_argument('--gpu', default=False, action='store_true')
    config = parser.parse_args()

if not config.test:
    model = SupervisedSNN(config)
    model.create_network()
    model.train()
    # save model
    torch.save(vars(model), config.model_save_dir)

else:
    load_dict = torch.load(config.model_save_dir)
    model = SupervisedSNN(config)
    model.__dict__.update(load_dict)
    test_acc = model.test()
    print("Test accuracy: {:}".format(test_acc))

