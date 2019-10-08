import torch
from single_layer import SingleLayerDSSM

# misc
from misc import AttrDict
import random
from tqdm import tqdm

# set random seed
random_seed = 1
torch.manual_seed(random_seed)

class SupervisedSNN():
    def __init__(self, args):
        config = AttrDict()

        # input
        input_dims = [28, 28, 1]
        input_dims_linear = int(np.prod(input_dims))
        # output
        num_classes = 10
        config.nb_subset_classes = 2
        config.subset_classes = random.sample(range(num_classes), config.nb_subset_classes)
        if config.nb_subset_classes == 2:
            config.output_dims = 1
        else:
            config.output_dims = config.nb_subset_classes

        config.nb_units = [100, 30]
        config.nb_units.insert(0, input_dims_linear)
        config.nb_layers = len(config.nb_units)
        config.nb_units.append(config.output_dims)

        # feedback parameter
        config.gamma = 0.00

        # host and device
        config.host = torch.device("cpu")
        config.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # learning rate and step size
        config.euler_lr = 0.1
        config.lr = 1e-3
        config.tol = 1e-5

        # training params
        config.nb_epochs = 10
        config.batch_size = args.batch_size

        # data dir
        config.data_dir = args.data_dir

        self.layers = {}

        # mnist train data lodaer
		self.train_loader = torch.utils.data.DataLoader(
		  torchvision.datasets.MNIST(self.config.data_dir, train=True, download=True,
		                             transform=torchvision.transforms.Compose([
		                               torchvision.transforms.ToTensor(),
		                               torchvision.transforms.Normalize(
		                                 (0.1307,), (0.3081,))
		                             ])),
		  batch_size=self.config.batch_size, shuffle=True, sampler = Data.sampler.SubsetRandomSampler(config.subset_classes))

		# mnist test data lodaer
		self.test_loader = torch.utils.data.DataLoader(
		  torchvision.datasets.MNIST(self.config.data_dir, train=False, download=True,
		                             transform=torchvision.transforms.Compose([
		                               torchvision.transforms.ToTensor(),
		                               torchvision.transforms.Normalize(
		                                 (0.1307,), (0.3081,))
		                             ])),
		  batch_size=self.config.batch_size, shuffle=True, sampler = Data.sampler.SubsetRandomSampler(config.subset_classes))

    """ helper func """
    def _label_embedding(self, labels):
        if self.config.nb_subset_classes == 2:
            label_emb = (labels == self.config.subset_classes[0]).float()
        else:
            label_emb = torch.nn.functional(labels).index_select(2, torch.tensor(self.config.subset_classes))

        return label_emb 

    """ main func """
    def create_network(self):
        for layer_id in range(self.config.nb_layers):
            layer_config = AttrDict()

            # layer-wise configs
            layer_config.layer_id = layer_id
            layer_config.input_dims = self.config.nb_units[layer_id]
            layer_config.output_dims = self.config.nb_units[layer_id + 1]

            # network-wise configs
            layer_config.network_config = self.config

            self.layers[layer_id] = SingleLayerDSSM(layer_config)

    def init_layers(self, batch_size):
        for layer_id in range(self.config.nb_layers):
            self.layers[layer_id].initialize(batch_size)

    def train(self, inp, out, epoch = 0):
        delta = np.ones(self.config.nb_layers) * np.inf

        self.layers[self.config.nb_layers - 1].set_z(out)

        for dynamic_step in range(3000):
            cur_inp = inp
            for layer_id in range(self.config.nb_layers - 1):

                delta[layer_id] = self.layers[layer_id].run_dynamics(cur_inp, self.layers[layer_id+1].feedback, step = dynamic_step)

                cur_inp = self.layers[layer_id].output
            
            delta[self.config.nb_layers - 1] = self.layers[self.config.nb_layers - 1].run_dynamics(cur_inp, step = dynamic_step)

            if delta.mean() < self.config.tol:
                break
        
        cur_inp = inp
        for layer_id in range(self.config.nb_layers):
            self.layers[layer_id].update_plasiticity(cur_inp, epoch = epoch)
            cur_inp = self.layers[layer_id].output
        
        return delta.sum()

    def run(self):
        for epoch in tqdm(self.config.nb_epochs):
            loss = 0
            for idx, (image, label) in enumerate(self.train_loder):
                batch_size = image.shape[0]
                image = image.to(self.config.device)
                image = image.view([batch_size, -1])

                label = self._label_embedding(label).to(self.config.device)

                self.init_layers(batch_size)

                # TO DO: train

                loss_per_batch = self.train(image, label, epoch)
                loss +=  loss_per_batch

            print("Epoch {:}: loss {:}".format(epoch, loss))


if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser()

	# training_parameters
	parser.add_argument('--batch_size', default=1, type=int)

	# train/test
	parser.add_argument('--train', default=False, action='store_true')

	# save/load model
	parser.add_argument('--model_save_dir', default='./save.pickle')
	parser.add_argument('--data_dir', default='./data')

	config = parser.parse_args()

if config.train:
	model = SupervisedSNN(config)
	model.create_network()
	model.run()
	# save model
	torch.save(vars(model), config.model_save_dir)

else:
    pass
	# load_dict = torch.load(config.model_save_dir)
	# model = DSSM(config)
	# model.__dict__.update(load_dict)
	# model.classify()

