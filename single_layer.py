

class SingleLayerDSSM():
    def __init__(self, config):
        self.input_dims = config.input_dims
        self.output_dims = config.output_dims

        self.layer_ind = config.layer_ind
        
        self.network_config = config.network_config
        
        W = torch.abs(torch.randn(self.output_dims, self.input_dims, device = self.network_config.device))
        self.W = self._matrix_normalize(W)
        L = torch.abs(torch.randn(self.output_dims, self.output_dims, device = self.network_config.device))
        self.L = self._matrix_normalize(L.T @ L)

        self.u = None
        self.r = None
        self.act_fn = torch.nn.Tanh()

    """ helper func """
    def _matrix_normalize(self, M):
        W = M / torch.sqrt(torch.sum(M**2, 1))
        return W

    """ layer func """
    def initialize(self, batch_size):
        # activations
		self.u = torch.randn(self.network_config.batch_size, self.output_dims, device = self.network_config.device)
		self.r = self.activation(self.u)

    def run_dynamics(self, prev_layer, feedback, step):
        dt = self.network_config.euler_lr
        r_save = self.r.clone()

        du = - self.u + self.W @ prev_layer - (self.L - torch.eye(self.output_dims, device = self.network_config.device)) @ self.r + feedback

        self.u += lr * du
        self.r = self.activation(self.u)

	def activation(self, u):
		r = torch.max(torch.min(u, torch.ones_like(u, device = self.network_config.device)), 
				 torch.zeros_like(u, device = self.network_config.device))

        # r = torch.max(u, torch.zeros_like(u, device = self.network_config.device))
		# r = self.act_fn(u)
		return r