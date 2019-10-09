import torch

class SingleLayerDSSM():
    def __init__(self, config):
        self.input_dims = config.input_dims
        self.output_dims = config.output_dims

        self.layer_ind = config.layer_ind
        
        self.network_config = config.network_config
        
        W = torch.abs(torch.randn(self.input_dims, self.output_dims, device = self.network_config.device))
        self.W = self._matrix_normalize(W)
        L = torch.abs(torch.randn(self.output_dims, self.output_dims, device = self.network_config.device))
        self.L = self._matrix_normalize(L.T @ L)

        self.u = None
        self.r = None
        self.act_fn = torch.nn.Tanh()

        self.z = None

    """ helper func """
    def _matrix_normalize(self, M):
        W = M / torch.sqrt(torch.sum(M**2, 1)).view([-1, 1])
        return W
    
    def _is_last_layer(self):
        return self.layer_ind == self.network_config.nb_layers - 1

    """ layer func """
    def set_z(self, z):
        self.z = z

    def initialize(self, batch_size):
        # activations
        self.u = torch.randn(batch_size, self.output_dims, device = self.network_config.device)
        self.r = self.activation(self.u)

    def run_dynamics(self, prev_layer, feedback = None, step = 0):
        dt = self.network_config.euler_lr
        r_save = self.r.clone()

        du = - self.u + prev_layer @ self.W  - self.r @ (self.L - torch.eye(self.output_dims, device = self.network_config.device))
        
        if feedback is not None:
            du += feedback

        self.u += dt * du
        self.r = self.activation(self.u)

        err_all = torch.norm(self.r - r_save, p=2, dim=1)/(1e-10 + torch.norm(r_save, p=2, dim=1))
        err = torch.mean(err_all) / dt 

        # print(self.r, err)

        return err.item()
    
    def update_plasiticity(self, prev_layer, epoch = 0):
        lr = self.network_config.lr
        gamma = self.network_config.gamma

        if gamma > 0:
            update_step = lr * gamma ** (1 + self.layer_ind - self.network_config.nb_layers)
        else:
            update_step = lr

        dW = prev_layer.t() @ self.r - self.W 

        if self.z is None:
            dL = self.r.t() @ self.r - self.L / (1 + gamma)
        else:
            assert(self._is_last_layer())
            dL = self.z.t() @ (- self.inv_activation(self.z) + (prev_layer @ self.W) - 
                        self.z @(self.L - torch.eye(self.output_dims, device = self.network_config.device)) ) 

        self.W += update_step * dW 
        self.L += update_step * dL

    def activation(self, u):
        r = torch.max(torch.min(u, torch.ones_like(u, device = self.network_config.device)), 
				 torch.zeros_like(u, device = self.network_config.device))

        # r = torch.max(u, torch.zeros_like(u, device = self.network_config.device))
		# r = self.act_fn(u)
        return r
    
    def inv_activation(self, r):

        return r
    
    @property
    def output(self):
        return self.r

    @property
    def feedback(self):
        return self.network_config.gamma * self.r @ self.W.t()