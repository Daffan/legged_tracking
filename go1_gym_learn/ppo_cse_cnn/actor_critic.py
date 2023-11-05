import torch
import torch.nn as nn
import numpy as np
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    # actor_hidden_dims = [256, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    # critic_hidden_dims = [256, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False
    use_cnn = False
    use_gru = False
    height_map_shape = (2, 61, 31)
    cnn_num_embedding = 256
    lstm_num_embedding = 256
    
    
class HeightMapEncoder(nn.Module):
    def __init__(self, height_map_shape, num_embedding=128, use_cnn=False, activation="elu"):
        super().__init__()
        self.height_map_shape = height_map_shape
        self.use_cnn = use_cnn
        self.num_embedding = num_embedding
        
        activation = get_activation(activation)
        
        if use_cnn:
            self.model = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(3360, num_embedding),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(np.prod(height_map_shape), 256),
                activation,
                nn.Linear(256, num_embedding),
                activation
            )
            
    def forward(self, x):
        batch_size = x.shape[0]
        if self.use_cnn:
            x = x.reshape(batch_size, *self.height_map_shape)
        else:
            x = x.reshape(batch_size, -1)
            
        return self.model(x)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 ac_args=AC_Args,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = ac_args.use_decoder
        super().__init__()

        self.ac_args = ac_args
        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        self.height_map_shape = ac_args.height_map_shape
        self.cnn_num_embedding = ac_args.cnn_num_embedding
        self.lstm_input_dim = int(num_obs - np.prod(self.height_map_shape) + self.cnn_num_embedding)
        self.lstm_num_embedding = ac_args.lstm_num_embedding if ac_args.use_gru else self.lstm_input_dim
        self.policy_input_dim = int(num_obs - np.prod(self.height_map_shape) + self.lstm_num_embedding)

        activation = get_activation(ac_args.activation)

        # height_map is in shape (2, 11, 10)
        self.height_map_encoder = HeightMapEncoder(
            self.height_map_shape,
            self.cnn_num_embedding,
            use_cnn=ac_args.use_cnn,
            activation=ac_args.activation
        )

        # two layers of LSTM
        if ac_args.use_gru:
            self.recurrent_latent_embedding = nn.GRU(self.lstm_input_dim, self.lstm_num_embedding, num_layers=1)
        else:  # use identical NN here
            self.recurrent_latent_embedding = nn.Identity()

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.policy_input_dim, ac_args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(ac_args.adaptation_module_branch_hidden_dims)):
            if l == len(ac_args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(ac_args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(ac_args.adaptation_module_branch_hidden_dims[l],
                              ac_args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.policy_input_dim, ac_args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(ac_args.actor_hidden_dims)):
            if l == len(ac_args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(ac_args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(ac_args.actor_hidden_dims[l], ac_args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.policy_input_dim, ac_args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(ac_args.critic_hidden_dims)):
            if l == len(ac_args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(ac_args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(ac_args.critic_hidden_dims[l], ac_args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(ac_args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def process_obs_history(self, observation_history):
        # observation_history in shape (batch_size, num_obs_history, num_obs)
        batch_size = observation_history.shape[0]
        observation_history = observation_history.view(batch_size, -1, self.num_obs)
        height_map = observation_history[:, :, -np.prod(self.height_map_shape):].reshape(-1, *self.height_map_shape)
        height_map_embedding = self.height_map_encoder(height_map).reshape(batch_size, -1, self.cnn_num_embedding)
        lstm_input = torch.cat([
            observation_history[:, :, :-np.prod(self.height_map_shape)],
            height_map_embedding
        ], dim=-1)
        if self.ac_args.use_gru:
            recurrent_latent, _ = self.recurrent_latent_embedding(lstm_input)
        else:
            recurrent_latent = self.recurrent_latent_embedding(lstm_input)
        policy_input = torch.cat([
            observation_history[:, -1, :-np.prod(self.height_map_shape)],
            recurrent_latent[:, -1, :]
        ], dim=-1)
        return policy_input

    def update_distribution(self, observation_history):
        recurrent_latent = self.process_obs_history(observation_history)
        latent = self.adaptation_module(recurrent_latent)
        mean = self.actor_body(torch.cat((recurrent_latent, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        recurrent_latent = self.process_obs_history(observation_history)
        latent = self.adaptation_module(recurrent_latent)
        actions_mean = self.actor_body(torch.cat((recurrent_latent, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        recurrent_latent = self.process_obs_history(observation_history)
        actions_mean = self.actor_body(torch.cat((recurrent_latent, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        recurrent_latent = self.process_obs_history(observation_history)
        value = self.critic_body(torch.cat((recurrent_latent, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        recurrent_latent = self.process_obs_history(observation_history)
        return self.adaptation_module(recurrent_latent)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
