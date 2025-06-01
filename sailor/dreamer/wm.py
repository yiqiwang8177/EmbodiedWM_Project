import numpy as np
import torch
import torch.optim
from einops import rearrange
from termcolor import cprint
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdm import trange

import sailor.dreamer.networks as networks
import sailor.dreamer.tools as tools


def to_np(x):
    return x.detach().cpu().numpy()


def gradient_penalty(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    f: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Calculates the gradient penalty for the given learner and expert state-action tensors.

    Args:
        learner_sa (torch.Tensor): The state-action tensor from the learner.
        expert_sa (torch.Tensor): The state-action tensor from the expert.
        f (nn.Module): The discriminator network.
        device (str, optional): The device to use. Defaults to "cuda".

    Returns:
        torch.Tensor: The gradient penalty.
    """
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data

    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated.float()).mode().to(device)

    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(f_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    gradients = gradients.view(batch_size, -1)

    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()


class WorldModel(nn.Module):
    def __init__(self, obs_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self.always_frozen_layers = []

        # Initialize the RSSM + Encoder + Decoder
        self.encoder = networks.MultiEncoder(
            shapes, **config.encoder, state_only=config.state_only
        )
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            pred_horizon=config.pred_horizon,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder, state_only=config.state_only
        )

        # Initialize the continue predictor
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # Initialize the Discriminator and its optimizer
        net_size = (
            feat_size
            if config.train_dp_mppi_params["discrim_state_only"]
            else feat_size + config.num_actions
        )
        self.reward_discrim = networks.MLP(
            net_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward_Discrim",
        )
        self.discrim_opt = tools.Optimizer(
            "discrim",
            self.reward_discrim.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer discrim_opt has {sum(param.numel() for param in self.reward_discrim.parameters())} variables."
        )

        # Scales for losses, others are scaled by 1.0.
        self._scales = dict(
            cont=config.cont_head["loss_scale"],
        )

    def parameters(self):
        # Return parameters of self.encoder, self.dynamics, self.heads
        return (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.heads.parameters())
        )

    def _get_post(
        self,
        data,
    ):
        # Get the post from the world model, dont use during training
        data = self.preprocess(data)

        with torch.no_grad():
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
            post = {k: v.detach() for k, v in post.items()}

        return post

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self, always_frozen_layers=self.always_frozen_layers):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}

        # Discriminator Update
        self._step += 1
        if (
            self._config.train_dp_mppi_params["use_discrim"]
            and self._step % self._config.train_dp_mppi_params["upate_discrim_every"]
            == 0
        ):
            # Get expert and learner data ids
            expert_data_ids = torch.where(torch.all(data["reward"] == 1, dim=1))[0]
            learner_data_ids = torch.where(torch.all(data["reward"] == -1, dim=1))[0]

            # Assert union of expert and learner data ids is equal to the total number of data points
            assert (
                torch.cat([expert_data_ids, learner_data_ids]).shape[0]
                == data["reward"].shape[0]
            )
            feat = self.dynamics.get_feat(post)

            # Get learner and expert state-action pairs
            learner_s = feat[learner_data_ids]
            expert_s = feat[expert_data_ids]

            if self._config.train_dp_mppi_params["discrim_state_only"]:
                learner_sa = learner_s
                expert_sa = expert_s
            else:
                learner_a = data["action"][learner_data_ids]
                learner_sa = torch.cat(
                    [learner_s, learner_a], dim=-1
                )  # BS/2 x BL x feat_dim + act_dim

                expert_a = data["action"][expert_data_ids]
                expert_sa = torch.cat(
                    [expert_s, expert_a], dim=-1
                )  # BS/2 x BL x feat_dim + act_dim

            # Merge BS and BL dimensions
            learner_sa = learner_sa.view(-1, learner_sa.shape[-1])
            expert_sa = expert_sa.view(-1, expert_sa.shape[-1])

            # Calculate discriminator loss
            f_learner = self.reward_discrim(learner_sa.float())
            f_expert = self.reward_discrim(expert_sa.float())
            gp = gradient_penalty(
                learner_sa, expert_sa, self.reward_discrim, device=self._config.device
            )  # Scalar
            pure_loss = torch.mean(f_learner.mode() - f_expert.mode())  # Scalar
            loss = pure_loss + 10 * gp  # Scalar
            metrics["discrim_gp"] = gp.item()
            metrics["discrim_pure_loss"] = pure_loss.item()
            metrics.update(self.discrim_opt(loss, self.reward_discrim.parameters()))

        return post, context, metrics

    def get_reward(self, data):
        if self._config.train_dp_mppi_params["use_discrim"]:
            return self.reward_discrim(data)
        return self.heads["reward"](data)

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()

        # Remove stacking dimension (select last state, and first action)
        if len(obs["state"].shape) == 4:
            if "agentview_image" in obs.keys():
                obs["agentview_image"] = obs["agentview_image"][..., -1]
            if "robot0_eye_in_hand_image" in obs.keys():
                obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][
                    ..., -1
                ]
            obs["state"] = obs["state"][..., -1]
        if "action" in obs.keys() and len(obs["action"].shape) == 4:
            obs["action"] = obs["action"][..., 0]

        for key in obs.keys():
            # If key contains 'image', normalize the image
            if "image" in key:
                obs[key] = torch.Tensor(obs[key]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs
