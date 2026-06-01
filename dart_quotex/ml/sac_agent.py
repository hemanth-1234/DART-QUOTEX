"""
dart_quotex/ml/sac_agent.py
Soft Actor-Critic (SAC) agent for trade decision-making.

The SAC agent receives a feature vector (state) and outputs:
  · action[0]  : trade direction  (-1 = PUT, +1 = CALL, ~0 = HOLD)
  · action[1]  : conviction / position size multiplier  (0 – 1)

SAC is chosen for its entropy maximisation objective, which prevents
the agent from collapsing into one action and provides natural uncertainty
quantification — key for risk management.

The agent is trained from the replay buffer (offline + online).
Online updates happen after each trade using the stored transitions.

Network architecture
--------------------
State → LayerNorm → MLP (256-256) → Actor (mean + log_std)
                                   → Twin Critics (Q1, Q2)
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional PyTorch import ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False
    logger.warning("PyTorch not installed — SACAgent running in stub mode")


# ──────────────────────────────────────────────────────────────────────────────
# Neural network modules (only defined when torch is available)
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class _MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class _Actor(nn.Module):
        LOG_STD_MIN = -5
        LOG_STD_MAX = 2

        def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.backbone = _MLP(state_dim, hidden)
            self.mu_head = nn.Linear(hidden, action_dim)
            self.log_std_head = nn.Linear(hidden, action_dim)

        def forward(
            self, state: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.backbone(state)
            mu = self.mu_head(h)
            log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = log_std.exp()
            return mu, std

        def sample(
            self, state: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            mu, std = self(state)
            dist = torch.distributions.Normal(mu, std)
            x = dist.rsample()
            action = torch.tanh(x)
            # Log prob with tanh squashing correction
            log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            return action, log_prob

    class _Critic(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.q1 = nn.Sequential(_MLP(state_dim + action_dim, hidden), nn.Linear(256, 1))
            self.q2 = nn.Sequential(_MLP(state_dim + action_dim, hidden), nn.Linear(256, 1))

        def forward(
            self, state: "torch.Tensor", action: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            sa = torch.cat([state, action], dim=-1)
            return self.q1(sa), self.q2(sa)


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, state_dim: int = 34) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Optional[Tuple]:
        if len(self._buf) < batch_size:
            return None
        idx = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ──────────────────────────────────────────────────────────────────────────────

class SACAgent:
    """
    SAC agent.  Falls back to a noise-augmented heuristic when PyTorch
    is not installed, so the system still functions end-to-end.
    """

    ACTION_DIM = 2   # [direction, conviction]

    def __init__(
        self,
        state_dim: int = 34,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 50_000,
        batch_size: int = 64,
    ) -> None:
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_size, state_dim)
        self._updates = 0

        if _TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._actor = _Actor(state_dim, self.ACTION_DIM).to(self.device)
            self._critic = _Critic(state_dim, self.ACTION_DIM).to(self.device)
            self._critic_target = _Critic(state_dim, self.ACTION_DIM).to(self.device)
            self._critic_target.load_state_dict(self._critic.state_dict())
            self._actor_opt = optim.Adam(self._actor.parameters(), lr=lr)
            self._critic_opt = optim.Adam(self._critic.parameters(), lr=lr)
            # Learnable temperature
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self._alpha_opt = optim.Adam([self._log_alpha], lr=lr)
            self._target_entropy = -self.ACTION_DIM
            logger.info("SACAgent initialised on %s", self.device)
        else:
            self._actor = None
            logger.warning("SACAgent running in stub mode (no PyTorch)")

    # ── public API ────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Given a state vector, return action = [direction(-1..1), conviction(0..1)].
        """
        if not _TORCH or self._actor is None:
            return self._stub_act(state)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self._actor(s)
                action = torch.tanh(mu)
            else:
                action, _ = self._actor.sample(s)
        action = action.cpu().numpy()[0]
        # Squash conviction to [0, 1]
        action[1] = (action[1] + 1) / 2
        return action

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(state, action, reward, next_state, done)

    def update(self, n_steps: int = 1) -> Optional[dict]:
        """Perform `n_steps` gradient updates.  Returns loss dict or None."""
        if not _TORCH or len(self.replay) < self.batch_size:
            return None

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for _ in range(n_steps):
            batch = self.replay.sample(self.batch_size)
            if batch is None:
                break
            stats = self._update_step(*batch)
            total_critic_loss += stats["critic_loss"]
            total_actor_loss += stats["actor_loss"]
            self._updates += 1

        return {
            "critic_loss": total_critic_loss / n_steps,
            "actor_loss": total_actor_loss / n_steps,
            "updates": self._updates,
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if _TORCH and self._actor is not None:
            torch.save(self._actor.state_dict(), path / "actor.pt")
            torch.save(self._critic.state_dict(), path / "critic.pt")
        with open(path / "replay.pkl", "wb") as f:
            pickle.dump(self.replay, f)
        logger.info("SACAgent saved to %s", path)

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not (path / "actor.pt").exists():
            return False
        if _TORCH and self._actor is not None:
            self._actor.load_state_dict(
                torch.load(path / "actor.pt", map_location=self.device)
            )
            self._critic.load_state_dict(
                torch.load(path / "critic.pt", map_location=self.device)
            )
            self._critic_target.load_state_dict(self._critic.state_dict())
        replay_pkl = path / "replay.pkl"
        if replay_pkl.exists():
            with open(replay_pkl, "rb") as f:
                self.replay = pickle.load(f)
        logger.info("SACAgent loaded from %s (replay size=%d)", path, len(self.replay))
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> dict:
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_a, next_log_pi = self._actor.sample(ns)
            q1_t, q2_t = self._critic_target(ns, next_a)
            min_q = torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            y = r + (1 - d) * self.gamma * min_q

        q1, q2 = self._critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self._critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 1.0)
        self._critic_opt.step()

        # Actor
        new_a, log_pi = self._actor.sample(s)
        q1_new, q2_new = self._critic(s, new_a)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()
        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        # Alpha
        alpha_loss = -(self._log_alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_opt.zero_grad()
        alpha_loss.backward()
        self._alpha_opt.step()
        self.alpha = self._log_alpha.exp().item()

        # Soft target update
        for p, tp in zip(self._critic.parameters(), self._critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _stub_act(self, state: np.ndarray) -> np.ndarray:
        """Heuristic fallback when no PyTorch: use RSI-momentum signal."""
        # state[7] = rsi_14_n (normalised 0-1)
        rsi_n = float(state[7]) if len(state) > 7 else 0.5
        direction = 1.0 if rsi_n < 0.3 else (-1.0 if rsi_n > 0.7 else 0.0)
        conviction = abs(rsi_n - 0.5) * 2
        return np.array([direction, conviction], dtype=np.float32)
