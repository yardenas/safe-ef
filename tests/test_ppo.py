# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO tests."""

from absl.testing import absltest, parameterized
from brax import envs

from ef14.algorithms.ppo import train as ppo


class PPOTest(parameterized.TestCase):
    """Tests for PPO module."""

    @parameterized.parameters(True, False)
    def testTrain(self, use_dict_obs):
        """Test PPO with a simple env."""
        fast = envs.get_environment("fast", use_dict_obs=use_dict_obs)
        _, _, metrics = ppo.train(
            fast,
            num_timesteps=2**15,
            episode_length=128,
            num_envs=64,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            discounting=0.95,
            unroll_length=5,
            batch_size=64,
            num_minibatches=8,
            num_updates_per_batch=4,
            normalize_observations=True,
            seed=2,
            num_evals=3,
            reward_scaling=10,
            normalize_advantage=False,
        )
        self.assertGreater(metrics["eval/episode_reward"], 135)
        self.assertEqual(fast.reset_count, 2)  # type: ignore
        self.assertEqual(fast.step_count, 2)  # type: ignore


if __name__ == "__main__":
    absltest.main()
