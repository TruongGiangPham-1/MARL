
from MAPPO import MAPPO


# inherit from MAPPO
class CMAPPO(MAPPO):
    def __init__(self, env, optimzer, policy, buffer,
                 single_agent_obs, single_agent_action,
                 collect_steps=128,
                 num_agents=4,
                 save_path=None, log_dir=None, log=False):
        super().__init__(env, optimzer, policy, buffer,
                         single_agent_obs, single_agent_action,
                         collect_steps=collect_steps,
                         num_agents=num_agents,
                         save_path=save_path, log_dir=log_dir, log=log)
        
    
    def compute_value_loss(self, target, new_values):
        """
        
        Args:
            target (torch.Tensor): Mini-batch value target. size (mini_batch_size, num_agents)
            new_values (torch.Tensor): New values from the critic.    # size (mini_batch_size, 1)
        """
        centralized_adv = target.mean(dim=0, keepdim=True)  # dim (mini_batch_size, 1)

        value_loss = 0.5 * ((new_values - centralized_adv)**2).mean()
        return value_loss


    