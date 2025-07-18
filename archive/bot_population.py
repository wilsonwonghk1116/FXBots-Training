"""
bot_population.py
Evolutionary population management for distributed trading bot training.
"""
import random
from typing import List
import ray
from synthetic_env import SyntheticForexEnv
from trading_bot import TradingBot
import numpy as np
import torch

@ray.remote
class EvaluationActor:
    """A remote actor for evaluating a single bot in its own environment."""
    def __init__(self, data_ref):
        # The data is retrieved from the local object store, not re-read from disk.
        data = ray.get(data_ref)
        self.env = SyntheticForexEnv(data=data)

    def evaluate_bot(self, bot_weights):
        """Creates a bot from weights, runs a simulation, and returns the fitness score."""
        # Create a temporary bot and load the weights
        bot = TradingBot(strategy_type="eval", device='cpu') # Evaluation can happen on CPU
        bot.load_state_dict(bot_weights)
        
        state = self.env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = bot.decide_action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            
        return total_reward

def create_population(size: int, strategy_types: List[str], device: str) -> List[TradingBot]:
    """Create a population of bots with diverse strategies."""
    population = []
    for i in range(size):
        strategy = strategy_types[i % len(strategy_types)]
        bot = TradingBot(strategy_type=strategy, device=device)
        population.append(bot)
    return population

def evaluate_fitness(actor_pool, population):
    """Evaluate fitness in parallel using a pool of remote actors."""
    bot_weights = [bot.state_dict() for bot in population]
    
    # Distribute the evaluation tasks to the actor pool
    futures = [actor.evaluate_bot.remote(weight) for actor, weight in zip(actor_pool, bot_weights)]
    
    # This assumes len(population) <= len(actor_pool). A more robust implementation
    # would use a queue or ray.util.ActorPool for larger populations.
    # For now, this is a direct parallelization.
    
    fitness_scores = ray.get(futures)
    return fitness_scores

def select_elite(population, fitness_scores, num_elite=5):
    """Selects top-performing bots (elitism) and returns their state_dicts for preservation."""
    elite_indices = np.argsort(fitness_scores)[-num_elite:]
    elite_weights = [population[i].state_dict() for i in elite_indices]
    return elite_weights

def tournament_selection(population, fitness_scores, num_to_select, tournament_size=7):
    """Selects individuals for the next generation's gene pool."""
    selected_indices = []
    population_with_scores = list(zip(range(len(population)), fitness_scores))
    
    for _ in range(num_to_select):
        competitors = random.sample(population_with_scores, tournament_size)
        winner_index = max(competitors, key=lambda x: x[1])[0]
        selected_indices.append(winner_index)
    return [population[i] for i in selected_indices]

def mutate_population(population_to_mutate, mutation_rate=0.1):
    """Mutate the population in-place."""
    for bot in population_to_mutate:
        bot.mutate(mutation_rate=mutation_rate)

def crossover_population(mating_pool, offspring_population):
    """
    Performs in-place crossover.
    Overwrites the state of bots in `offspring_population` with new crossed-over weights.
    """
    random.shuffle(mating_pool)
    
    for i in range(0, len(mating_pool), 2):
        if i+1 >= len(mating_pool) or i >= len(offspring_population):
            break 
            
        parent1 = mating_pool[i]
        parent2 = mating_pool[i+1]
        
        # Perform crossover and directly apply to the offspring bot's parameters
        child_bot_target = offspring_population[i]
        
        with torch.no_grad():
            for self_param, other_param, child_param in zip(parent1.parameters(), parent2.parameters(), child_bot_target.parameters()):
                mask = torch.rand_like(self_param) > 0.5
                child_param.data.copy_(torch.where(mask, self_param.data, other_param.data))
                
        # Handle the second child if there's space
        if i + 1 < len(offspring_population):
             child_bot_target_2 = offspring_population[i+1]
             with torch.no_grad():
                for self_param, other_param, child_param in zip(parent1.parameters(), parent2.parameters(), child_bot_target_2.parameters()):
                    mask = torch.rand_like(self_param) > 0.5
                    child_param.data.copy_(torch.where(mask, other_param.data, self_param.data)) # Swap for diversity 