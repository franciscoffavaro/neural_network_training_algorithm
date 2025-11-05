"""Genetic algorithm with tournament selection."""
import numpy as np
from .neural_network import NeuralNetwork

# TODO: maybe try adaptive mutation rate later

class GeneticAlgorithm1:
    """Tournament selection genetic algorithm.
    
    This implementation uses tournament selection where individuals compete
    in small random groups, with the fittest advancing to the next generation.
    
    Args:
        population_size: Number of individuals in population (default: 50)
        generations: Number of generations to evolve (default: 100)
        mutation_rate: Probability of gene mutation (default: 0.1)
        crossover_rate: Probability of crossover (default: 0.8)
        tournament_size: Number of competitors per tournament (default: 3)
        elite_size: Number of best individuals preserved (default: 2)
        display_interval: Generations between progress updates (default: 20)
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1,
                 crossover_rate=0.8, tournament_size=3, elite_size=2, 
                 display_interval=20):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.display_interval = display_interval
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def _initialize_population(self, weights_count):
        """Create initial population with random weights."""
        return [np.random.randn(weights_count) * 0.5 
                for _ in range(self.population_size)]
    
    def _tournament_selection(self, population, fitness_scores):
        """Select individual via tournament competition."""
        competitors = np.random.choice(len(population), self.tournament_size, 
                                      replace=False)
        best = max(competitors, key=lambda i: fitness_scores[i])
        return population[best].copy()
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover between two parents."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    
    def _mutate(self, individual):
        """Apply Gaussian mutation to individual."""
        mask = np.random.rand(len(individual)) < self.mutation_rate
        individual[mask] += np.random.randn(mask.sum()) * 0.3
        return individual
    
    def _evaluate_population(self, population, nn_template, X, y):
        """Calculate fitness for all individuals."""
        fitness_scores = []
        for weights in population:
            nn = NeuralNetwork(nn_template.input_size, 
                             nn_template.hidden_size, 
                             nn_template.output_size)
            nn.set_weights_from_array(weights)
            fitness_scores.append(nn.calculate_fitness(X, y))
        return fitness_scores
    
    def evolve(self, nn_template, X_train, y_train, verbose=True):
        # main evolution loop
        weights_count = nn_template.get_weights_count()
        population = self._initialize_population(weights_count)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Genetic Algorithm 1 - Tournament Selection")
            print(f"{'='*60}")
            print(f"Population: {self.population_size} | Generations: {self.generations}")
            print(f"Mutation: {self.mutation_rate} | Crossover: {self.crossover_rate}")
            print(f"Tournament size: {self.tournament_size} | Elite: {self.elite_size}")
        
        for gen in range(self.generations):
            fitness_scores = self._evaluate_population(population, nn_template, 
                                                      X_train, y_train)
            
            best_fit = max(fitness_scores)
            avg_fit = np.mean(fitness_scores)
            self.best_fitness_history.append(best_fit)
            self.avg_fitness_history.append(avg_fit)
            
            if verbose and self.display_interval > 0:
                if gen % self.display_interval == 0 or gen == self.generations - 1:
                    print(f"Gen {gen:3d} | Best: {best_fit:.6f} | Avg: {avg_fit:.6f}")
            
            # Elitism
            elite_idx = np.argsort(fitness_scores)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_idx]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                p1 = self._tournament_selection(population, fitness_scores)
                p2 = self._tournament_selection(population, fitness_scores)
                
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population.extend([c1, c2])
            
            population = new_population[:self.population_size]
        
        final_fitness = self._evaluate_population(population, nn_template, 
                                                  X_train, y_train)
        return population[np.argmax(final_fitness)]
