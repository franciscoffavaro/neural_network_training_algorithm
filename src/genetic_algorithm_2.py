"""Genetic algorithm with roulette wheel selection."""
import numpy as np
from .neural_network import NeuralNetwork


class GeneticAlgorithm2:
    """Roulette wheel selection genetic algorithm.
    
    Uses fitness-proportional selection where individuals are chosen
    with probability proportional to their fitness scores.
    
    Args:
        population_size: Number of individuals (default: 50)
        generations: Generations to evolve (default: 100)
        mutation_rate: Gene mutation probability (default: 0.15)
        crossover_rate: Crossover probability (default: 0.9)
        elite_size: Best individuals preserved (default: 3)
        display_interval: Progress update interval (default: 20)
    """
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.15,
                 crossover_rate=0.9, elite_size=3, display_interval=20):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.display_interval = display_interval
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def _initialize_population(self, weights_count):
        """Create initial population with random weights."""
        return [np.random.randn(weights_count) * 0.5 
                for _ in range(self.population_size)]
    
    def _roulette_selection(self, population, fitness_scores):
        """Select individual via fitness-proportional roulette wheel."""
        adjusted = np.array(fitness_scores) - min(fitness_scores) + 1e-6
        probs = adjusted / adjusted.sum()
        idx = np.random.choice(len(population), p=probs)
        return population[idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Two-point crossover between parents."""
        if np.random.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        pts = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
        child1 = np.concatenate([parent1[:pts[0]], parent2[pts[0]:pts[1]], 
                                parent1[pts[1]:]])
        child2 = np.concatenate([parent2[:pts[0]], parent1[pts[0]:pts[1]], 
                                parent2[pts[1]:]])
        return child1, child2
    
    def _mutate(self, individual):
        """Apply adaptive mutation proportional to gene magnitude."""
        mask = np.random.rand(len(individual)) < self.mutation_rate
        noise = np.random.randn(mask.sum()) * 0.2 * (1 + np.abs(individual[mask]) * 0.1)
        individual[mask] += noise
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
        # evolve population using roulette wheel selection
        weights_count = nn_template.get_weights_count()
        population = self._initialize_population(weights_count)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Genetic Algorithm 2 - Roulette Wheel Selection")
            print(f"{'='*60}")
            print(f"Population: {self.population_size} | Generations: {self.generations}")
            print(f"Mutation: {self.mutation_rate} | Crossover: {self.crossover_rate}")
            print(f"Elite: {self.elite_size}")
        
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
                p1 = self._roulette_selection(population, fitness_scores)
                p2 = self._roulette_selection(population, fitness_scores)
                
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                new_population.extend([c1, c2])
            
            population = new_population[:self.population_size]
        
        final_fitness = self._evaluate_population(population, nn_template, 
                                                  X_train, y_train)
        return population[np.argmax(final_fitness)]
