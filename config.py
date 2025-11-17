"""Configuration management from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration loaded from .env file."""
    
    # General settings
    DATA_FILE_PATH = os.getenv('DATA_FILE_PATH', 'database/dadosParaRede.ods')
    TRAIN_RATIO = float(os.getenv('TRAIN_RATIO', '0.8')) #80% train, 20% test
    RANDOM_SEED = os.getenv('RANDOM_SEED', '42')
    
    if RANDOM_SEED and RANDOM_SEED.lower() != 'none':
        RANDOM_SEED = int(RANDOM_SEED)
    else:
        RANDOM_SEED = None
    
    # Neural network
    NEURAL_HIDDEN_SIZE = int(os.getenv('NEURAL_HIDDEN_SIZE', '10')) #default 10 hidden neurons
    
    # GA1: Tournament selection
    AG1_POPULATION_SIZE = int(os.getenv('AG1_POPULATION_SIZE', '50')) #default 50 individuals
    AG1_GENERATIONS = int(os.getenv('AG1_GENERATIONS', '100')) #default 100 generations
    AG1_MUTATION_RATE = float(os.getenv('AG1_MUTATION_RATE', '0.1')) #default 10%
    AG1_CROSSOVER_RATE = float(os.getenv('AG1_CROSSOVER_RATE', '0.8')) #default 80%
    AG1_TOURNAMENT_SIZE = int(os.getenv('AG1_TOURNAMENT_SIZE', '3')) #default 3 competitors
    AG1_ELITE_SIZE = int(os.getenv('AG1_ELITE_SIZE', '2'))
    
    # GA2: Roulette wheel selection
    AG2_POPULATION_SIZE = int(os.getenv('AG2_POPULATION_SIZE', '50'))
    AG2_GENERATIONS = int(os.getenv('AG2_GENERATIONS', '100'))
    AG2_MUTATION_RATE = float(os.getenv('AG2_MUTATION_RATE', '0.15'))
    AG2_CROSSOVER_RATE = float(os.getenv('AG2_CROSSOVER_RATE', '0.9'))
    AG2_ELITE_SIZE = int(os.getenv('AG2_ELITE_SIZE', '3'))
    
    # Output settings
    VERBOSE = os.getenv('VERBOSE', 'true').lower() == 'true'
    DISPLAY_INTERVAL = int(os.getenv('DISPLAY_INTERVAL', '20'))
    
    # New rows generation for full-sequence predictions
    NEW_ROWS_COUNT = int(os.getenv('NEW_ROWS_COUNT', '0'))  # when > 0, auto-generate this many new rows
    
    @classmethod
    def print_config(cls):
        """Display current configuration."""
        print("\n" + "="*70)
        print("Configuration (from .env)")
        print("="*70)
        
        print(f"\nGeneral:")
        print(f"  Data file: {cls.DATA_FILE_PATH}")
        print(f"  Train ratio: {cls.TRAIN_RATIO:.1%}")
        print(f"  Random seed: {cls.RANDOM_SEED if cls.RANDOM_SEED else 'Random'}")
        
        print(f"\nNeural Network:")
        print(f"  Hidden neurons: {cls.NEURAL_HIDDEN_SIZE}")
        
        print(f"\nGA1 (Tournament):")
        print(f"  Population: {cls.AG1_POPULATION_SIZE} | Generations: {cls.AG1_GENERATIONS}")
        print(f"  Mutation: {cls.AG1_MUTATION_RATE:.1%} | Crossover: {cls.AG1_CROSSOVER_RATE:.1%}")
        print(f"  Tournament size: {cls.AG1_TOURNAMENT_SIZE} | Elite: {cls.AG1_ELITE_SIZE}")
        
        print(f"\nGA2 (Roulette):")
        print(f"  Population: {cls.AG2_POPULATION_SIZE} | Generations: {cls.AG2_GENERATIONS}")
        print(f"  Mutation: {cls.AG2_MUTATION_RATE:.1%} | Crossover: {cls.AG2_CROSSOVER_RATE:.1%}")
        print(f"  Elite: {cls.AG2_ELITE_SIZE}")
        
        print(f"\nOutput:")
        print(f"  Verbose: {cls.VERBOSE} | Display interval: {cls.DISPLAY_INTERVAL}")
        print(f"  New rows to generate (auto): {cls.NEW_ROWS_COUNT}")
        print("="*70 + "\n")
    
    @classmethod
    def validate(cls):
        """Validate configuration values."""
        errors = []
        
        if not (0.0 < cls.TRAIN_RATIO < 1.0):
            errors.append(f"TRAIN_RATIO must be between 0 and 1 (got {cls.TRAIN_RATIO})")
        
        if cls.NEURAL_HIDDEN_SIZE < 1:
            errors.append(f"NEURAL_HIDDEN_SIZE must be >= 1 (got {cls.NEURAL_HIDDEN_SIZE})")
        
        if cls.AG1_POPULATION_SIZE < 10:
            errors.append(f"AG1_POPULATION_SIZE must be >= 10 (got {cls.AG1_POPULATION_SIZE})")
        
        if cls.AG2_POPULATION_SIZE < 10:
            errors.append(f"AG2_POPULATION_SIZE must be >= 10 (got {cls.AG2_POPULATION_SIZE})")
        
        if not (0.0 <= cls.AG1_MUTATION_RATE <= 1.0):
            errors.append(f"AG1_MUTATION_RATE must be in [0,1] (got {cls.AG1_MUTATION_RATE})")
        
        if not (0.0 <= cls.AG2_MUTATION_RATE <= 1.0):
            errors.append(f"AG2_MUTATION_RATE must be in [0,1] (got {cls.AG2_MUTATION_RATE})")
        
        if errors:
            print("Configuration errors:")
            for err in errors:
                print(f"  - {err}")
            raise ValueError("Invalid configuration detected")
        
        return True


config = Config()
