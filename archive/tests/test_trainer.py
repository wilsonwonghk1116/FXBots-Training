from src.trainer import VRAMOptimizedTrainer

def test_trainer_init():
    trainer = VRAMOptimizedTrainer(population_size=10)
    assert trainer.population_size == 10
