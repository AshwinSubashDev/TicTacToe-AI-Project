from tictactoe_game import RandomAgent
from tictactoe_game import RuleBasedAgent
from minimax_agent import MinimaxAgent
from alphabeta_agent import AlphaBetaAgent
from testing_framework import GameEvaluator

# Create all agents
agents = {
    'random': RandomAgent("Random"),
    'rule_based': RuleBasedAgent("Rule-Based"),
    'minimax': MinimaxAgent("Minimax"),
    'alphabeta': AlphaBetaAgent("Alpha-Beta")
}

# Run tournament
evaluator = GameEvaluator()
results_df = evaluator.run_tournament(
    agents,
    games_per_matchup = 100,
    track_time = True          # True for Minimax vs Alpha-Beta comparison
)

# Exports
evaluator.export_results(results_df, "tournament_results.csv")
evaluator.plot_tournament_results(results_df, save_path="results.png")
evaluator.plot_performance_comparison(save_path="performance.png")

print("\n Tournament complete! Check the generated files:")
print("  - tournament_results.csv")
print("  - results.png")
print("  - performance.png")