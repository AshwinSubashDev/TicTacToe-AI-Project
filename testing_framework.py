import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
from collections import defaultdict

from tictactoe_game import TicTacToe, Player, GameResult, RandomAgent

class PerformanceMetrics:
    """ Track performance metrics for an agent including computation time """
    def __init__(self, agent_name:str):
        self.agent_name = agent_name
        self.total_time = 0.0
        self.move_count = 0
        self.times_per_move = []

    def record_move_time(self, time_taken: float):
        """ Record the time taken for a single move """
        self.total_time += time_taken
        self.move_count += 1
        self.times_per_move.append(time_taken)

    def get_average_time(self) -> float:
        """ Get average time per move """
        if self.move_count == 0:
            return 0.0
        return self.total_time / self.move_count
    
    def get_stats(self) -> dict:
        """ Returns the performance metrics as a dictionary """
        return{
            'agent' : self.agent_name,
            'total_time' : self.total_time,
            'move_count' : self.move_count,
            'avg_time_per_move' : self.get_average_time(),
            'min_time' : min(self.times_per_move) if self.times_per_move else 0.0,
            'max_time' : max(self.times_per_move) if self.times_per_move else 0.0
        }
    
class GameEvaluator:
    """
    Comprehensive evaluation framework for Tic-Tac-Toe agents.
    Features:
    - Head-to-head matchups with detailed statistics
    - Performance timing for computational efficiency comparison (mainly for Minmax and Alpha-beta comparisons)
    - Tournament style evaluation
    - CSV export and visualization of results
    """

    def __init__(self):
        self.match_results = [] # List to store results of each match
        self.performance_data = {} # Dict to store performance metrics per agent
    
    def play_single_game(self, agent1, agent2, track_time: bool = True, verbose: bool = False) -> Tuple[GameResult, Dict]:
        """
        Play a single game between two agents
        Args:
            agent1: First agent (Player X)
            agent2: Second agent (Player O)
            track_time: Whether to track computation time for each agent
            verbose: Whether to print the game board after each move
        Returns:
            Tuple of (GameResult, performance metrics dictionary)
        """
        game = TicTacToe()

        # Initialize performance metrics
        perf_metrics = {
            agent1.name: PerformanceMetrics(agent1.name),
            agent2.name: PerformanceMetrics(agent2.name)
        }

        if verbose:
            print(f"\n=== Game Start: {agent1.name} (X) vs {agent2.name} (O) ===")
            print(game.render())
            print()
        
        while game.check_winner() == GameResult.IN_PROGRESS:
            # Select current agent
            current_agent = agent1 if game.current_player == Player.X else agent2

            # Get action with optional timing
            if track_time:
                start_time = time.perf_counter()
                action = current_agent.select_action(game)
                elapsed = time.perf_counter() - start_time
                perf_metrics[current_agent.name].record_move_time(elapsed)
            else:
                action = current_agent.select_action(game)
            
            if verbose:
                print(f"{current_agent.name} ({game.current_player.name}) plays at {action}")

            # Make move
            state, result = game.make_move(action)

            if verbose and result == GameResult.IN_PROGRESS:
                print(game.render())
                print()
        
        result = game.check_winner()

        if verbose:
            print(game.render())
            print(f"\nGame Over! Result: {result.name}\n")
        
        return result, perf_metrics
    
    def play_match(self, agent1, agent2, num_games: int = 100, track_time: bool = True, verbose: bool = False) -> Dict:
        """
        Play multiple games between two agents
        Args:
            agent1: First agent (Player X)
            agent2: Second agent (Player O)
            num_games: Number of games to play
            track_time: Whether to track computation time for each agent
            verbose: Whether to print game details
        Returns:
            Dictionary with match statistics and performance metrics
        """
        results = {
            'agent1_name': agent1.name,
            'agent2_name': agent2.name,
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_wins_as_X': 0,
            'agent1_wins_as_O': 0,
            'agent2_wins_as_O': 0,
            'agent2_wins_as_X': 0,
            'num_games': num_games
        }
        
        # Aggregate performance metrics
        total_perf = {
            agent1.name: PerformanceMetrics(agent1.name),
            agent2.name: PerformanceMetrics(agent2.name)
        }

        for game_num in range(num_games):
            # Alternate starting players
            if game_num % 2 == 0:
                agent_X, agent_O = agent1, agent2
                agent1_is_X = True
            else:
                agent_X, agent_O = agent2, agent1
                agent1_is_X = False

            # Play game
            result, perf_metrics = self.play_single_game(agent_X, agent_O, track_time = track_time, verbose = False)

            # Update results
            if result == GameResult.DRAW:
                results['draws'] += 1
            elif result == GameResult.X_WIN:
                if agent1_is_X:
                    results['agent1_wins'] += 1
                    results['agent1_wins_as_X'] += 1
                else:
                    results['agent2_wins'] += 1
                    results['agent2_wins_as_X'] += 1
            elif result == GameResult.O_WIN:
                if agent1_is_X:
                    results['agent2_wins'] += 1
                    results['agent2_wins_as_O'] += 1
                else:
                    results['agent1_wins'] += 1
                    results['agent1_wins_as_O'] += 1
            
            # Aggregate performance metrics
            if track_time:
                for agent_name, metrics in perf_metrics.items():
                    total_perf[agent_name].total_time += metrics.total_time
                    total_perf[agent_name].move_count += metrics.move_count
                    total_perf[agent_name].times_per_move.extend(metrics.times_per_move)
            
        results['agent1_win_rate'] = results['agent1_wins'] / num_games
        results['agent2_win_rate'] = results['agent2_wins'] / num_games
        results['draw_rate'] = results['draws'] / num_games

        # Add performance metrics if tracked
        if track_time:
            results['performance'] = {
                agent1.name: total_perf[agent1.name].get_stats(),
                agent2.name: total_perf[agent2.name].get_stats()
            }
        self.match_results.append(results)
        return results
    
    def print_match_results(self, results: Dict):
        """ Print formatted match results with optional performance metrics """
        print(f"\n{'='*70}")
        print(f"Match Results: {results['agent1_name']} vs {results['agent2_name']}")
        print(f"{'='*70}")
        print(f"Total Games: {results['num_games']}")
        
        print(f"\n{results['agent1_name']}:")
        print(f"  Total Wins: {results['agent1_wins']} ({results['agent1_win_rate']:.1%})")
        print(f"  Wins as X: {results['agent1_wins_as_X']}")
        print(f"  Wins as O: {results['agent1_wins_as_O']}")
        
        print(f"\n{results['agent2_name']}:")
        print(f"  Total Wins: {results['agent2_wins']} ({results['agent2_win_rate']:.1%})")
        print(f"  Wins as X: {results['agent2_wins_as_X']}")
        print(f"  Wins as O: {results['agent2_wins_as_O']}")
        
        print(f"\nDraws: {results['draws']} ({results['draw_rate']:.1%})")
        
        # Print performance metrics if available
        if 'performance' in results:
            print(f"\n{'='*70}")
            print("Performance Metrics (Computation Time)")
            print(f"{'='*70}")
            for agent_name, perf in results['performance'].items():
                print(f"\n{agent_name}:")
                print(f"  Total Time: {perf['total_time']:.4f} seconds")
                print(f"  Total Moves: {perf['move_count']}")
                print(f"  Avg Time/Move: {perf['avg_time_per_move']*1000:.4f} ms")
                print(f"  Min Time: {perf['min_time']*1000:.4f} ms")
                print(f"  Max Time: {perf['max_time']*1000:.4f} ms")
        
        print(f"{'='*70}\n")

    def run_tournament(self, agents_dict: Dict, games_per_matchup: int = 100, track_time: bool = True) -> pd.DataFrame:
        """
        Run the complete tournament among all agents provided
        Args:
            agents_dict: Dictionary with keys 'random', 'rule-based', 'minimax', 'alpha-beta'
            games_per_matchup: Number of games per agent matchup (default: 100)
            track_time: Whether to track computation time (mainly for Minmax and Alpha-beta comparisons)
        Returns:
            DataFrame with all tournament results
        """
        print(f"\n{'='*70}")
        print(f"TIC-TAC-TOE AGENT TOURNAMENT")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"{'='*70}\n")

        tournament_results = []

        # Define matchups
        matchups = [
            ('random', 'random', 'Establish baseline win/draw ratio'),
            ('random', 'rule_based', 'Shows impact of heuristics'),
            ('random', 'minimax', 'Should show Minimax dominance'),
            ('random', 'alphabeta', 'Should match Minimax (same decisions)'),
            ('minimax', 'alphabeta', 'Compare compute time & identical outcomes')
        ]
        
        for idx, (agent1_key, agent2_key, purpose) in enumerate(matchups, 1):
            print(f"\n{'='*70}")
            print(f"MATCHUP {idx}/5: {agents_dict[agent1_key].name} vs {agents_dict[agent2_key].name}")
            print(f"Purpose: {purpose}")
            print(f"{'='*70}")

            # Create fresh instances of agents for each matchup if they're the same type
            if agent1_key == agent2_key:
                # Create two separate instances
                agent1 = type(agents_dict[agent1_key])(f"{agents_dict[agent1_key].name} A")
                agent2 = type(agents_dict[agent2_key])(f"{agents_dict[agent2_key].name} B")
            else:
                agent1 = agents_dict[agent1_key]
                agent2 = agents_dict[agent2_key]

            # Play match
            results = self.play_match(
                agent1, agent2,
                num_games = games_per_matchup,
                track_time = track_time,
                verbose = True
            )

            # Print results
            self.print_match_results(results)

            # Store for summary table
            tournament_results.append({
                'Matchup': f"{results['agent1_name']} vs {results['agent2_name']}",
                'Purpose': purpose,
                'Agent_1': results['agent1_name'],
                'Agent_1_Wins': results['agent1_wins'],
                'Agent_2': results['agent2_name'],
                'Agent_2_Wins': results['agent2_wins'],
                'Draws': results['draws'],
                'Agent_1_Win_Rate': f"{results['agent1_win_rate']:.1%}",
                'Agent_2_Win_Rate': f"{results['agent2_win_rate']:.1%}",
                'Draw_Rate': f"{results['draw_rate']:.1%}"
            })
        
        # Create summary DataFrame
        df = pd.DataFrame(tournament_results)
        return df
    
    def export_results(self, tournament_df: pd.DataFrame, filename: str = "tournament_results.csv"):
        """ Export tournament results to CSV """
        tournament_df.to_csv(filename, index=False)
        print(f"Tournament results exported to {filename}")
    
    def plot_tournament_results(self, tournament_df: pd.DataFrame, save_path: str = None):
        """ 
        Create comprehensive visualizations of tournament results
        Args:
            tournament_df: DataFrame with tournament results
            save_path: Optional path to save the plot image
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        # Extract matchup data
        matchups = tournament_df['Matchup'].tolist()
        agent1_wins = tournament_df['Agent_1_Wins'].tolist()
        agent2_wins = tournament_df['Agent_2_Wins'].tolist()
        draws = tournament_df['Draws'].tolist()

        # Plot 1: Stacked Bar Chart - Overall Results
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(matchups))
        width = 0.6

        p1 = ax1.bar(x, agent1_wins, width, label='Agent 1 Wins', color='#2ecc71', alpha=0.8)
        p2 = ax1.bar(x, draws, width, bottom=agent1_wins, label='Draws', color='#95a5a6', alpha=0.8)
        p3 = ax1.bar(x, agent2_wins, width, bottom=np.array(agent1_wins)+np.array(draws), 
                    label='Agent 2 Wins', color='#e74c3c', alpha=0.8)
        
        ax1.set_ylabel('Number of Games', fontsize=11, fontweight='bold')
        ax1.set_title('Tournament Results - All Matchups', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(matchups, ha='center', rotation = 0)
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2-6: Individual matchup breakdowns
                # Plot 2-6: Individual matchup breakdowns
        num_matchups = len(matchups)
        for idx in range(num_matchups):
            row = idx // 2 + 1
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            categories = ['Agent 1\nWins', 'Draws', 'Agent 2\nWins']
            values = [agent1_wins[idx], draws[idx], agent2_wins[idx]]
            colors = ['#2ecc71', '#95a5a6', '#e74c3c']

            bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ...
            for bar_idx, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                if val > 0:
                    # If bar is tall enough (>= 10), put text inside; otherwise above
                    if val >= 17:  # Big enough to fit text inside
                        y_pos = height / 2
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{int(val)}\n({val/sum(values)*100:.1f}%)',
                            ha='center', va='center', fontweight='bold', fontsize=9, 
                            color='white' if val > 20 else 'black')
                    else:  # Too small, put text above bar
                        y_pos = height + 2
                        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{int(val)}\n({val/sum(values)*100:.1f}%)',
                            ha='center', va='bottom', fontweight='bold', fontsize=8, color='black')
            
            ax.set_ylabel('Games', fontsize=9)
            ax.set_title(f"{matchups[idx]}", fontsize=10, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.25)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Tic-Tac-Toe Agent Tournament - Detailed Results', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

    def plot_performance_comparison(self, save_path: str = None):
        """
        Plot computation time comparison between agents
        Particularly useful for Minimax vs Alpha-Beta comparison
        """
        if not self.match_results:
            print("No match results available for performance comparison")
            return
        
        # Extract performance data
        perf_data = []
        for match in self.match_results:
            if 'performance' in match:
                for agent_name, perf in match['performance'].items():
                    perf_data.append({
                        'Agent': agent_name,
                        'Avg_Time_ms' : perf['avg_time_per_move'] * 1000,
                        'Total_Moves' : perf['move_count']
                    })
        
        if not perf_data:
            print("No performance data tracked. Set track_time = True in matches.")
            return

        df_perf = pd.DataFrame(perf_data)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14,5))

        # Plot 1: Average time per move
        agents = df_perf['Agent'].unique()
        avg_times = [df_perf[df_perf['Agent'] == agent]['Avg_Time_ms'].mean() 
                    for agent in agents]
        
        bars = ax1.bar(agents, avg_times, color='steelblue', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Average Time per Move (ms)', fontweight='bold')
        ax1.set_title('Computational Efficiency Comparison', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, avg_times):
            ax1.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.2f} ms', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Total computation time
        total_times = [df_perf[df_perf['Agent'] == agent]['Avg_Time_ms'].sum() 
                      / 1000 for agent in agents]  # Convert to seconds
        
        bars2 = ax2.bar(agents, total_times, color='coral', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Total Computation Time (seconds)', fontweight='bold')
        ax2.set_title('Total Time Across All Matches', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, total_times):
            ax2.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Performance comparison saved to {save_path}")
        
        plt.show()