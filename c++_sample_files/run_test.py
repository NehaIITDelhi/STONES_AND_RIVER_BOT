import subprocess
import re
import sys
import time
import math

# --- CONFIGURATION ---
NUM_GAMES = 50
YOUR_AGENT = "student_cpp"
OPPONENT_AGENT = "random"
# ---------------------

def avg(data_list):
    """Helper function to safely calculate an average."""
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)

def run_tournament():
    print(f"Starting tournament: {YOUR_AGENT} (Circle) vs. {OPPONENT_AGENT} (Square)")
    print(f"Running {NUM_GAMES} games... This may take a few minutes.\n")
    
    # List to store detailed results for each game
    # Each item will be: {"game_id": i, "winner": "student_cpp", "duration": 4.52}
    game_results = []
    
    # Regex to find the final score line
    score_regex = re.compile(r"Final Scores -> Circle: (\S+) \| Square: (\S+)")

    total_start_time = time.time()

    for i in range(NUM_GAMES):
        command = [
            "python3", 
            "gameEngine.py",
            "--mode", "aivai",
            "--circle", OPPONENT_AGENT,
            "--square", YOUR_AGENT,
            "--nogui"
        ]
        print(i)
        
        game_start_time = time.time() # Time this specific game
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            
            match = score_regex.search(result.stdout)
            
            if not match:
                print(f"Game {i+1}/{NUM_GAMES}: ERROR! Could not parse score. Skipping.")
                continue

            circle_score = float(match.group(1))
            square_score = float(match.group(2))
            
            winner = "draw"
            if circle_score > square_score:
                winner = YOUR_AGENT  # We are Circle
                sys.stdout.write('W')
            elif square_score > circle_score:
                winner = OPPONENT_AGENT # We are Circle
                sys.stdout.write('L')
            else:
                sys.stdout.write('D')
            
            sys.stdout.flush()
            
            # Store the detailed result
            game_results.append({
                "game_id": i + 1,
                "winner": winner,
                "duration": game_duration
            })

        except subprocess.CalledProcessError as e:
            print(f"\nGame {i+1}/{NUM_GAMES}: FAILED TO RUN!")
            print(e.stderr)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    print(f"\n\n--- Tournament Complete ---")

    # --- Basic Stats ---
    wins = [r for r in game_results if r['winner'] == YOUR_AGENT]
    losses = [r for r in game_results if r['winner'] == OPPONENT_AGENT]
    draws = [r for r in game_results if r['winner'] == "draw"]
    all_durations = [r['duration'] for r in game_results]

    print(f"Score (W-L-D): {len(wins)}-{len(losses)}-{len(draws)}")

    # --- Advanced Timing Stats ---
    if all_durations:
        win_times = [r['duration'] for r in wins]
        loss_times = [r['duration'] for r in losses]
        draw_times = [r['duration'] for r in draws]

        print("\n--- Performance ---")
        print(f"Total time:       {total_duration:.2f} seconds")
        print(f"Average game time:  {avg(all_durations):.2f} seconds")
        print(f"Fastest game:       {min(all_durations):.2f} seconds")
        print(f"Slowest game:       {max(all_durations):.2f} seconds")
        
        print("\n--- Time by Outcome ---")
        print(f"Avg. Win Time:      {avg(win_times):.2f} seconds ({len(wins)} games)")
        print(f"Avg. Loss Time:     {avg(loss_times):.2f} seconds ({len(losses)} games)")
        print(f"Avg. Draw Time:     {avg(draw_times):.2f} seconds ({len(draws)} games)")

    return game_results

if __name__ == "__main__":
    run_tournament()