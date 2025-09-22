"""
Student Agent Implementation for River and Stones Game

This file contains the essential utilities and template for implementing your AI agent.
Your task is to complete the StudentAgent class with intelligent move selection.

Game Rules:
- Goal: Get 4 of your stones into the opponent's scoring area
- Pieces can be stones or rivers (horizontal/vertical orientation)  
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers enable flow-based movement across the board

Your Task:
Implement the choose() method in the StudentAgent class to select optimal moves.
You may add any helper methods and modify the evaluation function as needed.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis


# rectangle in top of the grid and circle in bottom

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2
    # because the starting two rows are empty

def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"

# ==================== MOVE GENERATION HELPERS ====================



def _trace_river_flow(board: list[list[any]],
                      river_start_x: int, river_start_y: int,
                      original_piece_x: int, original_piece_y: int,
                      whos_turn: str,
                      rows: int, cols: int, score_cols: list[int],
                      is_a_push_move: bool = False) -> list[tuple[int, int]]:
    """
    Helper function to trace all possible paths along a river or a chain of rivers.
    It figures out all the empty spots a piece could land on.
    """
    possible_landings = []
    
    # A list of coordinates we need to check for rivers
    coords_to_check = [(river_start_x, river_start_y)]
    
    # A set to keep track of river coordinates we've already processed
    checked_coords = set()

    while coords_to_check:
        current_x, current_y = coords_to_check.pop(0)

        # Skip if we've already looked at this river piece or if it's off the board
        if (current_x, current_y) in checked_coords or not in_bounds(current_x, current_y, rows, cols):
            continue
        checked_coords.add((current_x, current_y))

        current_cell = board[current_y][current_x]
        
        # Special case for river pushes: the 'river' is actually the piece being pushed
        if is_a_push_move and current_x == river_start_x and current_y == river_start_y:
            current_cell = board[original_piece_y][original_piece_x]

        if current_cell is None:
            # This can happen if a river flows into an empty space
            if not is_opponent_score_cell(current_x, current_y, whos_turn, rows, cols, score_cols):
                if (current_x, current_y) not in possible_landings:
                     possible_landings.append((current_x, current_y))
            continue # Can't flow from an empty cell

        # If we hit a stone, the flow stops here
        if not hasattr(current_cell, 'side') or current_cell.side != "river":
            continue

        # Check horizontal or vertical flow based on the river's orientation
        flow_directions = [(1, 0), (-1, 0)] if current_cell.orientation == "horizontal" else [(0, 1), (-1, 0)]
        
        for move_x, move_y in flow_directions:
            new_x, new_y = current_x + move_x, current_y + move_y
            
            # Keep moving in one direction until we hit something or go off-board
            while in_bounds(new_x, new_y, rows, cols):
                # Stop if the path enters the opponent's goal
                if is_opponent_score_cell(new_x, new_y, whos_turn, rows, cols, score_cols):
                    break

                cell_in_path = board[new_y][new_x]

                if cell_in_path is None:
                    # Found an empty spot, it's a valid place to land
                    if (new_x, new_y) not in possible_landings:
                        possible_landings.append((new_x, new_y))
                    new_x += move_x # Keep checking further along the path
                    new_y += move_y
                    continue

                # Don't check the spot where the piece started its move
                if new_x == original_piece_x and new_y == original_piece_y:
                    new_x += move_x
                    new_y += move_y
                    continue
                
                # If we hit another river, add it to our list to check later
                if hasattr(cell_in_path, 'side') and cell_in_path.side == "river":
                    if (new_x, new_y) not in checked_coords:
                        coords_to_check.append((new_x, new_y))
                    break # Stop this direction, the new river will be checked from the queue

                # If we hit a stone, the path is blocked
                break
                
    return possible_landings


def get_valid_moves_for_piece(board: list[list[any]], start_pos_x: int, start_pos_y: int, current_player: str,
                                 rows: int, cols: int, score_cols: list[int]) -> list[dict[str, any]]:
    """
    Computes all valid moves for a single piece, including moves, pushes, flips, and rotates.
    Returns a list of action dictionaries.
    """
    my_piece = board[start_pos_y][start_pos_x]
    all_possible_actions = []

    # Basic check to make sure there's a piece and it's ours
    if my_piece is None or my_piece.owner != current_player:
        return all_possible_actions

    # --- Part 1: Calculate Moves and Pushes ---
    
    # Directions to check: right, left, down, up
    move_options = [(1, 0), (-1, 0), (0, 1), (-1, 0)]

    for dx, dy in move_options:
        target_x, target_y = start_pos_x + dx, start_pos_y + dy

        is_off_board = not in_bounds(target_x, target_y, rows, cols)
        is_illegal_goal = is_opponent_score_cell(target_x, target_y, current_player, rows, cols, score_cols)
        
        if is_off_board or is_illegal_goal:
            continue

        target_square = board[target_y][target_x]

        if target_square is None:
            # Simple move to an empty square
            move_action = {"action": "move", "from": [start_pos_x, start_pos_y], "to": [target_x, target_y]}
            all_possible_actions.append(move_action)
        
        elif hasattr(target_square, 'side') and target_square.side == "river":
            # Move along a river
            river_landings = _trace_river_flow(board, target_x, target_y, start_pos_x, start_pos_y, current_player, rows, cols, score_cols)
            for landing_spot in river_landings:
                lx, ly = landing_spot
                move_action = {"action": "move", "from": [start_pos_x, start_pos_y], "to": [lx, ly]}
                all_possible_actions.append(move_action)
        
        else: # Target is a stone, so it's a push move
            if my_piece.side == "stone":
                # Stone pushing a stone
                push_dest_x, push_dest_y = target_x + dx, target_y + dy
                if in_bounds(push_dest_x, push_dest_y, rows, cols) and board[push_dest_y][push_dest_x] is None:
                    if not is_opponent_score_cell(push_dest_x, push_dest_y, current_player, rows, cols, score_cols):
                        push_action = {"action": "push", "from": [start_pos_x, start_pos_y], "to": [target_x, target_y], "pushed_to": [push_dest_x, push_dest_y]}
                        all_possible_actions.append(push_action)
            
            else: # River pushing a stone
                owner_of_pushed_piece = target_square.owner
                river_landings = _trace_river_flow(board, target_x, target_y, start_pos_x, start_pos_y, owner_of_pushed_piece, rows, cols, score_cols, is_a_push_move=True)
                for landing_spot in river_landings:
                    lx, ly = landing_spot
                    push_action = {"action": "push", "from": [start_pos_x, start_pos_y], "to": [target_x, target_y], "pushed_to": [lx, ly]}
                    all_possible_actions.append(push_action)

    # --- Part 2: Add Flip and Rotate Actions ---

    # These actions don't depend on what's around the piece, only what the piece is.
    if my_piece.side == "stone":
        # A stone can be flipped to a river in two ways
        flip_h = {"action": "flip", "from": [start_pos_x, start_pos_y], "orientation": "horizontal"}
        flip_v = {"action": "flip", "from": [start_pos_x, start_pos_y], "orientation": "vertical"}
        all_possible_actions.append(flip_h)
        all_possible_actions.append(flip_v)
    else: # The piece is a river
        # A river can be flipped to a stone
        flip_action = {"action": "flip", "from": [start_pos_x, start_pos_y]}
        all_possible_actions.append(flip_action)
        
        # A river can also be rotated
        rotate_action = {"action": "rotate", "from": [start_pos_x, start_pos_y]}
        all_possible_actions.append(rotate_action)

    return all_possible_actions



def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all legal moves for the current player.
    
    Args:
        board: Current board state
        player: Current player ("circle" or "square")
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        List of all valid move dictionaries
    """
    all_moves = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player:
                piece_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols)
                all_moves.extend(piece_moves)
    
    return all_moves

# ==================== BOARD EVALUATION ====================

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0
    
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    
    return count

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    """
    Basic board evaluation function.
    
    Returns a score where higher values are better for the given player.
    Students can use this as a starting point and improve it.
    """
    score = 0.0
    opponent = get_opponent(player)
    
    # Count stones in scoring areas
    player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opponent_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    score += player_scoring_stones * 100  
    score -= opponent_scoring_stones * 100  
    
    # Count total pieces and positional factors
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Basic positional scoring
                if player == "circle":
                    score += (rows - y) * 0.1
                else:
                    score += y * 0.1
    
    return score

def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    """
    Simulate a move on a copy of the board.
    
    Args:
        board: Current board state
        move: Move to simulate
        player: Player making the move
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        (success: bool, new_board_state or error_message)
    """
    # Import the game engine's move validation function
    try:
        from gameEngine import validate_and_apply_move
        board_copy = copy.deepcopy(board)
        success, message = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols)
        return success, board_copy if success else message
    except ImportError:
        # Fallback to basic simulation if game engine not available
        return True, copy.deepcopy(board)

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation
    
    TODO: Implement your AI agent for the River and Stones game.
    The goal is to get 4 of your stones into the opponent's scoring area.
    
    You have access to these utility functions:
    - generate_all_moves(): Get all legal moves for current player
    - basic_evaluate_board(): Basic position evaluation 
    - simulate_move(): Test moves on board copy
    - count_stones_in_scoring_area(): Count stones in scoring positions
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        # TODO: Add any initialization you need
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions  
            score_cols: Column indices for scoring areas
            
        Returns:
            Dictionary representing your chosen move
        """
        moves = generate_all_moves(board, self.player, rows, cols, score_cols)
        
        if not moves:
            return None
        
        # TODO: Replace random selection with your AI algorithm
        return random.choice(moves)

# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()
