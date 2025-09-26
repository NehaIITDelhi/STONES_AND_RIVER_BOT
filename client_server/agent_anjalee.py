"""
River and Stones Game - AI Agent Framework

This module provides the base agent implementations and factory function.
The student implementation uses minimax with alpha-beta pruning.

Game Rules Summary:
- Two players: Circle (red) and Square (blue)
- Pieces can be stones or rivers (horizontal/vertical)
- Goal: Get 4 of your stones into opponent's scoring area
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers allow flow-based movement to distant locations
"""

import random
import copy
import math
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# ==================== GAME UTILITIES ====================
# These functions help agents understand and manipulate the game state

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

# ==================== RIVER FLOW SIMULATION ====================

def agent_river_flow(board, rx: int, ry: int, sx: int, sy: int, player: str, 
                    rows: int, cols: int, score_cols: List[int], river_push: bool = False) -> List[Tuple[int, int]]:
    """
    Simulate river flow from a given position.
    
    Args:
        board: Current board state
        rx, ry: River entry point
        sx, sy: Source position (where piece is moving from)
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices
        river_push: Whether this is for a river push move
    
    Returns:
        List of (x, y) coordinates where the piece can end up via river flow
    """
    destinations = []
    visited = set()
    queue = [(rx, ry)]
    
    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))
        
        cell = board[y][x]
        if river_push and x == rx and y == ry:
            cell = board[sy][sx]
            
        if cell is None:
            if is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                # Block entering opponent score cell
                pass
            else:
                destinations.append((x, y))
            continue
            
        if getattr(cell, "side", "stone") != "river":
            continue
            
        # River flow directions
        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]
        
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break
                    
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue
                    
                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue
                    
                if getattr(next_cell, "side", "stone") == "river":
                    queue.append((nx, ny))
                    break
                break
    
    # Remove duplicates
    unique_destinations = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            unique_destinations.append(d)
    
    return unique_destinations

# ==================== MOVE VALIDATION AND GENERATION ====================

def agent_compute_valid_moves(board, sx: int, sy: int, player: str, rows: int, cols: int, score_cols: List[int]) -> Dict[str, Any]:
    """
    Compute all valid moves for a piece at position (sx, sy).
    
    Returns:
        Dictionary with 'moves' (set of coordinates) and 'pushes' (list of tuples)
    """
    if not in_bounds(sx, sy, rows, cols):
        return {'moves': set(), 'pushes': []}
        
    piece = board[sy][sx]
    if piece is None or piece.owner != player:
        return {'moves': set(), 'pushes': []}
    
    moves = set()
    pushes = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for dx, dy in directions:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
            
        # Block moving into opponent score cell
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue
            
        target = board[ty][tx]
        
        if target is None:
            # Empty cell - direct move
            moves.add((tx, ty))
        elif getattr(target, "side", "stone") == "river":
            # River - compute flow destinations
            flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for dest in flow:
                moves.add(dest)
        else:
            # Occupied by stone - check push possibility
            if getattr(piece, "side", "stone") == "stone":
                # Stone pushing stone
                px, py = tx + dx, ty + dy
                if (in_bounds(px, py, rows, cols) and 
                    board[py][px] is None and 
                    not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    pushes.append(((tx, ty), (px, py)))
            else:
                # River pushing - compute flow for pushed piece
                flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols, river_push=True)
                for dest in flow:
                    if not is_opponent_score_cell(dest[0], dest[1], player, rows, cols, score_cols):
                        pushes.append(((tx, ty), dest))
    
    return {'moves': moves, 'pushes': pushes}

# ==================== MOVE APPLICATION (FOR SIMULATION) ====================

def agent_apply_move(board, move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, str]:
    """
    Apply a move to a board copy for simulation purposes.
    
    Args:
        board: Board state to modify
        move: Move dictionary
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices
    
    Returns:
        (success: bool, message: str)
    """
    action = move.get("action")
    
    if action == "move":
        return _apply_move_action(board, move, player, rows, cols, score_cols)
    elif action == "push":
        return _apply_push_action(board, move, player, rows, cols, score_cols)
    elif action == "flip":
        return _apply_flip_action(board, move, player, rows, cols, score_cols)
    elif action == "rotate":
        return _apply_rotate_action(board, move, player, rows, cols, score_cols)
    
    return False, "unknown action"

def _apply_move_action(board, move, player, rows, cols, score_cols):
    """Apply a move action."""
    fr = move.get("from")
    to = move.get("to")
    if not fr or not to:
        return False, "bad move format"
    
    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])
    
    if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
        return False, "out of bounds"
    
    if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
        return False, "cannot move into opponent score cell"
    
    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"
    
    if board[ty][tx] is None:
        # Simple move
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, "moved"
    
    # Move with push
    pushed_to = move.get("pushed_to")
    if not pushed_to:
        return False, "destination occupied; pushed_to required"
    
    ptx, pty = int(pushed_to[0]), int(pushed_to[1])
    dx, dy = tx - fx, ty - fy
    
    if (ptx, pty) != (tx + dx, ty + dy):
        return False, "invalid pushed_to"
    
    if not in_bounds(ptx, pty, rows, cols):
        return False, "pushed_to out of bounds"
    
    if is_opponent_score_cell(ptx, pty, player, rows, cols, score_cols):
        return False, "cannot push into opponent score"
    
    if board[pty][ptx] is not None:
        return False, "pushed_to not empty"
    
    board[pty][ptx] = board[ty][tx]
    board[ty][tx] = piece
    board[fy][fx] = None
    return True, "moved with push"

def _apply_push_action(board, move, player, rows, cols, score_cols):
    """Apply a push action."""
    fr = move.get("from")
    to = move.get("to")
    pushed_to = move.get("pushed_to")
    
    if not fr or not to or not pushed_to:
        return False, "bad push format"
    
    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])
    px, py = int(pushed_to[0]), int(pushed_to[1])
    
    if (is_opponent_score_cell(tx, ty, player, rows, cols, score_cols) or
        is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
        return False, "push would move into opponent score cell"
    
    if not (in_bounds(fx, fy, rows, cols) and 
            in_bounds(tx, ty, rows, cols) and 
            in_bounds(px, py, rows, cols)):
        return False, "out of bounds"
    
    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"
    
    if board[ty][tx] is None:
        return False, "'to' must be occupied"
    
    if board[py][px] is not None:
        return False, "pushed_to not empty"
    
    board[py][px] = board[ty][tx]
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None
    return True, "pushed"

def _apply_flip_action(board, move, player, rows, cols, score_cols):
    """Apply a flip action."""
    fr = move.get("from")
    if not fr:
        return False, "bad flip format"
    
    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]
    
    if piece is None or piece.owner != player:
        return False, "invalid piece"
    
    if piece.side == "stone":
        # Stone to river
        orientation = move.get("orientation")
        if orientation not in ("horizontal", "vertical"):
            return False, "stone->river needs orientation"
        
        # Check if new river would allow flow into opponent score
        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        
        # Revert for safety check
        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None
        
        for dx, dy in flow:
            if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                return False, "flip would allow flow into opponent score cell"
        
        # Apply flip
        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        return True, "flipped to river"
    else:
        # River to stone
        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None
        return True, "flipped to stone"

def _apply_rotate_action(board, move, player, rows, cols, score_cols):
    """Apply a rotate action."""
    fr = move.get("from")
    if not fr:
        return False, "bad rotate format"
    
    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]
    
    if piece is None or piece.owner != player or piece.side != "river":
        return False, "invalid rotate"
    
    # Try rotation
    old_orientation = piece.orientation
    piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
    
    # Check flow safety after rotation
    flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)
    
    for dx, dy in flow:
        if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
            # Revert rotation
            piece.orientation = old_orientation
            return False, "rotate would allow flow into opponent score cell"
    
    return True, "rotated"

# ==================== BASE AGENT CLASS ====================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Students should inherit from this class and implement the required methods.
    """
    
    def __init__(self, player: str):
        """
        Initialize agent.
        
        Args:
            player: Either "circle" or "square"
        """
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
            current_player_time : Remaining time for this player (in seconds)
            opponent_time : Remaining time for the opponent (in seconds)
        
        Returns:
            Dictionary representing the chosen move, or None if no moves available
            
        Move format examples:
            {"action": "move", "from": [x, y], "to": [x2, y2]}
            {"action": "push", "from": [x, y], "to": [x2, y2], "pushed_to": [x3, y3]}
            {"action": "flip", "from": [x, y], "orientation": "horizontal"}  # stone->river
            {"action": "flip", "from": [x, y]}  # river->stone
            {"action": "rotate", "from": [x, y]}  # rotate river
        """
        pass
    
    def generate_all_moves(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """
        Generate all legal moves for the current player.
        
        This is a helper method that students can use in their implementations.
        """
        moves = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece or piece.owner != self.player:
                    continue
                
                if piece.side == "stone":
                    # Generate moves for stones
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if not in_bounds(nx, ny, rows, cols):
                            continue
                        
                        # Block destination in opponent score
                        if is_opponent_score_cell(nx, ny, self.player, rows, cols, score_cols):
                            continue
                        
                        if board[ny][nx] is None:
                            moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
                        else:
                            if board[ny][nx].owner != self.player:
                                px, py = nx + dx, ny + dy
                                if (in_bounds(px, py, rows, cols) and 
                                    board[py][px] is None and 
                                    not is_opponent_score_cell(px, py, self.player, rows, cols, score_cols)):
                                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
                    
                    # Generate flip moves (stone -> river)
                    for orientation in ("horizontal", "vertical"):
                        # Check if flip is safe
                        temp = copy.deepcopy(board)
                        temp[y][x].side = "river"
                        temp[y][x].orientation = orientation
                        flow = agent_river_flow(temp, x, y, x, y, self.player, rows, cols, score_cols)
                        
                        if not any(is_opponent_score_cell(dx, dy, self.player, rows, cols, score_cols) for dx, dy in flow):
                            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
                
                else:  # River piece
                    # Flip river -> stone
                    moves.append({"action": "flip", "from": [x, y]})
                    
                    # Rotate if safe
                    new_orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
                    temp = copy.deepcopy(board)
                    temp[y][x].orientation = new_orientation
                    flow = agent_river_flow(temp, x, y, x, y, self.player, rows, cols, score_cols)
                    
                    if not any(is_opponent_score_cell(dx, dy, self.player, rows, cols, score_cols) for dx, dy in flow):
                        moves.append({"action": "rotate", "from": [x, y]})
        
        return moves
    
    def evaluate_board(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Evaluate the current board state from this player's perspective.
        Higher values indicate better positions for this player.
        
        This is a basic evaluation function that students can override.
        """
        score = 0.0
        top_row = top_score_row()
        bottom_row = bottom_score_row(rows)
        
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece:
                    continue
                
                if piece.owner == self.player and piece.side == "stone":
                    score += 1.0
                    
                    # Bonus for stones in own scoring area
                    if is_own_score_cell(x, y, self.player, rows, cols, score_cols):
                        score += 10.0
                    
                    # Small bonus for advancing toward opponent
                    if self.player == "circle":
                        score += (rows - y) * 0.1
                    else:
                        score += y * 0.1
                
                elif piece.owner == self.opponent and piece.side == "stone":
                    score -= 1.0
                    
                    # Penalty if opponent has stones in their scoring area
                    if is_own_score_cell(x, y, self.opponent, rows, cols, score_cols):
                        score -= 10.0
        
        return score
    
    def simulate_move(self, board: List[List[Any]], move: Dict[str, Any], rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
        """
        Simulate a move on a copy of the board.
        
        Returns:
            (success: bool, new_board or error_message)
        """
        board_copy = copy.deepcopy(board)
        success, message = agent_apply_move(board_copy, move, self.player, rows, cols, score_cols)
        
        if success:
            return True, board_copy
        else:
            return False, message

# ==================== IMPROVED STUDENT AGENT WITH MINIMAX ====================

class AnjaleeAgent(BaseAgent): 
    """Student Agent using Minimax with Alpha-Beta Pruning"""
    
    def __init__(self, player: str):
        super().__init__(player)
        self.max_depth = 2  # Adjust based on time constraints
        self.nodes_evaluated = 0
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], 
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move using minimax with alpha-beta pruning.
        """
        # Adjust depth based on remaining time
        if current_player_time > 30.0:
            self.max_depth = 3
        elif current_player_time > 10.0:
            self.max_depth = 2
        else:
            self.max_depth = 1  # Quick moves when time is low
        
        self.nodes_evaluated = 0
        best_move = None
        best_value = -math.inf
        
        # Get all possible moves
        all_moves = self.generate_all_moves(board, rows, cols, score_cols)
        
        if not all_moves:
            return None
        
        # Try each move with minimax
        for move in all_moves:
            # Simulate the move
            success, new_board = self.simulate_move(board, move, rows, cols, score_cols)
            if not success:
                continue
                
            # Evaluate the move using minimax
            value = self.minimax(new_board, self.max_depth - 1, -math.inf, math.inf, False, 
                                rows, cols, score_cols)
            
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move else random.choice(all_moves)
    
    def minimax(self, board: List[List[Any]], depth: int, alpha: float, beta: float, 
                maximizing_player: bool, rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        """
        self.nodes_evaluated += 1
        
        # Terminal node or depth limit reached
        if depth == 0 or self.is_terminal(board, rows, cols, score_cols):
            return self.evaluate_board_advanced(board, rows, cols, score_cols)
        
        if maximizing_player:
            max_eval = -math.inf
            moves = self.generate_all_moves_for_player(board, self.player, rows, cols, score_cols)
            
            for move in moves:
                success, new_board = self.simulate_move(board, move, rows, cols, score_cols)
                if not success:
                    continue
                    
                eval_val = self.minimax(new_board, depth - 1, alpha, beta, False, rows, cols, score_cols)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                
                if beta <= alpha:
                    break  # Beta cutoff
                    
            return max_eval
        else:
            min_eval = math.inf
            opponent = get_opponent(self.player)
            moves = self.generate_all_moves_for_player(board, opponent, rows, cols, score_cols)
            
            for move in moves:
                success, new_board = self.simulate_move(board, move, rows, cols, score_cols)
                if not success:
                    continue
                    
                eval_val = self.minimax(new_board, depth - 1, alpha, beta, True, rows, cols, score_cols)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                
                if beta <= alpha:
                    break  # Alpha cutoff
                    
            return min_eval
    
    def generate_all_moves_for_player(self, board: List[List[Any]], player: str, 
                                     rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """
        Generate all legal moves for a specific player.
        """
        moves = []
        
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece or piece.owner != player:
                    continue
                
                # Get valid moves for this piece
                valid_moves = agent_compute_valid_moves(board, x, y, player, rows, cols, score_cols)
                
                # Add direct moves
                for tx, ty in valid_moves['moves']:
                    moves.append({"action": "move", "from": [x, y], "to": [tx, ty]})
                
                # Add push moves
                for (tx, ty), (px, py) in valid_moves['pushes']:
                    moves.append({"action": "push", "from": [x, y], "to": [tx, ty], "pushed_to": [px, py]})
                
                # Add special actions
                if piece.side == "stone":
                    # Stone flip moves
                    for orientation in ("horizontal", "vertical"):
                        # Check if flip is safe
                        temp = copy.deepcopy(board)
                        temp[y][x].side = "river"
                        temp[y][x].orientation = orientation
                        flow = agent_river_flow(temp, x, y, x, y, player, rows, cols, score_cols)
                        
                        if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
                else:
                    # River flip and rotate moves
                    moves.append({"action": "flip", "from": [x, y]})  # River to stone
                    
                    # Rotate if safe
                    new_orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
                    temp = copy.deepcopy(board)
                    temp[y][x].orientation = new_orientation
                    flow = agent_river_flow(temp, x, y, x, y, player, rows, cols, score_cols)
                    
                    if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                        moves.append({"action": "rotate", "from": [x, y]})
        
        return moves
    
    def is_terminal(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> bool:
        """
        Check if the game is in a terminal state (win/loss).
        """
        # Count stones in scoring areas
        circle_score = 0
        square_score = 0
        top_row = top_score_row()
        bottom_row = bottom_score_row(rows)
        
        for y in [top_row, bottom_row]:
            for x in score_cols:
                piece = board[y][x]
                if piece and piece.side == "stone":
                    if piece.owner == "circle" and y == top_row:
                        circle_score += 1
                    elif piece.owner == "square" and y == bottom_row:
                        square_score += 1
        
        # Game ends when a player has 4 stones in opponent's scoring area
        return circle_score >= 4 or square_score >= 4
    
    def evaluate_board_advanced(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Enhanced evaluation function considering multiple game aspects.
        """
        score = 0.0
        top_row = top_score_row()
        bottom_row = bottom_score_row(rows)
        
        # Material balance and positional scoring
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece:
                    continue
                
                # Material value
                if piece.side == "stone":
                    stone_value = 10.0
                    
                    if piece.owner == self.player:
                        score += stone_value
                        
                        # Bonus for stones in scoring position
                        if is_own_score_cell(x, y, self.player, rows, cols, score_cols):
                            score += 50.0
                        
                        # Bonus for advancement toward opponent's side
                        if self.player == "circle":
                            advancement = (rows - 1 - y) * 0.5
                        else:
                            advancement = y * 0.5
                        score += advancement
                        
                        # Bonus for central control
                        center_bonus = 1.0 - abs(x - cols/2) / (cols/2)
                        score += center_bonus * 2.0
                        
                    else:  # Opponent's stone
                        score -= stone_value
                        
                        # Penalty for opponent stones in scoring position
                        if is_own_score_cell(x, y, self.opponent, rows, cols, score_cols):
                            score -= 50.0
        
        # River control evaluation
        river_score = self.evaluate_river_control(board, rows, cols, score_cols)
        score += river_score * 0.5
        
        # Mobility evaluation
        mobility = len(self.generate_all_moves_for_player(board, self.player, rows, cols, score_cols))
        opponent_mobility = len(self.generate_all_moves_for_player(board, self.opponent, rows, cols, score_cols))
        score += (mobility - opponent_mobility) * 0.1
        
        return score
    
    def evaluate_river_control(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Evaluate river control and connectivity.
        """
        score = 0.0
        
        for y in range(rows):
            for x in range(cols):
                piece = board[y][x]
                if not piece or piece.side != "river":
                    continue
                
                river_value = 3.0
                if piece.owner == self.player:
                    score += river_value
                    
                    # Bonus for rivers that create useful paths
                    connectivity = self.evaluate_river_connectivity(board, x, y, rows, cols, score_cols)
                    score += connectivity
                else:
                    score -= river_value
        
        return score
    
    def evaluate_river_connectivity(self, board: List[List[Any]], x: int, y: int, 
                                  rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Evaluate how useful a river is for creating paths.
        """
        # Simple connectivity measure - count reachable positions
        reachable = agent_river_flow(board, x, y, x, y, self.player, rows, cols, score_cols)
        connectivity = len(reachable) * 0.1
        
        # Bonus if river connects to scoring areas or key positions
        for rx, ry in reachable:
            if is_own_score_cell(rx, ry, self.player, rows, cols, score_cols):
                connectivity += 2.0
            elif rx == cols // 2 or ry == rows // 2:  # Center control
                connectivity += 0.5
        
        return connectivity

# ==================== EXAMPLE AGENT IMPLEMENTATIONS ====================

class RandomAgent(BaseAgent):
    """
    Simple random agent that chooses moves randomly.
    This serves as a baseline and example implementation.
    """
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        moves = self.generate_all_moves(board, rows, cols, score_cols)
        if not moves:
            return None
        return random.choice(moves)

# ==================== AGENT FACTORY ====================

def get_agent(player: str, strategy: str) -> BaseAgent:
    """
    Factory function to create agents based on strategy name.
    
    Args:
        player: "circle" or "square"
        strategy: Strategy name ("random", "student")
    
    Returns:
        Agent instance
    """
    strategy = strategy.lower()
    
    if strategy == "random":
        return RandomAgent(player)
    elif strategy == "student":
        return StudentAgent(player)
    elif strategy == "student_cpp":
        try:
            import student_agent_cpp as student_agent
            StudentAgentCpp = student_agent.StudentAgent
        except ImportError:
            StudentAgentCpp = None
        if StudentAgentCpp:
            return StudentAgentCpp(player)
        else:
            print("C++ StudentAgent not available. Falling back to Python StudentAgent.")
            return StudentAgent(player)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available: random, student")