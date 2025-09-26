import time
import random
import copy
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set

# ===================================================================
# UTILITY FUNCTIONS 
# ===================================================================
def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows: int) -> int:
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def board_hash(board: List[List[Any]]) -> str:
    """Create a hash of the board state for loop detection"""
    board_str = ""
    for row in board:
        for cell in row:
            if cell is None:
                board_str += "0"
            else:
                board_str += f"{cell.owner[0]}{cell.side[0]}"
                if hasattr(cell, 'orientation') and cell.orientation:
                    board_str += cell.orientation[0]
    return hashlib.md5(board_str.encode()).hexdigest()

def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(x1 - x2) + abs(y1 - y2)

# ===================================================================
# AGENT BASE CLASS
# ===================================================================
class BaseAgent:
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)

    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

# ===================================================================
# IMPROVED STUDENT AGENT IMPLEMENTATION
# ===================================================================
class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_table = {}
        self.turn_count = 0
        self.board_history_set = set()  # Use set for O(1) lookups
        self.recent_positions = []  # Keep last N positions as list
        self.last_moves = []  # Store last few moves to avoid repetition
        self.MAX_HISTORY_SIZE = 20
        self.MAX_RECENT_MOVES = 5

    # ==============================================================
    # MAIN SEARCH FUNCTION WITH IMPROVED LOOP PREVENTION
    # ==============================================================

    def choose(self, game_state: Dict[str, Any], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        
        self.turn_count += 1
        
        if isinstance(game_state, dict) and "board" in game_state:
            board = game_state["board"]
        else:
            board = game_state

        # Update position history with size limit
        current_hash = board_hash(board)
        self.recent_positions.append(current_hash)
        if len(self.recent_positions) > self.MAX_HISTORY_SIZE:
            old_hash = self.recent_positions.pop(0)
            self.board_history_set.discard(old_hash)
        self.board_history_set.add(current_hash)

        start_time = time.time()
        
        # Adaptive time management
        if current_player_time > 35:
            time_limit = 1.5
        elif current_player_time > 20:
            time_limit = 1.0
        elif current_player_time > 10:
            time_limit = 0.6
        else:
            time_limit = max(0.15, current_player_time / 20.0)

        # Get all valid moves
        moves = self.get_all_valid_moves_enhanced(board, self.player, rows, cols, score_cols)
        if not moves:
            return None
        
        # Check for immediate winning move
        for move in moves:
            if self.is_winning_move(board, move, self.player, rows, cols, score_cols):
                self.update_move_history(move)
                return move
        
        # Check for opponent winning threats and block them
        opp_moves = self.get_all_valid_moves_enhanced(board, self.opponent, rows, cols, score_cols)
        for opp_move in opp_moves:
            if self.is_winning_move(board, opp_move, self.opponent, rows, cols, score_cols):
                defensive_move = self.find_blocking_move(board, moves, opp_move, rows, cols, score_cols)
                if defensive_move:
                    self.update_move_history(defensive_move)
                    return defensive_move
        
        # Filter moves to prevent loops and repetitions
        filtered_moves = self.filter_non_repeating_moves(board, moves, rows, cols, score_cols)
        if filtered_moves:
            moves = filtered_moves
        
        # Determine game phase and strategy
        game_phase = self.get_game_phase(board, rows, cols)
        
        # Adjust depth based on time and game phase
        if current_player_time < 5:
            max_depth = 1
        elif current_player_time < 15:
            max_depth = 2
        elif game_phase == "endgame":
            max_depth = 4
        elif game_phase == "midgame":
            max_depth = 3
        else:
            max_depth = 2
        
        # Order moves with edge control consideration
        moves = self.order_moves_with_edge_control(board, moves, self.player, rows, cols, score_cols)
        
        # Limit branching factor based on time
        if current_player_time > 20:
            max_moves = min(10, len(moves))
        else:
            max_moves = min(6, len(moves))
        
        moves = moves[:max_moves]
        
        # Iterative deepening with improved move ordering
        best_move = moves[0]
        best_score = float('-inf')
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit * 0.85:
                break
                
            alpha = float('-inf')
            beta = float('inf')
            current_best = None
            
            for i, move in enumerate(moves):
                if time.time() - start_time > time_limit:
                    break
                    
                new_board = self.apply_move(board, move, self.player)
                
                # Check if this creates a repeated position
                new_hash = board_hash(new_board)
                if new_hash in self.board_history_set:
                    continue  # Skip moves that create loops
                
                if i == 0:
                    score = -self.negamax_with_balance(new_board, depth - 1, -beta, -alpha, 
                                                      self.opponent, start_time, time_limit, 
                                                      rows, cols, score_cols, depth)
                else:
                    score = -self.negamax_with_balance(new_board, depth - 1, -alpha - 1, -alpha,
                                                      self.opponent, start_time, time_limit,
                                                      rows, cols, score_cols, depth)
                    if alpha < score < beta:
                        score = -self.negamax_with_balance(new_board, depth - 1, -beta, -score,
                                                          self.opponent, start_time, time_limit,
                                                          rows, cols, score_cols, depth)
                
                if score > best_score:
                    best_score = score
                    current_best = move
                    
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            
            if current_best:
                best_move = current_best
                # Move best move to front for next iteration
                if best_move in moves:
                    moves.remove(best_move)
                    moves.insert(0, best_move)

        self.update_move_history(best_move)
        return best_move

    def update_move_history(self, move: Dict[str, Any]):
        """Update the history of recent moves"""
        self.last_moves.append(move)
        if len(self.last_moves) > self.MAX_RECENT_MOVES:
            self.last_moves.pop(0)

    def filter_non_repeating_moves(self, board: List[List[Any]], moves: List[Dict[str, Any]], 
                                  rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """Filter out moves that would create loops or repeat recent moves"""
        filtered = []
        
        for move in moves:
            # Check if move repeats recent moves
            is_repeat = False
            for recent_move in self.last_moves[-3:]:  # Check last 3 moves
                if self.moves_similar(move, recent_move):
                    is_repeat = True
                    break
            
            if is_repeat:
                continue
                
            # Check if move creates a position we've seen recently
            new_board = self.apply_move(board, move, self.player)
            new_hash = board_hash(new_board)
            
            # Don't create positions seen in last 8 moves
            if new_hash not in self.recent_positions[-8:]:
                filtered.append(move)
        
        # If all moves are filtered, return top moves by evaluation
        if not filtered:
            return moves[:5]
        
        return filtered

    def moves_similar(self, move1: Dict[str, Any], move2: Dict[str, Any]) -> bool:
        """Check if two moves are essentially the same or reversals"""
        if move1["action"] != move2["action"]:
            return False
        
        # Check for exact same move
        if move1.get("from") == move2.get("from"):
            if move1.get("to") == move2.get("to"):
                return True
        
        # Check for reversal moves (A->B followed by B->A)
        if move1.get("action") == "move" and move2.get("action") == "move":
            if (move1.get("from") == move2.get("to") and 
                move1.get("to") == move2.get("from")):
                return True
        
        return False

    # ==============================================================
    # BALANCED NEGAMAX WITH DEFENSIVE AWARENESS
    # ==============================================================

    def negamax_with_balance(self, board: List[List[Any]], depth: int, alpha: float, beta: float,
                             current_player: str, start_time: float, time_limit: float, 
                             rows: int, cols: int, score_cols: List[int], max_depth: int) -> float:
        
        if time.time() - start_time > time_limit:
            return self.evaluate_balanced(board, current_player, rows, cols, score_cols)
        
        # Check terminal conditions
        my_stones = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp = get_opponent(current_player)
        opp_stones = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        
        if my_stones >= 4:
            return 10000 - (max_depth - depth) * 10
        if opp_stones >= 4:
            return -10000 + (max_depth - depth) * 10
        
        if depth <= 0:
            return self.evaluate_balanced(board, current_player, rows, cols, score_cols)
        
        moves = self.get_all_valid_moves_enhanced(board, current_player, rows, cols, score_cols)
        if not moves:
            return self.evaluate_balanced(board, current_player, rows, cols, score_cols)
        
        # Order moves with edge control consideration
        moves = self.order_moves_with_edge_control(board, moves, current_player, rows, cols, score_cols)
        
        # Adaptive branching
        if depth <= 1:
            moves = moves[:4]
        else:
            moves = moves[:6]
        
        best_score = float('-inf')
        
        for move in moves:
            new_board = self.apply_move(board, move, current_player)
            
            # Skip if creates repeated position
            new_hash = board_hash(new_board)
            if new_hash in self.recent_positions[-5:]:
                continue
            
            score = -self.negamax_with_balance(new_board, depth - 1, -beta, -alpha,
                                              opp, start_time, time_limit,
                                              rows, cols, score_cols, max_depth)
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Update killer moves
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()
                break
        
        return best_score

    # ==============================================================
    # BALANCED EVALUATION WITH EDGE CONTROL
    # ==============================================================

    def evaluate_balanced(self, board: List[List[Any]], current_player: str, 
                          rows: int, cols: int, score_cols: List[int]) -> float:
        score = 0.0
        opp = get_opponent(current_player)
        
        # Scoring area evaluation
        my_scored = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp_scored = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        
        score += my_scored * 1000
        score -= opp_scored * 1100
        
        if my_scored >= 4: return 10000
        if opp_scored >= 4: return -10000
        if my_scored == 3: score += 500
        if opp_scored == 3: score -= 600
        
        # Edge control evaluation
        edge_control = self.evaluate_edge_control(board, current_player, rows, cols, score_cols)
        score += edge_control * 100
        
        # Defensive positioning
        defensive_score = self.evaluate_defensive_position(board, current_player, rows, cols, score_cols)
        score += defensive_score * 80
        
        # Manhattan distance and positioning
        my_manhattan = self.evaluate_manhattan_distances(board, current_player, rows, cols, score_cols)
        opp_manhattan = self.evaluate_manhattan_distances(board, opp, rows, cols, score_cols)
        score += my_manhattan * 6
        score -= opp_manhattan * 6
        
        # River network evaluation
        my_rivers = self.evaluate_river_network_balanced(board, current_player, rows, cols, score_cols)
        opp_rivers = self.evaluate_river_network_balanced(board, opp, rows, cols, score_cols)
        score += my_rivers * 50
        score -= opp_rivers * 50
        
        # Stones ready to score
        my_ready = self.count_stones_ready_to_score(board, current_player, rows, cols, score_cols)
        opp_ready = self.count_stones_ready_to_score(board, opp, rows, cols, score_cols)
        score += my_ready * 150
        score -= opp_ready * 150
        
        # Balance factor - avoid being too aggressive
        balance = self.evaluate_balance_factor(board, current_player, rows, cols)
        score += balance * 30
        
        return score

    def evaluate_edge_control(self, board: List[List[Any]], player: str, 
                              rows: int, cols: int, score_cols: List[int]) -> float:
        """Evaluate control of edge columns to prevent opponent flanking"""
        edge_score = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        # Check leftmost and rightmost columns
        edge_cols = [0, 1, cols-2, cols-1]
        
        for col in edge_cols:
            # Check defensive rows (rows between player and their goal)
            if player == "circle":
                check_rows = range(target_row + 1, min(target_row + 4, rows))
            else:
                check_rows = range(max(target_row - 3, 0), target_row)
            
            for row in check_rows:
                if in_bounds(col, row, rows, cols):
                    piece = board[row][col]
                    if piece:
                        if piece.owner == player:
                            # Bonus for controlling edges
                            edge_score += 1.5
                            if piece.side == "river" and hasattr(piece, 'orientation'):
                                if piece.orientation == "horizontal":
                                    # Horizontal rivers block edge paths
                                    edge_score += 2.0
                        elif piece.owner == get_opponent(player):
                            # Penalty for opponent controlling edges
                            edge_score -= 2.0
        
        # Check for edge lanes being open
        for col in [0, cols-1]:
            lane_blocked = False
            if player == "circle":
                for row in range(target_row + 1, bottom_score_row(rows)):
                    if in_bounds(col, row, rows, cols) and board[row][col]:
                        lane_blocked = True
                        break
            else:
                for row in range(top_score_row() + 1, target_row):
                    if in_bounds(col, row, rows, cols) and board[row][col]:
                        lane_blocked = True
                        break
            
            if not lane_blocked:
                # Penalty for open edge lanes
                edge_score -= 3.0
        
        return edge_score

    def evaluate_defensive_position(self, board: List[List[Any]], player: str,
                                    rows: int, cols: int, score_cols: List[int]) -> float:
        """Evaluate defensive positioning to prevent opponent breakthrough"""
        defensive_score = 0.0
        opp = get_opponent(player)
        
        # Define defensive zone
        if player == "circle":
            defensive_rows = range(top_score_row() + 1, min(top_score_row() + 4, rows))
        else:
            defensive_rows = range(max(bottom_score_row(rows) - 3, 0), bottom_score_row(rows))
        
        # Check piece distribution across columns
        col_coverage = [False] * cols
        
        for row in defensive_rows:
            for col in range(cols):
                if in_bounds(col, row, rows, cols):
                    piece = board[row][col]
                    if piece and piece.owner == player:
                        col_coverage[col] = True
                        defensive_score += 1.0
                        
                        # Bonus for pieces in critical defensive positions
                        if col in score_cols or col in [0, cols-1]:
                            defensive_score += 1.5
                        
                        # Rivers provide better defense
                        if piece.side == "river":
                            defensive_score += 0.5
        
        # Penalty for uncovered columns
        uncovered_critical = 0
        for col in range(cols):
            if not col_coverage[col]:
                if col in score_cols:
                    uncovered_critical += 1
                    defensive_score -= 2.0
                elif col in [0, cols-1]:
                    defensive_score -= 1.5
        
        # Check for opponent threats
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == opp and piece.side == "stone":
                    # Check if opponent stone is close to scoring
                    if player == "circle":
                        if row >= bottom_score_row(rows) - 2:
                            defensive_score -= 3.0
                    else:
                        if row <= top_score_row() + 2:
                            defensive_score -= 3.0
        
        return defensive_score

    def evaluate_balance_factor(self, board: List[List[Any]], player: str, rows: int, cols: int) -> float:
        """Evaluate balance between offense and defense"""
        balance_score = 0.0
        
        # Count pieces in different zones
        offensive_pieces = 0
        defensive_pieces = 0
        mid_pieces = 0
        
        mid_row = rows // 2
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player:
                    if player == "circle":
                        if row < mid_row - 1:
                            offensive_pieces += 1
                        elif row > mid_row + 1:
                            defensive_pieces += 1
                        else:
                            mid_pieces += 1
                    else:
                        if row > mid_row + 1:
                            offensive_pieces += 1
                        elif row < mid_row - 1:
                            defensive_pieces += 1
                        else:
                            mid_pieces += 1
        
        # Ideal ratio is balanced but slightly offensive
        total = offensive_pieces + defensive_pieces + mid_pieces
        if total > 0:
            off_ratio = offensive_pieces / total
            def_ratio = defensive_pieces / total
            mid_ratio = mid_pieces / total
            
            # Penalize extreme distributions
            if off_ratio > 0.7:  # Too aggressive
                balance_score -= 5.0
            elif def_ratio > 0.6:  # Too defensive
                balance_score -= 3.0
            elif off_ratio < 0.2:  # Not offensive enough
                balance_score -= 4.0
            else:
                # Reward balanced distribution
                balance_score += 3.0
            
            # Bonus for controlling middle
            balance_score += mid_ratio * 5.0
        
        return balance_score

    def evaluate_river_network_balanced(self, board: List[List[Any]], player: str,
                                        rows: int, cols: int, score_cols: List[int]) -> float:
        """Evaluate river network with focus on both offense and defense"""
        score = 0.0
        rivers = []
        
        for r in range(rows):
            for c in range(cols):
                p = board[r][c]
                if p and p.owner == player and p.side == "river":
                    rivers.append((r, c, p))
        
        # Evaluate river connectivity
        for i, (r1, c1, p1) in enumerate(rivers):
            for j, (r2, c2, p2) in enumerate(rivers[i+1:], i+1):
                dist = abs(r1 - r2) + abs(c1 - c2)
                if dist <= 3:
                    score += 1.0
                    # Bonus for orthogonal rivers (better coverage)
                    if hasattr(p1, 'orientation') and hasattr(p2, 'orientation'):
                        if p1.orientation != p2.orientation:
                            score += 0.5
        
        # Evaluate defensive river placement
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        for r, c, p in rivers:
            # Rivers in defensive positions
            if player == "circle":
                if r > target_row and r < target_row + 3:
                    score += 2.0
                    if hasattr(p, 'orientation') and p.orientation == "horizontal":
                        score += 1.0  # Horizontal rivers better for defense
            else:
                if r < target_row and r > target_row - 3:
                    score += 2.0
                    if hasattr(p, 'orientation') and p.orientation == "horizontal":
                        score += 1.0
            
            # Rivers on edges for blocking
            if c in [0, 1, cols-2, cols-1]:
                score += 1.5
        
        return score

    def evaluate_manhattan_distances(self, board: List[List[Any]], player: str, 
                                     rows: int, cols: int, score_cols: List[int]) -> float:
        """Enhanced Manhattan distance evaluation"""
        total_score = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    # Skip if already in scoring area
                    if is_own_score_cell(col, row, player, rows, cols, score_cols):
                        continue
                    
                    # Find minimum Manhattan distance to any scoring position
                    min_dist = float('inf')
                    for score_col in score_cols:
                        dist = manhattan_distance(col, row, score_col, target_row)
                        min_dist = min(min_dist, dist)
                    
                    # Score based on proximity (closer is better)
                    if min_dist > 0:
                        distance_score = max(0, 20 - min_dist * 2)
                        total_score += distance_score
                    
                    # Additional scoring for strategic positions
                    if col in score_cols:
                        total_score += 8
                    
                    # Bonus for being close to target row
                    row_dist = abs(row - target_row)
                    if row_dist <= 2:
                        total_score += (3 - row_dist) * 4
        
        return total_score

    def count_stones_ready_to_score(self, board: List[List[Any]], player: str, 
                                    rows: int, cols: int, score_cols: List[int]) -> float:
        """Count stones that can score in one move"""
        count = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        # Check adjacent rows to scoring area
        if player == "circle":
            check_row = target_row + 1
        else:
            check_row = target_row - 1
        
        if 0 <= check_row < rows:
            for col in score_cols:
                if in_bounds(col, check_row, rows, cols):
                    piece = board[check_row][col]
                    if piece and piece.owner == player and piece.side == "stone":
                        # Check if path to score is clear
                        if not board[target_row][col]:
                            count += 1.0
        
        return count

    # ==============================================================
    # ENHANCED MOVE ORDERING WITH EDGE CONTROL
    # ==============================================================

    def order_moves_with_edge_control(self, board: List[List[Any]], moves: List[Dict[str, Any]], 
                                      current_player: str, rows: int, cols: int, 
                                      score_cols: List[int]) -> List[Dict[str, Any]]:
        """Order moves with consideration for edge control and defense"""
        
        def move_priority(move):
            priority = 0
            
            # Winning move
            if self.is_winning_move(board, move, current_player, rows, cols, score_cols):
                return 10000
            
            # Killer moves
            depth = self.turn_count
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                priority += 5000
            
            # Direct scoring moves
            if move["action"] == "move" and "to" in move:
                to_col, to_row = move["to"]
                if is_own_score_cell(to_col, to_row, current_player, rows, cols, score_cols):
                    priority += 4000
            
            # Edge control moves
            if move["action"] in ["move", "push"]:
                if "to" in move:
                    to_col, to_row = move["to"]
                    # Prioritize edge control
                    if to_col in [0, 1, cols-2, cols-1]:
                        priority += 300
                        # Extra bonus for defensive edge positions
                        target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                        if current_player == "circle":
                            if to_row > target_row and to_row < target_row + 3:
                                priority += 200
                        else:
                            if to_row < target_row and to_row > target_row - 3:
                                priority += 200
            
            # River placement for defense
            if move["action"] == "flip" and "orientation" in move:
                from_col, from_row = move["from"]
                # Prioritize defensive river placement
                target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                if current_player == "circle":
                    if from_row > target_row and from_row < target_row + 3:
                        priority += 250
                        if move["orientation"] == "horizontal":
                            priority += 150  # Horizontal rivers better for defense
                else:
                    if from_row < target_row and from_row > target_row - 3:
                        priority += 250
                        if move["orientation"] == "horizontal":
                            priority += 150
                
                # Edge rivers for blocking
                if from_col in [0, 1, cols-2, cols-1]:
                    priority += 180
            
            # Smart pushing
            if move["action"] == "push":
                from_c, from_r = move['from']
                to_c, to_r = move['to']
                pushed_to_c, pushed_to_r = move['pushed_to']
                from_piece = board[from_r][from_c]
                pushed_piece = board[to_r][to_c]

                if from_piece and pushed_piece:
                    # Prioritize pushing opponent pieces, especially sideways
                    if pushed_piece.owner == get_opponent(current_player):
                        priority += 1800
                        if from_r == to_r:  # Horizontal push is good for clearing lanes
                            priority += 400
                    # Prioritize pushing own pieces towards the goal
                    elif pushed_piece.owner == current_player:
                        target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                        old_dist = abs(to_r - target_row)
                        new_dist = abs(pushed_to_r - target_row)
                        if new_dist < old_dist:
                            priority += 1200
            
            # Prioritize moves that progress towards the goal
            if move["action"] == "move":
                target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                from_row = move["from"][1]
                to_row = move["to"][1]
                old_dist = abs(from_row - target_row)
                new_dist = abs(to_row - target_row)
                if new_dist < old_dist:
                    priority += (old_dist - new_dist) * 150

            return priority
        
        return sorted(moves, key=move_priority, reverse=True)

    # ==============================================================
    # HELPER AND STRATEGY FUNCTIONS
    # ==============================================================

    def get_game_phase(self, board: List[List[Any]], rows: int, cols: int) -> str:
        """Determines the current phase of the game."""
        my_stones = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols_for(cols))
        opp_stones = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols_for(cols))
        total_scored = my_stones + opp_stones
        
        if total_scored >= 5: return "endgame"
        if total_scored >= 2: return "midgame"
        return "opening"

    def count_stones_in_score_area(self, board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
        """Counts the number of stones a player has in their scoring area."""
        count = 0
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        for x in score_cols:
            if in_bounds(x, score_row, rows, cols):
                p = board[score_row][x]
                if p and p.owner == player and p.side == "stone":
                    count += 1
        return count

    def is_winning_move(self, board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
        """Checks if a move results in a win."""
        if move['action'] not in ['move', 'push']:
            return False
        temp_board = self.apply_move(board, move, player)
        return self.count_stones_in_score_area(temp_board, player, rows, cols, score_cols) >= 4

    def find_blocking_move(self, board: List[List[Any]], my_moves: List[Dict[str, Any]], opp_winning_move: Dict[str, Any], rows: int, cols: int, score_cols: List[int]) -> Optional[Dict[str, Any]]:
        """Finds a move to block an opponent's winning move."""
        target_pos = None
        if 'to' in opp_winning_move:
            target_pos = tuple(opp_winning_move['to'])
        elif 'pushed_to' in opp_winning_move:
             target_pos = tuple(opp_winning_move['pushed_to'])

        if not target_pos:
            return None

        # Find any move that places a piece in the target position
        for move in my_moves:
            if move.get('action') == 'move' and 'to' in move and tuple(move.get('to')) == target_pos:
                return move
            if move.get('action') == 'push' and 'pushed_to' in move and tuple(move.get('pushed_to')) == target_pos:
                return move
        
        return None # Could not find a direct block

    def apply_move(self, board: List[List[Any]], move: Dict[str, Any], player: str) -> List[List[Any]]:
        """Applies a move to a copy of the board and returns the new board."""
        new_board = copy.deepcopy(board)
        fc, fr = move["from"]
        piece = new_board[fr][fc]
        action = move["action"]

        if action == "move":
            tc, tr = move["to"]
            new_board[tr][tc] = piece
            new_board[fr][fc] = None
        elif action == "push":
            tc, tr = move["to"]
            ptc, ptr = move["pushed_to"]
            pushed_piece = new_board[tr][tc]
            new_board[ptr][ptc] = pushed_piece
            new_board[tr][tc] = piece
            new_board[fr][fc] = None
            if piece.side == 'river':
                piece.side = 'stone'
                piece.orientation = None
        elif action == "flip":
            if piece.side == "stone":
                piece.side = "river"
                piece.orientation = move["orientation"]
            else:
                piece.side = "stone"
                piece.orientation = None
        elif action == "rotate":
            if hasattr(piece, 'orientation'):
                piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
        
        return new_board
    
    def get_all_valid_moves_enhanced(self, board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """Generates all possible valid moves for a player."""
        moves = []
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if piece and piece.owner == player:
                    moves.extend(self._get_moves_for_piece(board, r, c, player, rows, cols, score_cols))
        return moves

    def _get_moves_for_piece(self, board, row, col, player, rows, cols, score_cols):
        moves = []
        piece = board[row][col]
        
        # 1. Standard moves and pushes
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if not in_bounds(nc, nr, rows, cols) or is_opponent_score_cell(nc, nr, player, rows, cols, score_cols):
                continue
            
            target_cell = board[nr][nc]
            if not target_cell:
                moves.append({"action": "move", "from": [col, row], "to": [nc, nr]})
            elif target_cell.side == "river":
                flow_dests = self._trace_river_flow(board, nr, nc, player, rows, cols, score_cols)
                for dest_r, dest_c in flow_dests:
                    moves.append({"action": "move", "from": [col, row], "to": [dest_c, dest_r]})
            elif target_cell.side == "stone":
                if piece.side == "stone":
                    pr, pc = nr + dr, nc + dc
                    if in_bounds(pc, pr, rows, cols) and not board[pr][pc] and not is_opponent_score_cell(pc, pr, target_cell.owner, rows, cols, score_cols):
                        moves.append({"action": "push", "from": [col, row], "to": [nc, nr], "pushed_to": [pc, pr]})
                else: # River pushing stone
                    push_dests = self._trace_river_push(board, nr, nc, piece, target_cell.owner, rows, cols, score_cols)
                    for dest_r, dest_c in push_dests:
                        moves.append({"action": "push", "from": [col, row], "to": [nc, nr], "pushed_to": [dest_c, dest_r]})

        # 2. Flip and Rotate moves
        if piece.side == "stone":
            moves.append({"action": "flip", "from": [col, row], "orientation": "horizontal"})
            moves.append({"action": "flip", "from": [col, row], "orientation": "vertical"})
        else:
            moves.append({"action": "flip", "from": [col, row]})
            moves.append({"action": "rotate", "from": [col, row]})
            
        return moves

    def _trace_river_flow(self, board, start_r, start_c, player, rows, cols, score_cols):
        """Traces all possible destinations from a starting river cell."""
        q = [(start_r, start_c)]
        visited_rivers = {(start_r, start_c)}
        destinations = set()

        while q:
            r, c = q.pop(0)
            river = board[r][c]
            if not river or river.side != 'river': continue
            
            dirs = [(-1, 0), (1, 0)] if river.orientation == "vertical" else [(0, -1), (0, 1)]
            for dr, dc in dirs:
                cr, cc = r + dr, c + dc
                while in_bounds(cc, cr, rows, cols):
                    if is_opponent_score_cell(cc, cr, player, rows, cols, score_cols):
                        break
                    
                    cell = board[cr][cc]
                    if not cell:
                        destinations.add((cr, cc))
                    else:
                        if cell.side == "river" and (cr, cc) not in visited_rivers:
                            visited_rivers.add((cr, cc))
                            q.append((cr, cc))
                        break
                    cr, cc = cr + dr, cc + dc
        return list(destinations)

    def _trace_river_push(self, board, target_r, target_c, river_piece, pushed_player, rows, cols, score_cols):
        """Traces possible destinations for a piece being pushed by a river."""
        destinations = set()
        dirs = [(-1, 0), (1, 0)] if river_piece.orientation == "vertical" else [(0, -1), (0, 1)]

        for dr, dc in dirs:
            cr, cc = target_r + dr, target_c + dc
            while in_bounds(cc, cr, rows, cols):
                if is_opponent_score_cell(cc, cr, pushed_player, rows, cols, score_cols):
                    break
                
                cell = board[cr][cc]
                if not cell:
                    destinations.add((cr, cc))
                elif cell.side != "river":
                    break
                cr, cc = cr + dr, cc + dc
        return list(destinations)