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
    """Determines which columns are score areas based on board size"""
    if cols == 12:  # Small board
        w = 4
    elif cols == 14:  # Medium board
        w = 5
    else:  # Large board (16 cols)
        w = 6
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
# ENHANCED STUDENT AGENT WITH OPTIMAL EVALUATION
# ===================================================================
class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_table = {}
        self.turn_count = 0
        self.board_history_set = set()
        self.recent_positions = []
        self.last_moves = []
        self.MAX_HISTORY_SIZE = 20
        self.MAX_RECENT_MOVES = 5
        
        # Board-specific configurations (set dynamically)
        self.board_size = None
        self.SA_required = None
        self.weights = None

    def initialize_board_config(self, cols: int):
        """Initialize board-specific settings"""
        if self.board_size == cols:
            return  # Already initialized
            
        self.board_size = cols
        
        # Set required stones to win based on board size
        if cols == 12:  # Small board 13x12
            self.SA_required = 4
            self.weights = {
                'stone_in_SA': 1000,
                'stone_1_away': 200,
                'stone_2_away': 50,
                'distance_to_SA': 15,
                'river_path_value': 30,
                'river_nearby_stones': 10,
                'river_direction': 25,
                'opponent_stone_in_SA': -1200,
                'opponent_1_away': -250,
                'opponent_2_away': -60,
                'opponent_distance': -12,
                'blocking': 15,
                'mobility': 2,
                'push_opportunity': 8
            }
        elif cols == 14:  # Medium board 15x14
            self.SA_required = 5
            self.weights = {
                'stone_in_SA': 1000,
                'stone_1_away': 180,
                'stone_2_away': 45,
                'distance_to_SA': 18,
                'river_path_value': 35,
                'river_nearby_stones': 12,
                'river_direction': 30,
                'opponent_stone_in_SA': -1200,
                'opponent_1_away': -260,
                'opponent_2_away': -65,
                'opponent_distance': -15,
                'blocking': 18,
                'mobility': 3,
                'push_opportunity': 10
            }
        else:  # Large board 17x16
            self.SA_required = 6
            self.weights = {
                'stone_in_SA': 1000,
                'stone_1_away': 160,
                'stone_2_away': 40,
                'distance_to_SA': 20,
                'river_path_value': 40,
                'river_nearby_stones': 15,
                'river_direction': 35,
                'opponent_stone_in_SA': -1200,
                'opponent_1_away': -270,
                'opponent_2_away': -70,
                'opponent_distance': -18,
                'blocking': 20,
                'mobility': 4,
                'push_opportunity': 12
            }

    # ==============================================================
    # MAIN SEARCH FUNCTION
    # ==============================================================
    def choose(self, game_state: Dict[str, Any], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        
        self.turn_count += 1
        
        # Initialize board configuration
        self.initialize_board_config(cols)
        
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
        # BALANCED VERSION - Use more time early, be aggressive
        if current_player_time > 40:
            time_limit = 2.5  # Use time early to find good strategy
        elif current_player_time > 25:
            time_limit = 2.0  # Still thinking deeply
        elif current_player_time > 15:
            time_limit = 1.2  # Solid thinking
        elif current_player_time > 8:
            time_limit = 0.8  # Moderate thinking
        else:
            time_limit = max(0.3, current_player_time / 20.0)  # Quick but reasonable

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
        game_phase = self.get_game_phase(board, rows, cols, score_cols)
        
        # Adjust depth based on time and game phase
        # BALANCED VERSION - Works on MacBook Air but plays stronger
        if current_player_time < 5:
            max_depth = 2  # Still think ahead when low on time
        elif current_player_time < 12:
            max_depth = 3  # Better mid-game
        elif game_phase == "endgame":
            max_depth = 4  # Push to win in endgame
        elif game_phase == "midgame":
            max_depth = 3  # Solid midgame
        else:
            max_depth = 2  # Reasonable opening
        
        # Order moves intelligently
        moves = self.order_moves_intelligently(board, moves, self.player, rows, cols, score_cols, game_phase)
        
        # Limit branching factor based on time
        # BALANCED VERSION - Explore enough moves to find wins
        if current_player_time > 20:
            max_moves = min(8, len(moves))  # Explore more when time permits
        elif current_player_time > 10:
            max_moves = min(6, len(moves))  # Good balance
        else:
            max_moves = min(5, len(moves))  # Still reasonable when rushed
        
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
            current_best_score = float('-inf')
            
            for i, move in enumerate(moves):
                if time.time() - start_time > time_limit:
                    break
                    
                new_board = self.apply_move(board, move, self.player)
                
                # Check if this creates a repeated position
                new_hash = board_hash(new_board)
                if new_hash in self.board_history_set:
                    continue
                
                score = -self.alpha_beta(
                    new_board, depth - 1, -beta, -alpha, 
                    self.opponent, rows, cols, score_cols, 
                    start_time, time_limit
                )
                
                if score > current_best_score:
                    current_best_score = score
                    current_best = move
                    
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            
            if current_best and current_best_score > best_score:
                best_score = current_best_score
                best_move = current_best
        
        self.update_move_history(best_move)
        return best_move

    # ==============================================================
    # ALPHA-BETA SEARCH
    # ==============================================================
    def alpha_beta(self, board: List[List[Any]], depth: int, alpha: float, beta: float,
                   player: str, rows: int, cols: int, score_cols: List[int],
                   start_time: float, time_limit: float) -> float:
        
        # Time check
        if time.time() - start_time > time_limit:
            return self.evaluate_board_enhanced(board, self.player, rows, cols, score_cols)
        
        # Check terminal states
        my_stones = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols)
        opp_stones = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols)
        
        if my_stones >= self.SA_required:
            return 999999 if player == self.player else -999999
        if opp_stones >= self.SA_required:
            return -999999 if player == self.player else 999999
        
        # Depth limit reached
        if depth == 0:
            eval_score = self.evaluate_board_enhanced(board, self.player, rows, cols, score_cols)
            return eval_score if player == self.player else -eval_score
        
        # Transposition table lookup
        state_key = board_hash(board)
        if state_key in self.transposition_table:
            cached = self.transposition_table[state_key]
            if cached['depth'] >= depth:
                return cached['score'] if player == self.player else -cached['score']
        
        # Generate and order moves
        moves = self.get_all_valid_moves_enhanced(board, player, rows, cols, score_cols)
        if not moves:
            eval_score = self.evaluate_board_enhanced(board, self.player, rows, cols, score_cols)
            return eval_score if player == self.player else -eval_score
        
        # Quick move ordering
        # BALANCED VERSION - Explore enough to find good moves
        moves = self.quick_move_order(board, moves, player, rows, cols, score_cols)[:6]
        
        best_score = float('-inf')
        
        for move in moves:
            new_board = self.apply_move(board, move, player)
            score = -self.alpha_beta(
                new_board, depth - 1, -beta, -alpha,
                get_opponent(player), rows, cols, score_cols,
                start_time, time_limit
            )
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Store killer move
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                self.killer_moves[depth].append(move)
                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop(0)
                break
        
        # Store in transposition table
        self.transposition_table[state_key] = {
            'score': best_score if player == self.player else -best_score,
            'depth': depth
        }
        
        return best_score

    # ==============================================================
    # ENHANCED EVALUATION FUNCTION - THE SECRET SAUCE
    # ==============================================================
    def evaluate_board_enhanced(self, board: List[List[Any]], player: str, 
                               rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Comprehensive evaluation function that considers:
        1. Stones in scoring area (direct goal)
        2. Distance of stones to scoring area
        3. Stones close to scoring area (1-2 moves away)
        4. River network value (highways to SA)
        5. Opponent progress and threats
        6. Tactical opportunities (mobility, blocking)
        """
        score = 0.0
        w = self.weights
        
        # === PART 1: SCORING FEATURES (60% weight) ===
        
        # 1.1 Stones already in SA (critical)
        my_stones_SA = self.count_stones_in_score_area(board, player, rows, cols, score_cols)
        score += my_stones_SA * w['stone_in_SA']
        
        # 1.2 Stones very close to SA (high priority)
        my_stones_1_away = self.count_stones_n_moves_from_SA(board, player, 1, rows, cols, score_cols)
        my_stones_2_away = self.count_stones_n_moves_from_SA(board, player, 2, rows, cols, score_cols)
        score += my_stones_1_away * w['stone_1_away']
        score += my_stones_2_away * w['stone_2_away']
        
        # 1.3 Average distance of all stones to SA
        avg_distance = self.calculate_average_distance_to_SA(board, player, rows, cols, score_cols)
        if avg_distance > 0:
            max_possible_dist = rows + cols
            proximity_score = (max_possible_dist - avg_distance) * w['distance_to_SA']
            score += proximity_score
        
        # === PART 2: RIVER NETWORK FEATURES (25% weight) ===
        
        river_score = self.evaluate_river_network(board, player, rows, cols, score_cols)
        score += river_score
        
        # === PART 3: OPPONENT FEATURES (15% weight) ===
        
        # 3.1 Opponent stones in their SA (very bad)
        opp_stones_SA = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols)
        score += opp_stones_SA * w['opponent_stone_in_SA']
        
        # 3.2 Opponent proximity threats
        opp_stones_1_away = self.count_stones_n_moves_from_SA(board, self.opponent, 1, rows, cols, score_cols)
        opp_stones_2_away = self.count_stones_n_moves_from_SA(board, self.opponent, 2, rows, cols, score_cols)
        score += opp_stones_1_away * w['opponent_1_away']
        score += opp_stones_2_away * w['opponent_2_away']
        
        # 3.3 Opponent average distance (inverse value)
        opp_avg_distance = self.calculate_average_distance_to_SA(board, self.opponent, rows, cols, score_cols)
        if opp_avg_distance > 0:
            max_possible_dist = rows + cols
            # Penalize if opponent is close
            opp_proximity_penalty = (max_possible_dist - opp_avg_distance) * w['opponent_distance']
            score += opp_proximity_penalty
        
        # === PART 4: TACTICAL FEATURES ===
        
        # 4.1 Mobility
        my_moves = len(self.get_all_valid_moves_enhanced(board, player, rows, cols, score_cols))
        score += min(my_moves, 20) * w['mobility']
        
        # 4.2 Blocking opportunities
        blocking_value = self.calculate_blocking_value(board, player, rows, cols, score_cols)
        score += blocking_value * w['blocking']
        
        return score

    def evaluate_river_network(self, board: List[List[Any]], player: str,
                               rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Evaluate the strategic value of the river network.
        This is what separates good bots from great ones.
        """
        score = 0.0
        w = self.weights
        
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if not piece or piece.owner != player or piece.side != "river":
                    continue
                
                # Value 1: Direction toward opponent's SA
                direction_value = self.calculate_river_direction_value(
                    piece, c, r, player, rows, cols, score_cols
                )
                score += direction_value * w['river_direction']
                
                # Value 2: Path length (how far can you travel?)
                path_length = self.calculate_river_path_length(board, r, c, player, rows, cols, score_cols)
                score += min(path_length * w['river_path_value'], 100)
                
                # Value 3: Nearby stones that can use this river
                nearby_stones = self.count_nearby_player_stones(board, r, c, player, rows, cols, radius=2)
                score += nearby_stones * w['river_nearby_stones']
                
                # Value 4: River chains (connected rivers = exponential value)
                if self.is_part_of_river_chain(board, r, c, player, rows, cols):
                    score += 50  # Bonus for being part of a chain
        
        return score

    def calculate_river_direction_value(self, river_piece, x: int, y: int, player: str,
                                       rows: int, cols: int, score_cols: List[int]) -> float:
        """
        Calculate if this river points toward opponent's SA.
        Returns value from -10 (points away) to +10 (points directly toward)
        """
        # Determine opponent's SA location
        if player == "circle":
            target_row = bottom_score_row(rows)
            direction_needed = 1 if y < target_row else -1  # Need to go down
        else:
            target_row = top_score_row()
            direction_needed = -1 if y > target_row else 1  # Need to go up
        
        # Check river orientation
        if river_piece.orientation == "vertical":
            # Vertical rivers help with up/down movement
            if (player == "circle" and y < target_row) or \
               (player == "square" and y > target_row):
                return 10.0  # Perfect direction
            else:
                return -5.0  # Wrong direction
        else:
            # Horizontal rivers help positionally
            # Check if we're in the right column range
            if x in score_cols or abs(x - score_cols[len(score_cols)//2]) <= 2:
                return 5.0  # Good positioning
            else:
                return 2.0  # Neutral
    
    def calculate_river_path_length(self, board: List[List[Any]], r: int, c: int,
                                    player: str, rows: int, cols: int, score_cols: List[int]) -> int:
        """Calculate how many spaces you can travel on this river"""
        piece = board[r][c]
        if not piece or piece.side != "river":
            return 0
        
        if piece.orientation == "vertical":
            dirs = [(-1, 0), (1, 0)]  # up, down
        else:
            dirs = [(0, -1), (0, 1)]  # left, right
        
        total_length = 0
        
        for dr, dc in dirs:
            cr, cc = r + dr, c + dc
            length = 0
            while in_bounds(cc, cr, rows, cols):
                if is_opponent_score_cell(cc, cr, player, rows, cols, score_cols):
                    break
                    
                cell = board[cr][cc]
                if not cell:
                    length += 1
                elif cell.side == "river" and cell.owner == player:
                    length += 1  # Can continue on connected river
                else:
                    break
                    
                cr, cc = cr + dr, cc + dc
            
            total_length += length
        
        return total_length

    def count_nearby_player_stones(self, board: List[List[Any]], r: int, c: int,
                                   player: str, rows: int, cols: int, radius: int = 2) -> int:
        """Count player's stones within radius of position"""
        count = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if in_bounds(nc, nr, rows, cols):
                    cell = board[nr][nc]
                    if cell and cell.owner == player and cell.side == "stone":
                        count += 1
        return count

    def is_part_of_river_chain(self, board: List[List[Any]], r: int, c: int,
                               player: str, rows: int, cols: int) -> bool:
        """Check if this river connects to another player's river"""
        piece = board[r][c]
        if not piece or piece.side != "river":
            return False
        
        # Check adjacent cells for connected rivers
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if in_bounds(nc, nr, rows, cols):
                neighbor = board[nr][nc]
                if neighbor and neighbor.owner == player and neighbor.side == "river":
                    return True
        
        return False

    def count_stones_n_moves_from_SA(self, board: List[List[Any]], player: str, n_moves: int,
                                     rows: int, cols: int, score_cols: List[int]) -> int:
        """Count stones that are exactly n moves away from their SA"""
        count = 0
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if piece and piece.owner == player and piece.side == "stone":
                    if not is_own_score_cell(c, r, player, rows, cols, score_cols):
                        # Calculate minimum Manhattan distance to any SA cell
                        min_dist = float('inf')
                        for sc in score_cols:
                            dist = manhattan_distance(c, r, sc, score_row)
                            min_dist = min(min_dist, dist)
                        
                        if min_dist == n_moves:
                            count += 1
        
        return count

    def calculate_average_distance_to_SA(self, board: List[List[Any]], player: str,
                                         rows: int, cols: int, score_cols: List[int]) -> float:
        """Calculate weighted average distance of player's stones to their SA"""
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        total_distance = 0
        stone_count = 0
        
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if piece and piece.owner == player and piece.side == "stone":
                    if not is_own_score_cell(c, r, player, rows, cols, score_cols):
                        # Find minimum distance to any SA cell
                        min_dist = float('inf')
                        for sc in score_cols:
                            dist = manhattan_distance(c, r, sc, score_row)
                            min_dist = min(min_dist, dist)
                        
                        total_distance += min_dist
                        stone_count += 1
        
        if stone_count == 0:
            return 0
        
        return total_distance / stone_count

    def calculate_blocking_value(self, board: List[List[Any]], player: str,
                                 rows: int, cols: int, score_cols: List[int]) -> float:
        """Calculate value of blocking opponent's progress"""
        score = 0.0
        opponent = get_opponent(player)
        opp_score_row = top_score_row() if opponent == "circle" else bottom_score_row(rows)
        
        # Find opponent stones close to their SA
        for r in range(rows):
            for c in range(cols):
                piece = board[r][c]
                if piece and piece.owner == opponent and piece.side == "stone":
                    dist_to_SA = float('inf')
                    for sc in score_cols:
                        dist = manhattan_distance(c, r, sc, opp_score_row)
                        dist_to_SA = min(dist_to_SA, dist)
                    
                    # If opponent stone is close, check if we can block it
                    if dist_to_SA <= 3:
                        # Check if we have pieces nearby that can interfere
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if in_bounds(nc, nr, rows, cols):
                                blocker = board[nr][nc]
                                if blocker and blocker.owner == player:
                                    score += (4 - dist_to_SA) * 5  # Closer threats get more blocking value
        
        return score

    # ==============================================================
    # INTELLIGENT MOVE ORDERING
    # ==============================================================
    def order_moves_intelligently(self, board: List[List[Any]], moves: List[Dict[str, Any]],
                                  player: str, rows: int, cols: int, score_cols: List[int],
                                  game_phase: str) -> List[Dict[str, Any]]:
        """
        Order moves to explore the best ones first.
        This dramatically improves alpha-beta pruning efficiency.
        """
        def move_priority(move):
            priority = 0
            action = move.get('action')
            fc, fr = move['from']
            
            # Highest priority: Moves that score
            if action in ['move', 'push']:
                if 'to' in move:
                    tc, tr = move['to']
                    if is_own_score_cell(tc, tr, player, rows, cols, score_cols):
                        priority += 10000
                
                if 'pushed_to' in move:
                    ptc, ptr = move['pushed_to']
                    if is_own_score_cell(ptc, ptr, player, rows, cols, score_cols):
                        priority += 10000
            
            # High priority: Moves toward SA
            if action in ['move', 'push'] and 'to' in move:
                tc, tr = move['to']
                score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
                
                # Calculate improvement in position
                old_dist = min(manhattan_distance(fc, fr, sc, score_row) for sc in score_cols)
                new_dist = min(manhattan_distance(tc, tr, sc, score_row) for sc in score_cols)
                
                if new_dist < old_dist:
                    priority += (old_dist - new_dist) * 200
            
            # Medium priority: Using rivers to advance
            if action == 'move' and 'to' in move:
                tc, tr = move['to']
                # Check if we're moving onto a river
                target = board[tr][tc] if in_bounds(tc, tr, rows, cols) else None
                if target and target.side == 'river' and target.owner == player:
                    priority += 150
            
            # Push moves are valuable
            if action == 'push':
                priority += 100
            
            # Medium priority: Flipping to create strategic rivers
            if action == 'flip':
                # In opening, creating rivers is good
                if game_phase == "opening":
                    priority += 80
            
            # Killer move heuristic
            depth_key = 3  # Check recent killer moves
            if depth_key in self.killer_moves:
                if move in self.killer_moves[depth_key]:
                    priority += 300
            
            return priority
        
        return sorted(moves, key=move_priority, reverse=True)

    def quick_move_order(self, board: List[List[Any]], moves: List[Dict[str, Any]],
                        player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """Quick move ordering for use within alpha-beta (less comprehensive)"""
        def quick_priority(move):
            action = move.get('action')
            
            # Prioritize scoring moves
            if action in ['move', 'push']:
                if 'to' in move:
                    tc, tr = move['to']
                    if is_own_score_cell(tc, tr, player, rows, cols, score_cols):
                        return 10000
                if 'pushed_to' in move:
                    ptc, ptr = move['pushed_to']
                    if is_own_score_cell(ptc, ptr, player, rows, cols, score_cols):
                        return 10000
            
            # Prioritize forward movement
            if action in ['move', 'push'] and 'to' in move:
                fc, fr = move['from']
                tc, tr = move['to']
                score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
                
                old_dist = abs(fr - score_row)
                new_dist = abs(tr - score_row)
                
                if new_dist < old_dist:
                    return (old_dist - new_dist) * 100
            
            return 0
        
        return sorted(moves, key=quick_priority, reverse=True)

    # ==============================================================
    # HELPER FUNCTIONS
    # ==============================================================
    def get_game_phase(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> str:
        """Determines the current phase of the game."""
        my_stones = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols)
        opp_stones = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols)
        total_scored = my_stones + opp_stones
        
        # Adjust thresholds based on SA_required
        endgame_threshold = max(self.SA_required - 2, 3)
        midgame_threshold = max(self.SA_required - 4, 1)
        
        if total_scored >= endgame_threshold:
            return "endgame"
        if total_scored >= midgame_threshold:
            return "midgame"
        return "opening"

    def count_stones_in_score_area(self, board: List[List[Any]], player: str,
                                   rows: int, cols: int, score_cols: List[int]) -> int:
        """Counts the number of stones a player has in their scoring area."""
        count = 0
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        for x in score_cols:
            if in_bounds(x, score_row, rows, cols):
                p = board[score_row][x]
                if p and p.owner == player and p.side == "stone":
                    count += 1
        return count

    def is_winning_move(self, board: List[List[Any]], move: Dict[str, Any], player: str,
                       rows: int, cols: int, score_cols: List[int]) -> bool:
        """Checks if a move results in a win."""
        if move['action'] not in ['move', 'push']:
            return False
        temp_board = self.apply_move(board, move, player)
        stones = self.count_stones_in_score_area(temp_board, player, rows, cols, score_cols)
        return stones >= self.SA_required  # Uses correct board-specific value

    def find_blocking_move(self, board: List[List[Any]], my_moves: List[Dict[str, Any]],
                          opp_winning_move: Dict[str, Any], rows: int, cols: int,
                          score_cols: List[int]) -> Optional[Dict[str, Any]]:
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
            if move.get('action') == 'move' and 'to' in move:
                if tuple(move.get('to')) == target_pos:
                    return move
            if move.get('action') == 'push' and 'pushed_to' in move:
                if tuple(move.get('pushed_to')) == target_pos:
                    return move
        
        return None

    def filter_non_repeating_moves(self, board: List[List[Any]], moves: List[Dict[str, Any]],
                                   rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        """Filter out moves that would create repeated board positions"""
        filtered = []
        for move in moves:
            new_board = self.apply_move(board, move, self.player)
            new_hash = board_hash(new_board)
            
            # Allow the move if it's not in recent history
            if new_hash not in self.board_history_set:
                filtered.append(move)
        
        # If all moves create repeats, return all moves (forced repetition)
        return filtered if filtered else moves

    def update_move_history(self, move: Dict[str, Any]):
        """Track recent moves to avoid repetition"""
        self.last_moves.append(move)
        if len(self.last_moves) > self.MAX_RECENT_MOVES:
            self.last_moves.pop(0)

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
            # River becomes stone after pushing
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
            if hasattr(piece, 'orientation') and piece.orientation:
                piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
        
        return new_board

    # ==============================================================
    # MOVE GENERATION
    # ==============================================================
    def get_all_valid_moves_enhanced(self, board: List[List[Any]], player: str,
                                    rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
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
                    if in_bounds(pc, pr, rows, cols) and not board[pr][pc] and \
                       not is_opponent_score_cell(pc, pr, target_cell.owner, rows, cols, score_cols):
                        moves.append({"action": "push", "from": [col, row], "to": [nc, nr], "pushed_to": [pc, pr]})
                else:  # River pushing stone
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
            if not river or river.side != 'river':
                continue
            
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