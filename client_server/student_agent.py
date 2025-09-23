import time
import random
import copy
from typing import List, Dict, Any, Optional, Tuple

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


class BaseAgent:
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)

    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.transposition_table = {}
        self.killer_moves = {}  # Store killer moves for move ordering
        self.history_table = {}  # History heuristic for move ordering

    # ==============================================================
    # MAIN SEARCH FUNCTION
    # ==============================================================

    def choose(self, game_state: Dict[str, Any], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        
        if isinstance(game_state, dict) and "board" in game_state:
            board = game_state["board"]
        else:
            board = game_state

        start_time = time.time()
        
        # Aggressive time management - use less time per move
        time_limit = min(current_player_time / 30.0, 1.0)  # More aggressive time division
        
        # Get valid moves
        moves = self.get_all_valid_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None
        
        # Check for immediate win
        for move in moves:
            if self.is_winning_move(board, move, self.player, rows, cols, score_cols):
                return move
        
        # Check if we need to block opponent's win
        opp_moves = self.get_all_valid_moves(board, self.opponent, rows, cols, score_cols)
        critical_defensive_move = None
        for opp_move in opp_moves:
            if self.is_winning_move(board, opp_move, self.opponent, rows, cols, score_cols):
                # Find a move that prevents this
                for our_move in moves:
                    new_board = self.apply_move(board, our_move, self.player)
                    opp_stones = self.count_stones_in_score_area(new_board, self.opponent, rows, cols, score_cols)
                    if opp_stones < 3:  # Successfully blocks
                        critical_defensive_move = our_move
                        break
                if critical_defensive_move:
                    return critical_defensive_move
        
        # Dynamic depth based on time and game phase
        game_phase = self.get_game_phase(board, rows, cols)
        if current_player_time < 5:
            max_depth = 1
        elif current_player_time < 10:
            max_depth = 2
        elif game_phase == "endgame":
            max_depth = 4
        elif game_phase == "midgame":
            max_depth = 3
        else:  # opening
            max_depth = 2
        
        # Prioritize promising moves early
        moves = self.order_moves_simple(board, moves, self.player, rows, cols, score_cols)
        
        # Only consider top moves to save time
        max_moves_to_consider = min(10 if current_player_time > 20 else 5, len(moves))
        moves = moves[:max_moves_to_consider]
        
        best_move = moves[0]
        best_score = float('-inf')
        
        # Iterative deepening with aggressive pruning
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit * 0.8:
                break
                
            alpha = float('-inf')
            beta = float('inf')
            
            for move in moves:
                if time.time() - start_time > time_limit * 0.9:
                    break
                    
                new_board = self.apply_move(board, move, self.player)
                score = -self.negamax(new_board, depth - 1, -beta, -alpha, 
                                     self.opponent, start_time, time_limit, 
                                     rows, cols, score_cols, depth)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
        
        return best_move

    def negamax(self, board: List[List[Any]], depth: int, alpha: float, beta: float,
                current_player: str, start_time: float, time_limit: float, 
                rows: int, cols: int, score_cols: List[int], max_depth: int) -> float:
        
        # Time check
        if time.time() - start_time > time_limit:
            return self.fast_evaluate(board, current_player, rows, cols, score_cols)
        
        # Terminal node check
        my_stones = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp = get_opponent(current_player)
        opp_stones = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        
        if my_stones >= 4:
            return 10000 - (max_depth - depth) * 10
        if opp_stones >= 4:
            return -10000 + (max_depth - depth) * 10
        
        if depth <= 0:
            return self.fast_evaluate(board, current_player, rows, cols, score_cols)
        
        # Get moves and prune aggressively
        moves = self.get_all_valid_moves(board, current_player, rows, cols, score_cols)
        if not moves:
            return self.fast_evaluate(board, current_player, rows, cols, score_cols)
        
        # Order moves for better pruning
        moves = self.order_moves_simple(board, moves, current_player, rows, cols, score_cols)
        
        # Only consider top moves at deeper levels
        if depth < 2:
            moves = moves[:5]
        else:
            moves = moves[:8]
        
        best_score = float('-inf')
        
        for move in moves:
            new_board = self.apply_move(board, move, current_player)
            score = -self.negamax(new_board, depth - 1, -beta, -alpha,
                                opp, start_time, time_limit,
                                rows, cols, score_cols, max_depth)
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:  # Beta cutoff
                break
        
        return best_score

    # ==============================================================
    # SIMPLIFIED AND FASTER EVALUATION
    # ==============================================================

    def fast_evaluate(self, board: List[List[Any]], current_player: str, 
                     rows: int, cols: int, score_cols: List[int]) -> float:
        """Streamlined evaluation focusing on key winning factors"""
        score = 0.0
        opp = get_opponent(current_player)
        
        # 1. Stones in scoring area (most important)
        my_scored = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp_scored = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        score += my_scored * 1000
        score -= opp_scored * 1000
        
        # 2. Immediate threats and opportunities
        if my_scored == 3:
            score += 500  # One away from winning
        if opp_scored == 3:
            score -= 600  # Opponent one away from winning
        
        # 3. Stones ready to score (one move away)
        my_ready = self.count_stones_ready_to_score(board, current_player, rows, cols, score_cols)
        opp_ready = self.count_stones_ready_to_score(board, opp, rows, cols, score_cols)
        score += my_ready * 50
        score -= opp_ready * 50
        
        # 4. Simple position evaluation
        score += self.evaluate_position_simple(board, current_player, rows, cols, score_cols)
        
        return score

    def count_stones_ready_to_score(self, board, player, rows, cols, score_cols):
        """Optimized version that only checks stones that can score in one move"""
        count = 0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        # Check adjacent rows
        check_rows = [target_row + 1] if player == "circle" else [target_row - 1]
        
        for row in check_rows:
            if 0 <= row < rows:
                for col in score_cols:
                    piece = board[row][col]
                    if piece and piece.owner == player and piece.side == "stone":
                        count += 1
        
        # Check pieces that can use rivers to reach
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    # Quick check if near a river that points to score area
                    if abs(row - target_row) <= 3:  # Only check nearby pieces
                        count += 0.3  # Partial credit for being close
        
        return count

    def evaluate_position_simple(self, board, player, rows, cols, score_cols):
        """Simplified position evaluation"""
        score = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        center_col = cols // 2
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player:
                    if piece.side == "stone":
                        # Distance to target
                        distance = abs(row - target_row)
                        score += (10 - distance) * 2
                        
                        # Bonus for being in score columns
                        if col in score_cols:
                            score += 5
                        
                        # Slight center preference
                        score -= abs(col - center_col) * 0.1
                    else:  # river
                        # Rivers are valuable for mobility
                        score += 2
                        # Vertical rivers more valuable for forward movement
                        if player == "circle" and getattr(piece, 'orientation', None) == 'vertical':
                            score += 1
                        elif player == "square" and getattr(piece, 'orientation', None) == 'vertical':
                            score += 1
        
        return score

    def is_winning_move(self, board, move, player, rows, cols, score_cols):
        """Check if a move wins immediately"""
        if move["action"] == "move" and "to" in move:
            to_col, to_row = move["to"]
            if is_own_score_cell(to_col, to_row, player, rows, cols, score_cols):
                new_board = self.apply_move(board, move, player)
                return self.count_stones_in_score_area(new_board, player, rows, cols, score_cols) >= 4
        return False

    def get_game_phase(self, board, rows, cols):
        """Determine game phase for depth adjustment"""
        total_pieces = sum(1 for r in range(rows) for c in range(cols) if board[r][c])
        if total_pieces < 18:
            return "opening"
        elif total_pieces < 22:
            return "midgame"
        else:
            return "endgame"

    def order_moves_simple(self, board, moves, current_player, rows, cols, score_cols):
        """Simple but effective move ordering"""
        def move_priority(move):
            priority = 0
            
            # Highest priority: moves to score area
            if move["action"] == "move" and "to" in move:
                to_col, to_row = move["to"]
                if is_own_score_cell(to_col, to_row, current_player, rows, cols, score_cols):
                    priority += 1000
                    
            # High priority: pushes that might score
            if move["action"] == "push":
                priority += 100
                
            # Medium priority: moves toward score area
            if move["action"] == "move" and "to" in move:
                to_col, to_row = move["to"]
                target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                distance = abs(to_row - target_row)
                priority += (10 - distance) * 10
                
            # Small randomization to avoid predictability
            priority += random.random() * 0.1
            
            return priority
        
        moves.sort(key=move_priority, reverse=True)
        return moves

    def count_stones_in_score_area(self, board, player, rows, cols, score_cols):
        """Count stones in score area"""
        count = 0
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        for col in score_cols:
            if in_bounds(col, score_row, rows, cols):
                piece = board[score_row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    count += 1
        return count

    # ==============================================================
    # MOVE GENERATION & UTILITIES (keep existing but optimize)
    # ==============================================================

    def get_all_valid_moves(self, board: List[List[Any]], current_player: str,
                           rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves = []
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == current_player:
                    piece_moves = self.get_moves_for_piece(board, row, col, current_player, rows, cols, score_cols)
                    moves.extend(piece_moves)
        return moves

    def get_moves_for_piece(self, board: List[List[Any]], row: int, col: int, current_player: str,
                           rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves = []
        piece = board[row][col]
        if not piece or piece.owner != current_player:
            return moves

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if not in_bounds(new_col, new_row, rows, cols):
                continue
            if is_opponent_score_cell(new_col, new_row, current_player, rows, cols, score_cols):
                continue
            target = board[new_row][new_col]
            if not target:
                moves.append({"action": "move", "from": [col, row], "to": [new_col, new_row]})
            elif target.side == "river":
                destinations = self.trace_river_flow(board, new_row, new_col, row, col,
                                                    current_player, rows, cols, score_cols)
                for dest_row, dest_col in destinations:
                    moves.append({"action": "move", "from": [col, row], "to": [dest_col, dest_row]})
            elif target.side == "stone":
                if piece.side == "stone":
                    push_row, push_col = new_row + dr, new_col + dc
                    if (in_bounds(push_col, push_row, rows, cols) and
                        not board[push_row][push_col] and
                        not is_opponent_score_cell(push_col, push_row, target.owner, rows, cols, score_cols)):
                        moves.append({"action": "push", "from": [col, row], "to": [new_col, new_row],
                                      "pushed_to": [push_col, push_row]})
                else:
                    destinations = self.trace_river_flow(board, new_row, new_col, row, col,
                                                        target.owner, rows, cols, score_cols)
                    for dest_row, dest_col in destinations:
                        moves.append({"action": "push", "from": [col, row], "to": [new_col, new_row],
                                      "pushed_to": [dest_col, dest_row]})

        if piece.side == "stone":
            moves.append({"action": "flip", "from": [col, row], "orientation": "horizontal"})
            moves.append({"action": "flip", "from": [col, row], "orientation": "vertical"})
        else:
            moves.append({"action": "flip", "from": [col, row]})
            moves.append({"action": "rotate", "from": [col, row]})
        return moves

    def trace_river_flow(self, board, start_row, start_col, origin_row, origin_col,
                         moving_player, rows, cols, score_cols):
        destinations = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = start_row + dr, start_col + dc
            if (in_bounds(new_col, new_row, rows, cols) and
                not board[new_row][new_col] and
                not is_opponent_score_cell(new_col, new_row, moving_player, rows, cols, score_cols)):
                destinations.append((new_row, new_col))
        return destinations

    def apply_move(self, board, move, current_player):
        new_board = copy.deepcopy(board)
        from_col, from_row = move["from"]

        if move["action"] == "move":
            to_col, to_row = move["to"]
            new_board[to_row][to_col] = new_board[from_row][from_col]
            new_board[from_row][from_col] = None
        elif move["action"] == "push":
            to_col, to_row = move["to"]
            push_col, push_row = move["pushed_to"]
            new_board[push_row][push_col] = new_board[to_row][to_col]
            new_board[to_row][to_col] = new_board[from_row][from_col]
            new_board[from_row][from_col] = None
            if new_board[to_row][to_col].side == "river":
                new_board[to_row][to_col].side = "stone"
                if hasattr(new_board[to_row][to_col], "orientation"):
                    new_board[to_row][to_col].orientation = None
        elif move["action"] == "flip":
            piece = new_board[from_row][from_col]
            if piece.side == "stone":
                piece.side = "river"
                piece.orientation = move.get("orientation", None)
            else:
                piece.side = "stone"
                piece.orientation = None
        elif move["action"] == "rotate":
            piece = new_board[from_row][from_col]
            if piece.side == "river":
                piece.orientation = "vertical" if piece.orientation == "horizontal" else "horizontal"
        return new_board

    def order_moves(self, board, moves, current_player, rows, cols, score_cols):
        def move_score(move):
            if move["action"] == "move":
                to_col, to_row = move["to"]
                if is_own_score_cell(to_col, to_row, current_player, rows, cols, score_cols):
                    return 1000
            return random.random()
        moves.sort(key=move_score, reverse=True)

    def get_defensive_moves(self, board, moves, rows, cols, score_cols):
        defensive = []
        opp = self.opponent
        for move in moves:
            new_board = self.apply_move(board, move, self.player)
            if not self.blocks_opponent_win(new_board, opp, rows, cols, score_cols):
                defensive.append(move)
        return defensive

    def blocks_opponent_win(self, board, opp, rows, cols, score_cols):
        opp_stones = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        return opp_stones < 4
