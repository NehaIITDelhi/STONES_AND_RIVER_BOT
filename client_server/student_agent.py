import time
import random
import copy
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod


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


class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], 
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        pass


class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.transposition_table = {}
        self.nodes_evaluated = 0
    
    def choose(self, game_state: Dict[str, Any], rows: int, cols: int, score_cols: List[int], 
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        
        if isinstance(game_state, dict) and "board" in game_state:
            board = game_state["board"]
        else:
            board = game_state
            
        self.nodes_evaluated = 0
        start_time = time.time()
        
        # Time management
        time_limit = min(current_player_time / 20.0, 2.0)
        
        # Get all valid moves
        moves = self.get_all_valid_moves(board, self.player, rows, cols, score_cols)
        
        if not moves:
            return None
        
        # Check for immediate win
        for move in moves:
            new_board = self.apply_move(board, move, self.player)
            if self.count_stones_in_score_area(new_board, self.player, rows, cols, score_cols) >= 4:
                return move
        
        # Check for opponent threats
        defensive_moves = self.get_defensive_moves(board, moves, rows, cols, score_cols)
        if defensive_moves:
            moves = defensive_moves
        
        # Determine search depth based on time
        max_depth = 4
        if current_player_time < 5:
            max_depth = 1
        elif current_player_time < 15:
            max_depth = 2
        elif current_player_time < 30:
            max_depth = 3
        
        # Search for best move
        best_move = moves[0]
        best_score = float('-inf')
        
        # Order moves for better search
        self.order_moves(board, moves, self.player, rows, cols, score_cols)
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit * 0.9:
                break
                
            moves_to_search = min(len(moves), 15)
            for i in range(moves_to_search):
                if time.time() - start_time > time_limit * 0.95:
                    break
                    
                new_board = self.apply_move(board, moves[i], self.player)
                score = -self.negamax(new_board, depth - 1, float('-inf'), float('inf'), 
                                    -1, start_time, time_limit, rows, cols, score_cols)
                
                if score > best_score:
                    best_score = score
                    best_move = moves[i]
        
        return best_move
    
    def negamax(self, board: List[List[Any]], depth: int, alpha: float, beta: float, 
                color: int, start_time: float, time_limit: float, rows: int, cols: int, 
                score_cols: List[int]) -> float:
        
        self.nodes_evaluated += 1
        
        # Time check
        if self.nodes_evaluated % 1000 == 0 and time.time() - start_time > time_limit:
            return color * self.evaluate_board(board, rows, cols, score_cols)
        
        # Terminal check
        player_stones = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols)
        opponent_stones = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols)
        
        if player_stones >= 4:
            return color * (100000 - depth * 10)
        if opponent_stones >= 4:
            return color * (-100000 + depth * 10)
        
        if depth <= 0:
            return color * self.evaluate_board(board, rows, cols, score_cols)
        
        # Get current player
        current_player = self.player if color == 1 else self.opponent
        
        # Generate moves
        moves = self.get_all_valid_moves(board, current_player, rows, cols, score_cols)
        if not moves:
            return color * self.evaluate_board(board, rows, cols, score_cols)
        
        self.order_moves(board, moves, current_player, rows, cols, score_cols)
        
        best_score = float('-inf')
        moves_to_search = min(len(moves), 12)
        
        for i in range(moves_to_search):
            new_board = self.apply_move(board, moves[i], current_player)
            score = -self.negamax(new_board, depth - 1, -beta, -alpha, -color, 
                                start_time, time_limit, rows, cols, score_cols)
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break
        
        return best_score
    
    def evaluate_board(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]) -> float:
        score = 0.0
        
        # Stones in scoring areas
        player_scored = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols)
        opponent_scored = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols)
        
        score += player_scored * 1000
        score -= opponent_scored * 1000
        
        # Stones one move from scoring
        player_threats = self.count_stones_one_move_from_scoring(board, self.player, rows, cols, score_cols)
        opponent_threats = self.count_stones_one_move_from_scoring(board, self.opponent, rows, cols, score_cols)
        
        score += player_threats * 100
        score -= opponent_threats * 100
        
        # Positional evaluation
        score += self.evaluate_position(board, self.player, rows, cols, score_cols) * 10
        score -= self.evaluate_position(board, self.opponent, rows, cols, score_cols) * 10
        
        return score
    
    def evaluate_position(self, board: List[List[Any]], eval_player: str, rows: int, cols: int, score_cols: List[int]) -> float:
        score = 0.0
        target_row = top_score_row() if eval_player == "circle" else bottom_score_row(rows)
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and hasattr(piece, 'owner') and piece.owner == eval_player:
                    if hasattr(piece, 'side') and piece.side == "stone":
                        # Distance to scoring area
                        distance = abs(row - target_row)
                        score += 20.0 / (1.0 + distance)
                        
                        # Bonus for being in scoring columns
                        if col in score_cols:
                            score += 15
                        
                        # Advancement bonus
                        if eval_player == "circle":
                            score += (rows - row) * 2
                        else:
                            score += row * 2
                    else:  # River
                        score += 5
                        if col in score_cols:
                            score += 3
        
        return score
    
    def get_all_valid_moves(self, board: List[List[Any]], current_player: str, 
                           rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves = []
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and hasattr(piece, 'owner') and piece.owner == current_player:
                    piece_moves = self.get_moves_for_piece(board, row, col, current_player, rows, cols, score_cols)
                    moves.extend(piece_moves)
        
        return moves
    
    def get_moves_for_piece(self, board: List[List[Any]], row: int, col: int, current_player: str,
                           rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves = []
        piece = board[row][col]
        
        if not piece or not hasattr(piece, 'owner') or piece.owner != current_player:
            return moves
        
        # Movement directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if not in_bounds(new_col, new_row, rows, cols):
                continue
            if is_opponent_score_cell(new_col, new_row, current_player, rows, cols, score_cols):
                continue
            
            target = board[new_row][new_col]
            
            if not target:  # Empty cell
                moves.append({
                    "action": "move",
                    "from": [col, row],
                    "to": [new_col, new_row]
                })
            elif hasattr(target, 'side'):
                if target.side == "river":
                    # River movement - simplified for now
                    destinations = self.trace_river_flow(board, new_row, new_col, row, col, 
                                                       current_player, rows, cols, score_cols)
                    for dest_row, dest_col in destinations:
                        moves.append({
                            "action": "move",
                            "from": [col, row],
                            "to": [dest_col, dest_row]
                        })
                elif target.side == "stone":
                    # Push mechanics
                    if hasattr(piece, 'side') and piece.side == "stone":
                        # Stone pushes stone
                        push_row, push_col = new_row + dr, new_col + dc
                        if (in_bounds(push_col, push_row, rows, cols) and 
                            not board[push_row][push_col] and
                            not is_opponent_score_cell(push_col, push_row, target.owner, rows, cols, score_cols)):
                            moves.append({
                                "action": "push",
                                "from": [col, row],
                                "to": [new_col, new_row],
                                "pushed_to": [push_col, push_row]
                            })
                    else:  # River pushes stone
                        destinations = self.trace_river_flow(board, new_row, new_col, row, col,
                                                           target.owner, rows, cols, score_cols)
                        for dest_row, dest_col in destinations:
                            moves.append({
                                "action": "push",
                                "from": [col, row],
                                "to": [new_col, new_row],
                                "pushed_to": [dest_col, dest_row]
                            })
        
        # Flip moves
        if hasattr(piece, 'side'):
            if piece.side == "stone":
                moves.append({
                    "action": "flip",
                    "from": [col, row],
                    "orientation": "horizontal"
                })
                moves.append({
                    "action": "flip",
                    "from": [col, row],
                    "orientation": "vertical"
                })
            else:  # River
                moves.append({
                    "action": "flip",
                    "from": [col, row]
                })
                moves.append({
                    "action": "rotate",
                    "from": [col, row]
                })
        
        return moves
    
    def trace_river_flow(self, board: List[List[Any]], start_row: int, start_col: int,
                        origin_row: int, origin_col: int, moving_player: str,
                        rows: int, cols: int, score_cols: List[int]) -> List[Tuple[int, int]]:
        # Simplified river flow implementation
        destinations = []
        
        # Basic implementation - just return adjacent empty cells for now
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = start_row + dr, start_col + dc
            if (in_bounds(new_col, new_row, rows, cols) and 
                not board[new_row][new_col] and
                not is_opponent_score_cell(new_col, new_row, moving_player, rows, cols, score_cols)):
                destinations.append((new_row, new_col))
        
        return destinations
    
    def apply_move(self, board: List[List[Any]], move: Dict[str, Any], current_player: str) -> List[List[Any]]:
        new_board = copy.deepcopy(board)
        from_col, from_row = move["from"]
        
        if move["action"] == "move":
            to_col, to_row = move["to"]
            new_board[to_row][to_col] = new_board[from_row][from_col]
            new_board[from_row][from_col] = None
            
        elif move["action"] == "push":
            to_col, to_row = move["to"]
            push_col, push_row = move["pushed_to"]
            
            # Move pushed piece
            new_board[push_row][push_col] = new_board[to_row][to_col]
            # Move pushing piece
            new_board[to_row][to_col] = new_board[from_row][from_col]
            new_board[from_row][from_col] = None
            
            # River becomes stone after pushing
            if (hasattr(new_board[to_row][to_col], 'side') and 
                new_board[to_row][to_col].side == "river"):
                new_board[to_row][to_col].side = "stone"
                if hasattr(new_board[to_row][to_col], 'orientation'):
                    new_board[to_row][to_col].orientation = None
                    
        elif move["action"] == "flip":
            piece = new_board[from_row][from_col]
            if hasattr(piece, 'side'):
                if piece.side == "stone":
                    piece.side = "river"
                    if "orientation" in move:
                        piece.orientation = move["orientation"]
                else:
                    piece.side = "stone"
                    piece.orientation = None
                    
        elif move["action"] == "rotate":
            piece = new_board[from_row][from_col]
            if hasattr(piece, 'side') and piece.side == "river":
                if hasattr(piece, 'orientation'):
                    piece.orientation = ("vertical" if piece.orientation == "horizontal" 
                                       else "horizontal")
        
        return new_board
    
    def order_moves(self, board: List[List[Any]], moves: List[Dict[str, Any]], 
                   current_player: str, rows: int, cols: int, score_cols: List[int]):
        def move_score(move):
            score = 0
            
            # Immediate scoring moves
            if move["action"] == "move" and "to" in move:
                to_col, to_row = move["to"]
                if is_own_score_cell(to_col, to_row, current_player, rows, cols, score_cols):
                    score += 10000
            
            # Flip to stone in scoring area
            if move["action"] == "flip":
                from_col, from_row = move["from"]
                piece = board[from_row][from_col]
                if (hasattr(piece, 'side') and piece.side == "river" and
                    is_own_score_cell(from_col, from_row, current_player, rows, cols, score_cols)):
                    score += 8000
            
            # Push moves
            if move["action"] == "push":
                score += 100
                if "pushed_to" in move:
                    push_col, push_row = move["pushed_to"]
                    to_col, to_row = move["to"]
                    target = board[to_row][to_col]
                    if (hasattr(target, 'owner') and
                        is_own_score_cell(push_col, push_row, target.owner, rows, cols, score_cols)):
                        if target.owner == current_player:
                            score += 5000
                        else:
                            score -= 3000
            
            return score + random.random() * 0.1
        
        moves.sort(key=move_score, reverse=True)
    
    def get_defensive_moves(self, board: List[List[Any]], my_moves: List[Dict[str, Any]],
                           rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        defensive_moves = []
        
        # Check opponent threats
        opp_moves = self.get_all_valid_moves(board, self.opponent, rows, cols, score_cols)
        
        for opp_move in opp_moves:
            opp_board = self.apply_move(board, opp_move, self.opponent)
            if self.count_stones_in_score_area(opp_board, self.opponent, rows, cols, score_cols) >= 4:
                # Find defensive moves
                for my_move in my_moves:
                    if self.blocks_opponent_win(board, my_move, opp_move):
                        defensive_moves.append(my_move)
        
        return defensive_moves
    
    def blocks_opponent_win(self, board: List[List[Any]], my_move: Dict[str, Any], 
                           opp_move: Dict[str, Any]) -> bool:
        new_board = self.apply_move(board, my_move, self.player)
        
        # Check if opponent's piece was affected
        from_col, from_row = opp_move["from"]
        opp_piece = new_board[from_row][from_col]
        if not opp_piece or not hasattr(opp_piece, 'owner') or opp_piece.owner != self.opponent:
            return True
        
        # Check if destination is blocked
        if opp_move["action"] == "move" and "to" in opp_move:
            to_col, to_row = opp_move["to"]
            if new_board[to_row][to_col]:
                return True
        
        return False
    
    def count_stones_in_score_area(self, board: List[List[Any]], check_player: str,
                                  rows: int, cols: int, score_cols: List[int]) -> int:
        count = 0
        score_row = top_score_row() if check_player == "circle" else bottom_score_row(rows)
        
        for col in score_cols:
            if in_bounds(col, score_row, rows, cols):
                piece = board[score_row][col]
                if (piece and hasattr(piece, 'owner') and piece.owner == check_player and
                    hasattr(piece, 'side') and piece.side == "stone"):
                    count += 1
        
        return count
    
    def count_stones_one_move_from_scoring(self, board: List[List[Any]], check_player: str,
                                          rows: int, cols: int, score_cols: List[int]) -> int:
        count = 0
        counted = set()
        
        for row in range(rows):
            for col in range(cols):
                if (row, col) in counted:
                    continue
                    
                piece = board[row][col]
                if (piece and hasattr(piece, 'owner') and piece.owner == check_player and
                    hasattr(piece, 'side') and piece.side == "stone"):
                    moves = self.get_moves_for_piece(board, row, col, check_player, rows, cols, score_cols)
                    
                    for move in moves:
                        if move["action"] == "move" and "to" in move:
                            to_col, to_row = move["to"]
                            if is_own_score_cell(to_col, to_row, check_player, rows, cols, score_cols):
                                count += 1
                                counted.add((row, col))
                                break
        
        return count


def test_student_agent():
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        game_state = {"board": board}
        move = agent.choose(game_state, rows, cols, score_cols, 60.0, 60.0)
        
        if move:
            print("Agent successfully generated a move:", move)
        else:
            print("Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("StudentAgent created successfully")


if __name__ == "__main__":
    test_student_agent()