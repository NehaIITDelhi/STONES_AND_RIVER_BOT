import time
import random
import copy
from typing import List, Dict, Any, Optional, Tuple

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
# STUDENT AGENT IMPLEMENTATION
# ===================================================================
class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.transposition_table = {}
        self.killer_moves = {}
        self.history_table = {}
        self.river_chains_cache = {}
        self.turn_count = 0

    # ==============================================================
    # MAIN SEARCH FUNCTION
    # ==============================================================

    def choose(self, game_state: Dict[str, Any], rows: int, cols: int, score_cols: List[int],
               current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        
        self.turn_count += 1
        
        if isinstance(game_state, dict) and "board" in game_state:
            board = game_state["board"]
        else:
            board = game_state

        start_time = time.time()
        
        # Adaptive time management
        if current_player_time > 35:
            time_limit = 1.8
        elif current_player_time > 20:
            time_limit = 1.2
        elif current_player_time > 10:
            time_limit = 0.8
        else:
            time_limit = max(0.2, current_player_time / 15.0)

        moves = self.get_all_valid_moves_enhanced(board, self.player, rows, cols, score_cols)
        if not moves:
            return None
        
        for move in moves:
            if self.is_winning_move(board, move, self.player, rows, cols, score_cols):
                return move
        
        opp_moves = self.get_all_valid_moves_enhanced(board, self.opponent, rows, cols, score_cols)
        for opp_move in opp_moves:
            if self.is_winning_move(board, opp_move, self.opponent, rows, cols, score_cols):
                defensive_move = self.find_blocking_move(board, moves, opp_move, rows, cols, score_cols)
                if defensive_move:
                    return defensive_move
        
        if self.is_opponent_defensive(board, rows, cols, score_cols):
            aggressive_move = self.find_aggressive_river_push(board, moves, rows, cols, score_cols)
            if aggressive_move:
                return aggressive_move
        
        game_phase = self.get_game_phase(board, rows, cols)
        
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
        
        moves = self.order_moves_advanced(board, moves, self.player, rows, cols, score_cols)
        
        if current_player_time > 20:
            max_moves = min(12, len(moves))
        else:
            max_moves = min(8, len(moves))
        
        moves = moves[:max_moves]
        
        best_move = moves[0]
        best_score = float('-inf')
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit * 0.9:
                break
                
            alpha = float('-inf')
            beta = float('inf')
            current_best_for_depth = None
            
            for i, move in enumerate(moves):
                if time.time() - start_time > time_limit:
                    break
                    
                new_board = self.apply_move(board, move, self.player)
                
                if i == 0:
                    score = -self.negamax_enhanced(new_board, depth - 1, -beta, -alpha, 
                                                  self.opponent, start_time, time_limit, 
                                                  rows, cols, score_cols, depth)
                else:
                    score = -self.negamax_enhanced(new_board, depth - 1, -alpha - 1, -alpha,
                                                  self.opponent, start_time, time_limit,
                                                  rows, cols, score_cols, depth)
                    if alpha < score < beta:
                        score = -self.negamax_enhanced(new_board, depth - 1, -beta, -score,
                                                      self.opponent, start_time, time_limit,
                                                      rows, cols, score_cols, depth)
                
                if score > best_score:
                    best_score = score
                    current_best_for_depth = move
                    
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            
            if current_best_for_depth:
                best_move = current_best_for_depth
                if best_move in moves:
                    moves.remove(best_move)
                moves.insert(0, best_move)

        return best_move

    def negamax_enhanced(self, board: List[List[Any]], depth: int, alpha: float, beta: float,
                         current_player: str, start_time: float, time_limit: float, 
                         rows: int, cols: int, score_cols: List[int], max_depth: int) -> float:
        
        if time.time() - start_time > time_limit:
            return self.evaluate_enhanced(board, current_player, rows, cols, score_cols)
        
        my_stones = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp = get_opponent(current_player)
        opp_stones = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        
        if my_stones >= 4:
            return 10000 - (max_depth - depth) * 10
        if opp_stones >= 4:
            return -10000 + (max_depth - depth) * 10
        
        if depth <= 0:
            return self.evaluate_enhanced(board, current_player, rows, cols, score_cols)
        
        moves = self.get_all_valid_moves_enhanced(board, current_player, rows, cols, score_cols)
        if not moves:
            return self.evaluate_enhanced(board, current_player, rows, cols, score_cols)
        
        moves = self.order_moves_advanced(board, moves, current_player, rows, cols, score_cols)
        
        if depth <= 1:
            moves = moves[:5]
        else:
            moves = moves[:8]
        
        best_score = float('-inf')
        
        for move in moves:
            new_board = self.apply_move(board, move, current_player)
            score = -self.negamax_enhanced(new_board, depth - 1, -beta, -alpha,
                                          opp, start_time, time_limit,
                                          rows, cols, score_cols, max_depth)
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                if depth not in self.killer_moves:
                    self.killer_moves[depth] = []
                if move not in self.killer_moves[depth]:
                    self.killer_moves[depth].insert(0, move)
                    if len(self.killer_moves[depth]) > 2:
                        self.killer_moves[depth].pop()
                break
        
        return best_score

    # ==============================================================
    # ENHANCED EVALUATION WITH RIVER CHAINS
    # ==============================================================

    def evaluate_enhanced(self, board: List[List[Any]], current_player: str, 
                          rows: int, cols: int, score_cols: List[int]) -> float:
        score = 0.0
        opp = get_opponent(current_player)
        
        my_scored = self.count_stones_in_score_area(board, current_player, rows, cols, score_cols)
        opp_scored = self.count_stones_in_score_area(board, opp, rows, cols, score_cols)
        
        score += my_scored * 1000
        score -= opp_scored * 1100
        
        if my_scored >= 4: return 10000
        if opp_scored >= 4: return -10000
        if my_scored == 3: score += 600
        if opp_scored == 3: score -= 700
        
        my_chain_score = self.evaluate_river_chains(board, current_player, rows, cols, score_cols)
        opp_chain_score = self.evaluate_river_chains(board, opp, rows, cols, score_cols)
        score += my_chain_score * 80
        score -= opp_chain_score * 80
        
        my_ready = self.count_stones_ready_to_score_enhanced(board, current_player, rows, cols, score_cols)
        opp_ready = self.count_stones_ready_to_score_enhanced(board, opp, rows, cols, score_cols)
        score += my_ready * 150
        score -= opp_ready * 150
        
        score += self.evaluate_strategic_position(board, current_player, rows, cols, score_cols)
        score += self.evaluate_river_network(board, current_player, rows, cols) * 10
        score -= self.evaluate_river_network(board, opp, rows, cols) * 10
        
        return score

    def evaluate_river_chains(self, board, player, rows, cols, score_cols):
        total_score = 0.0
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    best_path_score = self.find_best_river_path_to_score(
                        board, row, col, player, rows, cols, score_cols)
                    total_score += best_path_score
        return total_score

    def find_best_river_path_to_score(self, board, start_row, start_col, player, rows, cols, score_cols):
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        q = [(start_row, start_col, 0, 0.0)]
        visited = {(start_row, start_col)}
        max_quality = 0.0

        while q:
            row, col, depth, path_quality = q.pop(0)
            if depth > 5: continue

            is_near_target = any(abs(row - target_row) <= 1 and abs(col - sc) <= 1 for sc in score_cols)
            if is_near_target:
                 max_quality = max(max_quality, path_quality)
                 continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row, next_col = row + dr, col + dc
                if in_bounds(next_col, next_row, rows, cols) and (next_row, next_col) not in visited:
                    piece = board[next_row][next_col]
                    if piece and piece.side == "river":
                        visited.add((next_row, next_col))
                        flow_quality = self.calculate_river_flow_quality(piece, next_row, next_col, target_row, player)
                        q.append((next_row, next_col, depth + 1, path_quality + flow_quality))
        return max_quality

    def calculate_river_flow_quality(self, river_piece, row, col, target_row, player):
        if not hasattr(river_piece, 'orientation'): return 0.1
        
        if player == "circle": is_correct_vertical = row < target_row
        else: is_correct_vertical = row > target_row
            
        if river_piece.orientation == "vertical": return 0.8 if is_correct_vertical else 0.4
        elif river_piece.orientation == "horizontal": return 0.3
        return 0.1

    def count_stones_ready_to_score_enhanced(self, board, player, rows, cols, score_cols):
        count = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        for col in score_cols:
            check_row = target_row + 1 if player == "circle" else target_row - 1
            if in_bounds(col, check_row, rows, cols):
                piece = board[check_row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    count += 1.0
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player and piece.side == "stone":
                    river_path_quality = self.find_best_river_path_to_score(board, row, col, player, rows, cols, score_cols)
                    if river_path_quality > 0.5: count += river_path_quality * 0.4
        return count

    def evaluate_strategic_position(self, board, player, rows, cols, score_cols):
        score = 0.0
        target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        
        for row in range(rows):
            for col in range(cols):
                piece = board[row][col]
                if piece and piece.owner == player:
                    if piece.side == "stone":
                        row_distance = abs(row - target_row)
                        score += (rows - row_distance) * 0.5
                        if col in score_cols: score += 2
                        if self.is_path_blocked_by_opponent(board, row, col, target_row, player): score -= 3
                    else:
                        score += 1
                        if hasattr(piece, 'orientation') and piece.orientation == "vertical": score += 1
        return score

    def evaluate_river_network(self, board, player, rows, cols):
        score, rivers = 0.0, []
        for r in range(rows):
            for c in range(cols):
                p = board[r][c]
                if p and p.owner == player and p.side == "river": rivers.append((r, c, p))
        
        for i, (r1, c1, p1) in enumerate(rivers):
            for j, (r2, c2, p2) in enumerate(rivers[i+1:], i+1):
                if abs(r1 - r2) + abs(c1 - c2) <= 3:
                    score += 0.5
                    if hasattr(p1, 'orientation') and hasattr(p2, 'orientation') and p1.orientation != p2.orientation:
                        score += 0.3
        return score

    # ==============================================================
    # ENHANCED MOVE GENERATION AND ORDERING
    # ==============================================================

    def get_all_valid_moves_enhanced(self, board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves = []
        for r in range(rows):
            for c in range(cols):
                if board[r][c] and board[r][c].owner == player:
                    moves.extend(self.get_moves_for_piece_enhanced(board, r, c, player, rows, cols, score_cols))
        return moves

    def get_moves_for_piece_enhanced(self, board: List[List[Any]], row: int, col: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
        moves, piece = [], board[row][col]
        if not piece: return moves

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if not in_bounds(nc, nr, rows, cols) or is_opponent_score_cell(nc, nr, player, rows, cols, score_cols):
                continue
            
            target = board[nr][nc]
            if not target:
                moves.append({"action": "move", "from": [col, row], "to": [nc, nr]})
            elif target.side == "river":
                for dr, dc_ in self.trace_river_flow_enhanced(board, nr, nc, row, col, player, rows, cols, score_cols):
                    moves.append({"action": "move", "from": [col, row], "to": [dc_, dr]})
            elif target.side == "stone":
                if piece.side == "stone":
                    pr, pc = nr + dr, nc + dc
                    if in_bounds(pc, pr, rows, cols) and not board[pr][pc] and not is_opponent_score_cell(pc, pr, target.owner, rows, cols, score_cols):
                        moves.append({"action": "push", "from": [col, row], "to": [nc, nr], "pushed_to": [pc, pr]})
                else:
                    for dr, dc_ in self.trace_river_push_enhanced(board, nr, nc, piece, target.owner, rows, cols, score_cols):
                        moves.append({"action": "push", "from": [col, row], "to": [nc, nr], "pushed_to": [dc_, dr]})

        if piece.side == "stone":
            moves.append({"action": "flip", "from": [col, row], "orientation": "horizontal"})
            moves.append({"action": "flip", "from": [col, row], "orientation": "vertical"})
        else:
            moves.append({"action": "flip", "from": [col, row]})
            moves.append({"action": "rotate", "from": [col, row]})
        return moves

    def trace_river_flow_enhanced(self, board, start_row, start_col, origin_row, origin_col, player, rows, cols, score_cols):
        destinations, q, visited = set(), [(start_row, start_col, 0)], {(start_row, start_col)}
        while q:
            r, c, depth = q.pop(0)
            if depth > 5: continue
            p = board[r][c]
            if not p or p.side != "river": continue
            
            flow_dirs = [(-1, 0), (1, 0)] if hasattr(p, 'orientation') and p.orientation == "vertical" else [(0, -1), (0, 1)]
            for dr, dc in flow_dirs:
                cr, cc = r, c
                while True:
                    cr, cc = cr + dr, cc + dc
                    if not in_bounds(cc, cr, rows, cols) or is_opponent_score_cell(cc, cr, player, rows, cols, score_cols): break
                    target = board[cr][cc]
                    if not target: destinations.add((cr, cc))
                    else:
                        if target.side == "river" and (cr, cc) not in visited:
                            visited.add((cr, cc)); q.append((cr, cc, depth + 1))
                        break
        return list(destinations)

    def trace_river_push_enhanced(self, board, target_row, target_col, river_piece, pushed_player, rows, cols, score_cols):
        destinations = []
        push_dirs = [(-1, 0), (1, 0)] if hasattr(river_piece, 'orientation') and river_piece.orientation == "vertical" else [(0, -1), (0, 1)]
        
        for dr, dc in push_dirs:
            for dist in range(1, max(rows, cols)):
                dest_row, dest_col = target_row + dr * dist, target_col + dc * dist
                if not in_bounds(dest_col, dest_row, rows, cols) or is_opponent_score_cell(dest_col, dest_row, pushed_player, rows, cols, score_cols):
                    break
                
                # *** NEW LOGIC: Don't push our own pieces too far past the goal ***
                if pushed_player == self.player:
                    if self.player == "circle" and dest_row < top_score_row(): break
                    if self.player == "square" and dest_row > bottom_score_row(rows): break
                
                if not board[dest_row][dest_col]:
                    destinations.append((dest_row, dest_col))
                elif board[dest_row][dest_col].side != "river": break
        return destinations

    def order_moves_advanced(self, board, moves, current_player, rows, cols, score_cols):
        def move_priority(move):
            priority = 0
            if self.is_winning_move(board, move, current_player, rows, cols, score_cols):
                return 10000
            
            depth = self.turn_count
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                priority += 5000

            if move["action"] == "move" and "to" in move and is_own_score_cell(move["to"][0], move["to"][1], current_player, rows, cols, score_cols):
                priority += 4000
            
            # *** NEW LOGIC: Prioritize horizontal pushes on opponents ***
            if move["action"] == "push":
                from_c, from_r = move['from']
                to_c, to_r = move['to']
                pushed_to_c, pushed_to_r = move['pushed_to']
                from_piece = board[from_r][from_c]
                pushed_piece = board[to_r][to_c]

                if from_piece and from_piece.side == 'river':
                    if pushed_piece and pushed_piece.owner == self.opponent:
                        is_vertical = from_c == pushed_to_c
                        if is_vertical: priority += 500   # Low priority for vertical opponent push
                        else: priority += 2500            # High priority for horizontal path clearing
                    else:
                        priority += 2000                  # Normal priority for pushing own piece
                else:
                    priority += 500                       # Stone push
            
            if move["action"] == "flip" and "orientation" in move:
                priority += 800 if move["orientation"] == "vertical" else 400
            
            if move["action"] == "move" and "to" in move:
                target_row = top_score_row() if current_player == "circle" else bottom_score_row(rows)
                old_dist = abs(move["from"][1] - target_row)
                new_dist = abs(move["to"][1] - target_row)
                if new_dist < old_dist: priority += (old_dist - new_dist) * 10
            
            return priority
        return sorted(moves, key=move_priority, reverse=True)

    # ==============================================================
    # HELPER AND STRATEGY FUNCTIONS
    # ==============================================================

    def apply_move(self, board: List[List[Any]], move: Dict[str, Any], player: str) -> List[List[Any]]:
        new_board = copy.deepcopy(board)
        fc, fr = move["from"]
        p = new_board[fr][fc]
        action = move["action"]

        if action == "move":
            tc, tr = move["to"]; new_board[tr][tc] = p; new_board[fr][fc] = None
        elif action == "push":
            tc, tr = move["to"]; ptc, ptr = move["pushed_to"]
            pp = new_board[tr][tc]
            new_board[ptr][ptc] = pp; new_board[tr][tc] = p; new_board[fr][fc] = None
            if p.side == 'river': p.side, p.orientation = 'stone', None
        elif action == "flip":
            if p.side == "stone": p.side, p.orientation = "river", move["orientation"]
            else: p.side, p.orientation = "stone", None
        elif action == "rotate":
            if hasattr(p, 'orientation'): p.orientation = "vertical" if p.orientation == "horizontal" else "horizontal"
        return new_board
    
    def is_winning_move(self, board, move, player, rows, cols, score_cols) -> bool:
        if move['action'] not in ['move', 'push']: return False
        temp_board = self.apply_move(board, move, player)
        return self.count_stones_in_score_area(temp_board, player, rows, cols, score_cols) >= 4
    
    def find_blocking_move(self, board, my_moves, opp_winning_move, rows, cols, score_cols):
        target_pos = tuple(opp_winning_move.get('to') or opp_winning_move.get('pushed_to'))
        if not target_pos: return None
        for move in my_moves:
            if move.get('action') == 'move' and tuple(move.get('to')) == target_pos: return move
            if move.get('action') == 'push' and tuple(move.get('pushed_to')) == target_pos: return move
        return None

    def is_opponent_defensive(self, board, rows, cols, score_cols):
        defensive_pieces = 0
        for r in range(rows):
            for c in range(cols):
                p = board[r][c]
                if p and p.owner == self.opponent:
                    if (self.opponent == "circle" and r > rows // 2) or (self.opponent == "square" and r < rows // 2):
                        defensive_pieces += 1
        return defensive_pieces > 7
    
    def find_aggressive_river_push(self, board, moves, rows, cols, score_cols):
        best_push, max_dist = None, -1
        for move in moves:
            if move['action'] == 'push':
                fc, fr = move['from']; pc, pr = move['pushed_to']; p = board[fr][fc]
                if p and p.side == 'river':
                    dist = abs(pr - fr) + abs(pc - fc)
                    if dist > max_dist: max_dist, best_push = dist, move
        return best_push

    def get_game_phase(self, board, rows, cols):
        my_stones = self.count_stones_in_score_area(board, self.player, rows, cols, score_cols_for(cols))
        opp_stones = self.count_stones_in_score_area(board, self.opponent, rows, cols, score_cols_for(cols))
        total_scored = my_stones + opp_stones
        if total_scored >= 4: return "endgame"
        if total_scored >= 2: return "midgame"
        return "opening"

    def is_path_blocked_by_opponent(self, board, row, col, target_row, player):
        step = 1 if target_row > row else -1
        for r in range(row + step, target_row, step):
            if in_bounds(col, r, len(board), len(board[0])) and board[r][col] and board[r][col].owner == self.opponent:
                return True
        return False
        
    def count_stones_in_score_area(self, board, player, rows, cols, score_cols):
        count = 0
        score_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        for x in score_cols:
            if in_bounds(x, score_row, rows, cols):
                p = board[score_row][x]
                if p and p.owner == player and p.side == "stone":
                    count += 1
        return count