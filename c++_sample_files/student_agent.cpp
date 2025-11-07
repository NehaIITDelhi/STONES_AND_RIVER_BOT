#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // This handles optional, vector, and map
#include "student_agent.h"
#include <sstream> // For string building
#include <set>

namespace py = pybind11;

// ===================================================================
// UTILITY FUNCTIONS 
// ===================================================================

bool StudentAgent::in_bounds(int x, int y, int rows, int cols) const {
    return 0 <= x && x < cols && 0 <= y && y < rows;
}

std::vector<int> StudentAgent::score_cols_for(int cols) const {
    int w = 4; // Assuming 4 spaces as per Python code
    int start = std::max(0, (cols - w) / 2);
    std::vector<int> result;
    for (int i = 0; i < w; ++i) {
        result.push_back(start + i);
    }
    return result;
}

int StudentAgent::top_score_row() const { return 2; }
int StudentAgent::bottom_score_row(int rows) const { return rows - 3; }

bool StudentAgent::is_opponent_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const {
    int opp_row = (p == "circle") ? bottom_score_row(rows) : top_score_row();
    if (y != opp_row) return false;
    for (int col : score_cols) {
        if (x == col) return true;
    }
    return false;
}

bool StudentAgent::is_own_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const {
    int own_row = (p == "circle") ? top_score_row() : bottom_score_row(rows);
    if (y != own_row) return false;
    for (int col : score_cols) {
        if (x == col) return true;
    }
    return false;
}

// C++ equivalent of the Python board hash
size_t StudentAgent::board_hash(const Board& board) const {
    std::stringstream ss;
    for (const auto& row : board) {
        for (const auto& cell : row) {
            if (!cell) {
                ss << "0";
            } else {
                ss << cell->owner[0] << cell->side[0];
                if (!cell->orientation.empty()) {
                    ss << cell->orientation[0];
                }
            }
        }
    }
    return std::hash<std::string>{}(ss.str());
}

int StudentAgent::manhattan_distance(int x1, int y1, int x2, int y2) const {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

// C++ equivalent of copy.deepcopy(board)
Board StudentAgent::deep_copy_board(const Board& board) {
    Board new_board;
    new_board.resize(board.size());
    for (size_t r = 0; r < board.size(); ++r) {
        new_board[r].resize(board[r].size());
        for (size_t c = 0; c < board[r].size(); ++c) {
            if (board[r][c]) {
                // Create a new Piece object, copying the old one
                new_board[r][c] = std::make_shared<Piece>(*board[r][c]);
            } else {
                new_board[r][c] = nullptr; // Is already nullptr, but good to be explicit
            }
        }
    }
    return new_board;
}

// ===================================================================
// MAIN SEARCH FUNCTION
// ===================================================================

std::optional<Move> StudentAgent::choose(
    const Board& board, int rows, int cols, const std::vector<int>& score_cols,
    double current_player_time, double opponent_time) 
{
    start_time_point = std::chrono::high_resolution_clock::now();
    turn_count++;

    // Update position history
    size_t current_hash = board_hash(board);
    recent_positions.push_back(current_hash);
    if (recent_positions.size() > MAX_HISTORY_SIZE) {
        size_t old_hash = recent_positions.front();
        recent_positions.pop_front();
        board_history_set.erase(old_hash);
    }
    board_history_set.insert(current_hash);

    // Adaptive time management
    if (current_player_time > 35) time_limit_seconds = 1.5;
    else if (current_player_time > 20) time_limit_seconds = 1.0;
    else if (current_player_time > 10) time_limit_seconds = 0.6;
    else time_limit_seconds = std::max(0.15, current_player_time / 20.0);

    auto moves = get_all_valid_moves_enhanced(board, this->player, rows, cols, score_cols);
    if (moves.empty()) {
        return std::nullopt;
    }

    // Check for immediate winning move
    for (const auto& move : moves) {
        if (is_winning_move(board, move, this->player, rows, cols, score_cols)) {
            update_move_history(move);
            return move;
        }
    }

    // Check for opponent winning threats and block them
    auto opp_moves = get_all_valid_moves_enhanced(board, this->opponent, rows, cols, score_cols);
    for (const auto& opp_move : opp_moves) {
        if (is_winning_move(board, opp_move, this->opponent, rows, cols, score_cols)) {
            auto defensive_move = find_blocking_move(board, moves, opp_move, rows, cols, score_cols);
            if (defensive_move) {
                update_move_history(*defensive_move);
                return *defensive_move;
            }
        }
    }

    // Filter moves to prevent loops
    auto filtered_moves = filter_non_repeating_moves(board, moves, rows, cols, score_cols);
    if (!filtered_moves.empty()) {
        moves = filtered_moves;
    }

    // Determine game phase and strategy
    std::string game_phase = get_game_phase(board, rows, cols, score_cols);

    // Adjust depth
    int max_depth;
    if (current_player_time < 5) max_depth = 1;
    else if (current_player_time < 15) max_depth = 2;
    else if (game_phase == "endgame") max_depth = 4;
    else if (game_phase == "midgame") max_depth = 3;
    else max_depth = 2;

    // Order moves
    moves = order_moves_with_edge_control(board, moves, this->player, rows, cols, score_cols);

    // Limit branching factor
    int max_moves = (current_player_time > 20) ? 10 : 6;
    if (moves.size() > max_moves) {
        moves.resize(max_moves);
    }

    // Iterative deepening
    Move best_move = moves[0];
    double best_score = -std::numeric_limits<double>::infinity();

    for (int depth = 1; depth <= max_depth; ++depth) {
        auto now = std::chrono::high_resolution_clock::now();
        double time_spent = std::chrono::duration<double>(now - start_time_point).count();
        if (time_spent > time_limit_seconds * 0.85) {
            break;
        }

        double alpha = -std::numeric_limits<double>::infinity();
        double beta = std::numeric_limits<double>::infinity();
        Move current_best;

        for (int i = 0; i < moves.size(); ++i) {
            now = std::chrono::high_resolution_clock::now();
            time_spent = std::chrono::duration<double>(now - start_time_point).count();
            if (time_spent > time_limit_seconds) {
                break;
            }

            Board new_board = apply_move(board, moves[i], this->player);
            
            size_t new_hash = board_hash(new_board);
            if (board_history_set.count(new_hash)) {
                continue;
            }
            
            double score;
            if (i == 0) {
                 score = -negamax_with_balance(new_board, depth - 1, -beta, -alpha, 
                                               this->opponent, rows, cols, score_cols, depth);
            } else {
                score = -negamax_with_balance(new_board, depth - 1, -alpha - 1, -alpha,
                                              this->opponent, rows, cols, score_cols, depth);
                if (alpha < score && score < beta) {
                    score = -negamax_with_balance(new_board, depth - 1, -beta, -score,
                                                  this->opponent, rows, cols, score_cols, depth);
                }
            }

            if (score > best_score) {
                best_score = score;
                current_best = moves[i];
            }
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                break;
            }
        }
        
        // Check for valid move
        if (current_best.action != Move::ActionType::MOVE || current_best.from != std::make_pair(0,0)) {
             best_move = current_best;
             // Move best move to front
             auto it = std::find(moves.begin(), moves.end(), best_move);
             if (it != moves.end()) {
                 moves.erase(it);
                 moves.insert(moves.begin(), best_move);
             }
        }
    }

    update_move_history(best_move);
    return best_move;
}

void StudentAgent::update_move_history(const Move& move) {
    last_moves.push_back(move);
    if (last_moves.size() > MAX_RECENT_MOVES) {
        last_moves.pop_front();
    }
}

std::vector<Move> StudentAgent::filter_non_repeating_moves(const Board& board, const std::vector<Move>& moves, int rows, int cols, const std::vector<int>& score_cols) {
    std::vector<Move> filtered;
    
    // Create a temporary set of recent positions for faster lookup
    std::set<size_t> recent_set;
    for(size_t i = 0; i < 8 && i < recent_positions.size(); ++i) {
        recent_set.insert(recent_positions[recent_positions.size() - 1 - i]);
    }

    for (const auto& move : moves) {
        bool is_repeat = false;
        // Check last 3 moves
        for(size_t i = 0; i < 3 && i < last_moves.size(); ++i) {
            if (moves_similar(move, last_moves[last_moves.size() - 1 - i])) {
                is_repeat = true;
                break;
            }
        }
        if (is_repeat) continue;

        Board new_board = apply_move(board, move, this->player);
        size_t new_hash = board_hash(new_board);

        if (recent_set.find(new_hash) == recent_set.end()) {
            filtered.push_back(move);
        }
    }

    if (filtered.empty() && !moves.empty()) {
        std::vector<Move> top_moves;
        for(size_t i = 0; i < 5 && i < moves.size(); ++i) {
            top_moves.push_back(moves[i]);
        }
        return top_moves;
    }
    return filtered;
}

bool StudentAgent::moves_similar(const Move& move1, const Move& move2) const {
    if (move1.action != move2.action) return false;
    if (move1.from == move2.from && move1.to == move2.to) return true;
    if (move1.action == Move::ActionType::MOVE && move2.action == Move::ActionType::MOVE) {
        if (move1.from == move2.to && move1.to == move2.from) return true;
    }
    return false;
}

// ===================================================================
// NEGAMAX IMPLEMENTATION
// ===================================================================

double StudentAgent::negamax_with_balance(Board board, int depth, double alpha, double beta,
                                          const std::string& current_player, int rows, int cols, 
                                          const std::vector<int>& score_cols, int max_depth) 
{
    auto now = std::chrono::high_resolution_clock::now();
    double time_spent = std::chrono::duration<double>(now - start_time_point).count();
    if (time_spent > time_limit_seconds) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }

    int my_stones = count_stones_in_score_area(board, current_player, rows, cols, score_cols);
    std::string opp = get_opponent(current_player);
    int opp_stones = count_stones_in_score_area(board, opp, rows, cols, score_cols);

    if (my_stones >= 4) return 10000.0 - (max_depth - depth) * 10.0;
    if (opp_stones >= 4) return -10000.0 + (max_depth - depth) * 10.0;

    if (depth <= 0) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }

    auto moves = get_all_valid_moves_enhanced(board, current_player, rows, cols, score_cols);
    if (moves.empty()) {
        return evaluate_balanced(board, current_player, rows, cols, score_cols);
    }
    
    moves = order_moves_with_edge_control(board, moves, current_player, rows, cols, score_cols);

    int max_moves = (depth <= 1) ? 4 : 6;
    if (moves.size() > max_moves) {
        moves.resize(max_moves);
    }

    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& move : moves) {
        Board new_board = apply_move(board, move, current_player);
        
        size_t new_hash = board_hash(new_board);
        // Check last 5 recent positions
        bool in_recent = false;
        for(size_t i = 0; i < 5 && i < recent_positions.size(); ++i) {
            if (recent_positions[recent_positions.size() - 1 - i] == new_hash) {
                in_recent = true;
                break;
            }
        }
        if (in_recent) continue;

        double score = -negamax_with_balance(new_board, depth - 1, -beta, -alpha,
                                            opp, rows, cols, score_cols, max_depth);

        if (score > best_score) {
            best_score = score;
        }
        if (score > alpha) {
            alpha = score;
        }

        if (alpha >= beta) {
            if (killer_moves.find(depth) == killer_moves.end()) {
                killer_moves[depth] = {};
            }
            auto& killers = killer_moves[depth];
            if (std::find(killers.begin(), killers.end(), move) == killers.end()) {
                killers.insert(killers.begin(), move);
                if (killers.size() > 2) {
                    killers.pop_back();
                }
            }
            break;
        }
    }
    return best_score;
}

// ===================================================================
// EVALUATION FUNCTIONS
// ===================================================================

double StudentAgent::evaluate_balanced(const Board& board, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols) {
    double score = 0.0;
    std::string opp = get_opponent(current_player);

    int my_scored = count_stones_in_score_area(board, current_player, rows, cols, score_cols);
    int opp_scored = count_stones_in_score_area(board, opp, rows, cols, score_cols);

    score += my_scored * 1000.0;
    score -= opp_scored * 1100.0;

    if (my_scored >= 4) return 10000.0;
    if (opp_scored >= 4) return -10000.0;
    if (my_scored == 3) score += 500.0;
    if (opp_scored == 3) score -= 600.0;

    score += evaluate_edge_control(board, current_player, rows, cols, score_cols) * 100.0;
    score += evaluate_defensive_position(board, current_player, rows, cols, score_cols) * 80.0;
    score += evaluate_manhattan_distances(board, current_player, rows, cols, score_cols) * 6.0;
    score -= evaluate_manhattan_distances(board, opp, rows, cols, score_cols) * 6.0;
    score += evaluate_river_network_balanced(board, current_player, rows, cols, score_cols) * 50.0;
    score -= evaluate_river_network_balanced(board, opp, rows, cols, score_cols) * 50.0;
    score += count_stones_ready_to_score(board, current_player, rows, cols, score_cols) * 150.0;
    score -= count_stones_ready_to_score(board, opp, rows, cols, score_cols) * 150.0;
    score += evaluate_balance_factor(board, current_player, rows, cols) * 30.0;

    return score;
}

double StudentAgent::evaluate_edge_control(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double edge_score = 0.0;
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    
    std::vector<int> edge_cols = {0, 1, cols - 2, cols - 1};
    
    for (int col : edge_cols) {
        int start_row = (player == "circle") ? (target_row + 1) : std::max(target_row - 3, 0);
        int end_row = (player == "circle") ? std::min(target_row + 4, rows) : target_row;

        for (int row = start_row; row < end_row; ++row) {
            if (in_bounds(col, row, rows, cols)) {
                auto piece = board[row][col];
                if (piece) {
                    if (piece->owner == player) {
                        edge_score += 1.5;
                        if (piece->side == "river" && piece->orientation == "horizontal") {
                            edge_score += 2.0;
                        }
                    } else {
                        edge_score -= 2.0;
                    }
                }
            }
        }
    }
    
    for (int col : {0, cols - 1}) {
        bool lane_blocked = false;
        int start_row = (player == "circle") ? (target_row + 1) : (top_score_row() + 1);
        int end_row = (player == "circle") ? bottom_score_row(rows) : target_row;

        for (int row = start_row; row < end_row; ++row) {
            if (in_bounds(col, row, rows, cols) && board[row][col]) {
                lane_blocked = true;
                break;
            }
        }
        if (!lane_blocked) edge_score -= 3.0;
    }
    return edge_score;
}

double StudentAgent::evaluate_defensive_position(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double defensive_score = 0.0;
    std::string opp = get_opponent(player);

    int start_row = (player == "circle") ? (top_score_row() + 1) : std::max(bottom_score_row(rows) - 3, 0);
    int end_row = (player == "circle") ? std::min(top_score_row() + 4, rows) : bottom_score_row(rows);

    std::vector<bool> col_coverage(cols, false);

    for (int row = start_row; row < end_row; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (in_bounds(col, row, rows, cols)) {
                auto piece = board[row][col];
                if (piece && piece->owner == player) {
                    col_coverage[col] = true;
                    defensive_score += 1.0;
                    if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end() || col == 0 || col == cols - 1) {
                        defensive_score += 1.5;
                    }
                    if (piece->side == "river") defensive_score += 0.5;
                }
            }
        }
    }

    for (int col = 0; col < cols; ++col) {
        if (!col_coverage[col]) {
            if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end()) {
                defensive_score -= 2.0;
            } else if (col == 0 || col == cols - 1) {
                defensive_score -= 1.5;
            }
        }
    }

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == opp && piece->side == "stone") {
                if (player == "circle" && row >= bottom_score_row(rows) - 2) {
                    defensive_score -= 3.0;
                } else if (player == "square" && row <= top_score_row() + 2) {
                    defensive_score -= 3.0;
                }
            }
        }
    }
    return defensive_score;
}

double StudentAgent::evaluate_balance_factor(const Board& board, const std::string& player, int rows, int cols) {
    double balance_score = 0.0;
    int offensive_pieces = 0, defensive_pieces = 0, mid_pieces = 0;
    int mid_row = rows / 2;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == player) {
                if (player == "circle") {
                    if (row < mid_row - 1) offensive_pieces++;
                    else if (row > mid_row + 1) defensive_pieces++;
                    else mid_pieces++;
                } else {
                    if (row > mid_row + 1) offensive_pieces++;
                    else if (row < mid_row - 1) defensive_pieces++;
                    else mid_pieces++;
                }
            }
        }
    }

    int total = offensive_pieces + defensive_pieces + mid_pieces;
    if (total > 0) {
        double off_ratio = (double)offensive_pieces / total;
        double def_ratio = (double)defensive_pieces / total;
        double mid_ratio = (double)mid_pieces / total;

        if (off_ratio > 0.7) balance_score -= 5.0;
        else if (def_ratio > 0.6) balance_score -= 3.0;
        else if (off_ratio < 0.2) balance_score -= 4.0;
        else balance_score += 3.0;
        balance_score += mid_ratio * 5.0;
    }
    return balance_score;
}

double StudentAgent::evaluate_river_network_balanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double score = 0.0;
    std::vector<std::tuple<int, int, PiecePtr>> rivers;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            auto p = board[r][c];
            if (p && p->owner == player && p->side == "river") {
                rivers.emplace_back(r, c, p);
            }
        }
    }

    for (size_t i = 0; i < rivers.size(); ++i) {
        auto [r1, c1, p1] = rivers[i];
        for (size_t j = i + 1; j < rivers.size(); ++j) {
            auto [r2, c2, p2] = rivers[j];
            if (std::abs(r1 - r2) + std::abs(c1 - c2) <= 3) {
                score += 1.0;
                if (p1->orientation != p2->orientation) score += 0.5;
            }
        }
    }

    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    for (auto [r, c, p] : rivers) {
        if (player == "circle") {
            if (r > target_row && r < target_row + 3) {
                score += 2.0;
                if (p->orientation == "horizontal") score += 1.0;
            }
        } else {
            if (r < target_row && r > target_row - 3) {
                score += 2.0;
                if (p->orientation == "horizontal") score += 1.0;
            }
        }
        if (c == 0 || c == 1 || c == cols - 2 || c == cols - 1) score += 1.5;
    }
    return score;
}

double StudentAgent::evaluate_manhattan_distances(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double total_score = 0.0;
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            auto piece = board[row][col];
            if (piece && piece->owner == player && piece->side == "stone") {
                if (is_own_score_cell(col, row, player, rows, cols, score_cols)) continue;

                int min_dist = std::numeric_limits<int>::max();
                for (int score_col : score_cols) {
                    min_dist = std::min(min_dist, manhattan_distance(col, row, score_col, target_row));
                }

                if (min_dist > 0) {
                    total_score += std::max(0, 20 - min_dist * 2);
                }
                
                if (std::find(score_cols.begin(), score_cols.end(), col) != score_cols.end()) {
                    total_score += 8.0;
                }

                int row_dist = std::abs(row - target_row);
                if (row_dist <= 2) {
                    total_score += (3 - row_dist) * 4.0;
                }
            }
        }
    }
    return total_score;
}

double StudentAgent::count_stones_ready_to_score(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    double count = 0.0;
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    int check_row = (player == "circle") ? (target_row + 1) : (target_row - 1);

    if (check_row >= 0 && check_row < rows) {
        for (int col : score_cols) {
            if (in_bounds(col, check_row, rows, cols)) {
                auto piece = board[check_row][col];
                if (piece && piece->owner == player && piece->side == "stone") {
                    if (!board[target_row][col]) {
                        count += 1.0;
                    }
                }
            }
        }
    }
    return count;
}

// ===================================================================
// MOVE ORDERING
// ===================================================================

std::vector<Move> StudentAgent::order_moves_with_edge_control(const Board& board, std::vector<Move> moves, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols) {
    
    std::map<Move*, int> move_priorities;
    
    // Lambda for priority calculation
    auto get_priority = [&](const Move& move) {
        int priority = 0;
        if (is_winning_move(board, move, current_player, rows, cols, score_cols)) return 10000;

        if (killer_moves.count(turn_count)) {
            const auto& killers = killer_moves[turn_count];
            if (std::find(killers.begin(), killers.end(), move) != killers.end()) {
                priority += 5000;
            }
        }

        if (move.action == Move::ActionType::MOVE) {
            if (is_own_score_cell(move.to.first, move.to.second, current_player, rows, cols, score_cols)) {
                priority += 4000;
            }
        }

        if (move.action == Move::ActionType::MOVE || move.action == Move::ActionType::PUSH) {
            int to_col = (move.action == Move::ActionType::MOVE) ? move.to.first : move.pushed_to.first;
            int to_row = (move.action == Move::ActionType::MOVE) ? move.to.second : move.pushed_to.second;
            if (to_col == 0 || to_col == 1 || to_col == cols - 2 || to_col == cols - 1) {
                priority += 300;
                int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
                if (current_player == "circle" && (to_row > target_row && to_row < target_row + 3)) {
                    priority += 200;
                } else if (current_player == "square" && (to_row < target_row && to_row > target_row - 3)) {
                    priority += 200;
                }
            }
        }

        if (move.action == Move::ActionType::FLIP) {
            int from_col = move.from.first;
            int from_row = move.from.second;
            int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
            if (current_player == "circle" && (from_row > target_row && from_row < target_row + 3)) {
                priority += 250;
                if (move.orientation == "horizontal") priority += 150;
            } else if (current_player == "square" && (from_row < target_row && from_row > target_row - 3)) {
                priority += 250;
                if (move.orientation == "horizontal") priority += 150;
            }
            if (from_col == 0 || from_col == 1 || from_col == cols - 2 || from_col == cols - 1) {
                priority += 180;
            }
        }
        
        if (move.action == Move::ActionType::PUSH) {
            auto from_piece = board[move.from.second][move.from.first];
            auto pushed_piece = board[move.to.second][move.to.first];
            if (from_piece && pushed_piece) {
                if (pushed_piece->owner == get_opponent(current_player)) {
                    priority += 1800;
                    if (move.from.second == move.to.second) priority += 400; // Horizontal
                } else {
                    int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
                    int old_dist = std::abs(move.to.second - target_row);
                    int new_dist = std::abs(move.pushed_to.second - target_row);
                    if (new_dist < old_dist) priority += 1200;
                }
            }
        }

        if (move.action == Move::ActionType::MOVE) {
            int target_row = (current_player == "circle") ? top_score_row() : bottom_score_row(rows);
            int old_dist = std::abs(move.from.second - target_row);
            int new_dist = std::abs(move.to.second - target_row);
            if (new_dist < old_dist) priority += (old_dist - new_dist) * 150;
        }
        
        return priority;
    };
    
    // Sort the moves
    std::sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        return get_priority(a) > get_priority(b);
    });

    return moves;
}

// ===================================================================
// STRATEGY & HELPER FUNCTIONS
// ===================================================================

std::string StudentAgent::get_game_phase(const Board& board, int rows, int cols, const std::vector<int>& score_cols) const {
    int my_stones = count_stones_in_score_area(board, this->player, rows, cols, score_cols);
    int opp_stones = count_stones_in_score_area(board, this->opponent, rows, cols, score_cols);
    if (my_stones + opp_stones >= 5) return "endgame";
    if (my_stones + opp_stones >= 2) return "midgame";
    return "opening";
}

int StudentAgent::count_stones_in_score_area(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    int count = 0;
    int score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    for (int x : score_cols) {
        if (in_bounds(x, score_row, rows, cols)) {
            auto p = board[score_row][x];
            if (p && p->owner == player && p->side == "stone") {
                count++;
            }
        }
    }
    return count;
}

bool StudentAgent::is_winning_move(const Board& board, const Move& move, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) {
    if (move.action != Move::ActionType::MOVE && move.action != Move::ActionType::PUSH) {
        return false;
    }
    Board temp_board = apply_move(board, move, player);
    return count_stones_in_score_area(temp_board, player, rows, cols, score_cols) >= 4;
}

std::optional<Move> StudentAgent::find_blocking_move(const Board& board, const std::vector<Move>& my_moves, const Move& opp_winning_move, int rows, int cols, const std::vector<int>& score_cols) {
    std::pair<int, int> target_pos = {-1, -1};
    if (opp_winning_move.action == Move::ActionType::MOVE) {
        target_pos = opp_winning_move.to;
    } else if (opp_winning_move.action == Move::ActionType::PUSH) {
        target_pos = opp_winning_move.pushed_to;
    }

    if (target_pos.first == -1) return std::nullopt;

    for (const auto& move : my_moves) {
        if (move.action == Move::ActionType::MOVE && move.to == target_pos) {
            return move;
        }
        if (move.action == Move::ActionType::PUSH && move.pushed_to == target_pos) {
            return move;
        }
    }
    return std::nullopt;
}

Board StudentAgent::apply_move(const Board& board, const Move& move, const std::string& player) {
    // This is the C++ equivalent of copy.deepcopy()
    Board new_board = deep_copy_board(board);

    int fr = move.from.second;
    int fc = move.from.first;
    auto piece = new_board[fr][fc]; // This is a shared_ptr

    if (!piece) return new_board; // Should not happen

    if (move.action == Move::ActionType::MOVE) {
        int tr = move.to.second;
        int tc = move.to.first;
        new_board[tr][tc] = piece;
        new_board[fr][fc] = nullptr;
    } 
    else if (move.action == Move::ActionType::PUSH) {
        int tr = move.to.second;
        int tc = move.to.first;
        int ptr = move.pushed_to.second;
        int ptc = move.pushed_to.first;
        auto pushed_piece = new_board[tr][tc];
        new_board[ptr][ptc] = pushed_piece;
        new_board[tr][tc] = piece;
        new_board[fr][fc] = nullptr;
        if (piece->side == "river") {
            piece->side = "stone";
            piece->orientation = "";
        }
    }
    else if (move.action == Move::ActionType::FLIP) {
        if (piece->side == "stone") {
            piece->side = "river";
            piece->orientation = move.orientation;
        } else {
            piece->side = "stone";
            piece->orientation = "";
        }
    }
    else if (move.action == Move::ActionType::ROTATE) {
        if (piece->orientation == "horizontal") {
            piece->orientation = "vertical";
        } else {
            piece->orientation = "horizontal";
        }
    }
    return new_board;
}

// ===================================================================
// MOVE GENERATION
// ===================================================================

std::vector<Move> StudentAgent::get_all_valid_moves_enhanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::vector<Move> moves;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (board[r][c] && board[r][c]->owner == player) {
                auto piece_moves = _get_moves_for_piece(board, r, c, player, rows, cols, score_cols);
                moves.insert(moves.end(), piece_moves.begin(), piece_moves.end());
            }
        }
    }
    return moves;
}

std::vector<Move> StudentAgent::_get_moves_for_piece(const Board& board, int row, int col, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::vector<Move> moves;
    auto piece = board[row][col];
    std::pair<int,int> from_pos = {col, row};

    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
        int nr = row + dr[i];
        int nc = col + dc[i];
        if (!in_bounds(nc, nr, rows, cols) || is_opponent_score_cell(nc, nr, player, rows, cols, score_cols)) {
            continue;
        }

        auto target_cell = board[nr][nc];
        if (!target_cell) {
            moves.emplace_back(Move::ActionType::MOVE, from_pos, std::make_pair(nc, nr));
        } else if (target_cell->side == "river") {
            auto flow_dests = _trace_river_flow(board, nr, nc, player, rows, cols, score_cols);
            for (auto dest : flow_dests) {
                moves.emplace_back(Move::ActionType::MOVE, from_pos, std::make_pair(dest.second, dest.first));
            }
        } else if (target_cell->side == "stone") {
            if (piece->side == "stone") {
                int pr = nr + dr[i];
                int pc = nc + dc[i];
                if (in_bounds(pc, pr, rows, cols) && !board[pr][pc] && !is_opponent_score_cell(pc, pr, target_cell->owner, rows, cols, score_cols)) {
                    moves.emplace_back(Move::ActionType::PUSH, from_pos, std::make_pair(nc, nr), std::make_pair(pc, pr));
                }
            } else { // River pushing stone
                auto push_dests = _trace_river_push(board, nr, nc, piece, target_cell->owner, rows, cols, score_cols);
                for (auto dest : push_dests) {
                    moves.emplace_back(Move::ActionType::PUSH, from_pos, std::make_pair(nc, nr), std::make_pair(dest.second, dest.first));
                }
            }
        }
    }

    if (piece->side == "stone") {
        moves.emplace_back(Move::ActionType::FLIP, from_pos, "horizontal");
        moves.emplace_back(Move::ActionType::FLIP, from_pos, "vertical");
    } else {
        moves.emplace_back(Move::ActionType::FLIP, from_pos);
        moves.emplace_back(Move::ActionType::ROTATE, from_pos);
    }
    return moves;
}

std::vector<std::pair<int, int>> StudentAgent::_trace_river_flow(const Board& board, int start_r, int start_c, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::deque<std::pair<int, int>> q = {{start_r, start_c}};
    std::set<std::pair<int, int>> visited_rivers = {{start_r, start_c}};
    std::set<std::pair<int, int>> destinations;

    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop_front();
        auto river = board[r][c];
        if (!river || river->side != "river") continue;

        std::vector<std::pair<int, int>> dirs;
        if (river->orientation == "vertical") dirs = {{-1, 0}, {1, 0}};
        else dirs = {{0, -1}, {0, 1}};

        for (auto [dr, dc] : dirs) {
            int cr = r + dr;
            int cc = c + dc;
            while (in_bounds(cc, cr, rows, cols)) {
                if (is_opponent_score_cell(cc, cr, player, rows, cols, score_cols)) break;
                
                auto cell = board[cr][cc];
                if (!cell) {
                    destinations.insert({cr, cc});
                } else {
                    if (cell->side == "river" && visited_rivers.find({cr, cc}) == visited_rivers.end()) {
                        visited_rivers.insert({cr, cc});
                        q.push_back({cr, cc});
                    }
                    break;
                }
                cr += dr;
                cc += dc;
            }
        }
    }
    return std::vector<std::pair<int, int>>(destinations.begin(), destinations.end());
}

std::vector<std::pair<int, int>> StudentAgent::_trace_river_push(const Board& board, int target_r, int target_c, const PiecePtr& river_piece, const std::string& pushed_player, int rows, int cols, const std::vector<int>& score_cols) const {
    std::set<std::pair<int, int>> destinations;
    std::vector<std::pair<int, int>> dirs;
    if (river_piece->orientation == "vertical") dirs = {{-1, 0}, {1, 0}};
    else dirs = {{0, -1}, {0, 1}};

    for (auto [dr, dc] : dirs) {
        int cr = target_r + dr;
        int cc = target_c + dc;
        while (in_bounds(cc, cr, rows, cols)) {
            if (is_opponent_score_cell(cc, cr, pushed_player, rows, cols, score_cols)) break;
            
            auto cell = board[cr][cc];
            if (!cell) {
                destinations.insert({cr, cc});
            } else if (cell->side != "river") {
                break;
            }
            cr += dr;
            cc += dc;
        }
    }
    return std::vector<std::pair<int, int>>(destinations.begin(), destinations.end());
}


// ===================================================================
// PYBIND11 MODULE BINDINGS
// ===================================================================

// ===================================================================
// PYBIND11 MODULE BINDINGS
// ===================================================================

PYBIND11_MODULE(student_agent_module, m) {
    m.doc() = "pybind11 module for StudentAgent";

    // Bind the Piece struct
    // We use std::shared_ptr<Piece> because the Board type uses it
    py::class_<Piece, std::shared_ptr<Piece>>(m, "Piece")
        .def(py::init<std::string, std::string, std::string>(),
             py::arg("owner"), py::arg("side"), py::arg("orientation") = "")
        .def_readwrite("owner", &Piece::owner)
        .def_readwrite("side", &Piece::side)
        .def_readwrite("orientation", &Piece::orientation);

    // Bind the Move::ActionType enum
    py::enum_<Move::ActionType>(m, "ActionType")
        .value("MOVE", Move::ActionType::MOVE)
        .value("PUSH", Move::ActionType::PUSH)
        .value("FLIP", Move::ActionType::FLIP)
        .value("ROTATE", Move::ActionType::ROTATE)
        .export_values();

    // Bind the Move struct
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("action", &Move::action)
        .def_readwrite("from_pos", &Move::from) 
        .def_readwrite("to_pos", &Move::to)
        .def_readwrite("pushed_to", &Move::pushed_to)
        .def_readwrite("orientation", &Move::orientation);

    // --- DELETE THESE LINES ---
    // py::class_<BaseAgent>(m, "BaseAgent")
    //     .def(py::init<std::string>());
    // --------------------------

    // Bind the StudentAgent class
    // We remove the BaseAgent from the template arguments
    py::class_<StudentAgent>(m, "StudentAgent") 
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose,
             py::arg("board"), py::arg("rows"), py::arg("cols"), 
             py::arg("score_cols"), py::arg("current_player_time"), 
             py::arg("opponent_time"));
}