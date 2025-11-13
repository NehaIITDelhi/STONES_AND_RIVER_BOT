#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace py = pybind11;

struct Move {
    std::string action;
    std::vector<int> from_pos;
    std::vector<int> to_pos;
    std::vector<int> pushed_to;
    std::string orientation;
};

using ExternalBoard = std::vector<std::vector<std::map<std::string, std::string>>>;

enum Player { NONE = 0, CIRCLE = 1, SQUARE = 2 };
enum PieceType { EMPTY = 0, STONE = 1, RIVER = 2 };
enum Orientation { NULL_ORI = 0, HORIZONTAL = 1, VERTICAL = 2 };

struct Piece {
    Player owner = NONE;
    PieceType side = EMPTY;
    Orientation orientation = NULL_ORI;
    bool is_empty() const { return side == EMPTY; }
};

using InternalBoard = std::vector<std::vector<Piece>>;

struct UndoMove {
    Move move;
    Piece from_piece;
    Piece to_piece;
    Piece pushed_piece;
};

class StudentAgent {
public:
    explicit StudentAgent(std::string player_side) : move_count(0) {
        if (player_side == "circle") {
            my_player = CIRCLE;
            opponent_player = SQUARE;
        } else {
            my_player = SQUARE;
            opponent_player = CIRCLE;
        }
    }

    Move choose(const ExternalBoard& board, int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {

        this->rows = rows;
        this->cols = cols;
        this->score_cols = score_cols;
        this->my_score_row = (my_player == CIRCLE) ? 2 : rows - 3;
        this->opp_score_row = (my_player == CIRCLE) ? rows - 3 : 2;

        InternalBoard internal_board = convert_board(board);

        // 1. Check for immediate winning moves
        std::vector<Move> winning_moves = find_winning_moves(internal_board, my_player);
        if (!winning_moves.empty()) {
            move_count++;
            update_move_history(winning_moves[0], internal_board);
            return winning_moves[0];
        }

        // 2. Check for opponent's winning moves and block them
        std::vector<Move> opponent_winning_moves = find_winning_moves(internal_board, opponent_player);
        if (!opponent_winning_moves.empty()) {
            std::vector<Move> blocking_moves = find_blocking_moves(internal_board, opponent_winning_moves);
            if (!blocking_moves.empty()) {
                move_count++;
                update_move_history(blocking_moves[0], internal_board);
                return blocking_moves[0];
            }
        }

        // 3. Use opening strategy for early game
        if (move_count < 6) {
            Move opening_move = find_strong_opening_move(internal_board);
            if (!opening_move.action.empty()) {
                move_count++;
                update_move_history(opening_move, internal_board);
                return opening_move;
            }
        }

        // 4. Adaptive depth based on game phase and time
        int depth = calculate_adaptive_depth(current_player_time, move_count);

        auto legal_moves = generate_all_valid_moves(internal_board, my_player);
        if (legal_moves.empty()) {
            move_count++;
            Move no_move = {"move", {}, {}, {}, {}};
            update_move_history(no_move, internal_board);
            return no_move;
        }
        
        Move best_move = find_best_move(internal_board, legal_moves, depth);
        
        move_count++;
        update_move_history(best_move, internal_board);
        return best_move;
    }

private:
    Player my_player;
    Player opponent_player;
    int move_count;
    int rows, cols;
    std::vector<int> score_cols;
    int my_score_row;
    int opp_score_row;
    
    // Enhanced oscillation prevention with state tracking
    std::vector<Move> recent_moves;
    std::vector<uint64_t> recent_board_states; // Hash of board states
    std::vector<std::vector<int>> recent_pushed_pieces;
    std::vector<std::vector<int>> recent_moved_pieces; // Track pieces that were moved
    const int MAX_HISTORY = 8;
    const int MAX_PUSH_HISTORY = 6;
    const int MAX_STATE_HISTORY = 10;
    const int MAX_MOVE_HISTORY = 8;

    // Enhanced heuristic weights
    const double W_STONE_IN_SA = 1000.0;
    const double W_THREAT = 200.0;
    const double W_DISTANCE = 2.0;
    const double W_RIVER_MOBILITY = 0.5;
    const double W_BLOCKING = 3.0;
    const double W_CENTER_CONTROL = 0.3;
    const double W_DEVELOPMENT = 0.2;
    const double W_WIN = 100000.0;

    // Enhanced oscillation detection that considers all move types and board states
    bool would_cause_oscillation(const Move& move, const InternalBoard& board) {
        if (recent_moves.size() < 2) return false;
        
        // 1. Check for direct move-move oscillation (A->B, B->A)
        if (move.action == "move") {
            const Move& last_move = recent_moves.back();
            if (last_move.action == "move" &&
                move.from_pos == last_move.to_pos && 
                move.to_pos == last_move.from_pos) {
                return true;
            }
            
            // Check for 3-move cycles in move history
            if (recent_moves.size() >= 3) {
                const Move& prev1 = recent_moves[recent_moves.size()-1];
                const Move& prev2 = recent_moves[recent_moves.size()-2];
                
                if (prev1.action == "move" && prev2.action == "move" &&
                    move.from_pos == prev2.to_pos && move.to_pos == prev2.from_pos &&
                    prev1.from_pos == move.to_pos && prev1.to_pos == move.from_pos) {
                    return true;
                }
            }
        }
        
        // 2. Enhanced push oscillation detection
        if (move.action == "push") {
            int pushed_x = move.pushed_to[0];
            int pushed_y = move.pushed_to[1];
            
            // Check if we're pushing back a piece that was just pushed by us
            for (int i = std::max(0, (int)recent_moves.size() - 4); i < recent_moves.size(); ++i) {
                const auto& prev_move = recent_moves[i];
                if (prev_move.action == "push" && 
                    prev_move.pushed_to[0] == pushed_x && 
                    prev_move.pushed_to[1] == pushed_y &&
                    prev_move.from_pos == move.pushed_to) { // We pushed it to where it came from
                    return true;
                }
            }
            
            // Check if this piece was recently pushed multiple times (avoid ping-pong)
            int push_count = 0;
            for (const auto& pushed_piece : recent_pushed_pieces) {
                if (pushed_piece[0] == pushed_x && pushed_piece[1] == pushed_y) {
                    push_count++;
                    if (push_count >= 2) return true;
                }
            }
            
            // Check for push-move-push cycles
            if (recent_moves.size() >= 3) {
                const Move& last_move = recent_moves.back();
                const Move& second_last = recent_moves[recent_moves.size()-2];
                
                if (last_move.action == "move" && second_last.action == "push" &&
                    move.action == "push" && second_last.pushed_to == move.from_pos) {
                    // Pattern: push A to B, move from C to A, push from A back to B
                    return true;
                }
            }
        }
        
        // 3. Flip/rotate oscillation detection
        if (move.action == "flip" || move.action == "rotate") {
            // Check if we're flipping/rotating the same piece repeatedly
            int flip_count = 0;
            for (int i = std::max(0, (int)recent_moves.size() - 4); i < recent_moves.size(); ++i) {
                const auto& prev_move = recent_moves[i];
                if ((prev_move.action == "flip" || prev_move.action == "rotate") &&
                    prev_move.from_pos == move.from_pos) {
                    flip_count++;
                    if (flip_count >= 2) return true; // Don't flip same piece more than twice in recent moves
                }
            }
            
            // Check for flip-flip cycles on same piece
            if (recent_moves.size() >= 2) {
                const Move& last_move = recent_moves.back();
                if ((last_move.action == "flip" || last_move.action == "rotate") &&
                    last_move.from_pos == move.from_pos &&
                    move.action == "flip") {
                    return true; // Flipping same piece consecutively
                }
            }
        }
        
        // 4. Board state oscillation (prevent returning to recent board states)
        if (recent_board_states.size() >= 3) {
            // Simulate the move and check if it creates a board state we've seen recently
            InternalBoard temp_board = board;
            apply_move_inplace(temp_board, move);
            uint64_t new_state_hash = compute_board_state_hash(temp_board);
            
            for (const auto& old_hash : recent_board_states) {
                if (new_state_hash == old_hash) {
                    return true; // This move recreates a recent board state
                }
            }
        }
        
        // 5. Pattern-based oscillation detection
        if (detects_oscillation_pattern(move)) {
            return true;
        }
        
        // 6. Check if we're moving the same piece too frequently
        if (is_piece_overused(move.from_pos)) {
            return true;
        }
        
        return false;
    }
    
    // Compute a simple hash of the board state for oscillation detection
    uint64_t compute_board_state_hash(const InternalBoard& board) {
        uint64_t hash = 0;
        const uint64_t prime = 2654435761U;
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const auto& piece = board[y][x];
                hash = hash * prime + (static_cast<uint64_t>(piece.owner) << 16);
                hash = hash * prime + (static_cast<uint64_t>(piece.side) << 8);
                hash = hash * prime + static_cast<uint64_t>(piece.orientation);
            }
        }
        return hash;
    }
    
    // Enhanced pattern detection for various oscillation types
    bool detects_oscillation_pattern(const Move& move) {
        if (recent_moves.size() < 4) return false;
        
        // Check for circular patterns in the last 4 moves
        int n = recent_moves.size();
        std::vector<Move> last_four;
        for (int i = std::max(0, n-4); i < n; ++i) {
            last_four.push_back(recent_moves[i]);
        }
        
        // Pattern: A->B, C->D, B->A, D->C type oscillations
        if (last_four.size() == 4) {
            const Move& m1 = last_four[0];
            const Move& m2 = last_four[1]; 
            const Move& m3 = last_four[2];
            const Move& m4 = last_four[3];
            
            // Cross oscillation pattern
            if (m1.action == "move" && m2.action == "move" && 
                m3.action == "move" && m4.action == "move") {
                if (m1.to_pos == m4.from_pos && m4.to_pos == m1.from_pos &&
                    m2.to_pos == m3.from_pos && m3.to_pos == m2.from_pos) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // Check if a piece is being used too frequently
    bool is_piece_overused(const std::vector<int>& piece_pos) {
        int use_count = 0;
        for (const auto& moved_piece : recent_moved_pieces) {
            if (moved_piece[0] == piece_pos[0] && moved_piece[1] == piece_pos[1]) {
                use_count++;
                if (use_count >= 3) return true; // Used same piece 3+ times recently
            }
        }
        return false;
    }
    
    void update_move_history(const Move& move, const InternalBoard& board) {
        // Update move history
        recent_moves.push_back(move);
        if (recent_moves.size() > MAX_HISTORY) {
            recent_moves.erase(recent_moves.begin());
        }
        
        // Update pushed pieces history
        if (move.action == "push") {
            std::vector<int> pushed_piece = {move.pushed_to[0], move.pushed_to[1]};
            recent_pushed_pieces.push_back(pushed_piece);
            if (recent_pushed_pieces.size() > MAX_PUSH_HISTORY) {
                recent_pushed_pieces.erase(recent_pushed_pieces.begin());
            }
        }
        
        // Update moved pieces history
        if (move.action == "move" || move.action == "push") {
            std::vector<int> moved_piece = {move.from_pos[0], move.from_pos[1]};
            recent_moved_pieces.push_back(moved_piece);
            if (recent_moved_pieces.size() > MAX_MOVE_HISTORY) {
                recent_moved_pieces.erase(recent_moved_pieces.begin());
            }
        }
        
        // Update board state history (only for meaningful moves)
        if (move.action != "flip" && move.action != "rotate") {
            uint64_t current_hash = compute_board_state_hash(board);
            recent_board_states.push_back(current_hash);
            if (recent_board_states.size() > MAX_STATE_HISTORY) {
                recent_board_states.erase(recent_board_states.begin());
            }
        }
    }

    // Enhanced move selection with fallback options
    Move find_best_move(InternalBoard& board, const std::vector<Move>& moves_to_check, int depth) {
        if (moves_to_check.empty()) return Move();
        
        std::vector<Move> best_moves;
        std::vector<Move> safe_moves; // Moves that don't cause oscillation
        double best_score = std::numeric_limits<double>::lowest();

        // Move ordering for better alpha-beta performance
        std::vector<Move> sorted_moves = moves_to_check;
        std::sort(sorted_moves.begin(), sorted_moves.end(), [&](const Move& a, const Move& b) {
            return score_move_heuristic(a) > score_move_heuristic(b);
        });

        // First pass: collect safe moves and find best score among them
        for (const auto& move : sorted_moves) {
            if (!would_cause_oscillation(move, board)) {
                safe_moves.push_back(move);
                
                UndoMove undo = apply_move_inplace(board, move);
                double score = alpha_beta_search(board, depth - 1, std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), false);
                unmake_move_inplace(board, undo);
                
                if (score > best_score) {
                    best_score = score;
                    best_moves.clear();
                    best_moves.push_back(move);
                } else if (score == best_score) {
                    best_moves.push_back(move);
                }
            }
        }

        // If we have safe moves, use the best one
        if (!best_moves.empty()) {
            // Add some randomness to break ties and prevent predictable patterns
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, best_moves.size() - 1);
            return best_moves[dis(gen)];
        }
        
        // If no safe moves, use fallback strategies
        if (!safe_moves.empty()) {
            // Use the highest heuristic scored safe move
            return *std::max_element(safe_moves.begin(), safe_moves.end(), 
                [&](const Move& a, const Move& b) {
                    return score_move_heuristic(a) < score_move_heuristic(b);
                });
        }
        
        // If all moves cause oscillation, use diversification strategy
        return select_diversified_move(sorted_moves, board);
    }

    // Select a move that breaks oscillation patterns
    Move select_diversified_move(const std::vector<Move>& moves, const InternalBoard& board) {
        // Prefer moves that change the game state significantly
        std::vector<Move> candidates;
        
        for (const auto& move : moves) {
            // Prefer moves that don't involve recently touched pieces
            bool involves_recent_piece = false;
            for (const auto& recent_move : recent_moves) {
                if (recent_move.from_pos == move.from_pos || 
                    recent_move.to_pos == move.from_pos ||
                    (!move.pushed_to.empty() && recent_move.pushed_to == move.pushed_to)) {
                    involves_recent_piece = true;
                    break;
                }
            }
            
            if (!involves_recent_piece) {
                candidates.push_back(move);
            }
        }
        
        if (!candidates.empty()) {
            return candidates[0]; // Use first candidate that breaks the pattern
        }
        
        // Last resort: use a non-push move if possible
        for (const auto& move : moves) {
            if (move.action != "push") {
                return move;
            }
        }
        
        // Absolute last resort: use random move
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, moves.size() - 1);
        return moves[dis(gen)];
    }

    // Time management
    int calculate_adaptive_depth(float current_time, int move_count) {
        float time_per_move;
        
        if (move_count < 20) {
            time_per_move = 2.0f;
        } else if (move_count < 40) {
            time_per_move = 1.5f;
        } else {
            time_per_move = 1.0f;
        }
        
        float base_time;
        if (this->rows == 13) {
            base_time = 120.0f;
        } else if (this->rows == 15) {
            base_time = 180.0f;
        } else if (this->rows == 17) {
            base_time = 240.0f;
        } else {
            base_time = 180.0f;
        }
        
        float time_remaining_ratio = current_time / base_time;
        
        if (time_remaining_ratio > 0.8f) return 4;
        if (time_remaining_ratio > 0.5f) return 3;
        if (time_remaining_ratio > 0.3f) return 2;
        return 1;
    }

    // Threat detection and blocking
    std::vector<Move> find_blocking_moves(InternalBoard& board, const std::vector<Move>& opponent_threats) {
        std::vector<Move> blocking_moves;
        auto my_moves = generate_all_valid_moves(board, my_player);
        
        for (const auto& threat : opponent_threats) {
            int target_x, target_y;
            if (threat.action == "move") {
                target_x = threat.to_pos[0];
                target_y = threat.to_pos[1];
            } else if (threat.action == "push") {
                target_x = threat.pushed_to[0];
                target_y = threat.pushed_to[1];
            } else {
                continue;
            }
            
            for (const auto& my_move : my_moves) {
                if (my_move.action == "push" && 
                    my_move.pushed_to[0] == target_x && my_move.pushed_to[1] == target_y) {
                    blocking_moves.push_back(my_move);
                } else if (my_move.action == "move" &&
                          my_move.to_pos[0] == target_x && my_move.to_pos[1] == target_y) {
                    blocking_moves.push_back(my_move);
                }
            }
        }
        return blocking_moves;
    }

    // Opening strategy
    Move find_strong_opening_move(InternalBoard& board) {
        std::vector<std::pair<int, int>> strong_openings = {
            {cols/2, rows/2}, {cols/2-1, rows/2}, {cols/2+1, rows/2},
            {cols/2, rows/2-1}, {cols/2, rows/2+1}
        };
        
        for (auto pos : strong_openings) {
            int x = pos.first;
            int y = pos.second;
            if (in_bounds(x, y) && board[y][x].is_empty()) {
                auto moves = generate_all_valid_moves(board, my_player);
                for (const auto& move : moves) {
                    if (move.action == "move" && move.to_pos[0] == x && move.to_pos[1] == y) {
                        return move;
                    }
                }
            }
        }
        return Move();
    }

    // --- START MODIFIED SECTION ---

    // Enhanced evaluation function
    double evaluate_board(const InternalBoard& board) {
        double score = 0.0;
        
        int my_stones_in_sa = count_stones_in_sa(board, my_player);
        int opp_stones_in_sa = count_stones_in_sa(board, opponent_player);
        int win_threshold = score_cols.size();

        // --- AGGRESSIVE STATE LOGIC ---
        // Check if we have *exactly* one piece left to score, as requested.
        bool one_piece_left_to_win = (my_stones_in_sa == win_threshold - 1);
        // Also check if opponent is NOT a threat, as you suggested.
        bool opp_is_not_a_threat = (opp_stones_in_sa < win_threshold - 2);
        
        // Combine into one "Go for the Kill" mode
        // This is true only if we are 1 piece away AND the opponent is not about to win
        bool go_for_the_kill = (one_piece_left_to_win && opp_is_not_a_threat);


        // 1. Immediate scoring (highest priority)
        score += (my_stones_in_sa - opp_stones_in_sa) * W_STONE_IN_SA;
        
        // 2. Threat detection
        double my_threats = calculate_threat_score(board, my_player);
        double opp_threats = calculate_threat_score(board, opponent_player);
        
        // If in "kill mode", make threats to score 10x more valuable
        if (go_for_the_kill) {
            score += (my_threats * 10.0 - opp_threats);
        } else {
            score += (my_threats - opp_threats);
        }
        
        // 3. Positional evaluation
        // Pass the "kill mode" flag to the positional advantage function
        score += evaluate_positional_advantage(board, go_for_the_kill);
        
        // 4. Center control
        score += calculate_center_control(board, my_player);
        score -= calculate_center_control(board, opponent_player);
        
        // 5. Piece development
        score += calculate_piece_development(board, my_player);
        score -= calculate_piece_development(board, opponent_player);
        
        return score;
    }

    double calculate_threat_score(const InternalBoard& board, Player player) {
        double threat_score = 0.0;
        int sa_row = (player == my_player) ? my_score_row : opp_score_row;
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const auto& piece = board[y][x];
                if (piece.owner != player || piece.side != STONE) continue;
                
                if (can_reach_sa_in_one_move(board, x, y, player)) {
                    threat_score += W_THREAT;
                }
            }
        }
        return threat_score;
    }

    bool can_reach_sa_in_one_move(const InternalBoard& board, int x, int y, Player player) {
        int sa_row = (player == my_player) ? my_score_row : opp_score_row;
        
        // Check direct movement to SA
        for (int sx : score_cols) {
            if (board[sa_row][sx].is_empty()) {
                int dist = std::abs(x - sx) + std::abs(y - sa_row);
                if (dist == 1) return true;
            }
        }
        
        // Check river movement to SA
        std::vector<std::pair<int, int>> dirs = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        for (auto dir : dirs) {
            int dx = dir.first;
            int dy = dir.second;
            int nx = x + dx, ny = y + dy;
            if (!in_bounds(nx, ny)) continue;
            
            const auto& adj_piece = board[ny][nx];
            if (adj_piece.side == RIVER) {
                InternalBoard temp_board = board;
                auto flow = river_flow(temp_board, nx, ny, nx, ny, player, false);
                for (const auto& dest : flow) {
                    if (is_own_score_cell(dest.first, dest.second, player) && 
                        board[dest.second][dest.first].is_empty()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // This function signature is changed to accept the new flag
    double evaluate_positional_advantage(const InternalBoard& board, bool go_for_the_kill) {
        double advantage = 0.0;
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const auto& piece = board[y][x];
                if (piece.is_empty()) continue;
                
                if (piece.owner == my_player) {
                    // Distance to scoring area penalty
                    int min_dist = calculate_min_distance_to_sa(x, y, my_player);
                    
                    // If in "kill mode", heavily penalize being far from the goal
                    if (go_for_the_kill) {
                        advantage -= min_dist * W_DISTANCE * 5.0; // 5x penalty
                    } else {
                        advantage -= min_dist * W_DISTANCE;
                    }
                    
                    // River mobility bonus
                    if (piece.side == RIVER) {
                        InternalBoard temp_board = board;
                        auto flow = river_flow(temp_board, x, y, x, y, my_player, false);
                        advantage += flow.size() * W_RIVER_MOBILITY;
                    }
                    
                    // Blocking bonus
                    if (is_blocking_opponent_path(x, y)) {
                        // If in "kill mode", blocking is 90% less important
                        if (go_for_the_kill) {
                            advantage += W_BLOCKING * 0.1;
                        } else {
                            advantage += W_BLOCKING;
                        }
                    }
                    
                } else {
                    // Same calculations for opponent (subtracted)
                    int min_dist = calculate_min_distance_to_sa(x, y, opponent_player);
                    advantage += min_dist * W_DISTANCE;
                    
                    if (piece.side == RIVER) {
                        InternalBoard temp_board = board;
                        auto flow = river_flow(temp_board, x, y, x, y, opponent_player, false);
                        advantage -= flow.size() * W_RIVER_MOBILITY;
                    }
                    
                    if (is_blocking_my_path(x, y)) {
                        advantage -= W_BLOCKING;
                    }
                }
            }
        }
        return advantage;
    }

    // --- END MODIFIED SECTION ---

    int calculate_min_distance_to_sa(int x, int y, Player player) {
        int sa_row = (player == my_player) ? my_score_row : opp_score_row;
        int min_dist = 1000;
        
        for (int sx : score_cols) {
            int dist = std::abs(x - sx) + std::abs(y - sa_row);
            min_dist = std::min(min_dist, dist);
        }
        return min_dist;
    }

    bool is_blocking_opponent_path(int x, int y) const {
        int opp_sa_row = (my_player == CIRCLE) ? rows - 3 : 2;
        
        for (int sx : score_cols) {
            if ((my_player == CIRCLE && y > opp_sa_row) || 
                (my_player == SQUARE && y < opp_sa_row)) {
                return true;
            }
        }
        return false;
    }

    bool is_blocking_my_path(int x, int y) const {
        int my_sa_row = (my_player == CIRCLE) ? 2 : rows - 3;
        
        for (int sx : score_cols) {
            if ((my_player == CIRCLE && y < my_sa_row) || 
                (my_player == SQUARE && y > my_sa_row)) {
                return true;
            }
        }
        return false;
    }

    double calculate_center_control(const InternalBoard& board, Player player) {
        double control = 0.0;
        int center_x = cols / 2;
        int center_y = rows / 2;
        
        for (int y = center_y - 2; y <= center_y + 2; ++y) {
            for (int x = center_x - 2; x <= center_x + 2; ++x) {
                if (!in_bounds(x, y)) continue;
                const auto& piece = board[y][x];
                if (!piece.is_empty() && piece.owner == player) {
                    double dist = std::max(std::abs(x - center_x), std::abs(y - center_y));
                    control += (3.0 - dist) * W_CENTER_CONTROL;
                }
            }
        }
        return control;
    }

    double calculate_piece_development(const InternalBoard& board, Player player) {
        double development = 0.0;
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const auto& piece = board[y][x];
                if (piece.owner != player) continue;
                
                // Bonus for forward development
                if (player == CIRCLE) {
                    development += y * W_DEVELOPMENT;
                } else {
                    development += (rows - 1 - y) * W_DEVELOPMENT;
                }
                
                // Bonus for piece coordination
                int friendly_neighbors = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx, ny = y + dy;
                        if (in_bounds(nx, ny) && board[ny][nx].owner == player) {
                            friendly_neighbors++;
                        }
                    }
                }
                development += friendly_neighbors * W_DEVELOPMENT * 0.5;
            }
        }
        return development;
    }

    int count_stones_in_sa(const InternalBoard& board, Player player) {
        int count = 0;
        int sa_row = (player == my_player) ? my_score_row : opp_score_row;
        
        for (int x : score_cols) {
            const auto& piece = board[sa_row][x];
            if (!piece.is_empty() && piece.owner == player && piece.side == STONE) {
                count++;
            }
        }
        return count;
    }

    // Core search logic
    double alpha_beta_search(InternalBoard& board, int depth, double alpha, double beta, bool maximizing_player) {
        std::string winner = check_win(board);
        if (winner != "None") {
            return (winner == ((my_player == CIRCLE) ? "circle" : "square")) ? W_WIN : -W_WIN;
        }

        if (depth == 0) {
            return evaluate_board(board);
        }

        Player player_to_move = maximizing_player ? my_player : opponent_player;
        auto legal_moves = generate_all_valid_moves(board, player_to_move);
        if (legal_moves.empty()) {
            return 0.0;
        }

        // Move ordering
        std::sort(legal_moves.begin(), legal_moves.end(), [&](const Move& a, const Move& b) {
            return score_move_heuristic(a) > score_move_heuristic(b);
        });

        if (maximizing_player) {
            double max_eval = std::numeric_limits<double>::lowest();
            for (const auto& move : legal_moves) {
                UndoMove undo = apply_move_inplace(board, move);
                double eval = alpha_beta_search(board, depth - 1, alpha, beta, false);
                unmake_move_inplace(board, undo);
                
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
            return max_eval;
        } else {
            double min_eval = std::numeric_limits<double>::max();
            for (const auto& move : legal_moves) {
                UndoMove undo = apply_move_inplace(board, move);
                double eval = alpha_beta_search(board, depth - 1, alpha, beta, true);
                unmake_move_inplace(board, undo);

                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
            return min_eval;
        }
    }

    int score_move_heuristic(const Move& move) const {
        int score = 0;
        if (move.action == "move" && is_own_score_cell(move.to_pos[0], move.to_pos[1], my_player)) score += 100;
        if (move.action == "push" && is_own_score_cell(move.pushed_to[0], move.pushed_to[1], my_player)) score += 100;
        if (move.action == "push") score += 20;
        if (move.action == "move") score += 10;
        if (move.action == "flip" || move.action == "rotate") score += 5;
        return score;
    }

    // Move generation functions
    std::vector<Move> generate_all_valid_moves(InternalBoard& board, Player player) {
        std::vector<Move> all_moves;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (board[y][x].owner == player) {
                    auto piece_moves = compute_valid_moves_for_piece(board, x, y, player);
                    all_moves.insert(all_moves.end(), piece_moves.begin(), piece_moves.end());
                }
            }
        }
        return all_moves;
    }

    std::vector<Move> compute_valid_moves_for_piece(InternalBoard& board, int sx, int sy, Player player) {
        std::vector<Move> moves;
        const auto &piece = board[sy][sx];
        PieceType piece_side = piece.side;
        
        // Normal moves and pushes
        std::vector<std::pair<int, int>> dirs = {{1,0}, {-1,0}, {0, 1}, {0, -1}};
        for (auto dir : dirs) {
            int dx = dir.first;
            int dy = dir.second;
            int tx = sx + dx, ty = sy + dy;
            if (!in_bounds(tx, ty) || is_opponent_score_cell(tx, ty, player)) continue;
                
            const auto &target_cell = board[ty][tx];
                
            if (target_cell.is_empty()) {
                moves.push_back({"move", {sx, sy}, {tx, ty}, {}, ""});
            }
            else if (target_cell.side == RIVER) {
                auto flow = river_flow(board, tx, ty, sx, sy, player, false);
                for (auto &dest : flow) {
                    moves.push_back({"move", {sx, sy}, {dest.first, dest.second}, {}, ""});
                }
            }
            else if (target_cell.side == STONE) {
                if (piece_side == STONE) {
                    int px = tx + dx, py = ty + dy;
                    if (in_bounds(px, py) && board[py][px].is_empty() && !is_opponent_score_cell(px, py, player)) {
                        bool invalid_push = (target_cell.owner == opponent_player && is_own_score_cell(px, py, player));
                        if (!invalid_push) {
                            moves.push_back({"push", {sx, sy}, {tx, ty}, {px, py}, ""});
                        }
                    }
                }
                else if (piece_side == RIVER) {
                    auto flow = river_flow(board, tx, ty, sx, sy, player, true);
                    for (auto &dest : flow) {
                        bool invalid_push = (target_cell.owner == opponent_player && is_own_score_cell(dest.first, dest.second, player));
                        if (!invalid_push) {
                            moves.push_back({"push", {sx, sy}, {tx, ty}, {dest.first, dest.second}, ""});
                        }
                    }
                }
            }
        }
        
        // Flip and rotate with safety checks
        if (piece_side == STONE) {
            if (!is_own_score_cell(sx, sy, player)) { 
                for (std::string orientation_str : {"horizontal", "vertical"}) {
                    bool is_safe = true;
                    InternalBoard temp_board = board;
                    temp_board[sy][sx].side = RIVER;
                    temp_board[sy][sx].orientation = (orientation_str == "horizontal") ? HORIZONTAL : VERTICAL;
                    
                    auto flow = river_flow(temp_board, sx, sy, sx, sy, player, false);
                    for (auto &dest : flow) {
                        if (is_opponent_score_cell(dest.first, dest.second, player)) {
                            is_safe = false;
                            break;
                        }
                    }
                    if (is_safe) {
                        moves.push_back({"flip", {sx, sy}, {sx, sy}, {}, orientation_str});
                    }
                }
            }
        } else {
            moves.push_back({"flip", {sx, sy}, {sx, sy}, {}, "NULL"});

            bool is_safe = true;
            InternalBoard temp_board = board;
            temp_board[sy][sx].orientation = (piece.orientation == HORIZONTAL) ? VERTICAL : HORIZONTAL;
            
            auto flow = river_flow(temp_board, sx, sy, sx, sy, player, false);
            for (auto &dest : flow) {
                if (is_opponent_score_cell(dest.first, dest.second, player)) {
                    is_safe = false;
                    break;
                }
            }
            if (is_safe) {
                moves.push_back({"rotate", {sx, sy}, {sx, sy}, {}, ""});
            }
        }

        return moves;
    }

    std::vector<Move> find_winning_moves(InternalBoard& board, Player player) {
        std::vector<Move> winning_moves;
        auto legal_moves = generate_all_valid_moves(board, player);

        for (const auto& move : legal_moves) {
            UndoMove undo = apply_move_inplace(board, move);
            std::string winner = check_win(board);
            unmake_move_inplace(board, undo);

            if (winner != "None" && ((winner == "circle" && player == CIRCLE) || (winner == "square" && player == SQUARE))) {
                winning_moves.push_back(move);
            }
        }
        return winning_moves;
    }

    // River flow simulation
    std::vector<std::pair<int, int>> river_flow(InternalBoard& board, int rx, int ry, int sx, int sy, Player player, bool river_push) {
        std::vector<std::pair<int, int>> destinations;
        std::set<std::pair<int, int>> visited;
        std::vector<std::pair<int, int>> queue = {{rx, ry}};

        while (!queue.empty()) {
            auto pos = queue.back();
            queue.pop_back();
            int x = pos.first;
            int y = pos.second;

            if (!in_bounds(x, y)) continue;
            if (visited.count({x, y})) continue;
            visited.insert({x, y});

            const auto &cell = board[y][x];
            std::vector<std::pair<int,int>> dirs;

            if (river_push && x == rx && y == ry) {
                const auto &pushing_piece = board[sy][sx];
                if (pushing_piece.is_empty() || pushing_piece.side != RIVER) continue;
                dirs = (pushing_piece.orientation == HORIZONTAL) ? std::vector<std::pair<int,int>>{{1,0},{-1,0}}
                                                                  : std::vector<std::pair<int,int>>{{0,1},{0,-1}};
            } else {
                if (cell.is_empty() || cell.side != RIVER) {
                    if (!river_push && !is_opponent_score_cell(x, y, player)) destinations.push_back({x,y});
                    continue;
                }
                dirs = (cell.orientation == HORIZONTAL) ? std::vector<std::pair<int,int>>{{1,0},{-1,0}}
                                                         : std::vector<std::pair<int,int>>{{0,1},{0,-1}};
            }

            for (auto dir : dirs) {
                int dx = dir.first;
                int dy = dir.second;
                int nx = x + dx;
                int ny = y + dy;
                while (in_bounds(nx, ny)) {
                    if (is_opponent_score_cell(nx, ny, player)) break;
                    const auto &next_cell = board[ny][nx];
                    if (next_cell.is_empty()) {
                        destinations.push_back({nx, ny});
                    } else if (next_cell.side == RIVER) {
                        queue.push_back({nx, ny});
                        break;
                    } else {
                        break;
                    }
                    nx += dx;
                    ny += dy;
                }
            }
        }
        return destinations;
    }

    // Move application
    UndoMove apply_move_inplace(InternalBoard& board, const Move& move) {
        UndoMove undo;
        undo.move = move;
        int fx = move.from_pos[0], fy = move.from_pos[1];
        undo.from_piece = board[fy][fx];

        if (move.action == "move") {
            int tx = move.to_pos[0], ty = move.to_pos[1];
            undo.to_piece = board[ty][tx];
            board[ty][tx] = board[fy][fx];
            board[fy][fx] = Piece();
        }
        else if (move.action == "push") {
            int tx = move.to_pos[0], ty = move.to_pos[1];
            int px = move.pushed_to[0], py = move.pushed_to[1];
            undo.pushed_piece = board[ty][tx];
            undo.to_piece = board[py][px];
            board[py][px] = board[ty][tx];
            board[ty][tx] = board[fy][fx];
            board[fy][fx] = Piece();
            if (undo.from_piece.side == RIVER) {
                board[ty][tx].side = STONE;
                board[ty][tx].orientation = NULL_ORI;
            }
        }
        else if (move.action == "flip") {
            if (undo.from_piece.side == STONE) {
                board[fy][fx].side = RIVER;
                board[fy][fx].orientation = (move.orientation == "horizontal") ? HORIZONTAL : VERTICAL;
            } else {
                board[fy][fx].side = STONE;
                board[fy][fx].orientation = NULL_ORI;
            }
        }
        else if (move.action == "rotate") {
            board[fy][fx].orientation = (undo.from_piece.orientation == HORIZONTAL) ? VERTICAL : HORIZONTAL;
        }
        return undo;
    }

    void unmake_move_inplace(InternalBoard& board, const UndoMove& undo) {
        int fx = undo.move.from_pos[0], fy = undo.move.from_pos[1];
        if (undo.move.action == "move") {
            int tx = undo.move.to_pos[0], ty = undo.move.to_pos[1];
            board[fy][fx] = undo.from_piece;
            board[ty][tx] = undo.to_piece;
        }
        else if (undo.move.action == "push") {
            int tx = undo.move.to_pos[0], ty = undo.move.to_pos[1];
            int px = undo.move.pushed_to[0], py = undo.move.pushed_to[1];
            board[fy][fx] = undo.from_piece;
            board[ty][tx] = undo.pushed_piece;
            board[py][px] = undo.to_piece;
        }
        else if (undo.move.action == "flip" || undo.move.action == "rotate") {
            board[fy][fx] = undo.from_piece;
        }
    }

    // Utility functions
    InternalBoard convert_board(const ExternalBoard& external_board) {
        InternalBoard internal(rows, std::vector<Piece>(cols));
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (!external_board[y][x].empty()) {
                    const auto& p = external_board[y][x];
                    internal[y][x].owner = (p.at("owner") == "circle") ? CIRCLE : SQUARE;
                    internal[y][x].side = (p.at("side") == "stone") ? STONE : RIVER;
                    if (internal[y][x].side == RIVER) {
                        internal[y][x].orientation = (p.at("orientation") == "horizontal") ? HORIZONTAL : VERTICAL;
                    }
                }
            }
        }
        return internal;
    }

    bool in_bounds(int x, int y) const {
       return x >= 0 && x < cols && y >= 0 && y < rows;
    }

    bool is_in_score_cols(int x) const {
        return std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end();
    }

    bool is_opponent_score_cell(int x, int y, Player player) const {
        int opponent_sa_row = (player == my_player) ? opp_score_row : my_score_row;
        return y == opponent_sa_row && is_in_score_cols(x);
    }

    bool is_own_score_cell(int x, int y, Player player) const {
        int own_sa_row = (player == my_player) ? my_score_row : opp_score_row;
        return y == own_sa_row && is_in_score_cols(x);
    }

    std::string check_win(const InternalBoard& board) {
        int circle_count = 0;
        int square_count = 0;
        int circle_sa_row = (my_player == CIRCLE) ? my_score_row : opp_score_row;
        int square_sa_row = (my_player == SQUARE) ? my_score_row : opp_score_row;

        for (int x : score_cols) {
            const auto& c_piece = board[circle_sa_row][x];
            if (!c_piece.is_empty() && c_piece.owner == CIRCLE && c_piece.side == STONE) {
                circle_count++;
            }
            const auto& s_piece = board[square_sa_row][x];
            if (!s_piece.is_empty() && s_piece.owner == SQUARE && s_piece.side == STONE) {
                square_count++;
            }
        }
        
        int win_threshold = score_cols.size();
        if (circle_count >= win_threshold) return "circle";
        if (square_count >= win_threshold) return "square";
        return "None";
    }
};

PYBIND11_MODULE(student_agent_module, m) {
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("action", &Move::action)
        .def_readwrite("from_pos", &Move::from_pos)
        .def_readwrite("to_pos", &Move::to_pos)
        .def_readwrite("pushed_to", &Move::pushed_to)
        .def_readwrite("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose);
}