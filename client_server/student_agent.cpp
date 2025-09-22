#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <memory>
#include <set>
#include <queue>
#include <random>
#include <algorithm>
#include <stdexcept>

// Using the standard namespace to avoid prefixing with std::
using namespace std;

// Represents a game piece. Corresponds to the piece objects in Python.
struct Piece {
    string owner;
    string side; // "stone" or "river"
    string orientation; // "horizontal" or "vertical", for rivers

    // Constructor
    Piece(string o, string s, string orient = "")
        : owner(move(o)), side(move(s)), orientation(move(orient)) {}
};

// Type alias for the board. unique_ptr manages memory automatically,
// similar to Python's garbage collector. A nullptr represents an empty cell.
using Board = vector<vector<unique_ptr<Piece>>>;

// Represents a game move. This struct replaces the Python dictionaries for actions.
// optional is used for fields that may or may not be present for a given action.
struct Move {
    string action;
    pair<int, int> from;
    optional<pair<int, int>> to;
    optional<pair<int, int>> pushed_to;
    optional<string> orientation;
};

// Forward declarations of all functions
// ==================== GAME UTILITIES ====================
bool in_bounds(int x, int y, int rows, int cols);
vector<int> score_cols_for(int cols);
int top_score_row();
int bottom_score_row(int rows);
bool is_opponent_score_cell(int x, int y, const string& player, int rows, int cols, const vector<int>& score_cols);
bool is_own_score_cell(int x, int y, const string& player, int rows, int cols, const vector<int>& score_cols);
string get_opponent(const string& player);

// ==================== MOVE GENERATION HELPERS ====================
vector<pair<int, int>> _trace_river_flow(const Board& board, int river_start_x, int river_start_y, int original_piece_x, int original_piece_y, const string& whos_turn, int rows, int cols, const vector<int>& score_cols, bool is_a_push_move = false);
vector<Move> get_valid_moves_for_piece(const Board& board, int start_pos_x, int start_pos_y, const string& current_player, int rows, int cols, const vector<int>& score_cols);
bool _are_boards_the_same(const Board& board1, const Board& board2, int rows, int cols);
vector<Move> generate_all_moves(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols);

// ==================== BOARD EVALUATION ====================
int count_stones_in_scoring_area(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols);
double basic_evaluate_board(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols);

// ==================== SIMULATION & AGENT CLASSES ====================
Board deep_copy_board(const Board& board);
pair<bool, Board> simulate_move(const Board& board, const Move& move, const string& player, int rows, int cols, const vector<int>& score_cols);
pair<bool, string> validate_and_apply_move(Board& board, const Move& move, const string& player, int rows, int cols, const vector<int>& score_cols);


// ==================== GAME UTILITIES IMPLEMENTATION ====================

/**
 * @brief Check if coordinates are within board boundaries.
 */
bool in_bounds(int x, int y, int rows, int cols) {
    return 0 <= x && x < cols && 0 <= y && y < rows;
}

/**
 * @brief Get the column indices for scoring areas.
 */
vector<int> score_cols_for(int cols) {
    int w = 4;
    int start = max(0, (cols - w) / 2);
    vector<int> result;
    for (int i = start; i < start + w; ++i) {
        result.push_back(i);
    }
    return result;
}

/**
 * @brief Get the row index for Circle's scoring area.
 */
int top_score_row() {
    return 2;
}

/**
 * @brief Get the row index for Square's scoring area.
 */
int bottom_score_row(int rows) {
    return rows - 3;
}

/**
 * @brief Check if a cell is in the opponent's scoring area.
 */
bool is_opponent_score_cell(int x, int y, const string& player, int rows, int cols, const vector<int>& score_cols) {
    int score_row = (player == "circle") ? bottom_score_row(rows) : top_score_row();
    return y == score_row && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
}

/**
 * @brief Check if a cell is in the player's own scoring area.
 */
bool is_own_score_cell(int x, int y, const string& player, int rows, int cols, const vector<int>& score_cols) {
    int score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    return y == score_row && (find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
}

/**
 * @brief Get the opponent player identifier.
 */
string get_opponent(const string& player) {
    return (player == "circle") ? "square" : "circle";
}

// ==================== MOVE GENERATION HELPERS IMPLEMENTATION ====================

/**
 * @brief Helper function to trace all possible paths along a river or a chain of rivers.
 */

vector<pair<int, int>> _trace_river_flow(
    const Board& board, int river_start_x, int river_start_y,
    int original_piece_x, int original_piece_y, const string& whos_turn,
    int rows, int cols, const vector<int>& score_cols, bool is_a_push_move) {
    
    vector<pair<int, int>> possible_landings;
    queue<pair<int, int>> coords_to_check;
    set<pair<int, int>> checked_coords;

    coords_to_check.push({river_start_x, river_start_y});

    while (!coords_to_check.empty()) {
        auto [current_x, current_y] = coords_to_check.front();
        coords_to_check.pop();

        if (checked_coords.count({current_x, current_y}) || !in_bounds(current_x, current_y, rows, cols)) {
            continue;
        }
        checked_coords.insert({current_x, current_y});

        const Piece* current_cell_piece = board[current_y][current_x].get();
        if (is_a_push_move && current_x == river_start_x && current_y == river_start_y) {
            current_cell_piece = board[original_piece_y][original_piece_x].get();
        }
        
        if (current_cell_piece == nullptr) {
            if (!is_opponent_score_cell(current_x, current_y, whos_turn, rows, cols, score_cols)) {
                 if (find(possible_landings.begin(), possible_landings.end(), make_pair(current_x, current_y)) == possible_landings.end()){
                    possible_landings.push_back({current_x, current_y});
                 }
            }
            continue;
        }

        if (current_cell_piece->side != "river") {
            continue;
        }

        vector<pair<int, int>> flow_directions = (current_cell_piece->orientation == "horizontal") ? 
            vector<pair<int, int>>{{1, 0}, {-1, 0}} : vector<pair<int, int>>{{0, 1}, {0, -1}};

        for (auto [move_x, move_y] : flow_directions) {
            int new_x = current_x + move_x;
            int new_y = current_y + move_y;

            while (in_bounds(new_x, new_y, rows, cols)) {
                if (is_opponent_score_cell(new_x, new_y, whos_turn, rows, cols, score_cols)) {
                    break;
                }

                const Piece* cell_in_path = board[new_y][new_x].get();

                if (cell_in_path == nullptr) {
                    if (find(possible_landings.begin(), possible_landings.end(), make_pair(new_x, new_y)) == possible_landings.end()){
                       possible_landings.push_back({new_x, new_y});
                    }
                    new_x += move_x;
                    new_y += move_y;
                    continue;
                }

                if (new_x == original_piece_x && new_y == original_piece_y) {
                    new_x += move_x;
                    new_y += move_y;
                    continue;
                }

                if (cell_in_path->side == "river") {
                    if (!checked_coords.count({new_x, new_y})) {
                        coords_to_check.push({new_x, new_y});
                    }
                    break;
                }
                break; // Stop if it's a stone
            }
        }
    }
    return possible_landings;
}


/**
 * @brief Computes all valid moves for a single piece based on assignment rules.
 */
vector<Move> get_valid_moves_for_piece(
    const Board& board, int start_pos_x, int start_pos_y,
    const string& current_player, int rows, int cols, const vector<int>& score_cols) {
    
    const Piece* my_piece = board[start_pos_y][start_pos_x].get();
    vector<Move> all_possible_actions;

    if (my_piece == nullptr || my_piece->owner != current_player) {
        return all_possible_actions;
    }
    
    // --- Part 1: Calculate Moves and Pushes ---
    vector<pair<int, int>> move_options = {{1, 0}, {-1, 0}, {0, -1}, {0, 1}};

    for (auto [dx, dy] : move_options) {
        int target_x = start_pos_x + dx;
        int target_y = start_pos_y + dy;
        
        if (!in_bounds(target_x, target_y, rows, cols) || is_opponent_score_cell(target_x, target_y, current_player, rows, cols, score_cols)) {
            continue;
        }

        const Piece* target_square = board[target_y][target_x].get();

        if (target_square == nullptr) {
            // Simple move to an empty square
            all_possible_actions.push_back({ "move", {start_pos_x, start_pos_y}, {{target_x, target_y}}, nullopt, nullopt });
        } else if (target_square->side == "river") {
            // Move along a river
            auto river_landings = _trace_river_flow(board, target_x, target_y, start_pos_x, start_pos_y, current_player, rows, cols, score_cols);
            for (auto [lx, ly] : river_landings) {
                 all_possible_actions.push_back({ "move", {start_pos_x, start_pos_y}, {{lx, ly}}, nullopt, nullopt });
            }
        } else if (target_square->side == "stone") { 
            // **CORRECTION**: A push is only possible if the target piece is a stone.
            
            if (my_piece->side == "stone") {
                // **Stone Pushing a Stone**
                int push_dest_x = target_x + dx;
                int push_dest_y = target_y + dy;
                // Check if the destination square is empty and valid
                if (in_bounds(push_dest_x, push_dest_y, rows, cols) && board[push_dest_y][push_dest_x] == nullptr) {
                    if (!is_opponent_score_cell(push_dest_x, push_dest_y, current_player, rows, cols, score_cols)) {
                        all_possible_actions.push_back({ "push", {start_pos_x, start_pos_y}, {{target_x, target_y}}, {{push_dest_x, push_dest_y}}, nullopt });
                    }
                }
            } else { // my_piece is a river
                // **River Pushing a Stone**
                string owner_of_pushed_piece = target_square->owner;
                auto river_landings = _trace_river_flow(board, target_x, target_y, start_pos_x, start_pos_y, owner_of_pushed_piece, rows, cols, score_cols, true);
                for (auto [lx, ly] : river_landings) {
                    all_possible_actions.push_back({ "push", {start_pos_x, start_pos_y}, {{target_x, target_y}}, {{lx, ly}}, nullopt });
                }
            }
        }
    }
    
    // --- Part 2: Add Flip and Rotate Actions ---
    if (my_piece->side == "stone") {
        // A stone can be flipped to a river (horizontal or vertical)
        all_possible_actions.push_back({ "flip", {start_pos_x, start_pos_y}, nullopt, nullopt, "horizontal" });
        all_possible_actions.push_back({ "flip", {start_pos_x, start_pos_y}, nullopt, nullopt, "vertical" });
    } else { // The piece is a river
        // A river can be flipped to a stone
        all_possible_actions.push_back({ "flip", {start_pos_x, start_pos_y}, nullopt, nullopt, nullopt });
        // A river can also be rotated
        all_possible_actions.push_back({ "rotate", {start_pos_x, start_pos_y}, nullopt, nullopt, nullopt });
    }

    return all_possible_actions;
}

/**
 * @brief Checks if two board states are identical.
 */
bool _are_boards_the_same(const Board& board1, const Board& board2, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const Piece* p1 = board1[r][c].get();
            const Piece* p2 = board2[r][c].get();
            if (p1 == nullptr && p2 == nullptr) continue;
            if (p1 == nullptr || p2 == nullptr) return false;
            if (p1->owner != p2->owner || p1->side != p2->side || p1->orientation != p2->orientation) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Generate all legal moves for the current player.
 */
vector<Move> generate_all_moves(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    vector<Move> all_moves;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (board[y][x] != nullptr && board[y][x]->owner == player) {
                auto piece_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols);
                all_moves.insert(all_moves.end(), piece_moves.begin(), piece_moves.end());
            }
        }
    }
    return all_moves;
}


// ==================== BOARD EVALUATION IMPLEMENTATION ====================

/**
 * @brief Count how many stones a player has in their scoring area.
 */
int count_stones_in_scoring_area(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    int count = 0;
    int score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    
    for (int x : score_cols) {
        if (in_bounds(x, score_row, rows, cols)) {
            const Piece* piece = board[score_row][x].get();
            if (piece && piece->owner == player && piece->side == "stone") {
                count++;
            }
        }
    }
    return count;
}

/**
 * @brief Basic board evaluation function.
 */
double basic_evaluate_board(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    string opponent_player = get_opponent(player);

    int player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols);
    int opponent_scoring_stones = count_stones_in_scoring_area(board, opponent_player, rows, cols, score_cols);

    score += player_scoring_stones * 100.0;
    score -= opponent_scoring_stones * 100.0;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const Piece* piece = board[y][x].get();
            if (piece && piece->owner == player && piece->side == "stone") {
                if (player == "circle") {
                    score += (rows - y) * 0.1;
                } else {
                    score += y * 0.1;
                }
            }
        }
    }
    return score;
}

// ==================== ENHANCED BOARD EVALUATION IMPLEMENTATION ====================

/**
 * @brief Count stones that can reach scoring area in one step
 */
int count_stones_one_move_from_scoring(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    int threat_count = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const auto& piece = board[y][x];
            if (piece && piece->owner == player) {
                // Generate all possible moves for this single piece
                // This requires a full move generator, which is assumed to exist.
                vector<Move> possible_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols);

                for (const auto& move : possible_moves) {
                    bool is_a_threat = false;
                    if (move.action == "move" && move.to.has_value()) {
                        if (is_own_score_cell(move.to->first, move.to->second, player, rows, cols, score_cols)) {
                            is_a_threat = true;
                        }
                    } else if (move.action == "push" && move.pushed_to.has_value()) {
                        // A push is a threat if the piece *being pushed* lands in the goal.
                        // We also need to check if that piece is a stone.
                        auto pushed_coord = move.to.value();
                        const auto& pushed_piece = board[pushed_coord.second][pushed_coord.first];
                        if (pushed_piece && pushed_piece->side == "stone") {
                             if (is_own_score_cell(move.pushed_to->first, move.pushed_to->second, player, rows, cols, score_cols)) {
                                is_a_threat = true;
                            }
                        }
                    } else if (move.action == "flip") {
                        // A flip is a threat if the piece is already in the scoring area and is a river.
                        if(piece->side == "river" && is_own_score_cell(x, y, player, rows, cols, score_cols)){
                            is_a_threat = true;
                        }
                    }
                    
                    if(is_a_threat){
                        threat_count++;
                        break; // Count this piece only once, even if it has multiple ways to score.
                    }
                }
            }
        }
    }
    return threat_count;
}

/**
 * @brief Calculate mobility score based on available moves
 */
double calculate_mobility_score(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    auto moves = generate_all_moves(board, player, rows, cols, score_cols);
    return moves.size() * 0.5; // Weight mobility appropriately
}

/**
 * @brief Evaluate piece positioning and formation
 */
double evaluate_piece_positioning(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    int target_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const Piece* piece = board[y][x].get();
            if (piece && piece->owner == player) {
                if (piece->side == "stone") {
                    // Reward stones closer to scoring area
                    double distance_bonus = 0.0;
                    if (player == "circle") {
                        // Circle wants to move down (increasing y)
                        distance_bonus = (double)(rows - y) / rows * 5.0;
                    } else {
                        // Square wants to move up (decreasing y)
                        distance_bonus = (double)y / rows * 5.0;
                    }
                    score += distance_bonus;
                    
                    // Bonus for stones in scoring columns
                    if (find(score_cols.begin(), score_cols.end(), x) != score_cols.end()) {
                        score += 3.0;
                    }
                    
                    // Penalty for stones far from center columns
                    int center_x = cols / 2;
                    double center_distance = abs(x - center_x);
                    score -= center_distance * 0.2;
                    
                } else { // River piece
                    // Evaluate river positioning
                    // Rivers are valuable for enabling movement
                    score += 2.0;
                    
                    // Bonus for rivers that create pathways toward scoring area
                    if (piece->orientation == "horizontal") {
                        // Horizontal rivers help lateral movement
                        score += 1.5;
                    } else {
                        // Vertical rivers help forward/backward movement
                        if (player == "circle" && y < target_row) {
                            score += 2.0; // Helps circle move toward their goal
                        } else if (player == "square" && y > target_row) {
                            score += 2.0; // Helps square move toward their goal
                        }
                    }
                }
            }
        }
    }
    
    return score;
}

/**
 * @brief Evaluate control of key board areas
 */
double evaluate_board_control(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    string opponent = get_opponent(player);
    
    // Control of center area
    int center_x = cols / 2;
    int center_y = rows / 2;
    
    for (int y = center_y - 2; y <= center_y + 2; ++y) {
        for (int x = center_x - 2; x <= center_x + 2; ++x) {
            if (in_bounds(x, y, rows, cols)) {
                const Piece* piece = board[y][x].get();
                if (piece) {
                    if (piece->owner == player) {
                        score += 1.0;
                    } else if (piece->owner == opponent) {
                        score -= 0.5;
                    }
                }
            }
        }
    }
    
    // Control of scoring approach lanes
    for (int x : score_cols) {
        int approach_row = (player == "circle") ? top_score_row() + 1 : bottom_score_row(rows) - 1;
        if (in_bounds(x, approach_row, rows, cols)) {
            const Piece* piece = board[approach_row][x].get();
            if (piece && piece->owner == player) {
                score += 3.0;
            }
        }
    }
    
    return score;
}

/**
 * @brief Evaluate defensive positioning
 */
double evaluate_defensive_position(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    string opponent = get_opponent(player);
    
    // Check if opponent stones are threatening our scoring area
    int our_score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    
    // Count opponent threats
    int opponent_threats = count_stones_one_move_from_scoring(board, opponent, rows, cols, score_cols);
    score -= opponent_threats * 8.0; // Heavy penalty for opponent threats
    
    // Bonus for blocking opponent paths
    for (int x : score_cols) {
        // Check positions that could block opponent approach
        int block_row = (player == "circle") ? top_score_row() - 1 : bottom_score_row(rows) + 1;
        if (in_bounds(x, block_row, rows, cols)) {
            const Piece* piece = board[block_row][x].get();
            if (piece && piece->owner == player) {
                score += 2.0;
            }
        }
    }
    
    return score;
}

/**
 * @brief Check for immediate winning or losing positions
 */
double evaluate_immediate_threats(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    
    int player_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols);
    int opponent_stones = count_stones_in_scoring_area(board, get_opponent(player), rows, cols, score_cols);
    
    // Massive bonus/penalty for being close to winning/losing
    if (player_stones == 3) {
        score += 500.0; // Very close to winning
    } else if (player_stones == 2) {
        score += 150.0;
    }
    
    if (opponent_stones == 3) {
        score -= 600.0; // Opponent very close to winning - defend!
    } else if (opponent_stones == 2) {
        score -= 200.0;
    }
    
    return score;
}

/**
 * @brief Comprehensive board evaluation function
 */
double advanced_evaluate_board(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double total_score = 0.0;
    string opponent = get_opponent(player);
    
    // 1. Primary scoring metric - stones in scoring area (highest weight)
    int player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols);
    int opponent_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols);
    total_score += player_scoring_stones * 1000.0;
    total_score -= opponent_scoring_stones * 1000.0;
    
    // 2. Secondary scoring metric - stones one move from scoring
    int player_ready_stones = count_stones_one_move_from_scoring(board, player, rows, cols, score_cols);
    int opponent_ready_stones = count_stones_one_move_from_scoring(board, opponent, rows, cols, score_cols);
    total_score += player_ready_stones * 50.0;
    total_score -= opponent_ready_stones * 50.0;
    
    // 3. Immediate threat evaluation (game-ending positions)
    total_score += evaluate_immediate_threats(board, player, rows, cols, score_cols);
    
    // 4. Piece positioning and formation
    total_score += evaluate_piece_positioning(board, player, rows, cols, score_cols);
    total_score -= evaluate_piece_positioning(board, opponent, rows, cols, score_cols) * 0.8;
    
    // 5. Board control and territorial advantage
    total_score += evaluate_board_control(board, player, rows, cols, score_cols);
    
    // 6. Defensive positioning
    total_score += evaluate_defensive_position(board, player, rows, cols, score_cols);
    
    // 7. Mobility and tactical flexibility
    total_score += calculate_mobility_score(board, player, rows, cols, score_cols);
    total_score -= calculate_mobility_score(board, opponent, rows, cols, score_cols) * 0.7;
    
    // 8. Endgame considerations
    int total_stones_in_play = 0;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (board[y][x] != nullptr) {
                total_stones_in_play++;
            }
        }
    }
    
    // In endgame (fewer pieces), prioritize direct scoring paths
    if (total_stones_in_play < 16) {
        total_score += player_ready_stones * 30.0; // Extra bonus for ready stones
        total_score -= opponent_ready_stones * 35.0; // Extra penalty for opponent ready stones
    }
    
    return total_score;
}

/**
 * @brief Quick evaluation function for time-critical situations
 */
double quick_evaluate_board(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    double score = 0.0;
    string opponent = get_opponent(player);
    
    // Focus only on most critical factors
    int player_scoring = count_stones_in_scoring_area(board, player, rows, cols, score_cols);
    int opponent_scoring = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols);
    int player_ready = count_stones_one_move_from_scoring(board, player, rows, cols, score_cols);
    int opponent_ready = count_stones_one_move_from_scoring(board, opponent, rows, cols, score_cols);
    
    score += player_scoring * 1000.0;
    score -= opponent_scoring * 1000.0;
    score += player_ready * 50.0;
    score -= opponent_ready * 50.0;
    
    // Quick threat assessment
    if (opponent_scoring == 3) score -= 500.0;
    if (player_scoring == 3) score += 400.0;
    
    return score;
}

// ==================== SIMULATION & AGENT IMPLEMENTATION ====================

/**
 * @brief Creates a deep copy of the board state. Necessary for simulation.
 */
Board deep_copy_board(const Board& board) {
    Board new_board(board.size());
    for (size_t i = 0; i < board.size(); ++i) {
        new_board[i].resize(board[i].size());
        for (size_t j = 0; j < board[i].size(); ++j) {
            if (board[i][j] != nullptr) {
                new_board[i][j] = make_unique<Piece>(*board[i][j]);
            } else {
                new_board[i][j] = nullptr;
            }
        }
    }
    return new_board;
}

/**
 * @brief Placeholder for the gameEngine's validate_and_apply_move function.
 * This is a simplified version for agent testing.
 */
pair<bool, string> validate_and_apply_move(
    Board& board, const Move& move, const string& player,
    int rows, int cols, const vector<int>& score_cols) {

    auto [from_x, from_y] = move.from;
    
    // Basic validation
    if (!in_bounds(from_x, from_y, rows, cols) || !board[from_y][from_x] || board[from_y][from_x]->owner != player) {
        return {false, "Invalid source piece."};
    }
    
    if (move.action == "move") {
        auto [to_x, to_y] = move.to.value();
        board[to_y][to_x] = std::move(board[from_y][from_x]);
    } else if (move.action == "push") {
        auto [to_x, to_y] = move.to.value();
        auto [pushed_to_x, pushed_to_y] = move.pushed_to.value();
        // The rule that the pushing river flips to a stone would be implemented here in a full game engine.
        board[pushed_to_y][pushed_to_x] = std::move(board[to_y][to_x]);
        board[to_y][to_x] = std::move(board[from_y][from_x]);
    } else if (move.action == "flip") {
        if (board[from_y][from_x]->side == "stone") {
            board[from_y][from_x]->side = "river";
            board[from_y][from_x]->orientation = move.orientation.value();
        } else {
            board[from_y][from_x]->side = "stone";
            board[from_y][from_x]->orientation = "";
        }
    } else if (move.action == "rotate") {
         board[from_y][from_x]->orientation = (board[from_y][from_x]->orientation == "horizontal") ? "vertical" : "horizontal";
    }

    return {true, ""};
}

/**
 * @brief Simulate a move on a copy of the board.
 */
pair<bool, Board> simulate_move(const Board& board, const Move& move, const string& player, int rows, int cols, const vector<int>& score_cols) {
    Board board_copy = deep_copy_board(board);
    auto [success, message] = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols);
    if (success) {
        return {true, std::move(board_copy)};
    }
    // In a real scenario, you'd handle the error message. Here we return original board.
    return {false, deep_copy_board(board)};
}


/**
 * @brief Abstract base class for all agents.
 */
class BaseAgent {
public:
    string player;
    string opponent;

    BaseAgent(string p) : player(move(p)), opponent(get_opponent(player)) {}
    virtual ~BaseAgent() = default;

    virtual optional<Move> choose(const Board& board, int rows, int cols, const vector<int>& score_cols, double current_player_time, double opponent_time) = 0;
};

/**
 * @brief Student Agent Implementation.
 */
class StudentAgent : public BaseAgent {
public:
    StudentAgent(string p) : BaseAgent(move(p)) {
        // TODO: Add any initialization you need
    }

    /**
     * @brief Choose the best move for the current board state.
     */
    optional<Move> choose(const Board& board, int rows, int cols, const vector<int>& score_cols, double current_player_time, double opponent_time) override {
        auto moves = generate_all_moves(board, this->player, rows, cols, score_cols);

        if (moves.empty()) {
            return nullopt; // Corresponds to returning None
        }

        // TODO: Replace random selection with your AI algorithm
        // C++ equivalent of random.choice(moves)
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(0, moves.size() - 1);
        return moves[distrib(gen)];
    }
};

// ==================== ENHANCED SIMULATION & AGENT IMPLEMENTATION ====================

/**
 * @brief Creates a deep copy of the board state. Necessary for simulation.
 */
Board deep_copy_board(const Board& board) {
    Board new_board(board.size());
    for (size_t i = 0; i < board.size(); ++i) {
        new_board[i].resize(board[i].size());
        for (size_t j = 0; j < board[i].size(); ++j) {
            if (board[i][j] != nullptr) {
                new_board[i][j] = make_unique<Piece>(
                    board[i][j]->owner, 
                    board[i][j]->side, 
                    board[i][j]->orientation
                );
            } else {
                new_board[i][j] = nullptr;
            }
        }
    }
    return new_board;
}

/**
 * @brief Comprehensive move validation and application function.
 * This handles all the complex game rules from the assignment.
 */
pair<bool, string> validate_and_apply_move(
    Board& board, const Move& move, const string& player,
    int rows, int cols, const vector<int>& score_cols) {

    auto [from_x, from_y] = move.from;
    
    // Basic validation - check source piece
    if (!in_bounds(from_x, from_y, rows, cols)) {
        return {false, "Source position out of bounds."};
    }
    
    Piece* source_piece = board[from_y][from_x].get();
    if (!source_piece) {
        return {false, "No piece at source position."};
    }
    
    if (source_piece->owner != player) {
        return {false, "Source piece does not belong to current player."};
    }

    // Handle different move types
    if (move.action == "move") {
        if (!move.to.has_value()) {
            return {false, "Move action requires destination."};
        }
        
        auto [to_x, to_y] = move.to.value();
        
        if (!in_bounds(to_x, to_y, rows, cols)) {
            return {false, "Destination out of bounds."};
        }
        
        if (is_opponent_score_cell(to_x, to_y, player, rows, cols, score_cols)) {
            return {false, "Cannot move into opponent's scoring area."};
        }
        
        if (board[to_y][to_x] != nullptr) {
            return {false, "Destination already occupied."};
        }
        
        // Validate move is legal (adjacent or via river)
        int dx = abs(to_x - from_x);
        int dy = abs(to_y - from_y);
        
        if (dx + dy == 1) {
            // Simple adjacent move - always valid if destination is empty
            board[to_y][to_x] = std::move(board[from_y][from_x]);
            return {true, ""};
        } else {
            // Must be a river move - validate path exists
            vector<pair<int, int>> valid_destinations = _trace_river_flow(
                board, from_x + (to_x > from_x ? 1 : (to_x < from_x ? -1 : 0)), 
                from_y + (to_y > from_y ? 1 : (to_y < from_y ? -1 : 0)),
                from_x, from_y, player, rows, cols, score_cols
            );
            
            if (find(valid_destinations.begin(), valid_destinations.end(), make_pair(to_x, to_y)) == valid_destinations.end()) {
                return {false, "Invalid river move - destination not reachable."};
            }
            
            board[to_y][to_x] = std::move(board[from_y][from_x]);
            return {true, ""};
        }
    }
    else if (move.action == "push") {
        if (!move.to.has_value() || !move.pushed_to.has_value()) {
            return {false, "Push action requires target and push destination."};
        }
        
        auto [to_x, to_y] = move.to.value();
        auto [pushed_to_x, pushed_to_y] = move.pushed_to.value();
        
        if (!in_bounds(to_x, to_y, rows, cols) || !in_bounds(pushed_to_x, pushed_to_y, rows, cols)) {
            return {false, "Push positions out of bounds."};
        }
        
        if (board[to_y][to_x] == nullptr) {
            return {false, "No piece to push at target position."};
        }
        
        if (board[to_y][to_x]->side != "stone") {
            return {false, "Can only push stone pieces."};
        }
        
        if (board[pushed_to_y][pushed_to_x] != nullptr) {
            return {false, "Push destination already occupied."};
        }
        
        if (is_opponent_score_cell(pushed_to_x, pushed_to_y, board[to_y][to_x]->owner, rows, cols, score_cols)) {
            return {false, "Cannot push piece into opponent's scoring area."};
        }
        
        // Validate push direction and distance
        if (source_piece->side == "stone") {
            // Stone push: exactly one step in same direction
            int dx = to_x - from_x;
            int dy = to_y - from_y;
            
            if (abs(dx) + abs(dy) != 1) {
                return {false, "Stone can only push adjacent pieces."};
            }
            
            if (pushed_to_x != to_x + dx || pushed_to_y != to_y + dy) {
                return {false, "Stone push must be in same direction."};
            }
            
            // Execute stone push
            board[pushed_to_y][pushed_to_x] = std::move(board[to_y][to_x]);
            board[to_y][to_x] = std::move(board[from_y][from_x]);
            return {true, ""};
        } else {
            // River push: can push multiple spaces along river direction
            int dx = to_x - from_x;
            int dy = to_y - from_y;
            
            if (abs(dx) + abs(dy) != 1) {
                return {false, "River can only push adjacent pieces."};
            }
            
            // Validate pushed destination is reachable via river flow
            vector<pair<int, int>> valid_destinations = _trace_river_flow(
                board, to_x, to_y, from_x, from_y, 
                board[to_y][to_x]->owner, rows, cols, score_cols, true
            );
            
            if (find(valid_destinations.begin(), valid_destinations.end(), 
                    make_pair(pushed_to_x, pushed_to_y)) == valid_destinations.end()) {
                return {false, "Invalid river push destination."};
            }
            
            // Execute river push and flip river to stone
            board[pushed_to_y][pushed_to_x] = std::move(board[to_y][to_x]);
            board[to_y][to_x] = std::move(board[from_y][from_x]);
            board[to_y][to_x]->side = "stone";
            board[to_y][to_x]->orientation = "";
            return {true, ""};
        }
    }
    else if (move.action == "flip") {
        if (source_piece->side == "stone") {
            // Flipping stone to river
            if (!move.orientation.has_value()) {
                return {false, "Flipping stone to river requires orientation."};
            }
            
            string orient = move.orientation.value();
            if (orient != "horizontal" && orient != "vertical") {
                return {false, "Invalid river orientation."};
            }
            
            source_piece->side = "river";
            source_piece->orientation = orient;
            return {true, ""};
        } else {
            // Flipping river to stone
            source_piece->side = "stone";
            source_piece->orientation = "";
            return {true, ""};
        }
    }
    else if (move.action == "rotate") {
        if (source_piece->side != "river") {
            return {false, "Can only rotate river pieces."};
        }
        
        // Rotate river 90 degrees
        source_piece->orientation = (source_piece->orientation == "horizontal") ? "vertical" : "horizontal";
        return {true, ""};
    }
    else {
        return {false, "Unknown action type."};
    }
}

/**
 * @brief Simulate a move on a copy of the board.
 */
pair<bool, Board> simulate_move(const Board& board, const Move& move, const string& player, 
                               int rows, int cols, const vector<int>& score_cols) {
    Board board_copy = deep_copy_board(board);
    auto [success, message] = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols);
    
    if (success) {
        return {true, std::move(board_copy)};
    } else {
        // Return original board on failure
        return {false, deep_copy_board(board)};
    }
}

/**
 * @brief Check if the game is in a terminal state (someone won).
 */
pair<bool, string> check_game_over(const Board& board, int rows, int cols, const vector<int>& score_cols) {
    int circle_stones = count_stones_in_scoring_area(board, "circle", rows, cols, score_cols);
    int square_stones = count_stones_in_scoring_area(board, "square", rows, cols, score_cols);
    
    if (circle_stones >= 4) {
        return {true, "circle"};
    }
    if (square_stones >= 4) {
        return {true, "square"};
    }
    
    return {false, ""};
}

/**
 * @brief Abstract base class for all agents.
 */
class BaseAgent {
public:
    string player;
    string opponent;

    BaseAgent(string p) : player(move(p)), opponent(get_opponent(player)) {}
    virtual ~BaseAgent() = default;

    virtual optional<Move> choose(const Board& board, int rows, int cols, const vector<int>& score_cols, 
                                 double current_player_time, double opponent_time) = 0;
};

/**
 * @brief Random Agent for testing purposes.
 */
class RandomAgent : public BaseAgent {
private:
    mutable random_device rd;
    mutable mt19937 gen;
    
public:
    RandomAgent(string p) : BaseAgent(move(p)), gen(rd()) {}

    optional<Move> choose(const Board& board, int rows, int cols, const vector<int>& score_cols, 
                         double current_player_time, double opponent_time) override {
        auto moves = generate_all_moves(board, this->player, rows, cols, score_cols);
        
        if (moves.empty()) {
            return nullopt;
        }
        
        uniform_int_distribution<> distrib(0, moves.size() - 1);
        return moves[distrib(gen)];
    }
};

/**
 * @brief Enhanced Student Agent Implementation with sophisticated AI.
 */
class StudentAgent : public BaseAgent {
private:
    static const int MAX_DEPTH = 4;
    mutable random_device rd;
    mutable mt19937 gen;
    
    /**
     * @brief Move ordering heuristic - prioritize promising moves first for better alpha-beta pruning.
     */
    vector<Move> order_moves(const vector<Move>& moves, const Board& board, 
                            int rows, int cols, const vector<int>& score_cols) {
        vector<pair<double, Move>> scored_moves;
        
        for (const auto& move : moves) {
            double score = 0.0;
            
            // Prioritize moves that directly score
            if (move.action == "move" && move.to.has_value()) {
                auto [to_x, to_y] = move.to.value();
                if (is_own_score_cell(to_x, to_y, this->player, rows, cols, score_cols)) {
                    score += 1000.0;
                }
            }
            
            // Prioritize moves that get pieces closer to scoring
            if (move.action == "move" && move.to.has_value()) {
                auto [to_x, to_y] = move.to.value();
                int target_row = (this->player == "circle") ? top_score_row() : bottom_score_row(rows);
                score += 100.0 / (1.0 + abs(to_y - target_row));
            }
            
            // Slightly prioritize captures and pushes
            if (move.action == "push") {
                score += 10.0;
            }
            
            scored_moves.emplace_back(score, move);
        }
        
        // Sort by score (descending)
        sort(scored_moves.begin(), scored_moves.end(), 
             [](const auto& a, const auto& b) { return a.first > b.first; });
        
        vector<Move> ordered_moves;
        for (const auto& [score, move] : scored_moves) {
            ordered_moves.push_back(move);
        }
        
        return ordered_moves;
    }
    
    /**
     * @brief Minimax with alpha-beta pruning and move ordering.
     */
    pair<double, optional<Move>> minimax(const Board& board, int depth, double alpha, double beta, 
                                       bool maximizing_player, const string& current_player,
                                       int rows, int cols, const vector<int>& score_cols,
                                       double time_remaining) {
        
        // Check time limit
        if (time_remaining < 0.1) {
            double eval = quick_evaluate_board(board, this->player, rows, cols, score_cols);
            return {eval, nullopt};
        }
        
        // Check for terminal states
        auto [game_over, winner] = check_game_over(board, rows, cols, score_cols);
        if (game_over) {
            if (winner == this->player) {
                return {10000.0 + depth, nullopt}; // Win bonus for faster wins
            } else {
                return {-10000.0 - depth, nullopt}; // Loss penalty for slower losses
            }
        }
        
        // Base case
        if (depth == 0) {
            double eval = advanced_evaluate_board(board, this->player, rows, cols, score_cols);
            return {eval, nullopt};
        }
        
        auto moves = generate_all_moves(board, current_player, rows, cols, score_cols);
        if (moves.empty()) {
            double eval = advanced_evaluate_board(board, this->player, rows, cols, score_cols);
            return {eval, nullopt};
        }
        
        // Move ordering for better pruning
        moves = order_moves(moves, board, rows, cols, score_cols);
        
        optional<Move> best_move = nullopt;
        
        if (maximizing_player) {
            double max_eval = -numeric_limits<double>::infinity();
            
            for (const auto& move : moves) {
                auto [success, new_board] = simulate_move(board, move, current_player, rows, cols, score_cols);
                if (!success) continue;
                
                string next_player = get_opponent(current_player);
                double time_per_move = time_remaining / moves.size();
                
                auto [eval, _] = minimax(new_board, depth - 1, alpha, beta, false, next_player, 
                                       rows, cols, score_cols, time_remaining - time_per_move);
                
                if (eval > max_eval) {
                    max_eval = eval;
                    best_move = move;
                }
                
                alpha = max(alpha, eval);
                if (beta <= alpha) {
                    break; // Alpha-beta pruning
                }
            }
            
            return {max_eval, best_move};
        } else {
            double min_eval = numeric_limits<double>::infinity();
            
            for (const auto& move : moves) {
                auto [success, new_board] = simulate_move(board, move, current_player, rows, cols, score_cols);
                if (!success) continue;
                
                string next_player = get_opponent(current_player);
                double time_per_move = time_remaining / moves.size();
                
                auto [eval, _] = minimax(new_board, depth - 1, alpha, beta, true, next_player, 
                                       rows, cols, score_cols, time_remaining - time_per_move);
                
                if (eval < min_eval) {
                    min_eval = eval;
                    best_move = move;
                }
                
                beta = min(beta, eval);
                if (beta <= alpha) {
                    break; // Alpha-beta pruning
                }
            }
            
            return {min_eval, best_move};
        }
    }
    
public:
    StudentAgent(string p) : BaseAgent(move(p)), gen(rd()) {}

    /**
     * @brief Choose the best move using minimax with alpha-beta pruning and advanced evaluation.
     */
    optional<Move> choose(const Board& board, int rows, int cols, const vector<int>& score_cols, 
                         double current_player_time, double opponent_time) override {
        
        auto moves = generate_all_moves(board, this->player, rows, cols, score_cols);
        if (moves.empty()) {
            return nullopt;
        }
        
        // Immediate win detection
        for (const auto& move : moves) {
            auto [success, new_board] = simulate_move(board, move, this->player, rows, cols, score_cols);
            if (success) {
                int stones_after = count_stones_in_scoring_area(new_board, this->player, rows, cols, score_cols);
                if (stones_after >= 4) {
                    return move; // Take winning move immediately
                }
            }
        }
        
        // Opponent threat analysis
        auto opponent_moves = generate_all_moves(board, this->opponent, rows, cols, score_cols);
        bool opponent_can_win = false;
        
        for (const auto& opp_move : opponent_moves) {
            auto [success, opp_board] = simulate_move(board, opp_move, this->opponent, rows, cols, score_cols);
            if (success) {
                int opp_stones = count_stones_in_scoring_area(opp_board, this->opponent, rows, cols, score_cols);
                if (opp_stones >= 4) {
                    opponent_can_win = true;
                    break;
                }
            }
        }
        
        // If opponent can win, try defensive moves first
        if (opponent_can_win) {
            vector<Move> defensive_moves;
            for (const auto& our_move : moves) {
                auto [our_success, our_board] = simulate_move(board, our_move, this->player, rows, cols, score_cols);
                if (our_success) {
                    bool blocks_win = true;
                    for (const auto& opp_move : opponent_moves) {
                        auto [opp_success, final_board] = simulate_move(our_board, opp_move, this->opponent, rows, cols, score_cols);
                        if (opp_success) {
                            int opp_stones = count_stones_in_scoring_area(final_board, this->opponent, rows, cols, score_cols);
                            if (opp_stones >= 4) {
                                blocks_win = false;
                                break;
                            }
                        }
                    }
                    if (blocks_win) {
                        defensive_moves.push_back(our_move);
                    }
                }
            }
            
            if (!defensive_moves.empty()) {
                moves = defensive_moves;
            }
        }
        
        // Determine search depth based on available time
        int search_depth = MAX_DEPTH;
        if (current_player_time < 15.0) {
            search_depth = 3;
        }
        if (current_player_time < 10.0) {
            search_depth = 2;
        }
        if (current_player_time < 5.0) {
            search_depth = 1;
        }
        
        // Minimax search
        double alpha = -numeric_limits<double>::infinity();
        double beta = numeric_limits<double>::infinity();
        double time_for_search = min(current_player_time * 0.1, 2.0); // Use at most 10% of remaining time
        
        auto [best_score, best_move] = minimax(board, search_depth, alpha, beta, true, this->player, 
                                             rows, cols, score_cols, time_for_search);
        
        if (best_move.has_value()) {
            return best_move;
        }
        
        // Fallback: evaluate each move and pick the best
        Move best_immediate = moves[0];
        double best_eval = -numeric_limits<double>::infinity();
        
        for (const auto& move : moves) {
            auto [success, new_board] = simulate_move(board, move, this->player, rows, cols, score_cols);
            if (success) {
                double eval = advanced_evaluate_board(new_board, this->player, rows, cols, score_cols);
                if (eval > best_eval) {
                    best_eval = eval;
                    best_immediate = move;
                }
            }
        }
        
        return best_immediate;
    }
};

// ==================== TESTING HELPERS ====================
// Dummy implementation of gameEngine parts for testing
namespace gameEngine {
    const int DEFAULT_ROWS = 12;
    const int DEFAULT_COLS = 8;

    Board default_start_board(int rows, int cols) {
        Board board(rows, vector<unique_ptr<Piece>>(cols));
        // Setup circle pieces (top player)
        for (int x = 2; x < 6; ++x) {
            board[3][x] = make_unique<Piece>("circle", "stone");
            board[4][x] = make_unique<Piece>("circle", "stone");
        }
        // Setup square pieces (bottom player)
        for (int x = 2; x < 6; ++x) {
            board[rows - 4][x] = make_unique<Piece>("square", "stone");
            board[rows - 5][x] = make_unique<Piece>("square", "stone");
        }
        return board;
    }
}

// Helper function to print a move for verification
void print_move(const Move& move) {
    cout << "Action: " << move.action
              << ", From: (" << move.from.first << ", " << move.from.second << ")";
    if (move.to) {
        cout << ", To: (" << move.to->first << ", " << move.to->second << ")";
    }
    if (move.pushed_to) {
        cout << ", Pushed To: (" << move.pushed_to->first << ", " << move.pushed_to->second << ")";
    }
    if (move.orientation) {
        cout << ", Orientation: " << move.orientation.value();
    }
    cout << endl;
}


/**
 * @brief Basic test to verify the student agent can be created and make moves.
 */
void test_student_agent() {
    cout << "Testing StudentAgent..." << endl;

    int rows = gameEngine::DEFAULT_ROWS;
    int cols = gameEngine::DEFAULT_COLS;
    auto score_cols = score_cols_for(cols);
    Board board = gameEngine::default_start_board(rows, cols);

    StudentAgent agent("circle");
    auto move_opt = agent.choose(board, rows, cols, score_cols, 1.0, 1.0);

    if (move_opt) {
        cout << "✓ Agent successfully generated a move:" << endl;
        print_move(move_opt.value());
    } else {
        cout << "✗ Agent returned no move" << endl;
    }
}

// ==================== COMPREHENSIVE TESTING HELPERS ====================

namespace Testing {

/**
 * @brief Print a visual representation of the board state
 */
void print_board(const Board& board, int rows, int cols, const vector<int>& score_cols) {
    cout << "\n=== BOARD STATE ===" << endl;
    cout << "   ";
    for (int x = 0; x < cols; ++x) {
        cout << setw(3) << x;
    }
    cout << endl;
    
    for (int y = 0; y < rows; ++y) {
        cout << setw(2) << y << " ";
        for (int x = 0; x < cols; ++x) {
            const Piece* piece = board[y][x].get();
            
            // Check if this is a scoring area
            bool is_circle_score = (y == top_score_row() && 
                                   find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
            bool is_square_score = (y == bottom_score_row(rows) && 
                                   find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
            
            if (piece == nullptr) {
                if (is_circle_score) cout << " ◯ "; // Circle scoring area
                else if (is_square_score) cout << " □ "; // Square scoring area
                else cout << " . ";
            } else {
                char owner_char = (piece->owner == "circle") ? 'C' : 'S';
                if (piece->side == "stone") {
                    cout << " " << owner_char << " ";
                } else { // river
                    char river_char = (piece->orientation == "horizontal") ? '-' : '|';
                    cout << owner_char << river_char << " ";
                }
            }
        }
        cout << endl;
    }
    
    cout << "\nLegend:" << endl;
    cout << "C/S = Circle/Square stone, C-/C|/S-/S| = Rivers, ◯/□ = Scoring areas, . = Empty" << endl;
}

/**
 * @brief Print detailed move information
 */
void print_move_details(const Move& move) {
    cout << "Move Details:" << endl;
    cout << "  Action: " << move.action << endl;
    cout << "  From: (" << move.from.first << ", " << move.from.second << ")" << endl;
    
    if (move.to.has_value()) {
        cout << "  To: (" << move.to->first << ", " << move.to->second << ")" << endl;
    }
    
    if (move.pushed_to.has_value()) {
        cout << "  Pushed To: (" << move.pushed_to->first << ", " << move.pushed_to->second << ")" << endl;
    }
    
    if (move.orientation.has_value()) {
        cout << "  Orientation: " << move.orientation.value() << endl;
    }
}

/**
 * @brief Create a test board with specific configuration
 */
Board create_test_board(int rows, int cols, const vector<tuple<int, int, string, string, string>>& pieces) {
    Board board(rows, vector<unique_ptr<Piece>>(cols));
    
    for (const auto& [x, y, owner, side, orientation] : pieces) {
        if (in_bounds(x, y, rows, cols)) {
            board[y][x] = make_unique<Piece>(owner, side, orientation);
        }
    }
    
    return board;
}

/**
 * @brief Test basic move generation
 */
void test_basic_move_generation() {
    cout << "\n=== TESTING BASIC MOVE GENERATION ===" << endl;
    
    int rows = 8, cols = 6;
    auto score_cols = score_cols_for(cols);
    
    // Create test board with a few pieces
    vector<tuple<int, int, string, string, string>> test_pieces = {
        {2, 3, "circle", "stone", ""},
        {3, 3, "square", "river", "horizontal"},
        {4, 3, "circle", "river", "vertical"}
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    // Test move generation for circle player
    auto moves = generate_all_moves(board, "circle", rows, cols, score_cols);
    cout << "\nGenerated " << moves.size() << " moves for circle:" << endl;
    
    for (size_t i = 0; i < min((size_t)5, moves.size()); ++i) {
        cout << "Move " << (i + 1) << ": ";
        print_move(moves[i]);
    }
    
    if (moves.size() > 5) {
        cout << "... and " << (moves.size() - 5) << " more moves" << endl;
    }
}

/**
 * @brief Test river movement mechanics
 */
void test_river_movement() {
    cout << "\n=== TESTING RIVER MOVEMENT ===" << endl;
    
    int rows = 8, cols = 6;
    auto score_cols = score_cols_for(cols);
    
    // Create board with river chain
    vector<tuple<int, int, string, string, string>> test_pieces = {
        {1, 4, "circle", "stone", ""},        // Stone to move
        {2, 4, "circle", "river", "horizontal"}, // River 1 (horizontal)
        {3, 4, "circle", "river", "horizontal"}, // River 2 (horizontal)
        {4, 4, "circle", "river", "vertical"},   // River 3 (vertical - change direction)
        {1, 3, "square", "stone", ""}          // Blocking stone
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    // Test river flow from stone position
    auto possible_moves = get_valid_moves_for_piece(board, 1, 4, "circle", rows, cols, score_cols);
    
    cout << "\nPossible moves for stone at (1, 4):" << endl;
    for (const auto& move : possible_moves) {
        if (move.action == "move") {
            print_move(move);
        }
    }
}

/**
 * @brief Test push mechanics
 */
void test_push_mechanics() {
    cout << "\n=== TESTING PUSH MECHANICS ===" << endl;
    
    int rows = 8, cols = 6;
    auto score_cols = score_cols_for(cols);
    
    // Test stone pushing stone
    cout << "\n--- Stone Pushing Stone ---" << endl;
    vector<tuple<int, int, string, string, string>> push_test1 = {
        {2, 4, "circle", "stone", ""},    // Pusher
        {3, 4, "square", "stone", ""}     // Target to push
        // Position (4, 4) should be empty for valid push
    };
    
    Board board1 = create_test_board(rows, cols, push_test1);
    print_board(board1, rows, cols, score_cols);
    
    auto push_moves1 = get_valid_moves_for_piece(board1, 2, 4, "circle", rows, cols, score_cols);
    cout << "Push moves available:" << endl;
    for (const auto& move : push_moves1) {
        if (move.action == "push") {
            print_move(move);
        }
    }
    
    // Test river pushing stone
    cout << "\n--- River Pushing Stone ---" << endl;
    vector<tuple<int, int, string, string, string>> push_test2 = {
        {2, 4, "circle", "river", "horizontal"}, // River pusher
        {3, 4, "square", "stone", ""},           // Target to push
        {4, 4, "circle", "river", "horizontal"}  // Continuation river
    };
    
    Board board2 = create_test_board(rows, cols, push_test2);
    print_board(board2, rows, cols, score_cols);
    
    auto push_moves2 = get_valid_moves_for_piece(board2, 2, 4, "circle", rows, cols, score_cols);
    cout << "River push moves available:" << endl;
    for (const auto& move : push_moves2) {
        if (move.action == "push") {
            print_move(move);
        }
    }
}

/**
 * @brief Test flip and rotate mechanics
 */
void test_flip_rotate() {
    cout << "\n=== TESTING FLIP AND ROTATE ===" << endl;
    
    int rows = 8, cols = 6;
    auto score_cols = score_cols_for(cols);
    
    vector<tuple<int, int, string, string, string>> test_pieces = {
        {2, 3, "circle", "stone", ""},
        {3, 3, "circle", "river", "horizontal"}
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    // Test flipping stone
    cout << "\nFlip/Rotate options for stone at (2, 3):" << endl;
    auto stone_moves = get_valid_moves_for_piece(board, 2, 3, "circle", rows, cols, score_cols);
    for (const auto& move : stone_moves) {
        if (move.action == "flip") {
            print_move(move);
        }
    }
    
    // Test flipping/rotating river
    cout << "\nFlip/Rotate options for river at (3, 3):" << endl;
    auto river_moves = get_valid_moves_for_piece(board, 3, 3, "circle", rows, cols, score_cols);
    for (const auto& move : river_moves) {
        if (move.action == "flip" || move.action == "rotate") {
            print_move(move);
        }
    }
}

/**
 * @brief Test move validation and application
 */
void test_move_validation() {
    cout << "\n=== TESTING MOVE VALIDATION ===" << endl;
    
    int rows = 8, cols = 6;
    auto score_cols = score_cols_for(cols);
    
    vector<tuple<int, int, string, string, string>> test_pieces = {
        {2, 4, "circle", "stone", ""},
        {3, 4, "square", "stone", ""}
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    // Test valid move
    Move valid_move = {"move", {2, 4}, {{1, 4}}, nullopt, nullopt};
    cout << "\nTesting valid move:" << endl;
    print_move(valid_move);
    
    auto [success1, new_board1] = simulate_move(board, valid_move, "circle", rows, cols, score_cols);
    cout << "Result: " << (success1 ? "SUCCESS" : "FAILED") << endl;
    
    if (success1) {
        cout << "Board after move:" << endl;
        print_board(new_board1, rows, cols, score_cols);
    }
    
    // Test invalid move
    Move invalid_move = {"move", {2, 4}, {{3, 4}}, nullopt, nullopt}; // Occupied destination
    cout << "\nTesting invalid move (occupied destination):" << endl;
    print_move(invalid_move);
    
    auto [success2, new_board2] = simulate_move(board, invalid_move, "circle", rows, cols, score_cols);
    cout << "Result: " << (success2 ? "SUCCESS" : "FAILED") << endl;
}

/**
 * @brief Test evaluation functions
 */
void test_evaluation_functions() {
    cout << "\n=== TESTING EVALUATION FUNCTIONS ===" << endl;
    
    int rows = 12, cols = 8;
    auto score_cols = score_cols_for(cols);
    
    // Create advantageous position for circle
    vector<tuple<int, int, string, string, string>> test_pieces = {
        // Circle stones in scoring area
        {3, 2, "circle", "stone", ""},
        {4, 2, "circle", "stone", ""},
        // Circle stones near scoring area
        {5, 3, "circle", "stone", ""},
        {2, 3, "circle", "stone", ""},
        // Square pieces
        {3, 9, "square", "stone", ""},
        {6, 7, "square", "stone", ""}
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    // Evaluate for both players
    double circle_eval = advanced_evaluate_board(board, "circle", rows, cols, score_cols);
    double square_eval = advanced_evaluate_board(board, "square", rows, cols, score_cols);
    
    cout << "\nEvaluation Scores:" << endl;
    cout << "Circle: " << circle_eval << endl;
    cout << "Square: " << square_eval << endl;
    
    // Test individual components
    int circle_scoring = count_stones_in_scoring_area(board, "circle", rows, cols, score_cols);
    int square_scoring = count_stones_in_scoring_area(board, "square", rows, cols, score_cols);
    int circle_ready = count_stones_one_move_from_scoring(board, "circle", rows, cols, score_cols);
    int square_ready = count_stones_one_move_from_scoring(board, "square", rows, cols, score_cols);
    
    cout << "\nScoring breakdown:" << endl;
    cout << "Circle stones in scoring area: " << circle_scoring << endl;
    cout << "Square stones in scoring area: " << square_scoring << endl;
    cout << "Circle stones one move from scoring: " << circle_ready << endl;
    cout << "Square stones one move from scoring: " << square_ready << endl;
}

/**
 * @brief Test agent decision making
 */
void test_agent_decisions() {
    cout << "\n=== TESTING AGENT DECISIONS ===" << endl;
    
    int rows = 12, cols = 8;
    auto score_cols = score_cols_for(cols);
    
    // Create near-win scenario for testing
    vector<tuple<int, int, string, string, string>> test_pieces = {
        // Circle close to winning (3 stones in scoring area)
        {3, 2, "circle", "stone", ""},
        {4, 2, "circle", "stone", ""},
        {5, 2, "circle", "stone", ""},
        // Circle stone that can win
        {4, 3, "circle", "stone", ""},
        // Square has some stones
        {3, 9, "square", "stone", ""},
        {4, 9, "square", "stone", ""}
    };
    
    Board board = create_test_board(rows, cols, test_pieces);
    print_board(board, rows, cols, score_cols);
    
    cout << "\nCircle is one move away from winning!" << endl;
    
    StudentAgent circle_agent("circle");
    auto chosen_move = circle_agent.choose(board, rows, cols, score_cols, 30.0, 30.0);
    
    if (chosen_move.has_value()) {
        cout << "\nAgent chose move:" << endl;
        print_move_details(chosen_move.value());
        
        // Simulate the move
        auto [success, new_board] = simulate_move(board, chosen_move.value(), "circle", rows, cols, score_cols);
        if (success) {
            cout << "\nBoard after agent's move:" << endl;
            print_board(new_board, rows, cols, score_cols);
            
            int stones_after = count_stones_in_scoring_area(new_board, "circle", rows, cols, score_cols);
            cout << "\nCircle stones in scoring area after move: " << stones_after << endl;
            if (stones_after >= 4) {
                cout << "🎉 CIRCLE WINS! 🎉" << endl;
            }
        }
    } else {
        cout << "Agent returned no move!" << endl;
    }
}

/**
 * @brief Performance test - measure move generation speed
 */
void test_performance() {
    cout << "\n=== PERFORMANCE TESTING ===" << endl;
    
    int rows = 12, cols = 8;
    auto score_cols = score_cols_for(cols);
    Board board = gameEngine::default_start_board(rows, cols);
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Generate moves multiple times
    const int iterations = 1000;
    int total_moves = 0;
    
    for (int i = 0; i < iterations; ++i) {
        auto moves = generate_all_moves(board, "circle", rows, cols, score_cols);
        total_moves += moves.size();
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    
    cout << "Performance Results:" << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Total moves generated: " << total_moves << endl;
    cout << "Average moves per iteration: " << (double)total_moves / iterations << endl;
    cout << "Total time: " << duration.count() << " microseconds" << endl;
    cout << "Time per iteration: " << duration.count() / iterations << " microseconds" << endl;
}

/**
 * @brief Test game simulation from start to finish
 */
void test_full_game_simulation() {
    cout << "\n=== FULL GAME SIMULATION ===" << endl;
    
    int rows = gameEngine::DEFAULT_ROWS;
    int cols = gameEngine::DEFAULT_COLS;
    auto score_cols = score_cols_for(cols);
    Board board = gameEngine::default_start_board(rows, cols);
    
    StudentAgent circle_agent("circle");
    RandomAgent square_agent("square");
    
    string current_player = "circle";
    int move_count = 0;
    const int max_moves = 50; // Limit for testing
    
    cout << "Starting game simulation..." << endl;
    print_board(board, rows, cols, score_cols);
    
    while (move_count < max_moves) {
        // Check for game over
        auto [game_over, winner] = check_game_over(board, rows, cols, score_cols);
        if (game_over) {
            cout << "\n🎉 GAME OVER! Winner: " << winner << " 🎉" << endl;
            break;
        }
        
        // Get move from current player
        optional<Move> chosen_move;
        if (current_player == "circle") {
            chosen_move = circle_agent.choose(board, rows, cols, score_cols, 30.0, 30.0);
        } else {
            chosen_move = square_agent.choose(board, rows, cols, score_cols, 30.0, 30.0);
        }
        
        if (!chosen_move.has_value()) {
            cout << "\nNo moves available for " << current_player << "!" << endl;
            break;
        }
        
        // Apply move
        auto [success, new_board] = simulate_move(board, chosen_move.value(), current_player, rows, cols, score_cols);
        if (!success) {
            cout << "\nInvalid move attempted by " << current_player << "!" << endl;
            break;
        }
        
        board = std::move(new_board);
        move_count++;
        
        cout << "\nMove " << move_count << " by " << current_player << ":" << endl;
        print_move(chosen_move.value());
        
        // Show board every few moves
        if (move_count % 10 == 0 || move_count <= 5) {
            print_board(board, rows, cols, score_cols);
        }
        
        // Switch players
        current_player = get_opponent(current_player);
    }
    
    cout << "\nSimulation completed after " << move_count << " moves." << endl;
    
    // Final scores
    int circle_stones = count_stones_in_scoring_area(board, "circle", rows, cols, score_cols);
    int square_stones = count_stones_in_scoring_area(board, "square", rows, cols, score_cols);
    cout << "Final scores - Circle: " << circle_stones << ", Square: " << square_stones << endl;
}

/**
 * @brief Run all tests
 */
void run_all_tests() {
    cout << "🧪 RUNNING COMPREHENSIVE TESTS FOR RIVERS AND STONES 🧪" << endl;
    cout << "======================================================" << endl;
    
    try {
        test_basic_move_generation();
        test_river_movement();
        test_push_mechanics();
        test_flip_rotate();
        test_move_validation();
        test_evaluation_functions();
        test_agent_decisions();
        test_performance();
        test_full_game_simulation();
        
        cout << "\n✅ ALL TESTS COMPLETED SUCCESSFULLY!" << endl;
        
    } catch (const exception& e) {
        cout << "\n❌ TEST FAILED WITH EXCEPTION: " << e.what() << endl;
    }
}

} // namespace Testing

// Helper function to print a move for verification (updated version)
void print_move(const Move& move) {
    cout << "Action: " << move.action
         << ", From: (" << move.from.first << ", " << move.from.second << ")";
    if (move.to) {
        cout << ", To: (" << move.to->first << ", " << move.to->second << ")";
    }
    if (move.pushed_to) {
        cout << ", Pushed To: (" << move.pushed_to->first << ", " << move.pushed_to->second << ")";
    }
    if (move.orientation) {
        cout << ", Orientation: " << move.orientation.value();
    }
    cout << endl;
}

/**
 * @brief Updated test function that runs comprehensive tests
 */
void test_student_agent() {
    cout << "Testing StudentAgent and game mechanics..." << endl;
    Testing::run_all_tests();
}

/**
 * @brief Interactive testing mode
 */
void interactive_test_mode() {
    cout << "\n=== INTERACTIVE TEST MODE ===" << endl;
    cout << "Available tests:" << endl;
    cout << "1. Basic move generation" << endl;
    cout << "2. River movement" << endl;
    cout << "3. Push mechanics" << endl;
    cout << "4. Flip and rotate" << endl;
    cout << "5. Move validation" << endl;
    cout << "6. Evaluation functions" << endl;
    cout << "7. Agent decisions" << endl;
    cout << "8. Performance test" << endl;
    cout << "9. Full game simulation" << endl;
    cout << "0. Run all tests" << endl;
    
    int choice;
    cout << "\nEnter test number: ";
    cin >> choice;
    
    switch (choice) {
        case 1: Testing::test_basic_move_generation(); break;
        case 2: Testing::test_river_movement(); break;
        case 3: Testing::test_push_mechanics(); break;
        case 4: Testing::test_flip_rotate(); break;
        case 5: Testing::test_move_validation(); break;
        case 6: Testing::test_evaluation_functions(); break;
        case 7: Testing::test_agent_decisions(); break;
        case 8: Testing::test_performance(); break;
        case 9: Testing::test_full_game_simulation(); break;
        case 0: Testing::run_all_tests(); break;
        default: cout << "Invalid choice!" << endl;
    }
}

int main() {
    // Run basic test when file is executed
    test_student_agent();
    return 0;
}