#include <iostream>
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
 * @brief Count stones that can reach scoring area in one move
 */
int count_stones_one_move_from_scoring(const Board& board, const string& player, int rows, int cols, const vector<int>& score_cols) {
    int count = 0;
    int score_row = (player == "circle") ? top_score_row() : bottom_score_row(rows);
    
    // Check adjacent positions to scoring area
    vector<pair<int, int>> adjacent_positions;
    for (int x : score_cols) {
        // Add positions above and below scoring row
        if (in_bounds(x, score_row - 1, rows, cols)) {
            adjacent_positions.push_back({x, score_row - 1});
        }
        if (in_bounds(x, score_row + 1, rows, cols)) {
            adjacent_positions.push_back({x, score_row + 1});
        }
    }
    
    // Add positions to the left and right of scoring area
    for (int y = score_row; y == score_row; ++y) {
        if (score_cols.size() > 0) {
            int leftmost = score_cols[0];
            int rightmost = score_cols[score_cols.size() - 1];
            
            if (in_bounds(leftmost - 1, y, rows, cols)) {
                adjacent_positions.push_back({leftmost - 1, y});
            }
            if (in_bounds(rightmost + 1, y, rows, cols)) {
                adjacent_positions.push_back({rightmost + 1, y});
            }
        }
    }
    
    // Check each adjacent position for player's stones
    for (auto [x, y] : adjacent_positions) {
        const Piece* piece = board[y][x].get();
        if (piece && piece->owner == player && piece->side == "stone") {
            count++;
        }
    }
    
    return count;
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

int main() {
    // Run basic test when file is executed
    test_student_agent();
    return 0;
}