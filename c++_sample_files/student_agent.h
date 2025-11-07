#ifndef STUDENT_AGENT_H
#define STUDENT_AGENT_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <deque>
#include <algorithm>
#include <chrono> // For time
#include <functional> // For std::hash
#include <limits> // For infinity

// ===================================================================
// HELPER STRUCTURES
// ===================================================================

// Represents a single piece on the board
struct Piece {
    std::string owner;
    std::string side;
    std::string orientation; // "horizontal", "vertical", or ""

    // Constructor
    Piece(std::string o, std::string s, std::string orient = "")
        : owner(std::move(o)), side(std::move(s)), orientation(std::move(orient)) {}
};

// Use shared_ptr to represent a piece, or nullptr for an empty square
using PiecePtr = std::shared_ptr<Piece>;
using Board = std::vector<std::vector<PiecePtr>>;

// Represents a single move
struct Move {
    enum class ActionType { MOVE, PUSH, FLIP, ROTATE };

    ActionType action;
    std::pair<int, int> from;
    std::pair<int, int> to;          // Used by MOVE, PUSH
    std::pair<int, int> pushed_to; // Used by PUSH
    std::string orientation;     // Used by FLIP

    // Default constructor (for sorting and storage)
    Move() : action(ActionType::MOVE), from({0,0}), to({0,0}), pushed_to({0,0}), orientation("") {}

    // Constructor for MOVE
    Move(ActionType a, std::pair<int, int> f, std::pair<int, int> t)
        : action(a), from(f), to(t), pushed_to({-1,-1}), orientation("") {}

    // Constructor for PUSH
    Move(ActionType a, std::pair<int, int> f, std::pair<int, int> t, std::pair<int, int> pt)
        : action(a), from(f), to(t), pushed_to(pt), orientation("") {}

    // Constructor for FLIP
    Move(ActionType a, std::pair<int, int> f, std::string o)
        : action(a), from(f), to({-1,-1}), pushed_to({-1,-1}), orientation(o) {}

    // Constructor for ROTATE or simple FLIP (stone)
    Move(ActionType a, std::pair<int, int> f)
        : action(a), from(f), to({-1,-1}), pushed_to({-1,-1}), orientation("") {}

    // Equality operator for finding in lists
    bool operator==(const Move& other) const {
        return action == other.action && from == other.from && to == other.to && 
               pushed_to == other.pushed_to && orientation == other.orientation;
    }
};

// ===================================================================
// BASE AGENT (for compatibility)
// ===================================================================

class BaseAgent {
public:
    std::string player;
    std::string opponent;

    BaseAgent(std::string p) : player(p), opponent(get_opponent(p)) {}
    virtual ~BaseAgent() {} // Virtual destructor for base class

    virtual std::optional<Move> choose(
        const Board& board, int rows, int cols, const std::vector<int>& score_cols,
        double current_player_time, double opponent_time) = 0;

    static std::string get_opponent(const std::string& p) {
        return (p == "circle") ? "square" : "circle";
    }
};

// ===================================================================
// STUDENT AGENT CLASS
// ===================================================================

class StudentAgent : public BaseAgent {
private:
    std::map<int, std::vector<Move>> killer_moves;
    std::set<size_t> board_history_set;
    std::deque<size_t> recent_positions;
    std::deque<Move> last_moves;
    
    int turn_count;
    const int MAX_HISTORY_SIZE = 20;
    const int MAX_RECENT_MOVES = 5;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    double time_limit_seconds;

    // --- Utility Functions ---
    bool in_bounds(int x, int y, int rows, int cols) const;
    std::vector<int> score_cols_for(int cols) const; // Assuming 4 spaces
    int top_score_row() const;
    int bottom_score_row(int rows) const;
    bool is_opponent_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const;
    bool is_own_score_cell(int x, int y, const std::string& p, int rows, int cols, const std::vector<int>& score_cols) const;
    size_t board_hash(const Board& board) const;
    int manhattan_distance(int x1, int y1, int x2, int y2) const;
    Board deep_copy_board(const Board& board);

    // --- Main Logic ---
    void update_move_history(const Move& move);
    std::vector<Move> filter_non_repeating_moves(const Board& board, const std::vector<Move>& moves, int rows, int cols, const std::vector<int>& score_cols);
    bool moves_similar(const Move& move1, const Move& move2) const;
    
    double negamax_with_balance(Board board, int depth, double alpha, double beta,
                                const std::string& current_player, int rows, int cols, 
                                const std::vector<int>& score_cols, int max_depth);
    
    // --- Evaluation ---
    double evaluate_balanced(const Board& board, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_edge_control(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_defensive_position(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_balance_factor(const Board& board, const std::string& player, int rows, int cols);
    double evaluate_river_network_balanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double evaluate_manhattan_distances(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    double count_stones_ready_to_score(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);

    // --- Move Generation & Ordering ---
    std::vector<Move> order_moves_with_edge_control(const Board& board, std::vector<Move> moves, const std::string& current_player, int rows, int cols, const std::vector<int>& score_cols);
    std::vector<Move> get_all_valid_moves_enhanced(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<Move> _get_moves_for_piece(const Board& board, int row, int col, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<std::pair<int, int>> _trace_river_flow(const Board& board, int start_r, int start_c, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    std::vector<std::pair<int, int>> _trace_river_push(const Board& board, int target_r, int target_c, const PiecePtr& river_piece, const std::string& pushed_player, int rows, int cols, const std::vector<int>& score_cols) const;

    // --- Strategy & Helpers ---
    std::string get_game_phase(const Board& board, int rows, int cols, const std::vector<int>& score_cols) const;
    int count_stones_in_score_area(const Board& board, const std::string& player, int rows, int cols, const std::vector<int>& score_cols) const;
    bool is_winning_move(const Board& board, const Move& move, const std::string& player, int rows, int cols, const std::vector<int>& score_cols);
    std::optional<Move> find_blocking_move(const Board& board, const std::vector<Move>& my_moves, const Move& opp_winning_move, int rows, int cols, const std::vector<int>& score_cols);
    Board apply_move(const Board& board, const Move& move, const std::string& player);

public:
    StudentAgent(std::string p) : BaseAgent(p), turn_count(0) {}

    std::optional<Move> choose(
        const Board& board, int rows, int cols, const std::vector<int>& score_cols,
        double current_player_time, double opponent_time) override;
};

#endif // STUDENT_AGENT_H