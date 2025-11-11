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

using Board = std::vector<std::vector<std::map<std::string, std::string>>>;

class StudentAgent {
public:
    explicit StudentAgent(std::string player_side) 
        : side(std::move(player_side)), move_count(0) {
        opponent_side = (side == "circle") ? "square" : "circle";
    }

    Move choose(const Board& board, int rows, int cols, const std::vector<int>& score_cols,
                float current_player_time, float opponent_time) {

        // if 1-step move to be in score area, preferentially choose that ('push' moves not considered yet)
        //flips: check for the score area (4 points) if river owned by us is present

        if(side == "circle"){
            int row = top_score_row();
            for(int c: score_cols){
                const auto &p = board[row][c];
                if(!p.empty()){
                    if(p.at("owner")==side && p.at("side")=="river"){
                        Move m;
                        m.action = "flip";
                        m.from_pos = {c, row};
                        m.to_pos = {c, row};
                        m.orientation = "NULL";
                        return m;
                    }
                }
            }
        }
        
        else{
            int row = bottom_score_row(rows);
            for(int c: score_cols){
                const auto &p = board[row][c];
                if(!p.empty()){
                    if(p.at("owner")==side && p.at("side")=="river"){
                        Move m;
                        m.action = "flip";
                        m.from_pos = {c, row};
                        m.to_pos = {c, row};
                        m.orientation = "NULL";
                        return m;
                    }
                }
            }
        }

        //moves and pushes
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if(!is_own_score_cell(x, y, side, score_cols, rows)){
                    const auto& piece = board[y][x];

                    if(!piece.empty()){
                        if (piece.at("owner") == side && piece.at("side") == "stone") {
                            
                            // Generate all possible moves for this stone
                            auto one_step_moves = compute_valid_moves_for_piece(board, x, y, side, rows, cols, score_cols);

                            for (const auto& move : one_step_moves) {
                                // move
                                if (move.action == "move") {
                                    if (is_own_score_cell(move.to_pos[0], move.to_pos[1], side, score_cols, rows)) {
                                        return move; // Preferentially take this move
                                    }
                                }
                                // push
                                else if (move.action == "push") {
                                    const auto& pushed_piece = board[move.to_pos[1]][move.to_pos[0]];
                                    // Ensure the piece being pushed is also our own stone
                                    if(!pushed_piece.empty()){
                                        if (pushed_piece.at("owner") == side && pushed_piece.at("side") == "stone") {
                                            if (is_own_score_cell(move.pushed_to[0], move.pushed_to[1], side, score_cols, rows)) {
                                                return move; // Preferentially take this push
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // In the score area row, our horizontal river would help
        int score_row = (side == "circle") ? top_score_row() : bottom_score_row(rows);

        for (int x = 0; x < cols; ++x) {
            if(std::find(score_cols.begin(), score_cols.end(), x) == score_cols.end()){
                const auto& piece = board[score_row][x];
                if (!piece.empty() && piece.at("owner") == side) {
                    if (piece.at("side") == "stone") {
                        Board temp_board = board;
                        temp_board[score_row][x]["side"] = "river";
                        temp_board[score_row][x]["orientation"] = "horizontal";
                        auto flow = river_flow(temp_board, x, score_row, x, score_row, side, score_cols);
                        bool is_safe = true;
                        for (const auto& dest : flow) {
                            if (is_opponent_score_cell(dest.first, dest.second, side, score_cols, rows)) {
                                is_safe = false;
                                break;
                            }
                        }
                        if (is_safe){
                            Move m;
                            m.action = "flip";
                            m.from_pos = {x, score_row};
                            m.to_pos = {};
                            m.orientation = "horizontal";
                            return m;
                        } 
                    }
                    else if (piece.at("side") == "river" && piece.at("orientation") == "vertical") {
                        Board temp_board = board;
                        temp_board[score_row][x]["orientation"] = "horizontal";
                        auto flow = river_flow(temp_board, x, score_row, x, score_row, side, score_cols);
                        bool is_safe = true;
                        for (const auto& dest : flow) {
                            if (is_opponent_score_cell(dest.first, dest.second, side, score_cols, rows)) {
                                is_safe = false;
                                break;
                            }
                        }
                        if (is_safe){
                            Move m;
                            m.action = "rotate";
                            m.from_pos = {x, score_row};
                            m.to_pos = {};
                            m.orientation = "";
                            return m;
                        } 
                    }
                }
            }
        }

        // river-stone moves
        for (int x : score_cols) {
            // Check if the current cell contains one of our stones
            const auto& stone_piece = board[score_row][x];
            if (!stone_piece.empty() && stone_piece.at("owner") == side && stone_piece.at("side") == "stone") {
                
                // Check left neighbor for a horizontal river
                int left_neighbor_x = x - 1;
                if (in_bounds(left_neighbor_x, score_row, rows, cols)) {
                    const auto& river_piece = board[score_row][left_neighbor_x];
                    if (!river_piece.empty() && river_piece.at("owner") == side &&
                        river_piece.at("side") == "river" && river_piece.at("orientation") == "horizontal") {
                        
                        // Check if the spot to the right of the stone is an empty scoring cell
                        int push_dest_x = x + 1;
                        if (is_own_score_cell(push_dest_x, score_row, side, score_cols, rows) && board[score_row][push_dest_x].empty()) {
                            // A river piece can push a stone
                            return {"push", {left_neighbor_x, score_row}, {x, score_row}, {push_dest_x, score_row}, ""};
                        }
                    }
                }

                // Check right neighbor for a horizontal river
                int right_neighbor_x = x + 1;
                if (in_bounds(right_neighbor_x, score_row, rows, cols)) {
                    const auto& river_piece = board[score_row][right_neighbor_x];
                    if (!river_piece.empty() && river_piece.at("owner") == side &&
                        river_piece.at("side") == "river" && river_piece.at("orientation") == "horizontal") {
                        
                        // Check if the spot to the left of the stone is an empty scoring cell
                        int push_dest_x = x - 1;
                        if (is_own_score_cell(push_dest_x, score_row, side, score_cols, rows) && board[score_row][push_dest_x].empty()) {
                            return {"push", {right_neighbor_x, score_row}, {x, score_row}, {push_dest_x, score_row}, ""};
                        }
                    }
                }
            }
        }
        
        // Calculate cutoff depth
        int time_component = (int)(0.9 * (1-current_player_time));
        int move_component = (int)(0.1 * std::max(0, 500 - move_count));
        int depth1 = (time_component + move_component) / 60; 
        int depth = int(std::max(1, std::min(depth1, 3)));
        if (current_player_time > 0.8) depth = 1;

        // Alpha-Beta search
        Move best_move;
        double best_score = std::numeric_limits<double>::lowest();
        
        auto legal_moves = generate_all_valid_moves(board, side, score_cols);
        
        for (const auto& move : legal_moves) {
            Board next_board = apply_move(board, move);
            double score = alpha_beta_search(next_board, depth - 1, std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max(), false, rows, cols, score_cols);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        move_count++;
        
        if (best_move.action.empty() && !legal_moves.empty()){
            return legal_moves[0]; // random move if search fails
        }

        return best_move;
    }

private:
    std::string side;
    std::string opponent_side;
    int move_count;

    bool in_bounds(int x, int y, int rows, int cols) {
       return x >= 0 && x < cols && y >= 0 && y < rows;
    }

    int top_score_row(){
        // """Get the row index for Circle's scoring area."""
        return 2;
    }

    int bottom_score_row(int rows){
        // """Get the row index for Square's scoring area."""
        return rows - 3;
    }

    bool is_opponent_score_cell(int x, int y, const std::string &player,const std::vector<int> &score_cols, int rows) {
        if (player== "circle") {
                return (y==rows-3) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        } else {
                return (y== 2) && (std::find(score_cols.begin(),score_cols.end(),x) != score_cols.end());
        }
    }

    bool is_own_score_cell(int x, int y, const std::string &player,const std::vector<int> &score_cols, int rows) {
        if (player== "circle") {
                return (y==2) && (std::find(score_cols.begin(), score_cols.end(), x) != score_cols.end());
        } else {
                return (y==rows-3) && (std::find(score_cols.begin(),score_cols.end(),x) != score_cols.end());
        }
    }

    std::string check_win(const Board& board, int rows, int cols, std::vector<int> score_cols){
        int top = 2; int bot = rows-3;
        int ccount=0; int scount=0;

        for (int x: score_cols){
            if(in_bounds(x, top, rows, cols)){
                auto p = board[top][x]; 
                if (!p.empty() && p.at("owner")=="circle" && p.at("side")=="stone") ccount+=1;}
            if(in_bounds(x, bot, rows, cols)){
                auto q = board[bot][x];
                if (!q.empty() && q.at("owner")=="square" && q.at("side")=="stone") scount+=1;}
        }

        if (ccount >= 4) return "circle";
        if (scount >= 4) return "square";

        return "None";
    }

    double alpha_beta_search(const Board& board, int depth, double alpha, double beta, bool maximizing_player,
                             int rows, int cols, const std::vector<int>& score_cols) {
        
        if (depth == 0 || (check_win(board, rows, cols, score_cols) == "circle" || check_win(board, rows, cols, score_cols) == "square")) { 
            return evaluate_board(board, rows, cols, score_cols);
        }

        if (maximizing_player) {
            double max_eval = std::numeric_limits<double>::lowest();
            auto legal_moves = generate_all_valid_moves(board, side, score_cols);
            for (const auto& move : legal_moves) {
                Board next_board = apply_move(board, move);
                double eval = alpha_beta_search(next_board, depth - 1, alpha, beta, false, rows, cols, score_cols);
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) {
                    break; // Prune
                }
            }
            return max_eval;

        } else { // Minimizing player
            double min_eval = std::numeric_limits<double>::max();
            auto legal_moves = generate_all_valid_moves(board, opponent_side, score_cols);
            for (const auto& move : legal_moves) {
                Board next_board = apply_move(board, move);
                double eval = alpha_beta_search(next_board, depth - 1, alpha, beta, true, rows, cols, score_cols);
                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) {
                    break; // Prune
                }
            }
            return min_eval;
        }
    }

    double evaluate_board(const Board& board, int rows, int cols, const std::vector<int>& score_cols) {
        // 1. Initialize score for both players
        double my_score = 0.0;
        double opponent_score = 0.0;

        // Define weights for heuristics
        const double W_DISTANCE = 2.0;
        const double W_BLOCKING = 1.0;
        const double W_RIVER = 0.1;
        const double W_BACK_ROW = 0.8;
        const double W_INPLACE = 10.0;
        const double W_WIN = 1000.0;

        // Check for a terminal win/loss state first for a decisive evaluation
        std::string winner = check_win(board, rows, cols, score_cols);
        if (winner == side) return W_WIN;
        if (winner == opponent_side) return -W_WIN;

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const auto& cell = board[y][x];
                if (cell.empty() || cell.find("owner") == cell.end()) continue;

                const std::string& owner = cell.at("owner");
                
                if (owner == side) {
                    int my_score_row = (side == "circle") ? 2 : rows - 3;
                    
                    int min_dist = 1000;
                    for (int sx : score_cols) {
                        min_dist = std::min(min_dist, std::abs(x - sx) + std::abs(y - my_score_row));
                    }
                    my_score -= W_DISTANCE * min_dist; // Penalize distance

                    if (cell.at("side") == "stone") {
                        if ((side == "circle" && y < 2) || (side == "square" && y > rows - 3)) {
                            my_score += W_BACK_ROW;
                        }
                    } else if (cell.at("side") == "river") {
                        auto flow = river_flow(board, x, y, x, y, side, score_cols);
                        my_score += W_RIVER * flow.size();
                    }
                    
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            if (is_opponent_score_cell(x + dx, y + dy, side, score_cols, rows)) {
                                my_score += W_BLOCKING;
                            }
                        }
                    }

                    if (is_own_score_cell(x, y, side, score_cols, rows)) {
                        my_score += W_INPLACE;
                    }

                } else if (owner == opponent_side) {
                    // same for opponent's score
                    int opponent_score_row = (opponent_side == "circle") ? 2 : rows - 3;

                    int min_dist = 1000;
                    for (int sx : score_cols) {
                        min_dist = std::min(min_dist, std::abs(x - sx) + std::abs(y - opponent_score_row));
                    }
                    opponent_score -= W_DISTANCE * min_dist; // Penalize their distance

                    if (cell.at("side") == "stone") {
                        if ((opponent_side == "circle" && y < 2) || (opponent_side == "square" && y > rows - 3)) {
                            opponent_score += W_BACK_ROW;
                        }
                    } else if (cell.at("side") == "river") {
                        auto flow = river_flow(board, x, y, x, y, opponent_side, score_cols);
                        opponent_score += W_RIVER * flow.size();
                    }

                    if (is_own_score_cell(x, y, opponent_side, score_cols, rows)) {
                        opponent_score += W_INPLACE;
                    }
                }
            }
        }

        // final score is the difference between our score and the opponent's.
        return my_score - opponent_score;
    }
        
    //river flow simulation
    std::vector<std::pair<int, int>> river_flow(
        const Board &board,
        int rx, int ry, int sx, int sy, const std::string &player,
        const std::vector<int> &score_cols, bool river_push = false, bool avoid_own_score_cells = false) {

        std::vector<std::pair<int, int>> destinations;
        std::set<std::pair<int, int>> visited;
        std::vector<std::pair<int, int>> queue = {{rx, ry}};

        if (board.empty() || board[0].empty()) return {};
        int rows = board.size();
        int cols = board[0].size();

        while (!queue.empty()) {
            auto [x, y] = queue.back();
            queue.pop_back();

            if (!in_bounds(x, y, rows, cols)) continue;
            if (avoid_own_score_cells && is_own_score_cell(x, y, player, score_cols, rows)) {
            continue;
            }
            if (visited.find({x, y}) != visited.end()) continue; // visited check moved earlier
            visited.insert({x, y});

            const auto &cell = board[y][x];

            std::vector<std::pair<int,int>> dirs;
            if (river_push && x == rx && y == ry) {
                const auto &pushing_piece = board[sy][sx];
                if (pushing_piece.empty() || pushing_piece.at("side") != "river") continue;

                dirs = (pushing_piece.at("orientation")=="horizontal") ? std::vector<std::pair<int,int>>{{1,0},{-1,0}}
                                                                        : std::vector<std::pair<int,int>>{{0,1},{0,-1}};
            } else {
                if (cell.empty() || cell.at("side") != "river") {
                    if (!is_opponent_score_cell(x, y, player, score_cols, rows)) destinations.push_back({x,y});
                    continue;
                }
                dirs = (cell.at("orientation")=="horizontal") ? std::vector<std::pair<int,int>>{{1,0},{-1,0}}
                                                               : std::vector<std::pair<int,int>>{{0,1},{0,-1}};
            }

            for (auto [dx,dy] : dirs) {
                int nx = x + dx;
                int ny = y + dy;

                while (in_bounds(nx, ny, rows, cols)) {
                    if (is_opponent_score_cell(nx, ny, player, score_cols, rows)) break;

                    if (avoid_own_score_cells && is_own_score_cell(nx, ny, player, score_cols, rows)) {
                    break;
                    }

                    const auto &next_cell = board[ny][nx];

                    if (next_cell.empty()) {
                        destinations.push_back({nx, ny});
                    } else if (next_cell.find("side") != next_cell.end() && next_cell.at("side") == "river") {
                        queue.push_back({nx, ny});
                        break; 
                    } else {
                        // If it's a stone or anything else, the flow stops here. Do not add it as a destination.
                        break;
                    }

                    nx += dx;
                    ny += dy;
                }
            }
        }

        std::set<std::pair<int,int>> unique_dests(destinations.begin(), destinations.end());
        return std::vector<std::pair<int,int>>(unique_dests.begin(), unique_dests.end());
    }


    // generate valid moves for a single piece 
    std::vector<Move> compute_valid_moves_for_piece(
        const Board &board,
        int sx, int sy, const std::string &player,
        int rows, int cols, const std::vector<int> &score_cols) {
        
        std::vector<Move> moves;
        if (!in_bounds(sx, sy, rows, cols)) return moves;
        
        const auto &piece = board[sy][sx];
        if (piece.empty() || piece.at("owner") != player) return moves;
        
        std::string piece_side = piece.at("side");
        
        // normal adjacent moves 
        std::vector<std::pair<int, int>> dirs = {{1,0}, {-1,0}, {0, 1}, {0, -1}};
        
        for (auto [dx, dy] : dirs) {
                int tx = sx +dx, ty = sy +dy;
                if (!in_bounds(tx, ty, rows, cols)) continue;
                if (is_opponent_score_cell(tx, ty, player, score_cols, rows)) continue;
                
                const auto &target = board[ty][tx];
                
                // case 1: empty cell - simple move
                if (board[ty][tx].empty()) {
                        Move m;
                        m.action = "move";
                        m.from_pos = {sx, sy};
                        m.to_pos = {tx, ty};
                        moves.push_back(m);
                }
                // case 2: stepping into river - river flow
                else if (target.at("side") == "river") {
                        auto flow = river_flow(board, tx, ty, sx, sy, player, score_cols);
                        for (auto &dest : flow) {
                                Move m;
                                m.action = "move";
                                m.from_pos = {sx, sy};
                                m.to_pos = {dest.first, dest.second};
                                moves.push_back(m);
                        }
                }
                // case 3: Pushes
                else {
                    if (target.find("side") == target.end() || target.at("side") != "stone") continue; // ensure the target being pushed is a stone 

                        if (piece_side == "stone") {
                                // stone pushing stone
                                int px = tx + dx, py = ty + dy;
                                if (in_bounds(px, py, rows, cols) && board[py][px].empty() &&
                                    !is_opponent_score_cell(px, py, player, score_cols, rows)) {

                                    // Check if the target is an opponent piece and the destination is our own score cell
                                    bool is_invalid_push = (target.at("owner") == opponent_side && 
                                                            is_own_score_cell(px, py, player, score_cols, rows));
                                    if (is_invalid_push) {
                                        continue; // Skip this invalid move
                                    }

                                    Move m;
                                    m.action = "push";
                                    m.from_pos = {sx, sy};
                                    m.to_pos = {tx, ty};
                                    m.pushed_to = {px, py};
                                    moves.push_back(m);
                                }
                        } else if (piece_side == "river") {
                                // river pushing stone
                                // First, determine if we are pushing an opponent's piece.
                            bool is_pushing_opponent = (target.at("owner") == opponent_side);
                            
                            // Call river_flow with the new flag set to true if we're pushing an opponent.
                            auto flow = river_flow(board, tx, ty, sx, sy, player, score_cols, true, is_pushing_opponent);

                            for (auto &dest : flow) {
                                if (!is_opponent_score_cell(dest.first, dest.second, player, score_cols, rows) && board[dest.second][dest.first].empty()) {
                                    // The check for the final destination is still required for non-river pushes
                                    // and as a redundant safety measure.
                                    if (is_pushing_opponent && is_own_score_cell(dest.first, dest.second, player, score_cols, rows)) {
                                        continue;
                                    }

                                    Move m;
                                    m.action = "push";
                                    m.from_pos = {sx, sy};
                                    m.to_pos = {tx, ty};
                                    m.pushed_to = {dest.first, dest.second};
                                    moves.push_back(m);
                                }
                            }
                        }
                }
        }
        
        // flips
        if (piece_side == "stone") {
            if(!is_own_score_cell(sx, sy, player, score_cols, rows)){
                for (std::string orientation : {"horizontal", "vertical"}) {
                        // check if flip is safe (won't allow flow into opponent score)
                        bool safe = true;

                        // vertical river check for unblocked opponent score
                        if (orientation == "vertical") {   
                            for (int c : score_cols) {     
                                int start_row = std::min(sy, (player=="square"?2:rows-3));
                                int end_row   = std::max(sy, (player=="square"?2:rows-3)); 
                                bool blocked = false; 
                                for (int r = start_row + 1; r < end_row; ++r) { 
                                    if (!board[r][sx].empty()) { blocked = true; break; } 
                                }
                                if (!blocked) { 
                                    safe = false; 
                                    break; 
                                }
                            }
                        }
                            
                        // create a temporary board to test the flip
                        auto temp_board = board;
                        temp_board[sy][sx]["side"] = "river";
                        temp_board[sy][sx]["orientation"] = orientation;
                        
                        auto flow = river_flow(temp_board, sx, sy, sx, sy, player, score_cols);
                        safe = !flow.empty();
                        for (auto &dest : flow) {
                                if (is_opponent_score_cell(dest.first, dest.second, player, score_cols, rows)) {
                                    safe = false;
                                    break;
                                }
                        }
                        if (safe) {
                                Move m;
                                m.action = "flip";
                                m.from_pos = {sx, sy};
                                m.to_pos = {sx, sy};
                                m.orientation = orientation;
                                moves.push_back(m);
                        }
                }
            }
        } 
        else {
                // river to stone flip (this is always allowed)
                Move m;
                m.action = "flip";
                m.from_pos = {sx, sy};
                m.to_pos = {sx, sy};
                moves.push_back(m);
        }
        
        // rotations
        if (piece_side == "river") {
                std::string new_orientation = (piece.at("orientation") == "horizontal") ? "vertical" : "horizontal";
                
                // check if rotation is safe
                bool safe = true;
                
                // create a temporary board to test the rotation
                auto temp_board = board;
                temp_board[sy][sx]["orientation"] = new_orientation;
                
                auto flow = river_flow(temp_board, sx, sy, sx, sy, player, score_cols);
                for (auto &dest : flow) {
                        if (is_opponent_score_cell(dest.first, dest.second, player, score_cols, rows)) {
                                safe = false;
                                break;
                        }
                }
                if (safe) {
                        Move m;
                        m.action = "rotate";
                        m.from_pos = {sx, sy};
                        m.to_pos = {sx, sy};
                        moves.push_back(m);
                }
        }
        
        return moves;
    }

    // generate all moves for player
    std::vector<Move> generate_all_valid_moves(
        const Board &board,
        const std::string &player, const std::vector<int> &score_cols) {
        
        std::vector<Move> all_moves;
        if (board.empty() || board[0].empty()) return {}; //check empty board
        int rows = board.size();
        int cols = board[0].size();
        
        for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                        if (board[y][x].empty()) continue;
                        if (board[y][x].at("owner") != player) continue;
                        
                        auto piece_moves = compute_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols);
                        all_moves.insert(all_moves.end(), piece_moves.begin(), piece_moves.end());
                }
        }
        return all_moves;
    }

    Board apply_move(Board board, const Move& move) {
        // the move action
        if (move.action == "move") {
            int fx = move.from_pos[0];
            int fy = move.from_pos[1];
            int tx = move.to_pos[0];
            int ty = move.to_pos[1];

            // simple move
            board[ty][tx] = board[fy][fx];
            board[fy][fx].clear();  // make previous cell empty
            }

        // the flip action
        else if(move.action == "flip") {
            int fx = move.from_pos[0], fy = move.from_pos[1];
            if(board[fy][fx].at("side")=="river"){
                board[fy][fx].at("side") = "stone";
                board[fy][fx].at("orientation") = "NULL";
            }
            else if(board[fy][fx].at("side")=="stone"){
                board[fy][fx].at("side") = "river";

                if(move.orientation!="horizontal" && move.orientation!="vertical"){
                    board[fy][fx].at("side") = "stone";
                    return board;
                }
                board[fy][fx].at("orientation") = move.orientation;
            }
        }

        // the rotate action
        else if(move.action == "rotate") {
            int fx = move.from_pos[0], fy = move.from_pos[1];
    
            std::string old_orientation = board[fy][fx].at("orientation");
            if(old_orientation == "horizontal") {
                board[fy][fx]["orientation"] = "vertical";
            } else {
                board[fy][fx]["orientation"] = "horizontal";
            }
        }

        // the push action
        else if(move.action == "push") {
            int fx = move.from_pos[0];
            int fy = move.from_pos[1];
            int tx = move.to_pos[0];
            int ty = move.to_pos[1];
            int px = move.pushed_to[0];
            int py = move.pushed_to[1];

            auto pushed_piece = board[ty][tx];
            board[ty][tx] = board[fy][fx];
            board[fy][fx].clear();
            board[py][px] = pushed_piece;

            }

        return board;
    }
};

// PyBind11 bindings to expose the C++ classes and methods to Python
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
