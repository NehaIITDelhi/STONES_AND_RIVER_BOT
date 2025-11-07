#!/bin/bash
# Simple Weight Testing Script for Rivers & Stones Bot

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
GAMES=50
BOARD_SIZE="small"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--games)
            GAMES="$2"
            shift 2
            ;;
        -b|--board-size)
            BOARD_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-g games] [-b board-size]"
            echo "  -g, --games       Number of games to play (default: 50)"
            echo "  -b, --board-size  Board size: small/medium/large (default: small)"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "  Rivers & Stones Bot Tester"
echo "========================================"
echo "Games: $GAMES"
echo "Board Size: $BOARD_SIZE"
echo "========================================"
echo ""

# Initialize counters
wins=0
draws=0
losses=0
total_score=0
games_completed=0

# Run games
for i in $(seq 1 $GAMES); do
    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "${YELLOW}Progress: $i/$GAMES games...${NC}\r"
    fi
    
    # Run game
    result=$(python3 gameEngine.py --mode aivai --circle student --square random --nogui 2>/dev/null)
    
    # Check if game completed
    if echo "$result" | grep -q "winner"; then
        games_completed=$((games_completed + 1))
        
        # Count wins
        if echo "$result" | grep -q "'winner': 'student'"; then
            wins=$((wins + 1))
        elif echo "$result" | grep -q "'winner': 'draw'"; then
            draws=$((draws + 1))
        else
            losses=$((losses + 1))
        fi
        
        # Extract score (if possible)
        score=$(echo "$result" | grep -oP "'student': \K[\d.]+")
        if [ ! -z "$score" ]; then
            total_score=$(echo "$total_score + $score" | bc)
        fi
    fi
done

echo ""  # New line after progress

# Calculate percentages
if [ $games_completed -gt 0 ]; then
    win_pct=$((wins * 100 / games_completed))
    draw_pct=$((draws * 100 / games_completed))
    loss_pct=$((losses * 100 / games_completed))
    avg_score=$(echo "scale=1; $total_score / $games_completed" | bc)
else
    win_pct=0
    draw_pct=0
    loss_pct=0
    avg_score=0
fi

# Display results
echo ""
echo "========================================"
echo "           RESULTS"
echo "========================================"
echo -e "Games Completed: ${GREEN}$games_completed${NC} / $GAMES"
echo ""
echo -e "Wins:   ${GREEN}$wins${NC} (${win_pct}%)"
echo -e "Draws:  ${YELLOW}$draws${NC} (${draw_pct}%)"
echo -e "Losses: ${RED}$losses${NC} (${loss_pct}%)"
echo ""
echo "Average Score: $avg_score"
echo "========================================"

# Provide recommendation
echo ""
if [ $win_pct -lt 60 ]; then
    echo -e "${RED}⚠️  Win rate below 60% - Bot needs aggressive tuning!${NC}"
    echo "   Recommendation: Use AGGRESSIVE configuration"
elif [ $win_pct -lt 75 ]; then
    echo -e "${YELLOW}⚡ Win rate 60-75% - Bot is good but can improve${NC}"
    echo "   Recommendation: Fine-tune with balanced adjustments"
else
    echo -e "${GREEN}✅ Win rate above 75% - Bot is performing well!${NC}"
    echo "   Recommendation: Optimize river usage for even better results"
fi

if [ $draw_pct -gt 25 ]; then
    echo -e "${YELLOW}⚠️  Draw rate above 25% - Bot is too passive!${NC}"
    echo "   Action: Increase stone_1_away and opponent penalties"
fi

echo ""