#!/bin/bash
# Simple Bot Tester - macOS Compatible

# How many games to run
GAMES=${1:-50}

echo "========================================"
echo "  Testing Bot - $GAMES games"
echo "========================================"

wins=0
draws=0
losses=0

for i in $(seq 1 $GAMES); do
    # Show progress every 10 games
    if [ $((i % 10)) -eq 0 ]; then
        echo "Progress: $i/$GAMES..."
    fi
    
    # Run game and capture result
    result=$(python3 gameEngine.py --mode aivai --circle student --square random --nogui 2>&1)
    
    # Count results
    if echo "$result" | grep -q "'winner': 'student'"; then
        wins=$((wins + 1))
    elif echo "$result" | grep -q "'winner': 'draw'"; then
        draws=$((draws + 1))
    elif echo "$result" | grep -q "'winner': 'random'"; then
        losses=$((losses + 1))
    fi
done

# Calculate percentages
win_pct=$((wins * 100 / GAMES))
draw_pct=$((draws * 100 / GAMES))
loss_pct=$((losses * 100 / GAMES))

# Show results
echo ""
echo "========================================"
echo "           RESULTS"
echo "========================================"
echo "Games: $GAMES"
echo ""
echo "Wins:   $wins ($win_pct%)"
echo "Draws:  $draws ($draw_pct%)"
echo "Losses: $losses ($loss_pct%)"
echo "========================================"
echo ""

# Give recommendation
if [ $win_pct -lt 60 ]; then
    echo "⚠️  Win rate below 60% - Use AGGRESSIVE config"
elif [ $win_pct -lt 75 ]; then
    echo "⚡ Win rate 60-75% - Fine-tune weights"
else
    echo "✅ Win rate above 75% - Excellent!"
fi

if [ $draw_pct -gt 25 ]; then
    echo "⚠️  Too many draws - Make bot more aggressive"
fi

echo ""