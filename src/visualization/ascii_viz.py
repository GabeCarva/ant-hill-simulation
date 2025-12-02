"""ASCII visualization for terminal display."""

from typing import Optional
from src.core.game import Game, GameState
from src.core.entities import Rock, Food, Ant, Anthill
from src.utils.game_config import Position


class ASCIIVisualizer:
    """Simple ASCII renderer for game state."""
    
    # Character mappings
    SYMBOLS = {
        'empty': '.',
        'rock': '#',
        'food': '*',
        'anthill_0': 'H',  # Player 0 anthill
        'anthill_1': 'h',  # Player 1 anthill
        'ant_0': 'A',      # Player 0 ants
        'ant_1': 'a',      # Player 1 ants
    }
    
    # Colors for terminal (ANSI escape codes)
    COLORS = {
        'rock': '\033[90m',      # Dark gray
        'food': '\033[93m',      # Yellow
        'anthill_0': '\033[94m',  # Blue
        'anthill_1': '\033[91m',  # Red
        'ant_0': '\033[96m',      # Cyan
        'ant_1': '\033[95m',      # Magenta
        'reset': '\033[0m'
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
    
    def render(self, game: Game, clear_screen: bool = True) -> str:
        """Render current game state as ASCII string."""
        state = game.get_state()
        board = state.board
        
        # Build the display
        lines = []
        
        if clear_screen:
            lines.append('\033[2J\033[H')  # Clear screen and move cursor to top
        
        # Header
        lines.append(f"Turn: {state.turn}")
        lines.append(f"Player 0 (Blue/Cyan): Ants={len(board.get_ants_by_player(0))}, "
                    f"Food={state.food_collected[0]}, Lost={state.ants_lost[0]}")
        lines.append(f"Player 1 (Red/Magenta): Ants={len(board.get_ants_by_player(1))}, "
                    f"Food={state.food_collected[1]}, Lost={state.ants_lost[1]}")
        
        if state.winner is not None:
            if state.winner == -1:
                lines.append("*** DRAW ***")
            else:
                lines.append(f"*** PLAYER {state.winner} WINS! ***")
        
        lines.append("-" * min(board.width + 2, 80))
        
        # Board rendering
        for y in range(board.height):
            row = []
            for x in range(board.width):
                pos = Position(x, y)
                char, color = self._get_display_char(board, pos)
                
                if self.use_colors and color:
                    row.append(f"{color}{char}{self.COLORS['reset']}")
                else:
                    row.append(char)
            
            lines.append(''.join(row))
        
        lines.append("-" * min(board.width + 2, 80))
        
        return '\n'.join(lines)
    
    def _get_display_char(self, board, pos: Position) -> tuple[str, Optional[str]]:
        """Get display character and color for a position."""
        entity = board.get_entity_at(pos)
        
        if entity is None:
            return self.SYMBOLS['empty'], None
        
        if isinstance(entity, Rock):
            return self.SYMBOLS['rock'], self.COLORS.get('rock')
        elif isinstance(entity, Food):
            return self.SYMBOLS['food'], self.COLORS.get('food')
        elif isinstance(entity, Anthill):
            key = f'anthill_{entity.player_id}'
            return self.SYMBOLS[key], self.COLORS.get(key)
        elif isinstance(entity, Ant):
            key = f'ant_{entity.player_id}'
            return self.SYMBOLS[key], self.COLORS.get(key)
        
        return '?', None
    
    def render_to_file(self, game: Game, filename: str):
        """Save ASCII representation to file."""
        content = self.render(game, clear_screen=False)
        # Remove ANSI color codes for file output
        import re
        clean_content = re.sub(r'\033\[[0-9;]+m', '', content)
        
        with open(filename, 'w') as f:
            f.write(clean_content)