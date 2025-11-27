import sc2
from sc2.data import Race
from sc2.main import run_game
from sc2 import maps
from sc2.bot_ai import BotAI

class MinimalBot(BotAI):
    async def on_step(self, iteration: int):
        # Do nothing, just keep the game running
        pass

def main():
    run_game(
        maps.get("AbyssalReefLE"),
        [
            sc2.player.Bot(Race.Protoss, MinimalBot()),
            sc2.player.Computer(Race.Terran, sc2.data.Difficulty.Medium)
        ],
        # SET THIS TO TRUE FOR GUI
        realtime=True,
        # SET THIS TO FALSE to prevent "headless" mode

    )

if __name__ == '__main__':
    main()
