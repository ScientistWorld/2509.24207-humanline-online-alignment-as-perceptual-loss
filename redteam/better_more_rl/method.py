from pathlib import Path
import sys
from redteam.train_variant import main

sys.argv = [__file__, str(Path(__file__).with_name("config.json"))]
main()
