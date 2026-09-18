import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.abspath(os.path.expanduser(os.environ.get("BK_DATA_DIR") or os.path.join(ROOT, os.pardir, "sound-symbolism-data")))
HUMAN = os.environ.get("BK_HUMAN_DIR") or os.path.join(DATA, "human")
LACEY = os.path.join(HUMAN, "lacey")
ACOUSTIC = os.path.join(HUMAN, "lacey_acoustic")
CWIEK = os.path.join(HUMAN, "cwiek", "web_by_trial.csv")

RESP = os.environ.get("BK_RESP_DIR") or os.path.join(ROOT, "responses_csv")
OUT = os.environ.get("BK_OUT_DIR") or os.path.join(ROOT, "results", "tables")
