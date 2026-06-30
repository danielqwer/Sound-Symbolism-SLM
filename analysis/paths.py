import os

# Edit data locations here (or set the BK_* env vars); everything else is relative.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HUMAN = os.environ.get("BK_HUMAN_DIR") or os.path.join(ROOT, "data", "human")
LACEY = os.path.join(HUMAN, "lacey")              # Lacey 2020: combo / image / pseudoword .mat
ACOUSTIC = os.path.join(HUMAN, "lacey_acoustic")  # pseudoword wavs, voiceReportData.xlsx, dsm_*.npy
CWIEK = os.path.join(HUMAN, "cwiek", "web_by_trial.csv")  # Cwiek 2022 web experiment

RESP = os.environ.get("BK_RESP_DIR") or os.path.join(ROOT, "responses_csv")  # judge-parsed outputs
OUT = os.environ.get("BK_OUT_DIR") or os.path.join(ROOT, "report", "tables")  # table output
