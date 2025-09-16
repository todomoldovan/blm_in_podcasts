This folder contains the code to perform preprocessing steps on the Structured Podcast Research Corpus.

The script `make_db.py` creates a database from `../data/episodeLevelData.jsonl`, `break_by_sentence.ipynb` creates a sentence-level csv file for each episode in the database, `filter_episodes.ipynb` perfoms the keyword search on the database and filters out episodes without expressions of collective action.
