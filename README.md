# Retail Recommendation Engine (Hybrid)

Overview
--------
Hybrid recommender combining collaborative filtering (item-item CF + SVD) and
content-based recommendations (TF-IDF on titles/descriptions + category features).

How to run (fast/sampled mode)
------------------------------
1. Create virtualenv and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run Phase 1 (explore):
   python notebooks/data_exploration.py

3. Run Phase 2 (preprocess):
   python notebooks/data_preprocessing.py

4. Run Phase 3 (collaborative filtering):
   python notebooks/collaborative_filtering.py

5. Run Phase 5 (content-based - sampled mode):
   python notebooks/content_based_filtering.py

6. Generate recommendations:
   python notebooks/recommendation_generation.py

Notes
-----
- Large data files are excluded. Add samples to `data/raw/` before running on local machine.
- For full-scale training use cloud (Colab or a VM) due to memory/time requirements.

## Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt