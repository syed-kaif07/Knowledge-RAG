# System Stability and Engineering Log

Daily log to record issues observed, root causes identified, fixes applied, and lessons learned while hardening this RAG system.

Currently : As i'm using free NVIDIA models, usually model gets removed on a regular basis from their server's. so working on different dimension embedding models/generation models makes a bit vary in the results. (currently used 2048 dim model)

Database : Currently the vectors are stored in the MongoDB Vector Search. (can be changed, facing retrieval latency!) 

Evals: Currently built offline evals, context is measured/evaluated on faithfulness, answer relevancy, context utilization. (will move to online evals soon.)

-------------------------------------

17-Nov-2026 - "update embedding model from nvidia/llama-nemotron-embed-vl-1b-v2 to nemotron-3.5-embed-1b"

1. Issue

used to get streamlit long error message's when api/server error occurs from NVIDIA

2. Fixed  

handled the error by showing showing short message with clean and minimal UI.
It is fixed on both 400/500 error's

3. Result

clean, readable and understandable error for user.

file added -> error_handler.py

-------------------------------------

18-Nov-2026 - "hybrid search backsoff to BM25 with RRF, no failure 503/429 errors"(service unavailable and too many requests errors)

1. Issue

Usually if service is unavailable from any API provider or too many requests happened then the model would use to stop the generation.

2. Fixed

No stopping of generation if API fails from any api server.

3. Result

Will attempt to retreive and generate 3 times if fails 3 times then model will use local BM25 keyword search instead of crashing.

4. Drawbacks 

Less content ore varied generation results. (observed)

modified : retrieval.py, generation.py, error_handler.py



