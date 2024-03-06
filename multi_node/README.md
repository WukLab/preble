# Data Parallel Request Handling

Forward similar requests to relvant node instead of forwarding to the same node.

TODO:
- [ ] Test different example workloads and managers and forwaring
- [ ] Add a sample Orcale Policy/Regex String Matching Policy
- [ ] Write tests for consistent hashing
    - [ ] In a simulated fashion test the eviction rate between consistent hashing and radix cache
        - [ ] Record the cache/evict rate from each individual runtime
- [ ] Record the model forwarding/decode time split for each request

# Multi Model Handler
Checklist:
- [ ] Write tests for model loading 
- [ ] Profile Model Loading Time
-  [ ] Load runtimes in parallel to reduce cold start time
    - Note: Potentially extract this to the parent model node loder to effeciently load multiple models in parallel
- [ ] Add Sample Interface with multi model loading

# Workload

Example Workload: `python3 example_workload.py`

