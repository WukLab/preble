REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be 13 types: 
(1) Calculate[formula], which calculates the formula and returns the result.
(2) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(3) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(4) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(5) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(6) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(7) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(8) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(9) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(10) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(11) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(12) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(13) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""